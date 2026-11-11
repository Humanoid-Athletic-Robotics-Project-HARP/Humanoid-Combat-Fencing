"""Cleanup operations on a Motion. Each is pure: it returns a new Motion plus a small dict
of what it changed, so the pipeline can log every step in the provenance row.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from fencing.motion.contacts import clean_runs, contact_phases, foot_speeds
from fencing.motion.filters import covering_smooth_max, interpolate_gaps, lowpass, smoothstep
from fencing.motion.ik import LegIK
from fencing.motion.io import JOINT_NAMES, Motion
from fencing.motion.kinematics import FKResult, G1Model, forward_kinematics, load_g1
from fencing.motion.normalize import fencing_line, yaw_of

# Unitree G1 actuator velocity limits (rad/s), from mjlab's g1_constants.py ElectricActuator
# specs: 5020 = 37, 7520-14 = 32, 7520-22 = 20, 4010 = 22. Waist roll/pitch and ankles are
# driven through linkages by two 5020s; mjlab assumes 1:1, so 37 nominal.
_ACTUATOR_VELOCITY_LIMIT = {
    "hip_pitch": 32.0, "hip_yaw": 32.0, "waist_yaw": 32.0,
    "hip_roll": 20.0, "knee": 20.0,
    "shoulder_pitch": 37.0, "shoulder_roll": 37.0, "shoulder_yaw": 37.0, "elbow": 37.0, "wrist_roll": 37.0,
    "wrist_pitch": 22.0, "wrist_yaw": 22.0,
    "waist_roll": 37.0, "waist_pitch": 37.0, "ankle_pitch": 37.0, "ankle_roll": 37.0,
}


def hardware_velocity_limits() -> np.ndarray:
    """(29,) motor no-load speed limit per CSV joint, rad/s."""
    limits = []
    for name in JOINT_NAMES:
        key = name.removesuffix("_joint").removeprefix("left_").removeprefix("right_")
        if key not in _ACTUATOR_VELOCITY_LIMIT:
            raise KeyError(f"no velocity limit known for joint {name!r}")
        limits.append(_ACTUATOR_VELOCITY_LIMIT[key])
    return np.array(limits)


# ---------------------------------------------------------------- jitter


def continuous_quaternions(quat: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie in the same hemisphere (q and -q are equal)."""
    quat = quat.copy()
    for t in range(1, quat.shape[0]):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] = -quat[t]
    return quat


def smooth_motion(motion: Motion, cutoff_hz: float = 8.0, root_cutoff_hz: float | None = None) -> tuple[Motion, dict]:
    """Zero-phase low-pass on joints, root position and root orientation.

    Quaternions are made sign-continuous first, then filtered component-wise and
    renormalised: for frame-to-frame rotations of a few degrees this equals filtering on
    the rotation manifold, and it never produces the 2*pi wrap a rotation-vector filter can.
    """
    root_cutoff_hz = root_cutoff_hz or cutoff_hz
    quat = continuous_quaternions(motion.root_quat)
    quat = lowpass(quat, root_cutoff_hz, motion.fps)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    joints = lowpass(motion.joint_pos, cutoff_hz, motion.fps)
    root = lowpass(motion.root_pos, root_cutoff_hz, motion.fps)
    out = motion.replace(root_pos=root, root_quat=quat, joint_pos=joints)
    change = np.abs(joints - motion.joint_pos)
    return out, {"cutoff_hz": cutoff_hz, "max_joint_change_rad": float(change.max()), "mean_joint_change_rad": float(change.mean())}


# ---------------------------------------------------------------- limits


def clamp_joint_limits(motion: Motion, margin_rad: float = 0.02, g1: G1Model | None = None) -> tuple[Motion, dict]:
    g1 = g1 or load_g1()
    lower, upper = g1.joint_lower + margin_rad, g1.joint_upper - margin_rad
    clipped = np.clip(motion.joint_pos, lower, upper)
    violated = clipped != motion.joint_pos
    worst = np.abs(clipped - motion.joint_pos)
    per_joint = {JOINT_NAMES[j]: int(violated[:, j].sum()) for j in np.flatnonzero(violated.any(axis=0))}
    return motion.replace(joint_pos=clipped), {
        "margin_rad": margin_rad,
        "clamped_values": int(violated.sum()),
        "max_violation_rad": float(worst.max()),
        "per_joint_frames": per_joint,
    }


def rate_limit(signal: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    """Zero-phase slew-rate limiter along axis 0.

    A forward-only limiter lags behind fast moves; the average of a forward and a backward
    pass is centred, and still obeys the limit because a convex combination of two
    Lipschitz-bounded signals keeps the same bound.
    """
    forward = signal.copy()
    for t in range(1, signal.shape[0]):
        forward[t] = forward[t - 1] + np.clip(signal[t] - forward[t - 1], -max_step, max_step)
    backward = signal.copy()
    for t in range(signal.shape[0] - 2, -1, -1):
        backward[t] = backward[t + 1] + np.clip(signal[t] - backward[t + 1], -max_step, max_step)
    return 0.5 * (forward + backward)


def clamp_joint_velocity(
    motion: Motion,
    hardware_fraction: float = 0.5,
    max_joint_speed: float | None = None,
) -> tuple[Motion, dict]:
    """Limit every joint to min(hardware_fraction * motor limit, max_joint_speed) rad/s.

    The roadmap's lunge risk: video lunges exceed what the G1 motors can track. Clamping
    (or time_scale first) keeps the reference trackable.
    """
    limits = hardware_velocity_limits() * hardware_fraction
    if max_joint_speed is not None:
        if max_joint_speed <= 0:
            raise ValueError(f"max_joint_speed must be positive, got {max_joint_speed}")
        limits = np.minimum(limits, max_joint_speed)
    before = np.abs(np.diff(motion.joint_pos, axis=0)) * motion.fps
    over = before > limits * (1 + 1e-9)
    if not over.any():
        return motion.copy(), {"limits_rad_s": limits.tolist(), "frames_over": 0, "max_speed_before": float(before.max(initial=0))}
    joints = rate_limit(motion.joint_pos, limits / motion.fps)
    return motion.replace(joint_pos=joints), {
        "limits_rad_s": limits.tolist(),
        "frames_over": int(over.any(axis=1).sum()),
        "joints_over": sorted({JOINT_NAMES[j] for j in np.flatnonzero(over.any(axis=0))}),
        "max_speed_before": float(before.max()),
        "max_joint_change_rad": float(np.abs(joints - motion.joint_pos).max()),
    }


# ---------------------------------------------------------------- time


def time_scale(motion: Motion, factor: float) -> tuple[Motion, dict]:
    """Slow down by `factor` (2.0 = twice as long) at the same fps.

    PCHIP for positions and joints (shape-preserving, so no overshoot past joint limits),
    slerp for the root orientation.
    """
    if factor <= 0:
        raise ValueError(f"time-scale factor must be positive, got {factor}")
    if abs(factor - 1.0) < 1e-9:
        return motion.copy(), {"factor": 1.0}
    n_out = int(round((motion.n_frames - 1) * factor)) + 1
    source_times = np.arange(motion.n_frames) / motion.fps
    sample_times = np.clip(np.arange(n_out) / motion.fps / factor, 0, source_times[-1])
    root = PchipInterpolator(source_times, motion.root_pos, axis=0)(sample_times)
    joints = PchipInterpolator(source_times, motion.joint_pos, axis=0)(sample_times)
    quat = Slerp(source_times, Rotation.from_quat(motion.root_quat))(sample_times).as_quat()
    out = Motion(root, quat, joints, motion.fps, dict(motion.meta))
    return out, {"factor": factor, "frames_before": motion.n_frames, "frames_after": n_out}


# ---------------------------------------------------------------- fencing line


def straighten_line(
    motion: Motion,
    fk: FKResult | None = None,
    lateral_keep: float = 0.25,
    yaw_cutoff_hz: float = 0.5,
    max_turn_deg: float = 60.0,
) -> tuple[Motion, dict]:
    """Put the root path on one straight fencing line and remove slow heading drift.

    Monocular video bends the path and lets the heading wander (depth along the camera axis
    is poorly observed, and the body's yaw follows where it is in the image). Real footwork
    runs along one line with a nearly constant heading, so:
    1. The line is `normalize.fencing_line` (root travel direction, or the feet on the spot).
    2. The slow part (< yaw_cutoff_hz) of the pelvis yaw relative to the line is replaced by
       its clip mean; each frame turns about the vertical through its own pelvis. Fast
       rotations (a hip turn in a lunge) are kept. The mean is not forced onto the feet: the
       feet line and the travel line differ by stance geometry as well as by depth error,
       and the two cannot be told apart here.
    3. The root's ground position is projected onto the line, keeping `lateral_keep` of the
       sideways offset (real sway is small; most of it is depth error).
    Planted feet move with the rotation, so this runs before foot-slide removal, whose root
    shift and leg IK re-plant them. A slow heading change or correction beyond max_turn_deg
    is a real turn (or a broken clip), not drift: then nothing is changed.
    """
    fk = fk or forward_kinematics(motion)
    angle, source = fencing_line(motion, fk)
    u, n = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    rel_xy = motion.root_pos[:, :2] - motion.root_pos[0, :2]
    lateral = rel_xy @ n
    info = {"line_deg": float(np.degrees(angle)), "line_source": source, "lateral_before_m": float(np.ptp(lateral))}

    rel_yaw = np.unwrap(np.angle(np.exp(1j * (yaw_of(motion.root_quat) - angle))))
    slow = lowpass(rel_yaw, yaw_cutoff_hz, motion.fps)
    correction = slow.mean() - slow
    turn, worst = float(np.degrees(np.ptp(slow))), float(np.degrees(np.abs(correction).max()))
    info.update({"slow_yaw_range_deg": turn, "max_correction_deg": worst, "mean_heading_to_line_deg": float(np.degrees(slow.mean()))})
    if turn > max_turn_deg or worst > max_turn_deg:
        return motion.copy(), {**info, "skipped": f"heading change {max(turn, worst):.0f} deg > {max_turn_deg:g}: a real turn, not drift"}

    quat = (Rotation.from_euler("z", correction[:, None]) * Rotation.from_quat(motion.root_quat)).as_quat()
    root = motion.root_pos.copy()
    root[:, :2] = motion.root_pos[0, :2] + np.outer(rel_xy @ u, u) + lateral_keep * np.outer(lateral, n)
    info["lateral_after_m"] = float(np.ptp(lateral_keep * lateral))
    return motion.replace(root_pos=root, root_quat=quat), info


# ---------------------------------------------------------------- vertical


def contact_floor_track(fk: FKResult, contacts: np.ndarray) -> np.ndarray:
    """(T,) height of the floor under the planted feet: lowest point of the lowest contact
    foot, linearly interpolated through frames with no contact (flight).

    Before the first and after the last contact the lower foot is the floor. A clip cut while
    the fencer is still recovering has no contact at its end; holding the last contact height
    there carried a lunge that monocular depth had sunk 10 cm into the recovery and lifted the
    end en garde off the floor.
    """
    lower_foot = fk.foot_lowest_z.min(axis=1)
    heights = np.where(contacts, fk.foot_lowest_z, np.inf).min(axis=1)
    valid = np.isfinite(heights)
    if not valid.any():
        # No detected contact at all: fall back to the lower foot (e.g. a 1 s clip mid-jump).
        return lower_foot
    track = interpolate_gaps(np.where(valid, heights, 0.0), valid)
    first, last = np.flatnonzero(valid)[[0, -1]]
    track[:first] = lower_foot[:first]
    track[last + 1 :] = lower_foot[last + 1 :]
    return track


def remove_vertical_drift(
    motion: Motion, contacts: np.ndarray, fk: FKResult | None = None, cutoff_hz: float = 0.3
) -> tuple[Motion, dict]:
    """Remove slow vertical drift of the whole body that the legs do not explain.

    Monocular pose estimation lets the root float up and down; the legs do not follow, so
    planted feet sink or hover. That shows up as the slow part of the planted-foot height
    track. The track's mean is kept (absolute height is align_to_ground's job), so on clean
    data this step is a no-op. A real pelvis drop (lunge) bends the legs and leaves the
    planted feet where they are, so it is not touched.
    """
    fk = fk or forward_kinematics(motion)
    track = contact_floor_track(fk, contacts)
    drift = lowpass(track, cutoff_hz, motion.fps)
    drift -= drift.mean()
    root = motion.root_pos.copy()
    root[:, 2] -= drift
    return motion.replace(root_pos=root), {"cutoff_hz": cutoff_hz, "drift_range_m": float(np.ptp(drift))}


def align_to_ground(
    motion: Motion, contacts: np.ndarray, fk: FKResult | None = None, smoothing_hz: float = 1.5
) -> tuple[Motion, dict]:
    """Put the lowest planted sole point on z = 0, with a smoothed per-frame correction."""
    fk = fk or forward_kinematics(motion)
    track = contact_floor_track(fk, contacts)
    correction = lowpass(track, smoothing_hz, motion.fps)
    root = motion.root_pos.copy()
    root[:, 2] -= correction
    return motion.replace(root_pos=root), {
        "smoothing_hz": smoothing_hz,
        "mean_correction_m": float(correction.mean()),
        "correction_range_m": float(np.ptp(correction)),
    }


def remove_floor_penetration(
    motion: Motion, fk: FKResult | None = None, tolerance_m: float = 0.0, ramp_s: float = 0.15
) -> tuple[Motion, dict]:
    """Lift the root wherever a sole point is below the floor, smoothly (no steps)."""
    fk = fk or forward_kinematics(motion)
    depth = np.maximum(0.0, -fk.foot_lowest_z.min(axis=1) - tolerance_m)
    if not depth.any():
        return motion.copy(), {"max_depth_m": 0.0, "frames_lifted": 0}
    lift = covering_smooth_max(depth, int(round(ramp_s * motion.fps)))
    root = motion.root_pos.copy()
    root[:, 2] += lift
    return motion.replace(root_pos=root), {"max_depth_m": float(depth.max()), "frames_lifted": int((lift > 1e-6).sum())}


# ---------------------------------------------------------------- foot slide


def _slide_corrections(xy: np.ndarray, contacts: np.ndarray, anchors: list[list], blend_frames: int):
    """Per-foot corrections (T, 2, D) and weights (T, 2) that pin each contact phase (D = 2 for
    xy; plant_still_feet passes D = 3).

    Inside a phase the correction moves the foot onto the phase anchor. In `blend_frames`
    before and after it the boundary correction fades out with a smoothstep, so the swing
    trajectory bends smoothly into the pinned position instead of jumping.
    """
    n = xy.shape[0]
    corrections = np.zeros(xy.shape)
    weights = np.zeros((n, 2))
    for f in range(2):
        phases = contact_phases(contacts[:, f])
        for i, ((start, end), anchor) in enumerate(zip(phases, anchors[f])):
            corrections[start:end, f] = anchor - xy[start:end, f]
            weights[start:end, f] = 1.0
            gap_before = start - (phases[i - 1][1] if i > 0 else -10**9)
            gap_after = (phases[i + 1][0] if i + 1 < len(phases) else 10**9) - end
            for k in range(1, min(blend_frames, gap_before // 2) + 1):
                t = start - k
                if t < 0:
                    break
                w = smoothstep(1.0 - k / (blend_frames + 1))
                corrections[t, f] = w * (anchor - xy[start, f])
                weights[t, f] = w
            for k in range(1, min(blend_frames, gap_after // 2) + 1):
                t = end - 1 + k
                if t >= n:
                    break
                w = smoothstep(1.0 - k / (blend_frames + 1))
                corrections[t, f] = w * (anchor - xy[end - 1, f])
                weights[t, f] = w
    return corrections, weights


def remove_foot_slide(
    motion: Motion,
    contacts: np.ndarray,
    blend_s: float = 0.15,
    use_root: bool = True,
    use_ik: bool = True,
    ik_passes: int = 2,
    g1: G1Model | None = None,
) -> tuple[Motion, dict]:
    """Pin every planted foot in the world for the whole contact phase.

    The anchor is the median sole-centre position over the phase (median: robust to the
    first and last frames, where the foot is still landing or already peeling off).
    1. Root translation absorbs the part of the slide shared by the planted feet (the usual
       video artefact: the whole body glides). With one foot planted it absorbs all of it.
    2. Leg IK (6 joints, foot orientation held) removes what is left, e.g. two planted feet
       sliding apart. The upper body is untouched by this step.
    """
    g1 = g1 or load_g1()
    fk = forward_kinematics(motion, g1)
    blend_frames = max(1, int(round(blend_s * motion.fps)))
    anchors = [
        [np.median(fk.sole_center[s:e, f, :2], axis=0) for s, e in contact_phases(contacts[:, f])] for f in range(2)
    ]
    before = _phase_slip(fk.sole_center[..., :2], contacts)
    out = motion.copy()
    root_shift = np.zeros((motion.n_frames, 2))
    if use_root:
        corrections, weights = _slide_corrections(fk.sole_center[..., :2], contacts, anchors, blend_frames)
        # Denominator max(sum, 1): a lone fading foot moves the root by w * c, two full-weight
        # feet by their average -- continuous in every case.
        root_shift = (weights[..., None] * corrections).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)[:, None]
        out.root_pos[:, :2] += root_shift
    ik_frames, worst_ik_error = 0, 0.0
    if use_ik:
        solver = LegIK(g1)
        for _ in range(ik_passes):
            fk = forward_kinematics(out, g1)
            corrections, weights = _slide_corrections(fk.sole_center[..., :2], contacts, anchors, blend_frames)
            qpos = g1.qpos_from_motion(out)
            foot_rot = fk.foot_rot
            for t, f in zip(*np.nonzero(np.linalg.norm(corrections, axis=-1) > 5e-4)):
                target = fk.sole_center[t, f].copy()
                target[:2] += corrections[t, f]
                qpos[t], err = solver.solve(qpos[t], int(f), target, foot_rot[t, f])
                worst_ik_error = max(worst_ik_error, err)
                ik_frames += 1
            out.joint_pos = qpos[:, g1.qpos_index]
    after = _phase_slip(forward_kinematics(out, g1).sole_center[..., :2], contacts)
    return out, {
        "blend_s": blend_s,
        "contact_phases": [len(a) for a in anchors],
        "slip_before_m": before,
        "slip_after_m": after,
        "max_root_shift_m": float(np.linalg.norm(root_shift, axis=1).max(initial=0.0)),
        "ik_solves": ik_frames,
        "worst_ik_error_m": worst_ik_error,
    }


def plant_still_feet(
    motion: Motion,
    hover_m: float = 0.02,
    still_mps: float = 0.35,
    speed_smoothing_hz: float = 2.0,
    min_duration_s: float = 0.3,
    blend_s: float = 0.15,
    g1: G1Model | None = None,
) -> tuple[Motion, dict]:
    """Put a foot that stands still above the floor onto it, pinned at its median xy.

    Video retargets can leave a planted foot hovering: in the held lunges of aY1P3wzvNBM and
    53FZTK_8qOI the front knee came out ~0.4 rad over-bent, so the front sole floated 10-15 cm
    for a whole second while wobbling a few cm sideways; a clip cut while the front foot is
    still landing holds it in the air through the end hold. Contact detection needs the sole
    within a few cm of the floor, so it never sees such a foot, and the slide fix pins only xy.
    A foot whose horizontal speed (smoothed at 2 Hz, so wobble does not count) stays under
    `still_mps` for `min_duration_s` while above `hover_m` is planted; swing feet move far
    faster. Leg IK only (as in remove_foot_slide). The floor is z = 0, so it runs after ground
    alignment; the edge blends are clamped so they never push a sole below the floor.
    """
    g1 = g1 or load_g1()
    fk = forward_kinematics(motion, g1)
    speeds = foot_speeds(fk, motion.fps, speed_smoothing_hz)
    min_length = max(1, int(round(min_duration_s * motion.fps)))
    max_gap = int(round(0.07 * motion.fps))  # same bridging as contact detection
    hovering = np.zeros((motion.n_frames, 2), dtype=bool)
    for f in range(2):
        still_above = (speeds[:, f] < still_mps) & (fk.foot_lowest_z[:, f] > hover_m)
        hovering[:, f] = clean_runs(still_above, min_length, max_gap)
    phases = [contact_phases(hovering[:, f]) for f in range(2)]
    info = {"planted_phases": [len(p) for p in phases], "max_lowered_m": 0.0, "ik_solves": 0, "worst_ik_error_m": 0.0}
    if not hovering.any():
        return motion.copy(), info
    # Sole centre xy plus the lowest sole point's height: pinning that to (median xy, 0)
    # puts the foot on the floor where it stood.
    points = np.concatenate([fk.sole_center[..., :2], fk.foot_lowest_z[..., None]], axis=-1)
    anchors = [[np.append(np.median(points[s:e, f, :2], axis=0), 0.0) for s, e in phases[f]] for f in range(2)]
    corrections, _ = _slide_corrections(points, hovering, anchors, max(1, int(round(blend_s * motion.fps))))
    corrections[..., 2] = np.maximum(corrections[..., 2], np.minimum(0.0, -fk.foot_lowest_z))  # lower, never lift
    solver = LegIK(g1)
    qpos = g1.qpos_from_motion(motion)
    for t, f in zip(*np.nonzero(np.linalg.norm(corrections, axis=-1) > 5e-4)):
        qpos[t], err = solver.solve(qpos[t], int(f), fk.sole_center[t, f] + corrections[t, f], fk.foot_rot[t, f])
        info["worst_ik_error_m"] = max(info["worst_ik_error_m"], err)
        info["ik_solves"] += 1
    out = motion.copy()
    out.joint_pos = qpos[:, g1.qpos_index]
    info["max_lowered_m"] = float(-corrections[..., 2].min())
    return out, info


def set_foot_heading(
    motion: Motion,
    foot_index: int,
    target_yaw_rad: float,
    step_deg: float = 10.0,
    g1: G1Model | None = None,
) -> tuple[Motion, dict]:
    """Turn one foot about the vertical through its sole centre until it points along
    `target_yaw_rad`, in every frame.

    Monocular pose gets leg twist worst of all: en_garde_v1's front foot came out 47 deg off
    the fencing line, which a saber fencer points it down, and every policy trained on that
    base settled further into the twist (right hip yaw +0.28 rad, ankle roll near its limit).
    Leg IK only (6 joints, as in remove_foot_slide): the sole stays where it was, tilt is
    kept, the rest of the body is untouched. The IK's orientation error is small-angle, so
    large turns are walked in `step_deg` increments. A heading the leg cannot reach is left
    short and reported as `worst_heading_error_deg`.
    """
    g1 = g1 or load_g1()
    fk = forward_kinematics(motion, g1)
    turn = np.angle(np.exp(1j * (target_yaw_rad - fk.foot_yaw[:, foot_index])))
    solver = LegIK(g1)
    qpos = g1.qpos_from_motion(motion)
    worst = 0.0
    for t in range(motion.n_frames):
        steps = max(1, int(np.ceil(abs(np.degrees(turn[t])) / step_deg)))
        for k in range(1, steps + 1):
            target_rot = Rotation.from_euler("z", turn[t] * k / steps).as_matrix() @ fk.foot_rot[t, foot_index]
            qpos[t], err = solver.solve(qpos[t], foot_index, fk.sole_center[t, foot_index], target_rot, iterations=40)
        worst = max(worst, err)
    out = motion.copy()
    out.joint_pos = qpos[:, g1.qpos_index]
    left_over = np.angle(np.exp(1j * (target_yaw_rad - forward_kinematics(out, g1).foot_yaw[:, foot_index])))
    return out, {"foot_index": foot_index, "target_yaw_deg": float(np.degrees(target_yaw_rad)),
                 "max_turn_deg": float(np.degrees(np.abs(turn).max())), "worst_ik_error_m": worst,
                 "worst_heading_error_deg": float(np.degrees(np.abs(left_over).max()))}


def neutralize_wrists(motion: Motion, sides: tuple[str, ...] = ("left", "right"),
                      axes: tuple[str, ...] = ("pitch", "yaw"), target_rad: float = 0.0) -> tuple[Motion, dict]:
    """Set wrist joints the source never measured to a constant.

    Video pose (MediaPipe / GVHMR body) has no wrist rotation, so the retargeted wrist pitch and
    yaw are an IK by-product. Their 5 Nm motors have an action scale of 0.075 rad, so angles far
    from the default pose (0) need actions of 10+ and are not learned; they only add joint error.
    Roll is left alone: it has the forearm motor's wider scale. Do not use on mocap or on a
    keyframed sword arm, where the wrist angles are real.
    """
    unknown = [side for side in sides if side not in ("left", "right")]
    if unknown:
        raise ValueError(f"sides must be 'left' or 'right', got {unknown}")
    joints = [f"{side}_wrist_{axis}_joint" for side in sides for axis in axes]
    columns = [JOINT_NAMES.index(joint) for joint in joints]
    joint_pos = motion.joint_pos.copy()
    change = np.abs(joint_pos[:, columns] - target_rad)
    joint_pos[:, columns] = target_rad
    return motion.replace(joint_pos=joint_pos), {
        "joints": joints,
        "target_rad": target_rad,
        "max_change_rad": float(change.max()) if change.size else 0.0,
    }


ARM_BODY_KEYS = ("shoulder", "elbow", "wrist", "hand", "rubber")


def clear_arm_torso_collision(
    motion: Motion,
    side: str,
    margin_m: float = 0.005,
    step_rad: float = 0.01,
    blend_s: float = 0.15,
    g1: G1Model | None = None,
) -> tuple[Motion, dict]:
    """Abduct one shoulder (roll only) just far enough that the arm clears the rest of the body.

    lunge_v2's sword arm came out of the oblique-front retarget hanging adducted inside the
    torso (up to 3.7 cm) for the whole clip; the robot cannot follow that and the policy
    missed the joint gate. Per frame the roll is stepped outward until no arm geom is within
    `margin_m` of a non-arm body; the per-frame offsets are then smoothed over `blend_s` by
    covering_smooth_max, which can only raise them, so the arm never moves back in; the
    result is clipped to the joint range (only where the arm is already at its outward limit,
    so the clip never puts it back into the body). Frames the roll cannot clear even at the
    limit are counted in `frames_still_touching`.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    g1 = g1 or load_g1()
    model = copy.deepcopy(g1.model)
    model.geom_margin[:] = margin_m  # report pairs closer than the margin, not only overlaps
    data = mujoco.MjData(model)
    arm = np.array([any(k in (n or "") for k in ARM_BODY_KEYS) and (n or "").startswith(f"{side}_")
                    for n in g1.body_names])
    column = JOINT_NAMES.index(f"{side}_shoulder_roll_joint")
    qadr = g1.qpos_index[column]
    outward = -1.0 if side == "right" else 1.0
    limit = g1.joint_lower[column] if side == "right" else g1.joint_upper[column]
    qpos = g1.qpos_from_motion(motion)

    def touching(q) -> bool:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if b1 and b2 and arm[b1] != arm[b2] and c.dist < margin_m:
                return True
        return False

    need = np.zeros(motion.n_frames)
    stuck = 0
    for t in range(motion.n_frames):
        q = qpos[t].copy()
        while touching(q):
            if outward * (limit - q[qadr]) <= step_rad:
                stuck += 1
                break
            q[qadr] += outward * step_rad
            need[t] += step_rad
    offset = covering_smooth_max(need, max(1, int(round(blend_s * motion.fps / 2))))
    out = motion.copy()
    out.joint_pos[:, column] = np.clip(out.joint_pos[:, column] + outward * offset,
                                       g1.joint_lower[column], g1.joint_upper[column])
    return out, {"side": side, "frames_fixed": int((need > 0).sum()), "frames_still_touching": stuck,
                 "max_offset_rad": float(offset.max())}


def _phase_slip(xy: np.ndarray, contacts: np.ndarray) -> float:
    """Largest distance any planted foot strays from where its contact phase began."""
    worst = 0.0
    for f in range(2):
        for s, e in contact_phases(contacts[:, f]):
            worst = max(worst, float(np.linalg.norm(xy[s:e, f] - xy[s, f], axis=1).max()))
    return worst
