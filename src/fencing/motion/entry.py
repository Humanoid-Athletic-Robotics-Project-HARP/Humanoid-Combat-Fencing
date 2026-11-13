"""Entry motion: from upstream's standby pose into a target motion's first frame.

On the real G1 every policy is switched in from motion_tracking_controller's standby controller,
which stands the robot feet together in its `default_position`. No Goal 1 motion starts anywhere
near that pose (docs/mtc.md, "Start from standby"). An entry reference bridges the two: it starts
exactly in the standby pose, feet flat on the floor, and ends exactly on the target's first frame,
held. A policy trained on it can be switched in from standby with almost no offset, and the motion
policy can then be switched in from the entry's hold, again on its own first frame (docs/mtc.md,
"Entry from standby").

The movement in between is built to be something the G1 can do, in the same way as
`fencing.motion.synthetic`: soles placed by leg IK (`LegIK`), one foot moving at a time with a
lifted swing, and the pelvis moved so that the centre of mass is over the stance foot while the
other foot is in the air (a weight shift, not feet sliding along the floor). Not fully static: in
the last 4 frames (0.13 s) of the back-foot step the CoM is up to 5 cm outside the stance sole
while the foot is still 0.6-2.8 cm up (see the release comment in build_entry):

    hold standby -> shift weight onto the left foot -> step the right (front) foot into place ->
    shift weight onto the right foot -> step the left (back) foot into place -> settle into the
    target pelvis pose -> hold the target frame

Waist and arms blend from the standby pose to the target over the whole transition (min-jerk).
The pelvis keeps the target's heading throughout, so the start faces the way the target pelvis
faces and only the feet turn.

Joint limits: nothing goes beyond a limit, but both ankle rolls sit exactly on their +-0.2618 rad
limit for part of the published entry (frames 97-126 left, 69-71 and 114-120 right; the stance
ankle as well as the swing one). The quality check counts values beyond a limit, not on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from fencing import paths
from fencing.motion.filters import smoothstep
from fencing.motion.ik import LegIK
from fencing.motion.io import JOINT_NAMES, Motion
from fencing.motion.kinematics import G1Model, foot_point_world, load_g1

LEFT, RIGHT = 0, 1
# The controller config real.launch.py loads (`config/<robot_type>/controllers.yaml`), read-only.
MTC_CONTROLLERS_YAML = paths.THIRD_PARTY / "motion_tracking_controller" / "config" / "g1" / "controllers.yaml"
# Leg IK misses allowed: a planted sole must stay put; a swing foot in the air may run onto a joint
# limit (the ankle roll's +-0.26 rad) and land its sole a little off the planned arc. The stance
# ankle roll also sits on its limit for a while (see the module docstring).
PLANTED_TOLERANCE_M = 2e-3
SWING_TOLERANCE_M = 1e-2
STEP_OF = {0: "step_left", 1: "step_right"}
# Fraction of a step at each end in which the swing foot only rises / lowers (no horizontal move).
SWING_EDGE_FRACTION = 0.3


def standby_joint_positions(config: Path = MTC_CONTROLLERS_YAML) -> np.ndarray:
    """standby_controller's default_position, in JOINT_NAMES order (matched by joint name)."""
    params = yaml.safe_load(Path(config).read_text())["standby_controller"]["ros__parameters"]
    by_name = dict(zip(params["joint_names"], params["default_position"], strict=True))
    missing = [n for n in JOINT_NAMES if n not in by_name]
    if missing:
        raise KeyError(f"{config}: standby_controller has no default_position for {missing}")
    return np.array([float(by_name[n]) for n in JOINT_NAMES])


@dataclass(frozen=True)
class EntryConfig:
    fps: float = 30.0
    start_hold_s: float = 1.0
    shift_s: float = 0.8  # weight onto the left (back) foot
    step_s: float = 0.8  # each step
    transfer_s: float = 0.8  # weight from the left foot onto the right foot
    settle_s: float = 0.8  # into the target pelvis pose
    end_hold_s: float = 2.5  # the target frame, held (>= the eval's 2 s end hold, docs/pipeline.md)
    # Swing foot clearance while it moves: above the contact detector's 5 cm exit height
    # (fencing.motion.contacts.ContactParams), so the quality check sees a step, not a slide.
    lift_m: float = 0.06
    # During single support the CoM goes to the point of the stance sole nearest the other foot,
    # this far inside the sole's capsule axes (which span -5.4..13.2 cm along the foot and only
    # +-2.6 cm across it; the sole edge is 1 cm further out). Nearest the other foot, not the sole
    # centre: the far leg then leans least, which keeps its ankle roll (limit +-0.26 rad) free.
    support_margin_m: float = 0.015
    # Waist and arms blend standby -> target from the start of the first phase to the end of the second.
    upper_body_phases: tuple[str, str] = ("shift", "settle")
    com_release_fraction: float = 0.25
    com_release_end_s: float = 0.3


@dataclass
class EntryResult:
    motion: Motion
    phases: list[tuple[str, int, int]]  # (label, start frame, end frame) half-open
    report: dict = field(default_factory=dict)

    def phase(self, label: str) -> tuple[int, int]:
        for name, start, end in self.phases:
            if name == label:
                return start, end
        raise KeyError(f"no phase {label!r}; phases: {[p[0] for p in self.phases]}")


class _Pose:
    """MuJoCo kinematics on the training G1, for poses given as (root pos, root quat xyzw, joints)."""

    def __init__(self, g1: G1Model):
        self.g1 = g1
        self.data = mujoco.MjData(g1.model)

    def qpos(self, root_pos, root_quat_xyzw, joints) -> np.ndarray:
        q = np.zeros(self.g1.model.nq)
        q[:3] = root_pos
        q[3], q[4:7] = root_quat_xyzw[3], root_quat_xyzw[:3]
        q[self.g1.qpos_index] = joints
        return q

    def run(self, qpos) -> mujoco.MjData:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.g1.model, self.data)
        mujoco.mj_comPos(self.g1.model, self.data)
        return self.data

    def feet(self, qpos):
        d = self.run(qpos)
        pos = np.array([foot_point_world(self.g1, d, f) for f in (LEFT, RIGHT)])
        rot = np.array([d.xmat[b].reshape(3, 3).copy() for b in self.g1.foot_body_ids])
        return pos, rot

    def com(self, qpos) -> np.ndarray:
        return self.run(qpos).subtree_com[self.g1.pelvis_body_id].copy()


def _yaw(quat_xyzw) -> float:
    return float(Rotation.from_quat(quat_xyzw).as_euler("ZYX")[0])


def standby_frame(target: Motion, frame: int = 0, joints: np.ndarray | None = None, g1: G1Model | None = None):
    """The standby pose standing flat on the floor, facing the target pelvis's heading, with the
    midpoint of its soles on the midpoint of the target frame's soles. Returns (root pos, root quat
    xyzw, joints)."""
    g1 = g1 or load_g1()
    joints = standby_joint_positions() if joints is None else np.asarray(joints, float)
    pose = _Pose(g1)
    # Feet flat: the pelvis orientation that turns the (symmetric) standby foot back to level,
    # then the target's heading on top.
    _, rot = pose.feet(pose.qpos(np.zeros(3), np.array([0, 0, 0, 1.0]), joints))
    # Pelvis = Rz(heading) Ry(pitch) Rx(roll), with roll and pitch chosen so that the sole's normal
    # (expressed in the pelvis frame) ends up vertical: the zyx yaw stays exactly the target's.
    normal = rot[LEFT][:, 2]
    roll = np.arctan2(normal[1], normal[2])
    pitch = -np.arctan2(normal[0], np.hypot(normal[1], normal[2]))
    root_quat = Rotation.from_euler("ZYX", [_yaw(target.root_quat[frame]), pitch, roll]).as_quat()
    pos, _ = pose.feet(pose.qpos(np.zeros(3), root_quat, joints))
    target_feet, _ = pose.feet(pose.qpos(target.root_pos[frame], target.root_quat[frame], target.joint_pos[frame]))
    lowest = min(_lowest_sole_z(pose, pose.qpos(np.zeros(3), root_quat, joints)))
    root_pos = np.zeros(3)
    root_pos[:2] = target_feet[:, :2].mean(axis=0) - pos[:, :2].mean(axis=0)
    root_pos[2] = -lowest
    return root_pos, root_quat, joints


def _lowest_sole_z(pose: _Pose, qpos) -> list[float]:
    d = pose.run(qpos)
    out = []
    for bid, foot in zip(pose.g1.foot_body_ids, pose.g1.feet):
        pts = foot.capsule_endpoints @ d.xmat[bid].reshape(3, 3).T + d.xpos[bid]
        out.append(float(pts[:, 2].min() - foot.radius))
    return out


def _sole_polygon(g1: G1Model, data: mujoco.MjData, foot: int) -> np.ndarray:
    bid = g1.foot_body_ids[foot]
    pts = g1.feet[foot].capsule_endpoints @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
    return pts[:, :2]


def support_margin(motion: Motion, stance: int, frames, g1: G1Model | None = None) -> np.ndarray:
    """How far (m) the centre of mass lies inside the stance foot's sole polygon (the convex hull
    of its sole capsule axes, ignoring their radius), per frame; negative = outside."""
    g1 = g1 or load_g1()
    pose = _Pose(g1)
    out = []
    for t in frames:
        q = pose.qpos(motion.root_pos[t], motion.root_quat[t], motion.joint_pos[t])
        com = pose.com(q)[:2]
        hull = ConvexHull(_sole_polygon(g1, pose.data, stance))
        out.append(float(-(hull.equations[:, :2] @ com + hull.equations[:, 2]).max()))
    return np.array(out)


def support_point(pose: _Pose, qpos, foot: int, toward_xy: np.ndarray, margin_m: float,
                  resolution_m: float = 0.001) -> np.ndarray:
    """The point at least margin_m inside `foot`'s sole polygon (in pose qpos) nearest toward_xy."""
    pose.run(qpos)
    hull = ConvexHull(_sole_polygon(pose.g1, pose.data, foot))
    pts = hull.points[hull.vertices]
    xs = np.arange(pts[:, 0].min(), pts[:, 0].max() + resolution_m, resolution_m)
    ys = np.arange(pts[:, 1].min(), pts[:, 1].max() + resolution_m, resolution_m)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    inside = grid[(grid @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1) <= -margin_m]
    if not len(inside):
        raise ValueError(f"foot {foot}'s sole has no point {margin_m} m inside its edges")
    return inside[np.argmin(np.linalg.norm(inside - toward_xy, axis=1))]


def _sole_centre_xy(pose: _Pose, qpos, foot: int) -> np.ndarray:
    pose.run(qpos)
    hull_pts = _sole_polygon(pose.g1, pose.data, foot)
    return hull_pts.mean(axis=0)


def build_entry(target: Motion, frame: int = 0, config: EntryConfig = EntryConfig(), g1: G1Model | None = None,
                standby_joints: np.ndarray | None = None) -> EntryResult:
    """Standby pose -> the target's frame `frame`, held. See the module docstring."""
    if abs(target.fps - config.fps) > 1e-9:
        raise ValueError(f"target is {target.fps} fps, the entry is built at {config.fps}")
    g1 = g1 or load_g1()
    pose = _Pose(g1)
    fps = config.fps
    s_pos, s_quat, s_joints = standby_frame(target, frame, standby_joints, g1)
    e_pos, e_quat, e_joints = target.root_pos[frame].copy(), target.root_quat[frame].copy(), target.joint_pos[frame].copy()
    q_start = pose.qpos(s_pos, s_quat, s_joints)
    q_end = pose.qpos(e_pos, e_quat, e_joints)
    feet0, rot0 = pose.feet(q_start)
    feet1, rot1 = pose.feet(q_end)
    com0, com1 = pose.com(q_start), pose.com(q_end)
    sole0 = [_sole_centre_xy(pose, q_start, f) for f in (LEFT, RIGHT)]
    sole1 = [_sole_centre_xy(pose, q_end, f) for f in (LEFT, RIGHT)]

    labels = ("start_hold", "shift", "step_right", "transfer", "step_left", "settle", "end_hold")
    seconds = (config.start_hold_s, config.shift_s, config.step_s, config.transfer_s, config.step_s,
               config.settle_s, config.end_hold_s)
    lengths = [max(1, int(round(s * fps))) for s in seconds]
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    phases = [(label, int(bounds[i]), int(bounds[i + 1])) for i, label in enumerate(labels)]
    n = int(bounds[-1])
    span = dict((p[0], (p[1], p[2])) for p in phases)
    move_start, move_end = span["shift"][0], span["settle"][1]

    def progress(t, start, end):  # 0 before start, smoothstep, 1 at end - 1 and after
        return float(smoothstep(np.clip(np.array([(t - start + 1) / (end - start)]), 0.0, 1.0))[0])

    # Support points during each single support (see EntryConfig.support_margin_m).
    com_left = support_point(pose, q_start, LEFT, sole0[RIGHT], config.support_margin_m)  # right foot in its standby place
    com_right = support_point(pose, q_end, RIGHT, sole0[LEFT], config.support_margin_m)  # right foot in its final place
    # The en garde stance is wider than the G1's hips, so a foot cannot be set down flat in it
    # while the CoM is still over the other foot (the ankle roll would pass its +-0.26 rad limit):
    # over the last com_release_fraction of each step, as the swing foot is lowered, the CoM heads
    # for its next target (the other sole; after the back-foot step the target frame's CoM, reached
    # com_release_end_s after the landing). The only part of a swing not over the stance foot:
    # at most 4 frames, foot under 3 cm, CoM at most 5.5 cm outside (tests/test_motion_entry.py).
    def release(label):
        a, b = span[label]
        return b - int(round(config.com_release_fraction * (b - a)))
    right_down, left_down = release("step_right"), release("step_left")
    release_end = min(span["settle"][1], span["step_left"][1] + int(round(config.com_release_end_s * fps)))
    feet_rot_slerp = [Slerp([0.0, 1.0], Rotation.from_matrix([rot0[f], rot1[f]])) for f in (LEFT, RIGHT)]
    root_slerp = Slerp([0.0, 1.0], Rotation.from_quat([s_quat, e_quat]))

    root_pos = np.zeros((n, 3))
    root_quat = np.zeros((n, 4))
    joints = np.zeros((n, len(JOINT_NAMES)))
    solver = LegIK(g1, damping=0.01, rotation_weight=0.5)
    leg_cols = np.concatenate([g1.leg_columns["left"], g1.leg_columns["right"]])
    upper_cols = np.setdiff1d(np.arange(len(JOINT_NAMES)), leg_cols)
    previous = s_joints.copy()
    pelvis_xy = s_pos[:2].copy()
    worst_ik = {"planted": 0.0, "swing": 0.0}
    for t in range(n):
        if t < move_start:
            root_pos[t], root_quat[t], joints[t] = s_pos, s_quat, s_joints
            continue
        if t >= move_end:
            root_pos[t], root_quat[t], joints[t] = e_pos, e_quat, e_joints
            continue
        whole = progress(t, move_start, move_end)
        quat = root_slerp(whole).as_quat()
        height = s_pos[2] + whole * (e_pos[2] - s_pos[2])
        q_joints = previous.copy()
        upper = progress(t, span[config.upper_body_phases[0]][0], span[config.upper_body_phases[1]][1])
        q_joints[upper_cols] = s_joints[upper_cols] + upper * (e_joints[upper_cols] - s_joints[upper_cols])

        # Foot targets: each foot moves only during its own step, with a lift arc.
        feet = np.array(feet0)
        rots = [rot0[LEFT], rot0[RIGHT]]
        # The swing is lift straight up, move and turn at full clearance, lower straight down: the
        # foot is never low and moving sideways at once, which would read as a sliding planted foot.
        for foot, label in ((RIGHT, "step_right"), (LEFT, "step_left")):
            a, b = span[label]
            edge = int(round(SWING_EDGE_FRACTION * (b - a)))
            s = progress(t, a + edge, b - edge)
            feet[foot] = feet0[foot] + s * (feet1[foot] - feet0[foot])
            if a <= t < b:
                feet[foot, 2] += config.lift_m * (progress(t, a, a + edge) - progress(t, b - edge, b))
            rots[foot] = feet_rot_slerp[foot](s).as_matrix()

        # Centre of mass target: over the left sole for the right step, over the right sole for
        # the left step, blended in the shift / transfer phases, then released (see release).
        if t < span["shift"][1]:
            com_xy = com0[:2] + progress(t, *span["shift"]) * (com_left - com0[:2])
        elif t < right_down:
            com_xy = com_left
        elif t < span["transfer"][1]:
            com_xy = com_left + progress(t, right_down, span["transfer"][1]) * (com_right - com_left)
        elif t < left_down:
            com_xy = com_right
        else:
            com_xy = com_right + progress(t, left_down, release_end) * (com1[:2] - com_right)

        for _ in range(4):  # pelvis xy <- CoM error, legs re-solved for the new pelvis
            qpos = pose.qpos(np.r_[pelvis_xy, height], quat, q_joints)
            for foot in (LEFT, RIGHT):
                qpos, _ = solver.solve(qpos, foot, feet[foot], rots[foot], iterations=60)
            q_joints = qpos[g1.qpos_index]
            pelvis_xy = pelvis_xy + (com_xy - pose.com(qpos)[:2])
        qpos = pose.qpos(np.r_[pelvis_xy, height], quat, q_joints)
        for foot in (LEFT, RIGHT):
            qpos, err = solver.solve(qpos, foot, feet[foot], rots[foot], iterations=60)
            a, b = span[STEP_OF[foot]]
            planted = not a <= t < b
            worst_ik["planted" if planted else "swing"] = max(worst_ik["planted" if planted else "swing"], err)
            limit = PLANTED_TOLERANCE_M if planted else SWING_TOLERANCE_M
            if err > limit:
                raise RuntimeError(f"entry IK misses the {'planted' if planted else 'swing'} foot {foot} by "
                                   f"{err:.4f} m (> {limit}) at frame {t}: target out of the legs' reach")
        joints[t] = qpos[g1.qpos_index]
        root_pos[t] = np.r_[pelvis_xy, height]
        root_quat[t] = quat
        previous = joints[t]

    # Exactly on the target at the end of the settle: the last moving frame already is (same feet,
    # same pelvis, same upper body), up to the IK tolerance.
    motion = Motion(root_pos, root_quat, joints, fps, {"entry": True})
    gap = float(np.abs(joints[move_end - 1] - e_joints).max())
    report = {
        "phases": [{"label": label, "start_s": a / fps, "end_s": b / fps} for label, a, b in phases],
        "duration_s": n / fps,
        "worst_ik_error_m": worst_ik,  # {"planted": m, "swing": m}
        "com_release_frames": {"step_right": int(right_down), "step_left": int(left_down)},
        "last_moving_frame_to_target_rad": gap,
        "standby_pelvis": {"pos": s_pos.tolist(), "quat_xyzw": s_quat.tolist()},
    }
    return EntryResult(motion, phases, report)
