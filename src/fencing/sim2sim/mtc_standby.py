"""Start from standby: score the switch from upstream's standby controller to the policy.

On the real G1 the operator stands the robot in standby_controller (`L1 + A`) and then presses
`R1 + A`, which asks the controller manager to activate walking_controller and deactivate
standby (unitree_bringup config/g1/joy.yaml). `scripts/mtc_sim2sim.sh --standby` reproduces that
in motion_tracking_controller's MuJoCo sim (docs/mtc.md, "Start from standby"). This module holds
what the host decides from such a recording:

- where the simulator holds the robot until standby takes it (`standing_qpos`);
- whether the robot was standing still under standby at the switch (`standby_state_report`),
  without which the run does not count;
- per-step joint error against the reference, and when the policy converged onto it
  (`step_joint_errors`, `convergence_step`);
- how hard the transition drove the joints, as a fraction of their limits (`peak_fraction`).

None of this changes how `fencing.sim2sim.mtc` scores a run: the tracking metrics, thresholds and
verdict are the same code. These are extra numbers about the transition.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

# A run counts only if, over the last STILL_WINDOW_S before the switch, the robot was under
# standby_controller and standing still. Standby ramps to its pose in 2 s (StandbyController.h,
# totalTime_ 2.0), so it must have been active for that ramp plus the window. The motion limits
# are a still, upright robot, far below what a robot still settling shows (the pilot run swayed
# at 4-7 cm/s 0.5-1 s after the ramp and 1.7 cm/s 3 s after it, docs/mtc.md).
STILL_WINDOW_S = 1.0
# The window ends half a policy step before the switch: velocities are central differences, and
# the last sample before the switch would otherwise also see the first one the policy moved.
STILL_WINDOW_GAP_S = 0.010
STANDBY_RAMP_S = 2.0
STILL_MAX_PELVIS_SPEED_M_S = 0.01
STILL_MAX_PELVIS_HEIGHT_RANGE_M = 0.002
STILL_MAX_PELVIS_TILT_RAD = 0.2
STILL_MAX_JOINT_SPEED_RAD_S = 0.05
# Converged: per-step joint error (mean |q - q_ref| over joints, the joint_angle_error_rad
# definition) under the line for the whole window. The line is 1.5x the 0.10 rad threshold:
# the policies run at 0.07-0.09 rad, so this is "on the reference", not "passing".
CONVERGED_JOINT_ERROR_RAD = 0.15
CONVERGED_WINDOW_S = 0.5
AT_LIMIT_FRACTION = 1.0 - 1e-6  # the simulator clamps torque at the limit; this is "at the clamp"

_EXACT_LOWEST_POINT = ("sphere", "capsule", "cylinder", "box")


def _geom_type_name(model, geom: int) -> str:
    import mujoco

    return mujoco.mjtGeom(int(model.geom_type[geom])).name.removeprefix("mjGEOM_").lower()


def _lowest_point(model, data, geom: int) -> float:
    kind = _geom_type_name(model, geom)
    centre, rot, size = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3), model.geom_size[geom]
    axis_z = abs(rot[2, 2])  # vertical component of the geom's local z axis
    if kind == "sphere":
        return float(centre[2] - size[0])
    if kind == "capsule":
        return float(centre[2] - axis_z * size[1] - size[0])
    if kind == "cylinder":
        return float(centre[2] - axis_z * size[1] - size[0] * np.sqrt(max(0.0, 1.0 - axis_z ** 2)))
    if kind == "box":
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * size
        return float((centre + corners @ rot.T)[:, 2].min())
    raise ValueError(f"geom {geom} on body {model.body(int(model.geom_bodyid[geom])).name} is a {kind}: "
                     f"standing_qpos bounds only {', '.join(_EXACT_LOWEST_POINT)} exactly")


def standing_qpos(model) -> np.ndarray:
    """The model's default pose (qpos0: unitree_description's straight-legged G1) moved
    vertically so that its lowest colliding point touches z = 0.

    The simulator holds the robot here until standby_controller's first command
    (scripts/mtc/hcf_mtc_hold). It must not be the standby pose itself: standby starts its ramp
    from the measured pose, so a robot held exactly in the standby pose gets zero torque and the
    hold would never release. Ramping from the model default is what upstream's simulator start
    would do if standby were spawned active."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    lowest = min(_lowest_point(model, data, g) for g in range(model.ngeom)
                 if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    free = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qpos = model.qpos0.copy()
    qpos[int(model.jnt_qposadr[free]) + 2] -= lowest
    return qpos


def _standby_active_s(activity: np.ndarray, switch_t: float) -> float:
    """How long standby_controller had been active, with walking_controller inactive, at the
    switch (0 when it was not the active controller just before it)."""
    rows = activity[activity[:, 0] < switch_t - 1e-9] if len(activity) else activity
    since = None
    for stamp, standby, walking in rows:
        if standby and not walking:
            since = stamp if since is None else since
        else:
            since = None
    return 0.0 if since is None else float(switch_t - since)


def standby_state_report(activity, base_t, base_pos, base_quat, base_lin_vel, joint_t, joint_vel, switch_t: float,
                         window_s: float = STILL_WINDOW_S) -> dict:
    """Was the robot standing still under standby_controller in the window before the switch?

    activity: rows (stamp, standby active, walking active) of /controller_manager/activity.
    base_*: ground-truth pelvis track (fencing.sim2sim.mtc.Replayer.base_track)."""
    end = switch_t - STILL_WINDOW_GAP_S
    in_window = (base_t >= end - window_s) & (base_t < end)
    joints_in_window = (joint_t >= end - window_s) & (joint_t < end)
    if in_window.sum() < 2 or joints_in_window.sum() < 2:
        return {"still": False, "reasons": [f"fewer than 2 samples in the {window_s} s before the switch"]}
    height = base_pos[in_window, 2]
    quat = base_quat[in_window]
    tilt = 2 * np.arccos(np.clip(np.linalg.norm(quat[:, [0, 3]], axis=1), 0, 1))
    report = {
        "window_s": window_s,
        "window_ends_before_switch_s": STILL_WINDOW_GAP_S,
        "standby_active_s": _standby_active_s(np.asarray(activity, float).reshape(-1, 3), switch_t),
        "pelvis_height_m": float(height.mean()),
        "pelvis_height_range_m": float(np.ptp(height)),
        "pelvis_speed_max_m_s": float(np.linalg.norm(base_lin_vel[in_window], axis=1).max()),
        "pelvis_tilt_max_rad": float(tilt.max()),
        "joint_speed_max_rad_s": float(np.abs(joint_vel[joints_in_window]).max()),
        "limits": {"standby_active_s": STANDBY_RAMP_S + window_s, "pelvis_speed_max_m_s": STILL_MAX_PELVIS_SPEED_M_S,
                   "pelvis_height_range_m": STILL_MAX_PELVIS_HEIGHT_RANGE_M,
                   "pelvis_tilt_max_rad": STILL_MAX_PELVIS_TILT_RAD,
                   "joint_speed_max_rad_s": STILL_MAX_JOINT_SPEED_RAD_S},
    }
    reasons = []
    if report["standby_active_s"] < STANDBY_RAMP_S + window_s:
        reasons.append(f"standby_controller active {report['standby_active_s']:.2f} s before the switch "
                       f"(needs its {STANDBY_RAMP_S} s ramp plus the {window_s} s window)")
    if report["pelvis_speed_max_m_s"] > STILL_MAX_PELVIS_SPEED_M_S:
        reasons.append(f"pelvis speed up to {report['pelvis_speed_max_m_s']:.4f} m/s "
                       f"(limit {STILL_MAX_PELVIS_SPEED_M_S})")
    if report["pelvis_height_range_m"] > STILL_MAX_PELVIS_HEIGHT_RANGE_M:
        reasons.append(f"pelvis height moved {report['pelvis_height_range_m'] * 1000:.1f} mm "
                       f"(limit {STILL_MAX_PELVIS_HEIGHT_RANGE_M * 1000:.0f} mm)")
    if report["pelvis_tilt_max_rad"] > STILL_MAX_PELVIS_TILT_RAD:
        reasons.append(f"pelvis tilt {report['pelvis_tilt_max_rad']:.3f} rad (limit {STILL_MAX_PELVIS_TILT_RAD})")
    if report["joint_speed_max_rad_s"] > STILL_MAX_JOINT_SPEED_RAD_S:
        reasons.append(f"joint speed up to {report['joint_speed_max_rad_s']:.3f} rad/s "
                       f"(limit {STILL_MAX_JOINT_SPEED_RAD_S})")
    return {"still": not reasons, "reasons": reasons, **report}


def step_joint_errors(joint_t, joint_pos, reference_joint_pos, times, num_steps: int) -> np.ndarray:
    """Per-step joint error as TrackingScorer scores it through fencing.sim2sim.mtc: step k on
    the state at times[k + 1] (interpolated) against reference frame k, the last frame held."""
    from fencing.sim2sim.mtc import interpolate_samples

    if len(times) < num_steps + 1:
        raise ValueError(f"{len(times)} tick times cannot score {num_steps} steps (needs {num_steps + 1})")
    state, _ = interpolate_samples(np.asarray(joint_t), np.asarray(joint_pos), np.asarray(times[1:num_steps + 1]))
    frames = np.minimum(np.arange(num_steps), len(reference_joint_pos) - 1)
    return np.abs(state - np.asarray(reference_joint_pos)[frames]).mean(axis=1)


def convergence_step(errors, threshold: float = CONVERGED_JOINT_ERROR_RAD, window_steps: int = 25) -> int | None:
    """First step from which the error stays under `threshold` for `window_steps` steps."""
    below = np.asarray(errors) < threshold
    for start in range(len(below) - window_steps + 1):
        if below[start:start + window_steps].all():
            return start
    return None


def peak_fraction(t, values, limits, names, start: float, end: float) -> dict:
    """Largest |value| / limit per joint over samples with start <= t <= end."""
    t, values = np.asarray(t), np.asarray(values)
    window = (t >= start) & (t <= end)
    if not window.any():
        raise ValueError(f"no samples between {start:.3f} and {end:.3f} s")
    fraction = np.abs(values[window]) / np.asarray(limits, float)
    peak = fraction.max(axis=0)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    at_limit = fraction >= AT_LIMIT_FRACTION
    worst = int(np.argmax(peak))
    return {
        "per_joint": {name: float(p) for name, p in zip(names, peak)},
        "max": float(peak[worst]),
        "joint": names[worst],
        "joints_at_limit": [n for n, hit in zip(names, at_limit.any(axis=0)) if hit],
        "at_limit_s": {n: float(c * dt) for n, c in zip(names, at_limit.sum(axis=0)) if c},
    }


def urdf_limits(urdf_text: str, names, attribute: str) -> np.ndarray:
    """`<limit attribute=...>` of each named joint in a URDF (velocity in rad/s, effort in N m)."""
    limits = {}
    for joint in ET.fromstring(urdf_text).iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get(attribute) is not None:
            limits[joint.get("name")] = float(limit.get(attribute))
    missing = [n for n in names if n not in limits]
    if missing:
        raise KeyError(f"no <limit {attribute}=...> in the URDF for joints {missing}")
    return np.array([limits[n] for n in names])
