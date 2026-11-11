"""Score a motion_tracking_controller (MTC) MuJoCo run with the same code as `fencing sim2sim`.

`scripts/mtc_sim2sim.sh` runs MTC's own C++ controller and simulator in the rootless ROS 2
rootfs (docs/mtc.md) and flattens the rosbag into bag.npz (scripts/mtc/bag_to_npz.py). This
module turns that recording into the sim2sim report:

- Ground-truth base pose: MTC's simulator publishes the `mid360` site pose (framepos +
  framequat sensors on torso_link, exact in simulation) as /mid360. With the joint angles
  from /joint_states, forward kinematics gives the pelvis pose. /odom (the state estimator)
  is only used to report the estimator's error, never for scoring.
- Policy steps: the controller runs every 20 ms of sim time from its activation (stamped on
  /controller_manager/activity). The exact phase inside the 20 ms is read from the PD
  torques, which jump when a new target arrives (`tick_phase`). The controller's observation
  at tick k carries reference frame k - 1 (frame 0 at ticks 0 and 1), see
  CONTROLLER_REFERENCE_LAG_STEPS, so step k (frame k, as in sim2sim) is scored from tick
  k + 1 and its post-step state is the state at tick k + 2. States at a tick are interpolated
  between the recorded samples either side (2 ms apart, more where rosbag dropped one).
- Reference: the motion bundled in the ONNX (what MTC tracks), moved the way
  MotionCommandTerm::reset moves it (start-frame anchor onto the robot anchor, yaw only).
- Scoring: recorded states are written into the exported mjlab scene (same kinematics) and
  passed to rollout.TrackingScorer, the class sim2sim's own rollout uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fencing.sim2sim.quat import normalize, quat_apply, quat_conjugate, quat_mul, yaw_quat

POLICY_PERIOD_S = 0.02  # MTC config/g1/controllers.yaml walking_controller update_rate 50
SITE_NAME = "mid360"
SITE_BODY = "torso_link"
# Half a policy step: a recorded state further than this from its tick is nearer the state of
# the neighbouring tick, so the step would be scored on the wrong state. Gaps elsewhere in the
# recording (rosbag drops under CPU load) do not matter: the controller runs in the physics loop.
MAX_STATE_TIME_MISMATCH_S = 0.010
EXIT_INVALID = 3  # mtc_sim2sim.sh re-records on this code
RELEASE_WINDOW_S = 0.5
RELEASE_MIN_MOTION_RAD = 1e-3
# MotionOnnxPolicy::forward runs the ONNX with time_step = timeStep_++ and stores the reference
# frame it returns; the next tick's observation reads that stored frame (MotionCommandTerm::
# getValue), and reset() leaves frame 0 stored. So tick k observes frame max(k - 1, 0): the
# controller runs the whole motion one tick behind activation. sim2sim (like mjlab) observes
# frame k at step k. Pairing tick k with frame k scored MTC 20 ms late (docs/mtc.md).
CONTROLLER_REFERENCE_LAG_STEPS = 1


def tick_phase(joint_t: np.ndarray, effort: np.ndarray, activation_t: float, period: float = POLICY_PERIOD_S,
               window_s: float = 4.0) -> float:
    """Offset (s) of the first policy tick after activation, from PD torque jumps.

    Between ticks the PD torque kp (target - q) - kd qd changes smoothly; when a new target
    arrives it jumps by kp * delta_target. Fold the jump size by time-since-activation modulo
    the period and take the sample slot with the largest mean jump.
    """
    after = (joint_t > activation_t) & (joint_t <= activation_t + window_s)
    if after.sum() < 3:
        raise ValueError("no /joint_states samples after activation: cannot find the policy tick phase")
    dt = float(np.median(np.diff(joint_t)))
    slots = int(round(period / dt))
    t, e = joint_t[after], effort[after]
    jumps = np.abs(np.diff(e, axis=0)).sum(axis=1)
    slot = np.round(((t[1:] - activation_t) % period) / dt).astype(int) % slots
    means = np.array([jumps[slot == s].mean() if np.any(slot == s) else -np.inf for s in range(slots)])
    return float(np.argmax(means) * dt)


def timing_problems(timing: dict, limit: float = MAX_STATE_TIME_MISMATCH_S) -> list[str]:
    """Reasons the recording cannot be scored faithfully (empty when it can)."""
    mismatch = timing["max_state_time_mismatch_s"]
    if mismatch > limit:
        return [f"a scored state was recorded {mismatch:.3f} s from its policy tick (limit {limit:.3f} s, "
                f"half a policy step): /joint_states or /mid360 messages were dropped"]
    return []


def require_release(joint_t: np.ndarray, joint_pos: np.ndarray, activation_t: float,
                    window_s: float = RELEASE_WINDOW_S, min_motion_rad: float = RELEASE_MIN_MOTION_RAD) -> None:
    """The robot must move once the controller is active; frozen joints mean the start hold (or
    the simulator) never let go, and scoring a statue would be meaningless."""
    after = (joint_t >= activation_t) & (joint_t <= activation_t + window_s)
    if after.sum() < 2:
        raise ValueError(f"no /joint_states samples in the {window_s} s after activation at {activation_t:.3f} s")
    q = joint_pos[after]
    motion = float(np.abs(q - q[0]).max())
    if motion < min_motion_rad:
        raise ValueError(f"joints moved {motion:.2e} rad in the {window_s} s after activation: the robot was "
                         f"never released (start hold stuck?)")


def policy_step_times(activation_t: float, phase: float, num_steps: int, period: float = POLICY_PERIOD_S):
    return activation_t + phase + period * np.arange(num_steps)


def scored_step_times(activation_t: float, phase: float, num_steps: int, period: float = POLICY_PERIOD_S,
                      lag_steps: int = CONTROLLER_REFERENCE_LAG_STEPS):
    """Tick at which step k begins, for k < num_steps: the tick whose observation carries
    reference frame k. The first tick (frame 0 a second time) is not scored."""
    return policy_step_times(activation_t, phase, num_steps + lag_steps, period)[lag_steps:]


def reference_placement_time(activation_t: float, phase: float, settle_steps: int, period: float = POLICY_PERIOD_S,
                             lag_steps: int = CONTROLLER_REFERENCE_LAG_STEPS) -> float:
    """Tick whose robot pose places the reference (start frame = settle_steps): the first tick whose
    observation carries that frame. Frame 0 is observed at ticks 0 and 1, and MotionCommandTerm::
    reset places the motion at activation, so tick 0; frame s > 0 first appears at tick s + lag."""
    tick = 0 if settle_steps == 0 else settle_steps + lag_steps
    return float(policy_step_times(activation_t, phase, tick + 1, period)[tick])


def _bracket(t_src: np.ndarray, t_query: np.ndarray):
    """Samples either side of each query time, the weight of the right one (clipped: held beyond
    the ends), and the largest distance from a query time to its nearest sample."""
    right = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    left = right - 1
    span = t_src[right] - t_src[left]
    weight = np.clip((t_query - t_src[left]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    gap = np.minimum(np.abs(t_query - t_src[left]), np.abs(t_src[right] - t_query))
    return left, right, weight, float(gap.max())


def interpolate_samples(t_src: np.ndarray, rows: np.ndarray, t_query: np.ndarray):
    """Rows linearly interpolated to each query time, and the largest |t_query - nearest sample|.

    Where a sample was dropped, the nearest sample was one of two equally near neighbours and
    float rounding picked it; interpolation is continuous in time, so the state at a tick no
    longer depends on rounding. The distance is still to the nearest *real* sample, so a run of
    dropped messages stays visible to timing_problems."""
    left, right, weight, gap = _bracket(t_src, t_query)
    w = weight.reshape(weight.shape + (1,) * (rows.ndim - 1))
    return rows[left] + w * (rows[right] - rows[left]), gap


def interpolate_quats(t_src: np.ndarray, quats: np.ndarray, t_query: np.ndarray):
    """Unit quaternions interpolated to each query time by nlerp, the right sample sign-aligned
    to the left (q and -q are one rotation). Exact at the samples; between 2 ms samples nlerp
    and slerp differ by ~1e-9 (the rotation per sample is a few mrad)."""
    left, right, weight, gap = _bracket(t_src, t_query)
    q0, q1 = quats[left], quats[right]
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
    return normalize(q0 + weight[:, None] * (q1 - q0)), gap


def nearest_samples(t_src: np.ndarray, rows: np.ndarray, t_query: np.ndarray):
    """Rows at the sample nearest each query time, and the largest |t_query - t_sample|."""
    right = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    left = right - 1
    pick = np.where(np.abs(t_src[left] - t_query) <= np.abs(t_src[right] - t_query), left, right)
    return rows[pick], float(np.abs(t_src[pick] - t_query).max())


def pelvis_pose_from_site(site_pos_w, site_quat_w, site_pos_in_pelvis, site_quat_in_pelvis):
    """Pelvis world pose from a site's world pose and the site's pose in the pelvis frame."""
    pelvis_quat = normalize(quat_mul(site_quat_w, quat_conjugate(site_quat_in_pelvis)))
    return site_pos_w - quat_apply(pelvis_quat, site_pos_in_pelvis), pelvis_quat


def align_reference(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, robot_anchor_pos_w,
                    robot_anchor_quat_w, anchor_index: int, start_frame: int = 0):
    """MotionCommandTerm::reset (MotionCommand.cpp:37-43): worldToInit = worldToAnchor(yaw)
    * initToAnchor(yaw)^-1, applied to the whole motion, but translating in x and y only.

    MTC also shifts z, by the difference between its *estimated* anchor height and the
    reference's; that only biases the policy's anchor observation. Both worlds have the floor
    at z = 0, so keeping the reference heights scores against the motion as sim2sim does (and
    keeps the ground-height estimate behind the foot-slip metric valid)."""
    ref_pos = body_pos_w[start_frame, anchor_index]
    ref_yaw = yaw_quat(body_quat_w[start_frame, anchor_index])
    rotation = normalize(quat_mul(yaw_quat(robot_anchor_quat_w), quat_conjugate(ref_yaw)))
    translation = robot_anchor_pos_w - quat_apply(rotation, ref_pos)
    translation[2] = 0.0
    shape = body_pos_w.shape[:-1]
    rot_b = np.broadcast_to(rotation, shape + (4,))
    return (quat_apply(rot_b, body_pos_w) + translation,
            normalize(quat_mul(rot_b, body_quat_w)),
            quat_apply(rot_b, body_lin_vel_w),
            quat_apply(rot_b, body_ang_vel_w))


def base_velocity(t: np.ndarray, pos: np.ndarray, quat: np.ndarray):
    """Free-joint velocities from sampled poses: world linear, body-frame angular (MuJoCo qvel)."""
    lin = np.gradient(pos, t, axis=0)
    q = _continuous(quat)
    dq = np.gradient(q, t, axis=0)
    # q_dot = 0.5 * q * (0, w_body)  =>  w_body = 2 * conj(q) * q_dot
    ang = 2.0 * quat_mul(quat_conjugate(q), dq)[:, 1:]
    return lin, ang


def _continuous(quat: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in the same hemisphere (q and -q are equal)."""
    out = quat.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


class RecordingIncomplete(ValueError):
    """The bag misses part of the run (rosbag under CPU load), not a fault of the policy: INVALID."""


def require_coverage(first_sample_t: float, first_tick_t: float) -> None:
    if first_sample_t > first_tick_t:
        raise RecordingIncomplete(f"recording starts at sim {first_sample_t:.3f} s, after the first policy tick at "
                         f"{first_tick_t:.3f} s: the bag must start before the controller activates")


def unique_by_time(rows: np.ndarray) -> np.ndarray:
    """Rows sorted by column 0 (time), one per distinct time (the last one received).

    The simulator republishes sensor topics without the clock advancing, so a time can repeat;
    finite differences over repeated stamps would divide by zero.
    """
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    keep = np.append(np.diff(rows[:, 0]) > 0, True)
    return rows[keep]


def real_time_factor(sim_t: np.ndarray, wall_t: np.ndarray) -> float:
    return float((sim_t[-1] - sim_t[0]) / max(wall_t[-1] - wall_t[0], 1e-9))


GAP_METRICS = ("success_rate", "joint_angle_error_rad", "body_pos_error_m", "anchor_pos_error_m",
               "anchor_ori_error_rad", "foot_slip_m_s", "blade_tip_error_m", "blade_tip_error_key_m")


def sim2sim_gap(mtc_nominal: dict, sim2sim_report: dict) -> dict:
    """MTC minus Python sim2sim, per metric both measured. sim2sim's span is reported alongside:
    its JSON may score a different number of seconds than this run."""
    theirs = sim2sim_report["nominal"]
    metrics = {k: {"mtc": mtc_nominal[k], "sim2sim": theirs[k], "diff": mtc_nominal[k] - theirs[k]}
               for k in GAP_METRICS if mtc_nominal.get(k) is not None and theirs.get(k) is not None}
    return {"metrics": metrics, "sim2sim_motion_steps": sim2sim_report.get("motion_steps"),
            "sim2sim_truncated": sim2sim_report.get("truncated"), "sim2sim_verdict": sim2sim_report.get("verdict")}


def record_seconds(motion_frames: int, seconds: float | None, hold_seconds: float, margin_s: float = 1.0) -> int:
    """Whole sim seconds the recorder must run past activation for this scoring span."""
    motion_s = motion_frames * POLICY_PERIOD_S
    if seconds is not None:
        motion_s = min(motion_s, seconds)
    return int(np.ceil(motion_s + hold_seconds + margin_s - 1e-9))


def start_qpos(model, joint_positions: dict[str, float], pelvis_pos, pelvis_quat) -> np.ndarray:
    """MuJoCo qpos of `model` (MTC's G1 MJCF) in the given pose: free joint = pelvis pose, hinges
    by name. Every hinge needs an angle; one left at qpos0 would start straight-legged."""
    import mujoco

    qpos = model.qpos0.copy()
    missing = set()
    for joint_id in range(model.njnt):
        adr = int(model.jnt_qposadr[joint_id])
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos[adr:adr + 3] = pelvis_pos
            qpos[adr + 3:adr + 7] = normalize(np.asarray(pelvis_quat, dtype=float))
        elif model.joint(joint_id).name in joint_positions:
            qpos[adr] = joint_positions[model.joint(joint_id).name]
        else:
            missing.add(model.joint(joint_id).name)
    if missing:
        raise ValueError(f"no start angle for model joints {sorted(missing)}")
    unknown = set(joint_positions) - {model.joint(j).name for j in range(model.njnt)}
    if unknown:
        raise KeyError(f"start angles for joints the model lacks: {sorted(unknown)}")
    return qpos


def write_start_qpos(export_dir: Path, out: Path, mtc_mjcf: Path) -> Path:
    """The reference motion's first frame as a qpos of MTC's MJCF, one line of numbers, for the
    hold plugin (scripts/mtc/hcf_mtc_hold) that keeps the simulator there until activation."""
    import mujoco

    from fencing.sim2sim import rollout as sim

    policy = sim.OnnxPolicy(export_dir / "policy.onnx")
    reference = policy.reference(1)
    pelvis = list(policy.contract.body_names).index("pelvis")
    pose = dict(zip(policy.contract.joint_names, reference.joint_pos[0].astype(float)))
    qpos = start_qpos(mujoco.MjModel.from_xml_path(str(mtc_mjcf)), pose, reference.body_pos_w[0, pelvis],
                      reference.body_quat_w[0, pelvis])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(f"{v:.9g}" for v in qpos) + "\n")
    return out


def site_in_body(mjcf: Path, site: str = SITE_NAME, body: str = SITE_BODY):
    """Fixed pose of `site` in `body` from MTC's MJCF (unitree_description g1.xml)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    site_id = model.site(site).id
    if model.body(int(model.site_bodyid[site_id])).name != body:
        raise ValueError(f"{mjcf}: site {site} is not on {body}")
    return model.site_pos[site_id].copy(), model.site_quat[site_id].copy()


class Replayer:
    """Writes recorded robot states into the exported scene so TrackingScorer can read them."""

    def __init__(self, model, state, recording: dict, site_pos_in_torso, site_quat_in_torso):
        import mujoco

        self.mujoco = mujoco
        self.model, self.state = model, state
        self.data = mujoco.MjData(model)
        names = [str(n) for n in recording["joint_names"]]
        self.columns = [names.index(n) for n in state.contract.joint_names]
        self.rec = recording
        self.site_pos_in_torso, self.site_quat_in_torso = site_pos_in_torso, site_quat_in_torso
        joint = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.qadr, self.vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.pelvis = state.body_ids[list(state.contract.body_names).index("pelvis")]
        self.torso = next(b for b in range(model.nbody) if model.body(b).name.split("/")[-1] == SITE_BODY)
        self._base_cache = None

    def _site_in_pelvis(self, joint_pos):
        """torso_link pose in the pelvis frame for these joint angles, then the site on it."""
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.qadr:self.qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[self.state.qpos_adr] = joint_pos
        self.mujoco.mj_kinematics(self.model, d)
        torso_pos, torso_quat = d.xpos[self.torso].copy(), d.xquat[self.torso].copy()
        return (torso_pos + quat_apply(torso_quat, self.site_pos_in_torso),
                normalize(quat_mul(torso_quat, self.site_quat_in_torso)))

    def base_track(self):
        """Pelvis pose and free-joint velocity at every /mid360 sample (ground truth). The joint
        angles for the forward kinematics are interpolated to the sample's time: the nearest
        /joint_states sample could be 2 ms off where one was dropped."""
        if self._base_cache is None:
            mid = unique_by_time(self.rec["mid360"])
            joints, _ = interpolate_samples(self.rec["joint_t"], self.rec["joint_pos"][:, self.columns], mid[:, 0])
            pos, quat = [], []
            for row, q in zip(mid, joints):
                rel_pos, rel_quat = self._site_in_pelvis(q)
                p, o = pelvis_pose_from_site(row[1:4], normalize(row[4:8]), rel_pos, rel_quat)
                pos.append(p)
                quat.append(o)
            pos, quat = np.array(pos), _continuous(np.array(quat))
            lin, ang = base_velocity(mid[:, 0], pos, quat)
            self._base_cache = (mid[:, 0], pos, quat, lin, ang)
        return self._base_cache

    def set_state(self, t: float) -> float:
        """Load the recorded state at sim time t into data (with mj_forward), interpolated between
        the samples either side; returns the distance from t to the nearest real sample."""
        bt, pos, quat, lin, ang = self.base_track()
        query = np.array([t])
        base_pos, gap_b = interpolate_samples(bt, pos, query)
        base_quat, _ = interpolate_quats(bt, quat, query)
        base_lin, _ = interpolate_samples(bt, lin, query)
        base_ang, _ = interpolate_samples(bt, ang, query)
        joint_t = self.rec["joint_t"]
        joint_pos, gap_j = interpolate_samples(joint_t, self.rec["joint_pos"][:, self.columns], query)
        joint_vel, _ = interpolate_samples(joint_t, self.rec["joint_vel"][:, self.columns], query)
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qvel[:] = 0
        d.qpos[self.qadr:self.qadr + 3] = base_pos[0]
        d.qpos[self.qadr + 3:self.qadr + 7] = base_quat[0]
        d.qvel[self.vadr:self.vadr + 3] = base_lin[0]
        d.qvel[self.vadr + 3:self.vadr + 6] = base_ang[0]
        d.qpos[self.state.qpos_adr] = joint_pos[0]
        d.qvel[self.state.qvel_adr] = joint_vel[0]
        self.mujoco.mj_forward(self.model, d)
        return max(gap_b, gap_j)


def score_recording(export_dir: Path, recording: dict, mtc_mjcf: Path, seconds: float | None,
                    hold_seconds: float, settle_seconds: float = 0.0) -> dict:
    import mujoco

    from fencing.sim2sim import rollout as sim
    from fencing.sim2sim.observation import MujocoState

    export = sim.load_export(export_dir)
    policy = sim.OnnxPolicy(export.onnx)
    contract = policy.contract
    model = mujoco.MjModel.from_xml_path(str(export.scene_xml))
    state = MujocoState(model, contract)
    reference = policy.reference(int(export.info["motion_frames"]))
    step_dt = POLICY_PERIOD_S
    motion_steps, truncated = reference.num_frames, False
    if seconds is not None and round(seconds / step_dt) < motion_steps:
        motion_steps, truncated = max(1, int(round(seconds / step_dt))), True
    hold_steps = int(round(hold_seconds / step_dt))

    if len(recording["activation_t"]) == 0:
        raise ValueError("walking_controller never became active in the recording")
    activation = float(recording["activation_t"][0])
    names = [str(n) for n in recording["joint_names"]]
    columns = [names.index(n) for n in contract.joint_names]
    # Coverage before anything that reads the samples after activation: a bag that starts late
    # would otherwise surface as a misleading "no samples" or "never released" error.
    require_coverage(max(recording["joint_t"][0], recording["mid360"][0, 0]), activation)
    require_release(recording["joint_t"], recording["joint_pos"][:, columns], activation)
    phase = tick_phase(recording["joint_t"], recording["joint_effort"][:, columns], activation)
    ticks = policy_step_times(activation, phase, motion_steps + hold_steps + 1)
    times = scored_step_times(activation, phase, motion_steps + hold_steps + 1)
    available = min(recording["joint_t"][-1], recording["mid360"][-1, 0])
    recorded_steps = int(np.searchsorted(times, available, side="right")) - 1  # needs times[k + 1]

    settle_steps = min(int(round(settle_seconds / step_dt)), max(recorded_steps - 1, 0), motion_steps - 1)
    # The reference is placed like MotionCommandTerm::reset places it at activation, but from the
    # robot's pose when scoring starts. With the start hold (scripts/mtc/hcf_mtc_hold) the robot
    # is activated in the reference's first frame and the two placements agree; an unheld run
    # (--no-hold) starts in unitree_description's pose and its first steps move it (dance1: ~0.4 m
    # in 1 s), an offset the global metrics would otherwise carry. Placement is in the JSON.
    # The pose is taken at the first tick whose observation carries the start frame: activation
    # for frame 0 (as MotionCommandTerm::reset), times[settle_steps] for a later one.
    replay = Replayer(model, state, recording, *site_in_body(mtc_mjcf))
    replay.set_state(ticks[0])
    pelvis_at_activation = replay.data.qpos[replay.qadr:replay.qadr + 7].copy()
    anchor_at_activation = state.anchor_pose(replay.data)[0]
    replay.set_state(reference_placement_time(activation, phase, settle_steps))
    anchor = list(contract.body_names).index(contract.anchor_body_name)
    robot_anchor_pos, robot_anchor_quat = state.anchor_pose(replay.data)
    aligned = align_reference(reference.body_pos_w, reference.body_quat_w, reference.body_lin_vel_w,
                              reference.body_ang_vel_w, robot_anchor_pos, robot_anchor_quat, anchor,
                              start_frame=settle_steps)
    reference = sim.Reference(reference.joint_pos, reference.joint_vel, *aligned,
                              source=f"{reference.source} (placed like MotionCommandTerm::reset at step "
                                     f"{settle_steps})")

    key_frames = sim.export_key_frames(export)
    scorer = sim.TrackingScorer(model, state, reference, motion_steps, hold_steps, step_dt, settle_steps,
                                key_frames=key_frames.steps if key_frames else None)
    worst_gap = 0.0
    for step in range(min(scorer.total_steps, recorded_steps)):
        worst_gap = max(worst_gap, replay.set_state(times[step]))
        scorer.begin_step(step, *state.anchor_pose(replay.data))
        worst_gap = max(worst_gap, replay.set_state(times[step + 1]))
        if not scorer.end_step(step, replay.data):
            break
    scored = scorer.result()
    complete = scored["terminated_at_step"] is not None or scored["steps_run"] >= scorer.total_steps
    result = {
        "reference_source": reference.source,
        "metadata_full_precision": False,  # the C++ parses the 3-decimal strings
        "physics": {"step_dt": step_dt, "simulator": f"MTC mujoco_sim_ros2 ({mtc_mjcf})"},
        **{k: scored[k] for k in ("motion_frames", "motion_steps", "hold_steps", "steps_run", "terminated_at_step",
                                  "terminated_at_s", "termination_terms", "has_blade")},
        "truncated": truncated,
        "settle_seconds": settle_steps * step_dt,
        "start": {  # how the robot entered the motion (unscored)
            "pelvis_height_at_activation_m": float(pelvis_at_activation[2]),
            "pelvis_tilt_at_activation_rad": float(2 * np.arccos(np.clip(np.linalg.norm(
                pelvis_at_activation[[3, 6]]), 0, 1))),
            "anchor_moved_during_settle_m": float(np.linalg.norm((robot_anchor_pos - anchor_at_activation)[:2])),
        },
        "recording_complete": bool(complete),
        "nominal": scored["nominal"],
        "timing": timing_report(recording, activation, phase, worst_gap),
        "estimator": estimator_report(recording, replay, ticks[:max(recorded_steps, 1)]),
    }
    if key_frames:
        result["key_frames"] = key_frames.report()
    return result


def timing_report(recording: dict, activation: float, phase: float, worst_gap: float) -> dict:
    gaps = np.diff(recording["joint_t"])
    report = {
        "activation_sim_s": activation,
        "first_policy_tick_after_activation_s": phase,
        "controller_reference_lag_steps": CONTROLLER_REFERENCE_LAG_STEPS,
        "joint_states_dt_median_s": float(np.median(gaps)),
        "joint_states_gaps_over_3ms": int((gaps > 0.003).sum()),
        "joint_states_max_gap_s": float(gaps.max()),
        "mid360_max_gap_s": float(np.diff(unique_by_time(recording["mid360"])[:, 0]).max()),
        "max_state_time_mismatch_s": worst_gap,
        "max_state_time_mismatch_limit_s": MAX_STATE_TIME_MISMATCH_S,
        "real_time_factor": (real_time_factor(recording["clock"], recording["clock_wall"])
                             if len(recording.get("clock_wall", [])) > 1 else None),
    }
    period = recording.get("stat_period")
    if period is not None and len(period):
        # pal_statistics reports NaN min/max until enough samples exist: keep only finite values.
        def finite(values, reduce):
            values = values[np.isfinite(values)]
            return float(reduce(values)) if len(values) else None
        report["controller_periodicity_hz"] = {
            "average": finite(period[-1:, 0], np.mean), "min": finite(period[:, 1], np.min),
            "max": finite(period[:, 2], np.max)}
    return report


def estimator_report(recording: dict, replay: Replayer, times: np.ndarray) -> dict:
    """MTC state estimator (/odom, pelvis) against ground truth at the policy ticks."""
    odom = recording.get("odom")
    if odom is None or len(odom) < 2:
        return {}
    bt, pos, quat, lin, _ = replay.base_track()
    est, _ = nearest_samples(odom[:, 0], odom, times)
    true_pos, _ = nearest_samples(bt, pos, times)
    true_quat, _ = nearest_samples(bt, quat, times)
    true_lin_w, _ = nearest_samples(bt, lin, times)
    # nav_msgs/Odometry twist is in the child (pelvis) frame.
    true_lin_b = quat_apply(quat_conjugate(true_quat), true_lin_w)
    pos_err = np.linalg.norm(est[:, 1:4] - true_pos, axis=1)
    pos_err_xy = np.linalg.norm(est[:, 1:3] - true_pos[:, :2], axis=1)
    vel_err = np.linalg.norm(est[:, 8:11] - true_lin_b, axis=1)
    dot = np.abs(np.sum(normalize(est[:, 4:8]) * true_quat, axis=1)).clip(0, 1)
    return {
        # What the policy's first observation is built from: a held, still robot should read ~0.
        "at_first_tick": {"lin_vel_m_s": float(np.linalg.norm(est[0, 8:11])),
                          "height_error_m": float(est[0, 3] - true_pos[0, 2])},
        "pelvis_pos_error_m": {"mean": float(pos_err.mean()), "max": float(pos_err.max())},
        "pelvis_pos_error_xy_m": {"mean": float(pos_err_xy.mean()), "max": float(pos_err_xy.max())},
        "pelvis_ori_error_rad": {"mean": float((2 * np.arccos(dot)).mean())},
        "pelvis_lin_vel_error_m_s": {"mean": float(vel_err.mean()), "max": float(vel_err.max())},
    }


def load_recording(npz: Path) -> dict:
    with np.load(npz) as data:
        return {k: data[k] for k in data.files}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fencing.sim2sim.mtc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="export dir, run dir, run dir name or run name (as for sim2sim)")
    parser.add_argument("--checkpoint", help="with a run: iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--recording", type=Path, help="bag.npz from scripts/mtc/bag_to_npz.py")
    parser.add_argument("--mtc-mjcf", type=Path, help="the MJCF MTC's simulator loaded (default: "
                        "unitree_description's g1.xml in $HCF_DATA/mtc/noble, or for a saber policy the "
                        "saber plant in $HCF_DATA/mtc/models, see fencing.sim2sim.mtc_plant)")
    parser.add_argument("--seconds", type=float, help="score only the first S seconds of the motion")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=0.0,
                        help="first seconds after activation that are neither scored nor checked for termination "
                             "(docs/mtc.md, scoring conventions; the default 0 scores from step 0 like "
                             "fencing sim2sim)")
    parser.add_argument("--write-start-qpos", type=Path, metavar="TXT",
                        help="write the reference's first frame as a qpos of MTC's MJCF (for mtc_sim2sim.sh) and exit")
    parser.add_argument("--plan", action="store_true",
                        help="print '<export dir> <seconds to record past activation>' and exit (for mtc_sim2sim.sh)")
    parser.add_argument("--out", type=Path, help="output JSON (default: <export dir>/mtc.json)")
    parser.add_argument("--extra", type=Path, help="JSON merged into the report (run settings)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from fencing.sim2sim import mtc_plant
    from fencing.sim2sim.cli import resolve_export_dir, summary_line
    from fencing.train import thresholds

    parser = build_parser()
    args = parser.parse_args(argv)

    export_dir = resolve_export_dir(args.target, args.checkpoint)
    if args.plan:
        frames = int(json.loads((export_dir / "export.json").read_text())["motion_frames"])
        print(export_dir, record_seconds(frames, args.seconds, args.hold_seconds))
        return 0
    plant = mtc_plant.policy_plant(export_dir)
    mtc_mjcf = args.mtc_mjcf or (mtc_plant.MODELS_DIR / mtc_plant.SABER_MJCF_NAME if plant == mtc_plant.SABER
                                 else mtc_plant.ROOTFS_G1_MJCF)
    mtc_plant.require_plant(plant, mtc_mjcf)
    if args.write_start_qpos:
        print(write_start_qpos(export_dir, args.write_start_qpos, mtc_mjcf))
        return 0
    if args.recording is None:
        parser.error("--recording is required unless --plan or --write-start-qpos")
    out = args.out or export_dir / "mtc.json"
    extra = json.loads(args.extra.read_text()) if args.extra else {}
    try:
        result = score_recording(export_dir, load_recording(args.recording), mtc_mjcf, args.seconds,
                                 args.hold_seconds, args.settle_seconds)
    except RecordingIncomplete as error:
        out.write_text(json.dumps({"kind": "mtc_sim2sim", "export_dir": str(export_dir), "plant": plant,
                                   "recording": str(args.recording), **extra, "nominal": {},
                                   "invalid_reasons": [str(error)], "verdict": "INVALID"}, indent=2) + "\n")
        print(f"[mtc]   INVALID {error}\n[mtc] INVALID -> {out}")
        return EXIT_INVALID
    checks = thresholds.check(result, thresholds.load_thresholds(None, None))
    nominal_checks = [c for c in checks if c.metric.startswith("nominal.")]
    incomplete = list(thresholds.incomplete_reasons(result))
    invalid = timing_problems(result["timing"])
    if not result["recording_complete"]:
        incomplete.append(f"recording ended after {result['steps_run']} of "
                          f"{result['motion_steps'] + result['hold_steps']} policy steps")
    report = {
        "kind": "mtc_sim2sim",
        "export_dir": str(export_dir),
        "plant": plant,
        "simulator": "motion_tracking_controller C++ + mujoco_sim_ros2 (ROS 2 Jazzy, rootless rootfs)",
        "recording": str(args.recording),
        **extra,
        **result,
        "thresholds": [c.__dict__ for c in nominal_checks],
        "invalid_reasons": invalid,
        # INVALID, not FAIL: the policy was not judged, the recording was too coarse to judge it.
        "verdict": "INVALID" if invalid else thresholds.verdict(nominal_checks, thresholds.not_applicable(result),
                                                                incomplete),
    }
    sim2sim_json = export_dir / "sim2sim.json"
    if sim2sim_json.is_file():
        report["gap_vs_sim2sim"] = sim2sim_gap(result["nominal"], json.loads(sim2sim_json.read_text()))
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    timing = result["timing"]
    print(f"[mtc] activation at sim {timing['activation_sim_s']:.3f} s, first tick +"
          f"{timing['first_policy_tick_after_activation_s'] * 1000:.0f} ms, real-time factor "
          f"{timing['real_time_factor'] if timing['real_time_factor'] is None else round(timing['real_time_factor'], 2)}")
    if result["estimator"]:
        e = result["estimator"]
        print(f"[mtc] state estimator vs truth: pelvis pos {e['pelvis_pos_error_m']['mean'] * 100:.1f} cm, "
              f"lin vel {e['pelvis_lin_vel_error_m_s']['mean']:.3f} m/s (means); at the first tick: speed "
              f"{e['at_first_tick']['lin_vel_m_s']:.3f} m/s, height error {e['at_first_tick']['height_error_m'] * 100:+.1f} cm")
    if result["terminated_at_step"] is not None:
        print(f"[mtc] terminated at {result['terminated_at_s']:.2f} s ({', '.join(result['termination_terms'])})")
    print(f"[mtc] {summary_line(result['nominal'])} (after a {result['settle_seconds']:.1f} s unscored settle)")
    if "gap_vs_sim2sim" in report:
        gaps = ", ".join(f"{k} {v['diff']:+.3g}" for k, v in report["gap_vs_sim2sim"]["metrics"].items())
        print(f"[mtc] gap vs sim2sim.json (mtc - sim2sim): {gaps}")
    for check in nominal_checks:
        print(f"[mtc]   {check.line()}")
    for reason in invalid:
        print(f"[mtc]   INVALID {reason}")
    print(f"[mtc] {report['verdict']} -> {out}")
    return EXIT_INVALID if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
