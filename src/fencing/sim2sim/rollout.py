"""Run an exported mjlab tracking ONNX in plain CPU MuJoCo and score it like mjlab.

Inputs are only what a deployment has: policy.onnx (weights, the bundled reference
motion and the metadata contract), the exported scene.xml (robot + floor + mjlab's solver
options and PD actuators) and export.json (timestep/decimation). Per policy step:

    obs_t   = build_observation(robot state, reference frame t, last action)
    a_t     = onnx(obs_t, t)                       (no clipping: mjlab's G1 config has none)
    target  = default_joint_pos + action_scale * a_t
    decimation x { ctrl <- target (scene PD actuators) | ctrl <- kp(target-q) - kd qd ; mj_step }
    mj_forward, then score the post-step robot against reference frame t (mjlab's pairing)

The reset mirrors mjlab play mode: robot placed on reference frame 0 (root pose and world
velocity, joints clipped to the soft limits), no pose/velocity perturbation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from fencing import paths
from fencing.sim2sim.contract import PolicyContract, parse_metadata
from fencing.sim2sim.observation import (
    EE_BODY_NAMES, MujocoState, build_observation, relative_body_poses, resolve_id, tracking_terminated,
)
from fencing.sim2sim.quat import quat_apply_inverse
from fencing.train import metrics

SOFT_JOINT_POS_LIMIT_FACTOR = 0.9  # mjlab G1_ARTICULATION
TRAINING_EPISODE_S = 10.0  # tracking_env_cfg episode_length_s: mjlab's evaluation window
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
PELVIS_BODY = "pelvis"
WRIST_BODY = "right_wrist_yaw_link"
TIP_SITE = "blade_tip"


@dataclass
class Reference:
    """Reference motion for the tracked bodies (contract.body_names order), env origin 0."""

    joint_pos: np.ndarray  # (T, J)
    joint_vel: np.ndarray  # (T, J)
    body_pos_w: np.ndarray  # (T, B, 3)
    body_quat_w: np.ndarray  # (T, B, 4)
    body_lin_vel_w: np.ndarray  # (T, B, 3)
    body_ang_vel_w: np.ndarray  # (T, B, 3)
    source: str

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass
class RolloutConfig:
    seconds: float | None = None  # cap on the motion part (None = whole reference)
    hold_seconds: float = 2.0  # hold the last frame afterwards to measure recovery
    pd_mode: str = "actuator"  # "actuator" (scene PD) or "explicit" (torques from metadata kp/kd)
    reset_joint_noise: float = 0.0  # mjlab play adds U(-0.1, 0.1); 0 keeps sim2sim deterministic
    seed: int = 0
    video: Path | None = None
    video_seconds: float | None = None  # None = the whole rollout
    video_fps: float = 25.0
    video_size: tuple[int, int] = (640, 480)
    stop_on_termination: bool = True
    key_frames: tuple[int, ...] | None = None  # blade key frames to score (fencing.saber.scored_keys)


@dataclass
class Export:
    directory: Path
    onnx: Path
    scene_xml: Path
    info: dict = field(default_factory=dict)


def load_export(directory: Path) -> Export:
    directory = Path(directory)
    onnx = directory / "policy.onnx"
    scene = directory / "scene" / "scene.xml"
    info_path = directory / "export.json"
    for required in (onnx, scene, info_path):
        if not required.is_file():
            raise FileNotFoundError(f"{required} missing; run `fencing train export <run> --checkpoint N` first")
    return Export(directory, onnx, scene, json.loads(info_path.read_text()))


def export_key_frames(export: Export, motion_npz: Path | None = None, num_frames: int | None = None):
    """Blade key frames of the motion an export tracks (fencing.saber.scored_keys), or None.

    The motion is `motion_npz` when given (sim2sim --motion), else the NPZ the run trained on
    (export.json `motion_file`, which is what the ONNX bundles)."""
    from fencing.saber.scored_keys import resolve_for_npz

    step_dt = export.info.get("step_dt")
    frames = num_frames if num_frames is not None else export.info.get("motion_frames")
    if step_dt is None or frames is None:
        return None
    return resolve_for_npz(motion_npz or export.info.get("motion_file"), float(step_dt), int(frames))


class OnnxPolicy:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # one tiny MLP: threads only add latency
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.contract: PolicyContract = parse_metadata(self.metadata)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._zero_obs = np.zeros((1, self.session.get_inputs()[0].shape[1]), dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return int(self._zero_obs.shape[1])

    def run(self, obs: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {"obs": obs.reshape(1, -1).astype(np.float32),
                                          "time_step": np.array([[time_step]], dtype=np.float32)})
        return {name: value[0] for name, value in zip(self.output_names, outputs)}

    def reference(self, num_frames: int) -> Reference:
        """The motion bundled in the ONNX (what motion_tracking_controller tracks)."""
        frames = [self.run(self._zero_obs, t) for t in range(num_frames)]
        stack = {k: np.stack([f[k] for f in frames]).astype(np.float64) for k in frames[0] if k != "actions"}
        return Reference(source="onnx", **stack)


def reference_from_npz(npz: Path, body_names: tuple[str, ...]) -> Reference:
    """An mjlab motion NPZ; its body arrays follow the upstream G1 body order (no world)."""
    data = np.load(npz)
    g1 = mujoco.MjModel.from_xml_path(str(paths.G1_MJCF))
    g1_bodies = [g1.body(i).name for i in range(1, g1.nbody)]
    if data["body_pos_w"].shape[1] != len(g1_bodies):
        raise ValueError(f"{npz} has {data['body_pos_w'].shape[1]} bodies; the G1 has {len(g1_bodies)}")
    index = [g1_bodies.index(name) for name in body_names]
    return Reference(
        joint_pos=data["joint_pos"].astype(np.float64),
        joint_vel=data["joint_vel"].astype(np.float64),
        body_pos_w=data["body_pos_w"][:, index].astype(np.float64),
        body_quat_w=data["body_quat_w"][:, index].astype(np.float64),
        body_lin_vel_w=data["body_lin_vel_w"][:, index].astype(np.float64),
        body_ang_vel_w=data["body_ang_vel_w"][:, index].astype(np.float64),
        source=str(npz),
    )


def actuator_ids(model: mujoco.MjModel, state: MujocoState, contract: PolicyContract) -> np.ndarray:
    """Actuator driving each policy joint (scene actuators are not in joint order)."""
    joint_to_actuator = {}
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_to_actuator[int(model.actuator_trnid[actuator, 0])] = actuator
    ids = []
    for name in contract.joint_names:
        joint = resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint not in joint_to_actuator:
            raise ValueError(f"joint {name!r} has no actuator in the scene")
        ids.append(joint_to_actuator[joint])
    return np.array(ids)


def check_gains(model: mujoco.MjModel, actuators: np.ndarray, contract: PolicyContract) -> dict:
    """The scene's PD gains must be the metadata's (3-decimal strings: allow rounding)."""
    kp_scene = model.actuator_gainprm[actuators, 0]
    kd_scene = -model.actuator_biasprm[actuators, 2]
    report = {"kp_max_abs_diff": None, "kd_max_abs_diff": None}
    for key, scene, meta in (("kp", kp_scene, contract.joint_stiffness), ("kd", kd_scene, contract.joint_damping)):
        if meta is None:
            continue
        diff = np.abs(scene - meta)
        report[f"{key}_max_abs_diff"] = float(diff.max())
        tolerance = 1e-3 + 1e-3 * np.abs(meta)
        if np.any(diff > tolerance):
            worst = int(np.argmax(diff - tolerance))
            raise ValueError(f"scene {key} for {contract.joint_names[worst]} is {scene[worst]:.4f} but the ONNX "
                             f"metadata says {meta[worst]:.4f}: scene.xml and policy.onnx do not belong together")
    return report


def soft_joint_limits(model: mujoco.MjModel, state: MujocoState, factor: float = SOFT_JOINT_POS_LIMIT_FACTOR):
    joint_ids = [resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in state.contract.joint_names]
    ranges = model.jnt_range[joint_ids]
    mean, half = ranges.mean(axis=1), 0.5 * (ranges[:, 1] - ranges[:, 0]) * factor
    limited = model.jnt_limited[joint_ids].astype(bool)
    low = np.where(limited, mean - half, -np.inf)
    high = np.where(limited, mean + half, np.inf)
    return low, high


def free_joint_adr(model: mujoco.MjModel) -> tuple[int, int]:
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    raise ValueError("the scene has no free joint (floating base)")


def reset_to_frame(model, data, state: MujocoState, reference: Reference, frame: int, rng, joint_noise: float,
                   root_index: int) -> None:
    """mjlab MotionCommand._resample_command with play-mode ranges."""
    mujoco.mj_resetData(model, data)
    qadr, vadr = free_joint_adr(model)
    root_quat = reference.body_quat_w[frame, root_index]
    data.qpos[qadr:qadr + 3] = reference.body_pos_w[frame, root_index]
    data.qpos[qadr + 3:qadr + 7] = root_quat
    data.qvel[vadr:vadr + 3] = reference.body_lin_vel_w[frame, root_index]
    data.qvel[vadr + 3:vadr + 6] = quat_apply_inverse(root_quat, reference.body_ang_vel_w[frame, root_index])
    joint_pos = reference.joint_pos[frame].copy()
    if joint_noise:
        joint_pos += rng.uniform(-joint_noise, joint_noise, size=joint_pos.shape)
    low, high = soft_joint_limits(model, state)
    data.qpos[state.qpos_adr] = np.clip(joint_pos, low, high)
    data.qvel[state.qvel_adr] = reference.joint_vel[frame]
    mujoco.mj_forward(model, data)


def tip_offset_in_body(model: mujoco.MjModel, site: int, body: int) -> np.ndarray:
    """Constant position of `site` in `body`'s frame (the saber is welded to the wrist)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rotation = data.xmat[body].reshape(3, 3)
    return rotation.T @ (data.site_xpos[site] - data.xpos[body])


def find_site(model: mujoco.MjModel, name: str) -> int | None:
    try:
        return resolve_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    except KeyError:
        return None


class TrackingScorer:
    """Scores a robot trajectory against the reference like mjlab, one policy step at a time.

    Shared by `rollout` (Python MuJoCo) and `fencing.sim2sim.mtc` (states recorded from
    motion_tracking_controller's simulator, replayed into the same scene), so both report
    identical metric definitions. Per step: `begin_step` with the pre-step robot anchor (how
    MotionCommand re-anchors the reference), then `end_step` with the post-step state.

    `settle_steps` (MTC only, 0 in sim2sim): steps at the start that are neither scored nor
    checked for termination. MTC's simulator cannot start the robot in the reference's first
    frame, so its first steps are a transition into the motion that mjlab never trains on.

    `key_frames` (reference frames, from fencing.saber.scored_keys): also score the blade tip
    around them. None keeps the result exactly as it was without key frames.
    """

    def __init__(self, model: mujoco.MjModel, state: MujocoState, reference: Reference,
                 motion_steps: int, hold_steps: int, step_dt: float, settle_steps: int = 0,
                 key_frames=None):
        contract = state.contract
        self.model, self.state, self.reference = model, state, reference
        self.motion_steps, self.hold_steps, self.step_dt = motion_steps, hold_steps, step_dt
        self.total_steps = motion_steps + hold_steps
        self.settle_steps = settle_steps
        self.last_frame = motion_steps - 1
        body_names = list(contract.body_names)
        self.anchor_index = body_names.index(contract.anchor_body_name)
        foot_indices = [body_names.index(n) for n in FOOT_BODIES]
        self.ee_indices = [body_names.index(n) for n in EE_BODY_NAMES]
        self.pelvis_index = body_names.index(PELVIS_BODY)
        tip_site = find_site(model, TIP_SITE)
        self.tip_in_wrist = self.wrist_index = None
        if tip_site is not None and WRIST_BODY in body_names:
            self.wrist_index = body_names.index(WRIST_BODY)
            self.tip_in_wrist = tip_offset_in_body(model, tip_site,
                                                   resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY))
        ground_z = metrics.estimate_ground_height(reference.body_pos_w[:, foot_indices, 2])
        self.accumulator = metrics.TrackingAccumulator(1, body_names, foot_indices, ground_z, key_frames=key_frames)
        self.terminated_at = metrics.NOT_TERMINATED
        self.termination_terms: list[str] = []
        self.final_height_error = self.final_speed = None
        self.steps_run = 0
        self._frame = 0
        self._ref_rel_pos = self._ref_rel_quat = None

    def frame(self, step: int) -> int:
        return min(step, self.last_frame)

    @property
    def alive(self) -> bool:
        return self.terminated_at == metrics.NOT_TERMINATED

    def begin_step(self, step: int, robot_anchor_pos_w: np.ndarray, robot_anchor_quat_w: np.ndarray) -> None:
        reference, frame = self.reference, self.frame(step)
        self._frame = frame
        # Re-anchored reference as MotionCommand holds it during this step (pre-step robot anchor).
        self._ref_rel_pos, self._ref_rel_quat = relative_body_poses(
            reference.body_pos_w[frame, self.anchor_index], reference.body_quat_w[frame, self.anchor_index],
            robot_anchor_pos_w, robot_anchor_quat_w, reference.body_pos_w[frame], reference.body_quat_w[frame])

    def end_step(self, step: int, data: mujoco.MjData) -> bool:
        """Score the post-step state of `step`; returns False once the episode has terminated."""
        state, reference, frame = self.state, self.reference, self._frame
        ref_rel_pos, ref_rel_quat = self._ref_rel_pos, self._ref_rel_quat
        ref_anchor_pos = reference.body_pos_w[frame, self.anchor_index]
        ref_anchor_quat = reference.body_quat_w[frame, self.anchor_index]
        self.steps_run = step + 1
        if step < self.settle_steps:
            return self.alive
        robot_anchor_pos, robot_anchor_quat = state.anchor_pose(data)
        robot_body_pos = state.body_pos_w(data)
        alive = self.alive
        if alive:
            terms = tracking_terminated(ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat,
                                        ref_rel_pos[self.ee_indices], robot_body_pos[self.ee_indices])
            if any(terms.values()):
                self.terminated_at = step
                self.termination_terms = [k for k, v in terms.items() if v]
                alive = False
        if step < self.motion_steps:
            robot_body_quat = state.body_quat_w(data)
            tips = {}
            if self.tip_in_wrist is not None:
                wrist, tip = self.wrist_index, self.tip_in_wrist
                tips = {
                    "ref_tip_rel_w": metrics.blade_tip_from_wrist(ref_rel_pos[wrist], ref_rel_quat[wrist], tip)[None],
                    "ref_tip_w": metrics.blade_tip_from_wrist(reference.body_pos_w[frame, wrist],
                                                              reference.body_quat_w[frame, wrist], tip)[None],
                    "robot_tip_w": metrics.blade_tip_from_wrist(robot_body_pos[wrist], robot_body_quat[wrist],
                                                                tip)[None],
                }
            self.accumulator.add(
                active=np.array([alive]),
                ref_joint_pos=reference.joint_pos[frame][None], robot_joint_pos=state.joint_pos(data)[None],
                ref_body_pos_w=reference.body_pos_w[frame][None], ref_body_pos_rel_w=ref_rel_pos[None],
                robot_body_pos_w=robot_body_pos[None],
                ref_anchor_pos_w=ref_anchor_pos[None], ref_anchor_quat_w=ref_anchor_quat[None],
                robot_anchor_pos_w=robot_anchor_pos[None], robot_anchor_quat_w=robot_anchor_quat[None],
                ref_body_lin_vel_w=reference.body_lin_vel_w[frame][None],
                robot_body_lin_vel_w=state.body_lin_vel_w(data)[None],
                frame=frame,
                **tips,
            )
        if step == self.total_steps - 1 and self.hold_steps:
            pelvis = self.pelvis_index
            self.final_height_error = robot_body_pos[pelvis, 2] - reference.body_pos_w[self.last_frame, pelvis, 2]
            self.final_speed = float(np.linalg.norm(state.body_lin_vel_w(data)[pelvis]))
        return alive

    def result(self) -> dict:
        window_steps = int(round(TRAINING_EPISODE_S / self.step_dt))
        hold_evaluated = self.hold_steps > 0
        reached_end = self.final_height_error is not None
        outcomes = metrics.episode_outcomes(
            np.array([self.terminated_at]), self.motion_steps, window_steps,
            # A rollout that stopped early never reached the end of the hold: not recovered.
            np.array([self.final_height_error if reached_end else np.inf]) if hold_evaluated else None,
            np.array([self.final_speed if reached_end else np.inf]) if hold_evaluated else None,
        )
        nominal = {**metrics.outcome_summary(outcomes, hold_evaluated=hold_evaluated), **self.accumulator.summary()}
        if reached_end:
            nominal["final_pelvis_height_error_m"] = float(self.final_height_error)
            nominal["final_pelvis_speed_m_s"] = self.final_speed
        terminated = self.terminated_at != metrics.NOT_TERMINATED
        return {
            "motion_frames": self.reference.num_frames,
            "motion_steps": self.motion_steps,
            "hold_steps": self.hold_steps,
            "steps_run": self.steps_run,
            "terminated_at_step": self.terminated_at if terminated else None,
            "terminated_at_s": metrics.termination_time_s(self.terminated_at, self.step_dt) if terminated else None,
            "termination_terms": self.termination_terms,
            "has_blade": self.tip_in_wrist is not None,
            "nominal": nominal,
        }


def rollout(export: Export, reference: Reference | None = None, config: RolloutConfig | None = None,
            policy: OnnxPolicy | None = None) -> dict:
    config = config or RolloutConfig()
    policy = policy or OnnxPolicy(export.onnx)
    contract = policy.contract
    model = mujoco.MjModel.from_xml_path(str(export.scene_xml))
    data = mujoco.MjData(model)
    info = export.info
    decimation = int(info.get("decimation", 4))
    step_dt = float(info.get("step_dt", model.opt.timestep * decimation))
    if abs(model.opt.timestep * decimation - step_dt) > 1e-9:
        raise ValueError(f"scene timestep {model.opt.timestep} x decimation {decimation} != step_dt {step_dt}")
    state = MujocoState(model, contract)
    actuators = actuator_ids(model, state, contract)
    gain_report = check_gains(model, actuators, contract)
    if config.pd_mode == "explicit":
        if contract.joint_stiffness is None or contract.joint_damping is None:
            raise ValueError("explicit PD needs joint_stiffness/joint_damping in the metadata")
        model.actuator_gainprm[actuators, 0] = 1.0
        model.actuator_biasprm[actuators, :3] = 0.0  # ctrl is now a torque, still clamped by forcerange
    elif config.pd_mode != "actuator":
        raise ValueError(f"pd_mode must be 'actuator' or 'explicit', got {config.pd_mode!r}")

    if reference is None:
        reference = policy.reference(int(info["motion_frames"]))
    motion_steps = reference.num_frames
    truncated = False
    if config.seconds is not None and round(config.seconds / step_dt) < motion_steps:
        motion_steps, truncated = max(1, int(round(config.seconds / step_dt))), True
    hold_steps = int(round(config.hold_seconds / step_dt))
    scorer = TrackingScorer(model, state, reference, motion_steps, hold_steps, step_dt, key_frames=config.key_frames)
    total_steps, pelvis_index, anchor_index = scorer.total_steps, scorer.pelvis_index, scorer.anchor_index
    rng = np.random.default_rng(config.seed)
    reset_to_frame(model, data, state, reference, 0, rng, config.reset_joint_noise, pelvis_index)

    renderer = writer = None
    render_every = 1
    video_steps = total_steps
    if config.video is not None:
        from fencing.sim2sim.video import SceneRenderer, VideoWriter

        width, height = config.video_size
        render_every = max(1, int(round(1.0 / (config.video_fps * step_dt))))
        if config.video_seconds is not None:
            video_steps = min(total_steps, int(round(config.video_seconds / step_dt)))
        renderer = SceneRenderer(model, width, height, track_body=int(state.body_ids[pelvis_index]))
        writer = VideoWriter(config.video, 1.0 / (render_every * step_dt), width, height)

    last_action = np.zeros(contract.num_joints)
    explicit = config.pd_mode == "explicit"
    kp, kd = contract.joint_stiffness, contract.joint_damping
    qadr, _ = free_joint_adr(model)
    try:
        for step in range(total_steps):
            frame = scorer.frame(step)
            inputs = state.inputs(data, reference.joint_pos[frame], reference.joint_vel[frame],
                                  reference.body_pos_w[frame, anchor_index], reference.body_quat_w[frame, anchor_index],
                                  last_action)
            scorer.begin_step(step, inputs.robot_anchor_pos_w, inputs.robot_anchor_quat_w)
            obs = build_observation(inputs, contract)
            action = policy.run(obs, frame)["actions"].astype(np.float64)
            target = contract.default_joint_pos + contract.action_scale * action
            for _ in range(decimation):
                if explicit:
                    q = data.qpos[state.qpos_adr]
                    qd = data.qvel[state.qvel_adr]
                    data.ctrl[actuators] = kp * (target - q) - kd * qd
                else:
                    data.ctrl[actuators] = target
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            last_action = action
            alive = scorer.end_step(step, data)
            if writer is not None and step < video_steps and step % render_every == 0:
                ghost = data.qpos.copy()
                ghost[qadr:qadr + 3] = reference.body_pos_w[frame, pelvis_index]
                ghost[qadr + 3:qadr + 7] = reference.body_quat_w[frame, pelvis_index]
                ghost[state.qpos_adr] = reference.joint_pos[frame]
                writer.write(renderer.render(data, ghost))
            if not alive and config.stop_on_termination and (writer is None or step >= video_steps):
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    scored = scorer.result()
    return {
        "reference_source": reference.source,
        "pd_mode": config.pd_mode,
        "metadata_full_precision": contract.full_precision,
        "gains_vs_metadata": gain_report,
        "physics": {"timestep": float(model.opt.timestep), "decimation": decimation, "step_dt": step_dt,
                    "integrator": int(model.opt.integrator)},
        "motion_frames": scored["motion_frames"],
        "motion_steps": motion_steps,
        "hold_steps": hold_steps,
        "truncated": truncated,
        "steps_run": scored["steps_run"],
        "terminated_at_step": scored["terminated_at_step"],
        "terminated_at_s": scored["terminated_at_s"],
        "termination_terms": scored["termination_terms"],
        "has_blade": scored["has_blade"],
        "video": str(config.video) if config.video is not None else None,
        "nominal": scored["nominal"],
    }
