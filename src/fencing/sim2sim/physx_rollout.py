"""Roll an exported mjlab tracking ONNX out in CPU PhysX and score it exactly like `rollout`.

Same loop, observation builder, reset and TrackingScorer as the MuJoCo sim2sim; only the
plant differs (`physx_plant`): per policy step

    obs_t  = build_observation(PhysX robot state, reference frame t, last action)
    target = default_joint_pos + action_scale * onnx(obs_t, t)
    decimation x { PhysX joint drives (kp, kd from the metadata, implicit) ; step }

The IMU terms are MuJoCo's gyro and velocimeter at the scene's pelvis IMU site, computed
from the PhysX pelvis link state. The scorer reads the PhysX state through `PhysxState`,
which has the MujocoState methods TrackingScorer uses; the MuJoCo model is still loaded,
but only for site offsets (IMU, blade tip) and the soft joint limits.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from fencing.sim2sim import physx_plant
from fencing.sim2sim.observation import (
    MujocoState,
    ObservationInputs,
    build_observation,
    resolve_id,
)
from fencing.sim2sim.quat import quat_apply, quat_apply_inverse, quat_mul
from fencing.sim2sim.rollout import (
    Export,
    OnnxPolicy,
    Reference,
    RolloutConfig,
    TrackingScorer,
    soft_joint_limits,
)

SCENE_USD_NAME = "physx_scene.usda"  # written next to the output JSON
IMU_SITE = "imu_in_pelvis"  # the site of mjlab's robot/imu_ang_vel and robot/imu_lin_vel sensors


def imu_readings(link_quat, origin_lin_vel_w, ang_vel_w, site_pos, site_quat) -> tuple[np.ndarray, np.ndarray]:
    """(velocimeter, gyro) of a site on a link, in the site frame, from the link's world state."""
    site_velocity = np.asarray(origin_lin_vel_w) + np.cross(ang_vel_w, quat_apply(link_quat, site_pos))
    site_world = quat_mul(link_quat, site_quat)
    return quat_apply_inverse(site_world, site_velocity), quat_apply_inverse(site_world, ang_vel_w)


def check_step_dt(timestep: float, decimation: int, step_dt: float) -> None:
    """The physics step times the decimation must be the policy step, as in rollout.rollout."""
    if abs(timestep * decimation - step_dt) > 1e-9:
        raise ValueError(f"scene timestep {timestep} x decimation {decimation} != step_dt {step_dt}")


def check_imu_on_root(model: mujoco.MjModel, site: int, root_link_name: str) -> None:
    """The IMU readings are computed from the articulation root link's state, so the site must be on it."""
    body_name = physx_plant.strip_prefix(model.body(int(model.site_bodyid[site])).name)
    if body_name != root_link_name:
        raise ValueError(f"IMU site {model.site(site).name!r} is on body {body_name!r}, not the articulation "
                         f"root link {root_link_name!r}")


class PhysxState:
    """The MujocoState surface TrackingScorer reads, taken from a physx_plant.Snapshot."""

    def __init__(self, plant: physx_plant.PhysxPlant, contract):
        self.contract = contract
        self.body_links = plant.link_index(contract.body_names)
        self.anchor_link = plant.link_names.index(contract.anchor_body_name)

    def joint_pos(self, snapshot):
        return snapshot.q.copy()

    def anchor_pose(self, snapshot):
        return snapshot.link_pos[self.anchor_link].copy(), snapshot.link_quat[self.anchor_link].copy()

    def body_pos_w(self, snapshot):
        return snapshot.link_pos[self.body_links].copy()

    def body_quat_w(self, snapshot):
        return snapshot.link_quat[self.body_links].copy()

    def body_lin_vel_w(self, snapshot):
        return snapshot.link_lin_vel[self.body_links].copy()


def rollout_physx(export: Export, reference: Reference | None = None, config: RolloutConfig | None = None,
                  policy: OnnxPolicy | None = None, usd_options: physx_plant.UsdOptions | None = None,
                  usd_path: Path | None = None) -> dict:
    config = config or RolloutConfig()
    if config.pd_mode != "actuator":
        raise ValueError("the PhysX engine drives the joints with PhysX's implicit PD drives (pd_mode 'actuator')")
    if config.video is not None:
        raise ValueError("the PhysX engine does not render video")
    physx_plant.require_ovphysx()
    policy = policy or OnnxPolicy(export.onnx)
    contract = policy.contract
    if contract.joint_stiffness is None or contract.joint_damping is None:
        raise ValueError("the PhysX plant takes kp/kd from the ONNX metadata, which has none")
    model = mujoco.MjModel.from_xml_path(str(export.scene_xml))
    info = export.info
    decimation = int(info.get("decimation", 4))
    step_dt = float(info.get("step_dt", model.opt.timestep * decimation))
    check_step_dt(model.opt.timestep, decimation, step_dt)
    physics_dt = step_dt / decimation
    usd_path = usd_path or export.directory / SCENE_USD_NAME
    authored = physx_plant.build_usd(model, contract.joint_names, contract.joint_stiffness, contract.joint_damping,
                                     usd_path, usd_options)
    plant = physx_plant.PhysxPlant(usd_path, contract.joint_names)
    try:
        properties = physx_plant.property_report(plant, model, contract.joint_names, contract.joint_stiffness,
                                                 contract.joint_damping)
        result = _run(plant, model, export, reference, config, policy, decimation, step_dt, physics_dt)
    finally:
        plant.close()
    return {**result, "physx": {"authored": authored, "properties_vs_mujoco": properties,
                                "physics_dt": physics_dt}}


def _run(plant, model, export, reference, config, policy, decimation, step_dt, physics_dt) -> dict:
    contract = policy.contract
    state = PhysxState(plant, contract)
    site = resolve_id(model, mujoco.mjtObj.mjOBJ_SITE, IMU_SITE)
    site_pos, site_quat = model.site_pos[site].copy(), model.site_quat[site].copy()
    if reference is None:
        reference = policy.reference(int(export.info["motion_frames"]))
    motion_steps, truncated = reference.num_frames, False
    if config.seconds is not None and round(config.seconds / step_dt) < motion_steps:
        motion_steps, truncated = max(1, round(config.seconds / step_dt)), True
    hold_steps = round(config.hold_seconds / step_dt)
    scorer = TrackingScorer(model, state, reference, motion_steps, hold_steps, step_dt, key_frames=config.key_frames)
    pelvis, anchor = scorer.pelvis_index, scorer.anchor_index

    # reset_to_frame with play-mode ranges: root on frame 0 (world velocities), joints clipped.
    rng = np.random.default_rng(config.seed)
    joint_pos = reference.joint_pos[0].copy()
    if config.reset_joint_noise:
        joint_pos += rng.uniform(-config.reset_joint_noise, config.reset_joint_noise, size=joint_pos.shape)
    low, high = soft_joint_limits(model, MujocoState(model, contract))
    plant.reset(reference.body_pos_w[0, pelvis], reference.body_quat_w[0, pelvis], reference.body_lin_vel_w[0, pelvis],
                reference.body_ang_vel_w[0, pelvis], np.clip(joint_pos, low, high), reference.joint_vel[0])
    snapshot = plant.snapshot()
    last_action = np.zeros(contract.num_joints)
    root = plant.root_link
    check_imu_on_root(model, site, plant.link_names[root])
    for step in range(scorer.total_steps):
        frame = scorer.frame(step)
        imu_lin, imu_ang = imu_readings(snapshot.link_quat[root], snapshot.link_lin_vel[root],
                                        snapshot.link_ang_vel[root], site_pos, site_quat)
        anchor_pos, anchor_quat = state.anchor_pose(snapshot)
        inputs = ObservationInputs(
            ref_joint_pos=reference.joint_pos[frame], ref_joint_vel=reference.joint_vel[frame],
            ref_anchor_pos_w=reference.body_pos_w[frame, anchor], ref_anchor_quat_w=reference.body_quat_w[frame, anchor],
            robot_anchor_pos_w=anchor_pos, robot_anchor_quat_w=anchor_quat, imu_lin_vel=imu_lin, imu_ang_vel=imu_ang,
            joint_pos=snapshot.q, joint_vel=snapshot.qd, last_action=last_action)
        scorer.begin_step(step, anchor_pos, anchor_quat)
        action = policy.run(build_observation(inputs, contract), frame)["actions"].astype(np.float64)
        plant.set_targets(contract.default_joint_pos + contract.action_scale * action)
        for _ in range(decimation):
            plant.step(physics_dt)
        snapshot = plant.snapshot()
        last_action = action
        if not scorer.end_step(step, snapshot) and config.stop_on_termination:
            break

    scored = scorer.result()
    return {
        "reference_source": reference.source,
        "pd_mode": config.pd_mode,
        "metadata_full_precision": contract.full_precision,
        "gains_vs_metadata": {"kp_max_abs_diff": None, "kd_max_abs_diff": None},  # PhysX gains *are* the metadata's
        "physics": {"timestep": physics_dt, "decimation": decimation, "step_dt": step_dt, "solver": "TGS"},
        "motion_frames": scored["motion_frames"],
        "motion_steps": motion_steps,
        "hold_steps": hold_steps,
        "truncated": truncated,
        "steps_run": scored["steps_run"],
        "terminated_at_step": scored["terminated_at_step"],
        "terminated_at_s": scored["terminated_at_s"],
        "termination_terms": scored["termination_terms"],
        "has_blade": scored["has_blade"],
        "video": None,
        "nominal": scored["nominal"],
    }
