"""Rebuild mjlab's tracking actor observation from plain MuJoCo state + the reference.

Mirrors, term by term, `make_tracking_env_cfg()`'s actor group (mjlab v1.6.0):

    command              [ref joint_pos, ref joint_vel]                  mdp.generated_commands
    motion_anchor_pos_b  ref anchor position in the robot anchor frame   mdp.motion_anchor_pos_b
    motion_anchor_ori_b  first two columns of R(robot anchor^-1 * ref)   mdp.motion_anchor_ori_b
    base_lin_vel         sensor robot/imu_lin_vel (velocimeter, pelvis)  mdp.builtin_sensor
    base_ang_vel         sensor robot/imu_ang_vel (gyro, pelvis)         mdp.builtin_sensor
    joint_pos            q - default (encoder bias: none in sim2sim)     mdp.joint_pos_rel
    joint_vel            qd (default joint vel is zero)                  mdp.joint_vel_rel
    actions              last raw policy action                          mdp.last_action

Which terms are present, and in which order, comes from the ONNX metadata
(`observation_names`), so the No-State-Estimation variant (no anchor pos, no lin vel)
works too. Each term is clipped then scaled like mjlab's ObservationManager.

Also here: the tracking command's re-anchored reference (`body_pos_relative_w`) and the
task's terminations, both needed to score a rollout the way mjlab does.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fencing.sim2sim.contract import PolicyContract
from fencing.sim2sim.quat import (
    matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul,
    subtract_frame_transforms, yaw_quat,
)

ENTITY_PREFIX = "robot/"  # mjlab names every robot element "robot/<name>" in the scene
# Observation term -> MuJoCo sensor, as configured in tracking_env_cfg.py.
SENSOR_TERMS = {"base_lin_vel": "imu_lin_vel", "base_ang_vel": "imu_ang_vel"}
KNOWN_TERMS = ("command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel",
               "base_ang_vel", "joint_pos", "joint_vel", "actions")

# Terminations of Mjlab-Tracking-Flat-Unitree-G1 (tracking_env_cfg.py + g1/env_cfgs.py).
ANCHOR_Z_THRESHOLD = 0.25
ANCHOR_ORI_THRESHOLD = 0.8
EE_Z_THRESHOLD = 0.25
EE_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass(frozen=True)
class ObservationInputs:
    """Everything the actor observation depends on, for one env (no batch axis)."""

    ref_joint_pos: np.ndarray  # (J,)
    ref_joint_vel: np.ndarray  # (J,)
    ref_anchor_pos_w: np.ndarray  # (3,)
    ref_anchor_quat_w: np.ndarray  # (4,) wxyz
    robot_anchor_pos_w: np.ndarray  # (3,)
    robot_anchor_quat_w: np.ndarray  # (4,)
    imu_lin_vel: np.ndarray  # (3,) IMU frame
    imu_ang_vel: np.ndarray  # (3,)
    joint_pos: np.ndarray  # (J,) policy joint order
    joint_vel: np.ndarray  # (J,)
    last_action: np.ndarray  # (J,)


def observation_terms(inputs: ObservationInputs, default_joint_pos: np.ndarray) -> dict[str, np.ndarray]:
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
        inputs.robot_anchor_pos_w, inputs.robot_anchor_quat_w,
        inputs.ref_anchor_pos_w, inputs.ref_anchor_quat_w,
    )
    return {
        "command": np.concatenate([inputs.ref_joint_pos, inputs.ref_joint_vel]),
        "motion_anchor_pos_b": anchor_pos_b,
        # mat[..., :2] keeps the first two *columns*; reshape flattens row-major.
        "motion_anchor_ori_b": matrix_from_quat(anchor_quat_b)[..., :2].reshape(-1),
        "base_lin_vel": np.asarray(inputs.imu_lin_vel, dtype=np.float64),
        "base_ang_vel": np.asarray(inputs.imu_ang_vel, dtype=np.float64),
        "joint_pos": np.asarray(inputs.joint_pos, dtype=np.float64) - default_joint_pos,
        "joint_vel": np.asarray(inputs.joint_vel, dtype=np.float64),
        "actions": np.asarray(inputs.last_action, dtype=np.float64),
    }


def build_observation(inputs: ObservationInputs, contract: PolicyContract) -> np.ndarray:
    """The flat float32 actor observation, terms in the contract's order."""
    terms = observation_terms(inputs, contract.default_joint_pos)
    parts = []
    for name, scale, (low, high), history in zip(
        contract.observation_names, contract.observation_scales, contract.observation_clips,
        contract.observation_history_lengths,
    ):
        if name not in terms:
            raise NotImplementedError(f"observation term {name!r} is not implemented (know {KNOWN_TERMS})")
        if history:
            raise NotImplementedError(f"observation term {name!r} uses history_length={history}")
        value = np.clip(terms[name], low, high) * np.asarray(scale, dtype=np.float64)
        parts.append(value.reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def relative_body_poses(ref_anchor_pos_w, ref_anchor_quat_w, robot_anchor_pos_w, robot_anchor_quat_w,
                        ref_body_pos_w, ref_body_quat_w) -> tuple[np.ndarray, np.ndarray]:
    """MotionCommand.update_relative_body_poses: the reference re-anchored to the robot's
    anchor x/y and heading (keeping the reference anchor height). Leading axes are batch,
    bodies on axis -2. Returns (body_pos_relative_w, body_quat_relative_w)."""
    ref_anchor_pos_w = np.asarray(ref_anchor_pos_w, dtype=np.float64)[..., None, :]
    ref_anchor_quat_w = np.asarray(ref_anchor_quat_w, dtype=np.float64)[..., None, :]
    delta_pos = np.array(robot_anchor_pos_w, dtype=np.float64)[..., None, :].copy()
    delta_pos[..., 2] = ref_anchor_pos_w[..., 2]
    delta_ori = yaw_quat(quat_mul(np.asarray(robot_anchor_quat_w)[..., None, :], quat_inv(ref_anchor_quat_w)))
    delta_ori = np.broadcast_to(delta_ori, np.shape(ref_body_quat_w))
    body_quat = quat_mul(delta_ori, ref_body_quat_w)
    body_pos = delta_pos + quat_apply(delta_ori, np.asarray(ref_body_pos_w) - ref_anchor_pos_w)
    return body_pos, body_quat


def tracking_terminated(ref_anchor_pos_w, ref_anchor_quat_w, robot_anchor_pos_w, robot_anchor_quat_w,
                        ref_ee_pos_relative_w, robot_ee_pos_w) -> dict[str, bool]:
    """The task's three failure terminations for one env (time_out excluded)."""
    gravity = np.array([0.0, 0.0, -1.0])
    ref_gravity_b = quat_apply_inverse(ref_anchor_quat_w, gravity)
    robot_gravity_b = quat_apply_inverse(robot_anchor_quat_w, gravity)
    return {
        "anchor_pos": bool(abs(ref_anchor_pos_w[2] - robot_anchor_pos_w[2]) > ANCHOR_Z_THRESHOLD),
        "anchor_ori": bool(abs(ref_gravity_b[2] - robot_gravity_b[2]) > ANCHOR_ORI_THRESHOLD),
        "ee_body_pos": bool(np.any(np.abs(np.asarray(ref_ee_pos_relative_w)[:, 2]
                                          - np.asarray(robot_ee_pos_w)[:, 2]) > EE_Z_THRESHOLD)),
    }


def resolve_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    """Id of `name` or `robot/<name>` (exported mjlab scenes prefix the entity name)."""
    for candidate in (name, ENTITY_PREFIX + name):
        index = mujoco.mj_name2id(model, objtype, candidate)
        if index >= 0:
            return index
    raise KeyError(f"{objtype.name} {name!r} (or {ENTITY_PREFIX}{name}) not in the model")


def body_origin_lin_vel_w(model: mujoco.MjModel, data: mujoco.MjData, body_ids) -> np.ndarray:
    """(B, 3) world-frame linear velocity of each body frame's origin.

    mjOBJ_XBODY, not mjOBJ_BODY: the latter is the inertial frame (centre of mass), which
    sits up to ~8 cm from the G1 link origins and would bias foot slip and pelvis speed.
    """
    out = np.empty((len(body_ids), 3))
    velocity = np.empty(6)
    for i, body in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(body), velocity, 0)
        out[i] = velocity[3:]
    return out


class MujocoState:
    """Reads the robot side of the observation (and metric inputs) from MjData."""

    def __init__(self, model: mujoco.MjModel, contract: PolicyContract):
        self.model = model
        self.contract = contract
        joint_ids = [resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in contract.joint_names]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.qvel_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.anchor_body = resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, contract.anchor_body_name)
        self.body_ids = np.array([resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in contract.body_names])
        self.sensor_slices = {}
        for term, sensor_name in SENSOR_TERMS.items():
            sensor = resolve_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            start = model.sensor_adr[sensor]
            self.sensor_slices[term] = slice(start, start + model.sensor_dim[sensor])

    def joint_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def joint_vel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_adr].copy()

    def anchor_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.xpos[self.anchor_body].copy(), data.xquat[self.anchor_body].copy()

    def body_pos_w(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.body_ids].copy()

    def body_quat_w(self, data: mujoco.MjData) -> np.ndarray:
        return data.xquat[self.body_ids].copy()

    def body_lin_vel_w(self, data: mujoco.MjData) -> np.ndarray:
        """World-frame linear velocity of each tracked body's origin (mjlab body_link_lin_vel_w)."""
        return body_origin_lin_vel_w(self.model, data, self.body_ids)

    def inputs(self, data: mujoco.MjData, ref_joint_pos, ref_joint_vel, ref_anchor_pos_w,
               ref_anchor_quat_w, last_action) -> ObservationInputs:
        """Call after mj_forward so kinematics and sensors match qpos/qvel."""
        anchor_pos, anchor_quat = self.anchor_pose(data)
        return ObservationInputs(
            ref_joint_pos=np.asarray(ref_joint_pos, dtype=np.float64),
            ref_joint_vel=np.asarray(ref_joint_vel, dtype=np.float64),
            ref_anchor_pos_w=np.asarray(ref_anchor_pos_w, dtype=np.float64),
            ref_anchor_quat_w=np.asarray(ref_anchor_quat_w, dtype=np.float64),
            robot_anchor_pos_w=anchor_pos,
            robot_anchor_quat_w=anchor_quat,
            imu_lin_vel=data.sensordata[self.sensor_slices["base_lin_vel"]].copy(),
            imu_ang_vel=data.sensordata[self.sensor_slices["base_ang_vel"]].copy(),
            joint_pos=self.joint_pos(data),
            joint_vel=self.joint_vel(data),
            last_action=np.asarray(last_action, dtype=np.float64),
        )
