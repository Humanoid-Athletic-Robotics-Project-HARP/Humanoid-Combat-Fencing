"""mjlab task: the standard G1 tracking task with the saber bolted to the right wrist.

Runs in the mjlab venv (needs torch + mjlab). Registered from our code so third_party
stays untouched:

    from fencing.saber.mjlab_task import register, TASK_ID
    register()  # safe to call any number of times
    # then train TASK_ID like Mjlab-Tracking-Flat-Unitree-G1

Differences from Mjlab-Tracking-Flat-Unitree-G1:
  * robot spec = upstream G1 MJCF + fencing.saber.spec.attach_saber (body "saber", sites
    "blade_tip" and "guard"); the saber is the last body, so plain-robot NPZs still work;
  * startup event "saber_mass" randomizes the saber's mass and inertia together
    (dr.pseudo_inertia, uniform density scale) over MASS_RANDOMIZATION_KG;
  * play configs drop that event, so evaluation runs with the nominal saber;
  * right wrist pitch/yaw action scale SABER_WRIST_ACTION_SCALE instead of upstream 0.075
    (see saber_action_scale);
  * extra reward term "blade_tip_pos" (blade_tip_position_error_exp): the blade tip as the
    eval gate measures it, so a wrist rotation error the 0.4 rad body-orientation term
    barely notices is paid for at the tip, 0.75 m out (train and play configs);
  * runs are logged under experiment "g1_saber_tracking".
"""

from __future__ import annotations

import functools
import math
import re

import mujoco

from fencing.saber.spec import (
    MASS_RANDOMIZATION_KG, NOMINAL_SABER, PARENT_BODY, SABER_BODY, SaberSpec, attach_saber, tip_in_hand,
)

TASK_ID = "Fencing-Tracking-Flat-G1-Saber"
TASK_ID_NO_STATE_ESTIMATION = "Fencing-Tracking-Flat-G1-Saber-No-State-Estimation"
EXPERIMENT_NAME = "g1_saber_tracking"
SABER_MASS_EVENT = "saber_mass"
# rad of PD target per unit action for the right wrist pitch/yaw. Upstream uses
# 0.25 * effort / stiffness = 0.075 for the 5 Nm wrist motors, so the ~1 rad wrist angles
# the fixed 30 deg mount needs for an en garde guard (ADR 0002) take actions of ~13-18 and
# were not learned in 1000 iterations (extension_v1: wrist pitch error 0.71 rad). 0.25 puts
# them in the range the other arm joints use. PD gains and torque limits are unchanged, and
# the scale is exported in the ONNX metadata, so deployment picks it up.
SABER_WRIST_ACTION_SCALE = 0.25
SABER_WRIST_JOINTS = ("right_wrist_pitch_joint", "right_wrist_yaw_joint")
BLADE_TIP_REWARD = "blade_tip_pos"
# Same scale as the upstream body terms (motion_body_pos: weight 1.0, std 0.3 m). std 0.15 m
# puts the steep part of exp(-e^2/std^2) around the 10 cm blade-tip gate: 0.64 at 10 cm,
# 0.29 at the 16.8 cm extension_v2 left, 0.89 at 5 cm. That 0.3 rad wrist yaw error costs
# ~0.04 of motion_body_ori (one of 14 bodies, std 0.4 rad) but ~0.7 of this term.
BLADE_TIP_REWARD_WEIGHT = 1.0
BLADE_TIP_REWARD_STD_M = 0.15


def saber_action_scale(base: dict[str, float]) -> dict[str, float]:
    """Upstream per-joint action scale (regex keys) with the saber wrist joints widened.

    The ".*_wrist_pitch_joint"-style keys are split into explicit left/right entries so no
    joint is matched by two keys. Returns a new dict; `base` is not modified.
    """
    scale: dict[str, float] = {}
    for key, value in base.items():
        sides = [joint for joint in SABER_WRIST_JOINTS if re.fullmatch(key, joint)]
        if not sides:
            scale[key] = value
            continue
        for joint in sides:
            scale[joint] = SABER_WRIST_ACTION_SCALE
        left = [j.replace("right_", "left_") for j in sides]
        scale.update({j: value for j in left if re.fullmatch(key, j)})
    return scale


def saber_alpha_range(saber: SaberSpec = NOMINAL_SABER,
                      mass_range_kg: tuple[float, float] = MASS_RANDOMIZATION_KG
                      ) -> tuple[float, float]:
    """dr.pseudo_inertia scales mass (and inertia) by exp(2*alpha); invert that."""
    low, high = mass_range_kg
    if not 0 < low <= high:
        raise ValueError(f"mass range must satisfy 0 < low <= high, got {mass_range_kg}")
    nominal = saber.total_mass
    return (0.5 * math.log(low / nominal), 0.5 * math.log(high / nominal))


def saber_g1_spec(saber: SaberSpec = NOMINAL_SABER) -> mujoco.MjSpec:
    # mjlab's own get_spec, so the robot is exactly what the upstream task trains.
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    return attach_saber(get_spec(), saber)


def saber_g1_robot_cfg(saber: SaberSpec = NOMINAL_SABER):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    cfg = get_g1_robot_cfg()
    cfg.spec_fn = functools.partial(saber_g1_spec, saber)
    return cfg


@functools.lru_cache(maxsize=None)
def _tip_tensor(tip_in_wrist: tuple[float, float, float], device, dtype):
    import torch

    return torch.tensor(tip_in_wrist, device=device, dtype=dtype)


def blade_tip_position_error_exp(env, command_name: str, std: float, wrist_body: str,
                                 tip_in_wrist: tuple[float, float, float]):
    """exp(-|tip_ref - tip_robot|^2 / std^2) per env, the tip fixed at `tip_in_wrist` in
    `wrist_body`'s frame.

    Same definition as evaluate's blade_tip_error_m: the reference wrist pose is the
    motion command's body_pos/quat_relative_w (re-anchored to the robot like
    motion_body_pos), the robot's is its world wrist pose.
    """
    from mjlab.utils.lab_api.math import quat_apply

    command = env.command_manager.get_term(command_name)
    body_names = list(command.cfg.body_names)
    if wrist_body not in body_names:
        raise KeyError(f"blade tip reward: {wrist_body!r} is not a tracked body of command "
                       f"{command_name!r} (body_names={body_names})")
    wrist = body_names.index(wrist_body)
    robot_pos = command.robot_body_pos_w[:, wrist]
    tip = _tip_tensor(tuple(tip_in_wrist), robot_pos.device, robot_pos.dtype).expand_as(robot_pos)
    ref_tip = command.body_pos_relative_w[:, wrist] + quat_apply(command.body_quat_relative_w[:, wrist], tip)
    robot_tip = robot_pos + quat_apply(command.robot_body_quat_w[:, wrist], tip)
    error = (ref_tip - robot_tip).square().sum(dim=-1)
    return (-error / std**2).exp()


def blade_tip_reward_cfg(saber: SaberSpec = NOMINAL_SABER):
    from mjlab.managers.reward_manager import RewardTermCfg

    return RewardTermCfg(
        func=blade_tip_position_error_exp,
        weight=BLADE_TIP_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "std": BLADE_TIP_REWARD_STD_M,
            "wrist_body": PARENT_BODY,
            # plain floats so the logged env.yaml stays readable
            "tip_in_wrist": tuple(float(x) for x in tip_in_hand(saber)),
        },
    )


def saber_tracking_env_cfg(saber: SaberSpec = NOMINAL_SABER, has_state_estimation: bool = True,
                           play: bool = False,
                           mass_range_kg: tuple[float, float] = MASS_RANDOMIZATION_KG):
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=has_state_estimation, play=play)
    cfg.scene.entities["robot"] = saber_g1_robot_cfg(saber)
    joint_pos_action = cfg.actions["joint_pos"]
    joint_pos_action.scale = saber_action_scale(dict(joint_pos_action.scale))
    cfg.rewards[BLADE_TIP_REWARD] = blade_tip_reward_cfg(saber)
    if not play:
        # Startup = one draw per env for the whole run; with thousands of envs that covers
        # the range, and it keeps the mass constant within an episode like a real weapon.
        cfg.events[SABER_MASS_EVENT] = EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(SABER_BODY,)),
                "alpha_range": saber_alpha_range(saber, mass_range_kg),
            },
        )
    return cfg


def saber_rl_cfg():
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = EXPERIMENT_NAME
    return cfg


def register() -> list[str]:
    """Register the saber tracking tasks with mjlab. Idempotent; returns the task ids."""
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    registered = set(list_tasks())
    for task_id, has_state_estimation in ((TASK_ID, True), (TASK_ID_NO_STATE_ESTIMATION, False)):
        if task_id in registered:
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=saber_tracking_env_cfg(has_state_estimation=has_state_estimation),
            play_env_cfg=saber_tracking_env_cfg(has_state_estimation=has_state_estimation,
                                                play=True),
            rl_cfg=saber_rl_cfg(),
            runner_cls=MotionTrackingOnPolicyRunner,
        )
    return [TASK_ID, TASK_ID_NO_STATE_ESTIMATION]
