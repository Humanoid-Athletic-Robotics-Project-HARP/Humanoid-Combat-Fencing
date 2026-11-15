"""mjlab task: the saber task with a bending blade (fencing.saber.flex).

Runs in the mjlab venv. Registered like the saber task (fencing.train.mjlab_env registers both):

    from fencing.saber.mjlab_flex_task import register, FLEX_TASK_ID
    register()  # safe to call any number of times

Differences from Fencing-Tracking-Flat-G1-Saber (fencing.saber.mjlab_task):
  * the blade is FLEX_SABER's: spring hinges between FLEX_SABER.flex_segments pieces
    (docs/saber_flex.md). check_training_blade refuses settings that blow up at the training
    timestep anywhere in the randomization range;
  * the policy's inputs hold the 29 robot joints only, since the robot has no sensor on the blade;
    the critic, which is not exported, also gets the blade hinge angles and speeds;
  * the motion command compares and resets the 29 robot joints; the blade starts straight and still;
  * "blade_tip_pos" rewards the real tip (the blade_tip site at the end of the chain) against the
    reference tip, which stays the straight blade's tip on the reference wrist: where the tip should
    be does not depend on the blade;
  * startup event "saber_mass" scales the whole saber's mass and inertia (handle and every blade
    piece) by one factor per env over the saber task's mass range, and picks each direction's
    wobble frequency (at the nominal mass) from FLATWISE_HZ_RANGE and EDGEWISE_HZ_RANGE. The hinge
    damping follows both, so the damping ratio stays FLEX_SABER.flex_damping_ratio;
  * the ONNX metadata lists the 29 robot joints, so the policy file reads like a saber-task policy;
  * the viewer shows reference frames instead of the ghost robot (the ghost expects 29 joints).

Rewards, actions, terminations and every other term are the saber task's.
"""

from __future__ import annotations

import dataclasses
import functools
import math

from fencing.saber import flex
from fencing.saber.mjlab_task import (
    BLADE_TIP_REWARD,
    BLADE_TIP_REWARD_STD_M,
    BLADE_TIP_REWARD_WEIGHT,
    SABER_MASS_EVENT,
    _tip_tensor,
    saber_alpha_range,
    saber_g1_spec,
    saber_tracking_env_cfg,
)
from fencing.saber.spec import (
    MASS_RANDOMIZATION_KG,
    NOMINAL_SABER,
    PARENT_BODY,
    SABER_BODY,
    TIP_SITE,
    tip_in_hand,
)

FLEX_TASK_ID = "Fencing-Tracking-Flat-G1-Saber-Bending"
# Foam practice saber estimates (docs/saber_flex.md, "Results"): 4-6 Hz towards the flat side,
# 10-16 Hz towards the edge, at the nominal 0.2 kg. The middle of each range is the nominal blade.
FLATWISE_HZ_RANGE = (4.0, 6.0)
EDGEWISE_HZ_RANGE = (10.0, 16.0)
# One piece: the stiffest edge (16 Hz) on the lightest saber (0.12 kg) wobbles at 20.7 Hz, under
# the 30 Hz one piece survives at a 5 ms step; two pieces blow up above 12.3 Hz (docs/saber_flex.md,
# "Stability"). One and two pieces gave the same pass/fail on the eval gates.
FLEX_SABER = dataclasses.replace(
    NOMINAL_SABER, flex_segments=1,
    flex_f1_flatwise_hz=sum(FLATWISE_HZ_RANGE) / 2, flex_f1_edgewise_hz=sum(EDGEWISE_HZ_RANGE) / 2)

ROBOT_JOINTS = r"(?!saber_flex_).*"  # every joint but the blade hinges (flex.add_bending_blade names)
BLADE_JOINTS = r"saber_flex_.*"
EDGEWISE_JOINTS = r"saber_flex_\d+_y"  # bend towards the edge
FLATWISE_JOINTS = r"saber_flex_\d+_z"  # bend towards the flat side
SABER_BODIES = (SABER_BODY, r"saber_blade_\d+")


def check_training_blade(saber=FLEX_SABER, timestep: float = 0.005, integrator: str = "implicitfast",
                         mass_range_kg=MASS_RANDOMIZATION_KG, flatwise_hz=FLATWISE_HZ_RANGE,
                         edgewise_hz=EDGEWISE_HZ_RANGE) -> None:
    """Raise ValueError if the fastest blade training can draw blows up at `timestep`.

    A lighter saber with the same hinge stiffness wobbles faster (frequency ~ 1/sqrt(mass)), so the
    fastest case is the stiffest draw in each direction on the lightest saber.
    """
    import mujoco

    lightest = mass_range_kg[0]
    speedup = math.sqrt(saber.total_mass / lightest)
    fastest = flex.FlexBlade(saber.flex_segments, flatwise_hz[1] * speedup, saber.flex_damping_ratio,
                             edgewise_hz[1] * speedup)
    integrators = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
    flex.check_stable(fastest, timestep, saber.with_total_mass(lightest), int(integrators[integrator]))


def flex_robot_cfg(saber=FLEX_SABER):
    """The saber task's robot with the bending blade (saber_g1_robot_cfg refuses one)."""
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    cfg = get_g1_robot_cfg()
    cfg.spec_fn = functools.partial(saber_g1_spec, saber)
    return cfg


# --- motion command ------------------------------------------------------------------------------

def bending_motion_command_cfg(cfg):
    """A copy of the tracking task's MotionCommandCfg that builds BendingMotionCommand."""
    import torch
    from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

    class BendingMotionCommand(MotionCommand):
        """MotionCommand for a robot with extra passive joints (the blade hinges).

        The reference holds the 29 robot joints. Comparisons use those joints only, and a reset
        writes them while the blade hinges start straight and still.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            ids, _ = self.robot.find_joints(ROBOT_JOINTS, preserve_order=True)
            if len(ids) != self.motion.joint_pos.shape[1]:
                raise ValueError(f"the motion has {self.motion.joint_pos.shape[1]} joints, the robot "
                                 f"{len(ids)} (without the blade hinges)")
            self.robot_joint_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

        @property
        def robot_joint_pos(self):
            return self.robot.data.joint_pos[:, self.robot_joint_ids]

        @property
        def robot_joint_vel(self):
            return self.robot.data.joint_vel[:, self.robot_joint_ids]

        def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel,
                                          joint_pos, joint_vel):
            shape = (joint_pos.shape[0], self.robot.num_joints)
            full_pos = torch.zeros(shape, dtype=joint_pos.dtype, device=joint_pos.device)
            full_vel = torch.zeros(shape, dtype=joint_vel.dtype, device=joint_vel.device)
            full_pos[:, self.robot_joint_ids] = joint_pos
            full_vel[:, self.robot_joint_ids] = joint_vel
            super()._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel,
                                                  full_pos, full_vel)

    @dataclasses.dataclass(kw_only=True)
    class BendingMotionCommandCfg(MotionCommandCfg):
        def build(self, env):
            return BendingMotionCommand(self, env)

    values = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.init}
    new = BendingMotionCommandCfg(**values)
    new.viz.mode = "frames"
    return new


# --- reward --------------------------------------------------------------------------------------

def bending_blade_tip_error_exp(env, command_name: str, std: float, wrist_body: str,
                                tip_in_wrist: tuple[float, float, float], tip_site: str):
    """exp(-|tip_ref - tip_robot|^2 / std^2) per env, the robot's tip read from `tip_site`.

    The reference tip is the saber task's (`tip_in_wrist` fixed in the reference wrist frame);
    the robot's is where the bending blade's tip actually is.
    """
    from mjlab.utils.lab_api.math import quat_apply

    command = env.command_manager.get_term(command_name)
    body_names = list(command.cfg.body_names)
    if wrist_body not in body_names:
        raise KeyError(f"blade tip reward: {wrist_body!r} is not a tracked body of command "
                       f"{command_name!r} (body_names={body_names})")
    wrist = body_names.index(wrist_body)
    ref_pos = command.body_pos_relative_w[:, wrist]
    tip = _tip_tensor(tuple(tip_in_wrist), ref_pos.device, ref_pos.dtype).expand_as(ref_pos)
    ref_tip = ref_pos + quat_apply(command.body_quat_relative_w[:, wrist], tip)
    robot_tip = command.robot.data.site_pos_w[:, command.robot.site_names.index(tip_site)]
    error = (ref_tip - robot_tip).square().sum(dim=-1)
    return (-error / std**2).exp()


def bending_blade_tip_reward_cfg(saber=FLEX_SABER):
    from mjlab.managers.reward_manager import RewardTermCfg

    return RewardTermCfg(
        func=bending_blade_tip_error_exp,
        weight=BLADE_TIP_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "std": BLADE_TIP_REWARD_STD_M,
            "wrist_body": PARENT_BODY,
            "tip_in_wrist": tuple(float(x) for x in tip_in_hand(saber)),
            "tip_site": TIP_SITE,
        },
    )


# --- saber randomization -------------------------------------------------------------------------

def _defaults(env, field: str, env_ids, ids):
    """Default (unrandomized) model values as (env, entity, ...)."""
    import torch

    default = env.sim.get_default_field(field)
    if field in env.sim.per_world_default_fields:
        env_grid, id_grid = torch.meshgrid(env_ids, ids, indexing="ij")
        return default[env_grid, id_grid]
    return default[ids].unsqueeze(0).expand((env_ids.numel(),) + default[ids].shape)


def randomize_bending_saber(env, env_ids, alpha_range: tuple[float, float],
                            flatwise_scale_range: tuple[float, float], edgewise_scale_range: tuple[float, float],
                            saber_cfg, flatwise_cfg, edgewise_cfg) -> None:
    """One mass scale for the whole saber, and each direction's wobble, per env.

    Mass and inertia of every saber body scale by a = exp(2 alpha), centres of mass unchanged (what
    dr.pseudo_inertia does with alpha alone, but one draw for all the bodies). Each direction's
    wobble frequency at the nominal mass scales by s (hinge stiffness by s^2), and the hinge damping
    by s * sqrt(a), which keeps the damping ratio of the stiffness-proportional damping.
    """
    import torch
    from mjlab.utils.lab_api.math import sample_uniform

    robot = env.scene[saber_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    env_ids = env_ids.to(env.device, dtype=torch.int)
    n = env_ids.numel()
    model = env.sim.model

    def uniform(r):
        return sample_uniform(torch.tensor(r[0], device=env.device), torch.tensor(r[1], device=env.device),
                              (n,), env.device)

    mass_scale = torch.exp(2 * uniform(alpha_range))
    bodies = robot.indexing.body_ids[saber_cfg.body_ids]
    env_grid, body_grid = torch.meshgrid(env_ids, bodies, indexing="ij")
    model.body_mass[env_grid, body_grid] = _defaults(env, "body_mass", env_ids, bodies) * mass_scale[:, None]
    model.body_inertia[env_grid, body_grid] = (_defaults(env, "body_inertia", env_ids, bodies)
                                               * mass_scale[:, None, None])
    for joints_cfg, scale_range in ((flatwise_cfg, flatwise_scale_range), (edgewise_cfg, edgewise_scale_range)):
        s = uniform(scale_range)
        joints = robot.indexing.joint_ids[joints_cfg.joint_ids]
        dofs = robot.indexing.joint_v_adr[joints_cfg.joint_ids]
        env_grid, joint_grid = torch.meshgrid(env_ids, joints, indexing="ij")
        model.jnt_stiffness[env_grid, joint_grid] = (_defaults(env, "jnt_stiffness", env_ids, joints)
                                                     * (s * s)[:, None])
        env_grid, dof_grid = torch.meshgrid(env_ids, dofs, indexing="ij")
        model.dof_damping[env_grid, dof_grid] = (_defaults(env, "dof_damping", env_ids, dofs)
                                                 * (s * mass_scale.sqrt())[:, None])


def saber_randomization_event_cfg(saber=FLEX_SABER, mass_range_kg=MASS_RANDOMIZATION_KG,
                                  flatwise_hz=FLATWISE_HZ_RANGE, edgewise_hz=EDGEWISE_HZ_RANGE):
    from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    func = requires_model_fields("body_mass", "body_inertia", "jnt_stiffness", "dof_damping",
                                 recompute=RecomputeLevel.set_const)(randomize_bending_saber)
    edge_nominal = saber.flex_f1_edgewise_hz or saber.flex_f1_flatwise_hz
    return EventTermCfg(
        mode="startup",
        func=func,
        params={
            "alpha_range": saber_alpha_range(saber, mass_range_kg),
            "flatwise_scale_range": tuple(f / saber.flex_f1_flatwise_hz for f in flatwise_hz),
            "edgewise_scale_range": tuple(f / edge_nominal for f in edgewise_hz),
            "saber_cfg": SceneEntityCfg("robot", body_names=SABER_BODIES),
            "flatwise_cfg": SceneEntityCfg("robot", joint_names=(FLATWISE_JOINTS,)),
            "edgewise_cfg": SceneEntityCfg("robot", joint_names=(EDGEWISE_JOINTS,)),
        },
    )


# --- task ----------------------------------------------------------------------------------------

def flex_tracking_env_cfg(saber=FLEX_SABER, has_state_estimation: bool = True, play: bool = False,
                          mass_range_kg=MASS_RANDOMIZATION_KG):
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.tracking import mdp

    cfg = saber_tracking_env_cfg(has_state_estimation=has_state_estimation, play=play, mass_range_kg=mass_range_kg)
    check_training_blade(saber, cfg.sim.mujoco.timestep, cfg.sim.mujoco.integrator, mass_range_kg)
    cfg.scene.entities["robot"] = flex_robot_cfg(saber)
    cfg.commands["motion"] = bending_motion_command_cfg(cfg.commands["motion"])

    for group in ("actor", "critic"):
        for name in ("joint_pos", "joint_vel"):
            cfg.observations[group].terms[name].params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=(ROBOT_JOINTS,))
    blade = SceneEntityCfg("robot", joint_names=(BLADE_JOINTS,))
    cfg.observations["critic"].terms["blade_pos"] = ObservationTermCfg(func=mdp.joint_pos_rel,
                                                                       params={"asset_cfg": blade})
    cfg.observations["critic"].terms["blade_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel, params={"asset_cfg": dataclasses.replace(blade)})

    cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(ROBOT_JOINTS,))
    cfg.rewards[BLADE_TIP_REWARD] = bending_blade_tip_reward_cfg(saber)
    if SABER_MASS_EVENT in cfg.events:  # train configs only, as in the saber task
        cfg.events[SABER_MASS_EVENT] = saber_randomization_event_cfg(saber, mass_range_kg)
    return cfg


def _robot_joint_indices(joint_names) -> list[int]:
    return [i for i, name in enumerate(joint_names) if not name.startswith("saber_flex_")]


def drop_blade_joints(metadata: dict) -> dict:
    """Policy metadata (mjlab's get_base_metadata) without the blade hinges, which have no motor.

    `joint_names` and `default_joint_pos` list every joint; the policy, the action scale and the
    PD gains cover the 29 motors. Apply before anything copies those lists (e.g. hcf_full_precision).
    """
    keep = _robot_joint_indices(metadata["joint_names"])
    out = dict(metadata)
    for key in ("joint_names", "default_joint_pos"):
        out[key] = [metadata[key][i] for i in keep]
    return out


def robot_joints_only(onnx_path) -> None:
    """drop_blade_joints for an ONNX file whose metadata is already attached (the training-time save)."""
    import onnx

    model = onnx.load(str(onnx_path))
    props = {entry.key: entry for entry in model.metadata_props}
    keep = _robot_joint_indices(props["joint_names"].value.split(","))
    for key in ("joint_names", "default_joint_pos"):
        values = props[key].value.split(",")
        props[key].value = ",".join(values[i] for i in keep)
    onnx.save(model, str(onnx_path))


def flex_runner_cls():
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    class BendingMotionTrackingRunner(MotionTrackingOnPolicyRunner):
        """Saves the ONNX with the 29 robot joints in its metadata."""

        def save(self, path: str, infos=None):
            super().save(path, infos)
            onnx_path = self._get_export_paths(path)[2]
            if onnx_path.is_file():
                robot_joints_only(onnx_path)

    return BendingMotionTrackingRunner


def register() -> list[str]:
    """Register the bending-blade task with mjlab. Idempotent; returns the task ids."""
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    from fencing.saber.mjlab_task import saber_rl_cfg

    if FLEX_TASK_ID not in set(list_tasks()):
        register_mjlab_task(
            task_id=FLEX_TASK_ID,
            env_cfg=flex_tracking_env_cfg(),
            play_env_cfg=flex_tracking_env_cfg(play=True),
            rl_cfg=saber_rl_cfg(),
            runner_cls=flex_runner_cls(),
        )
    return [FLEX_TASK_ID]
