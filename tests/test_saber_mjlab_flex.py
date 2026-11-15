"""The bending-blade training task (fencing.saber.mjlab_flex_task). Skipped in the fencing venv
(no mjlab/torch); run in the mjlab venv:

    PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_saber_mjlab_flex.py
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest

_HOW = ("run in the mjlab venv: PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python -m pytest -q "
        "-p no:cacheprovider tests/test_saber_mjlab_flex.py")
pytest.importorskip("torch", reason=f"no torch; {_HOW}")
pytest.importorskip("mjlab", reason=f"no mjlab; {_HOW}")

import numpy as np
import torch
from test_saber_mjlab import _fake_tip_env, _yaw_quat, describe_task_cfg


def _joint_names(saber):
    from fencing.saber.mjlab_task import saber_g1_spec

    model = saber_g1_spec(saber).compile()
    return [model.joint(i).name for i in range(1, model.njnt)]  # without the floating base


# --- registration and what differs from the saber task -------------------------------------------

def test_register_is_idempotent_and_the_repo_registers_it_with_the_saber_tasks():
    from mjlab.tasks.registry import list_tasks

    from fencing.saber.mjlab_flex_task import FLEX_TASK_ID, register
    from fencing.train import mjlab_env

    assert register() == register() == [FLEX_TASK_ID]
    assert FLEX_TASK_ID == "Fencing-Tracking-Flat-G1-Saber-Bending"
    assert FLEX_TASK_ID in mjlab_env.register_tasks() and FLEX_TASK_ID in list_tasks()


# Everything a policy sees, is paid for, is perturbed by and runs on, other than these, is the saber task's.
DIFFERENT = {"scene", "commands", "observations", "events", "rewards"}


@pytest.mark.parametrize("play", [False, True])
def test_the_bending_task_is_the_saber_task_except_the_blade_terms(play):
    from fencing.saber.mjlab_flex_task import FLEX_TASK_ID, register
    from fencing.saber.mjlab_task import BLADE_TIP_REWARD, SABER_MASS_EVENT, TASK_ID
    from fencing.saber.mjlab_task import register as register_saber

    register_saber()
    register()
    saber, bending = describe_task_cfg(TASK_ID, play), describe_task_cfg(FLEX_TASK_ID, play)
    assert {k: v for k, v in bending.items() if k not in DIFFERENT | {"robot_model"}} == \
        {k: v for k, v in saber.items() if k not in DIFFERENT | {"robot_model"}}
    # rewards: only the blade-tip term changes; events: only the encoder bias joints and the saber randomization
    assert {k: v for k, v in bending["rewards"].items() if k != BLADE_TIP_REWARD} == \
        {k: v for k, v in saber["rewards"].items() if k != BLADE_TIP_REWARD}
    assert "bending_blade_tip_error_exp" in bending["rewards"][BLADE_TIP_REWARD]
    changed_events = {k for k in saber["events"] if bending["events"].get(k) != saber["events"][k]}
    assert changed_events == ({"encoder_bias", SABER_MASS_EVENT} if not play else {"encoder_bias"})
    assert set(bending["events"]) == set(saber["events"])
    # observations: the actor has the same terms (same input size); the critic gets the blade hinges
    assert bending["robot_model"]["njnt"] == saber["robot_model"]["njnt"] + 2  # 1 piece = 2 hinges
    assert bending["robot_model"]["nu"] == saber["robot_model"]["nu"]  # hinges have no motor


def test_the_policy_sees_the_29_robot_joints_and_the_critic_also_the_blade():
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_flex_task import (
        BLADE_JOINTS,
        FLEX_SABER,
        FLEX_TASK_ID,
        ROBOT_JOINTS,
        register,
    )
    from fencing.saber.mjlab_task import TASK_ID
    from fencing.saber.mjlab_task import register as register_saber
    from fencing.saber.spec import NOMINAL_SABER

    names = _joint_names(FLEX_SABER)

    robot = [n for n in names if re.fullmatch(ROBOT_JOINTS, n)]
    blade = [n for n in names if re.fullmatch(BLADE_JOINTS, n)]
    assert len(robot) == 29 and blade == ["saber_flex_0_y", "saber_flex_0_z"]
    assert robot == _joint_names(NOMINAL_SABER)  # the same 29 joints, in the same order, as the saber task

    register_saber()
    register()
    for play in (False, True):
        cfg, saber_cfg = load_env_cfg(FLEX_TASK_ID, play=play), load_env_cfg(TASK_ID, play=play)
        actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
        assert list(actor) == list(saber_cfg.observations["actor"].terms)
        assert list(critic) == [*saber_cfg.observations["critic"].terms, "blade_pos", "blade_vel"]
        for terms in (actor, critic):
            for name in ("joint_pos", "joint_vel"):
                assert terms[name].params["asset_cfg"].joint_names == (ROBOT_JOINTS,)
        for name in ("blade_pos", "blade_vel"):
            assert critic[name].params["asset_cfg"].joint_names == (BLADE_JOINTS,)


# --- stability over the randomization range -------------------------------------------------------

def test_the_training_blade_is_stable_over_the_whole_randomization_range():
    import dataclasses

    from fencing.saber.mjlab_flex_task import FLEX_SABER, check_training_blade

    check_training_blade(FLEX_SABER)  # 1 piece, 5 ms, implicitfast
    # Two pieces are fine at the nominal blade but blow up on the lightest, stiffest draw (20.7 Hz).
    with pytest.raises(ValueError, match="blows up"):
        check_training_blade(dataclasses.replace(FLEX_SABER, flex_segments=2))


# --- reward ---------------------------------------------------------------------------------------

def _fake_bending_env(ref_pos, ref_quat, robot_pos, robot_quat, tip_w):
    env = _fake_tip_env(ref_pos, ref_quat, robot_pos, robot_quat)
    command = env.command_manager.get_term("motion")
    tip_w = torch.as_tensor(tip_w, dtype=torch.float32)
    filler = torch.full((tip_w.shape[0], 1, 3), 9.0)  # another site: must be ignored
    command.robot = SimpleNamespace(site_names=("guard", "blade_tip"),
                                    data=SimpleNamespace(site_pos_w=torch.cat([filler, tip_w[:, None]], dim=1)))
    return env


def test_the_bending_tip_reward_scores_the_real_tip_against_the_straight_reference_tip():
    from fencing.saber.mjlab_flex_task import bending_blade_tip_error_exp
    from fencing.saber.mjlab_task import blade_tip_position_error_exp

    tip = (0.8, 0.0, 0.0)
    std = 0.15
    pos = [[0.1, 0.2, 1.0]] * 3
    quat = [_yaw_quat(0.0)] * 3
    straight_tip = [0.9, 0.2, 1.0]
    # env 0: the blade is straight; env 1: bent 5 cm sideways; env 2: bent 10 cm down
    real_tips = [straight_tip, [0.9, 0.25, 1.0], [0.9, 0.2, 0.9]]
    env = _fake_bending_env(pos, quat, pos, quat, real_tips)

    reward = bending_blade_tip_error_exp(env, "motion", std, "right_wrist_yaw_link", tip, "blade_tip").numpy()

    np.testing.assert_allclose(reward, [1.0, math.exp(-(0.05 / std) ** 2), math.exp(-(0.10 / std) ** 2)], rtol=1e-5)
    # With a straight blade it is the saber task's reward.
    straight = blade_tip_position_error_exp(env, "motion", std, "right_wrist_yaw_link", tip).numpy()
    assert reward[0] == pytest.approx(straight[0])


def test_the_bending_tip_reward_uses_the_saber_task_reference_tip_and_the_real_tip_site():
    from fencing.saber.mjlab_flex_task import bending_blade_tip_reward_cfg
    from fencing.saber.mjlab_task import blade_tip_reward_cfg
    from fencing.saber.spec import NOMINAL_SABER, TIP_SITE

    bending, straight = bending_blade_tip_reward_cfg().params, blade_tip_reward_cfg(NOMINAL_SABER).params
    assert bending["tip_site"] == TIP_SITE == "blade_tip"
    assert {k: v for k, v in bending.items() if k != "tip_site"} == straight
    assert bending_blade_tip_reward_cfg().weight == blade_tip_reward_cfg().weight


def test_the_bending_tip_reward_fails_loudly_when_the_wrist_is_not_tracked():
    from fencing.saber.mjlab_flex_task import bending_blade_tip_error_exp

    env = _fake_bending_env([[0, 0, 1]], [_yaw_quat(0)], [[0, 0, 1]], [_yaw_quat(0)], [[0.8, 0, 1]])
    with pytest.raises(KeyError, match="left_wrist_yaw_link"):
        bending_blade_tip_error_exp(env, "motion", 0.15, "left_wrist_yaw_link", (0.8, 0, 0), "blade_tip")


# --- saber randomization --------------------------------------------------------------------------

def _fake_randomization_env(num_envs):
    """Bodies 0-2 = [wrist, saber, blade piece]; joints 0-2 = [wrist, flat hinge, edge hinge]."""
    defaults = {"body_mass": torch.tensor([0.5, 0.1, 0.1]),
                "body_inertia": torch.tensor([[1.0, 1.0, 1.0], [2e-4, 3e-4, 4e-4], [1e-5, 5e-4, 5e-4]]),
                "jnt_stiffness": torch.tensor([0.0, 18.5, 115.7]),
                "dof_damping": torch.tensor([0.3, 0.059, 0.147])}
    model = SimpleNamespace(**{k: v.repeat(num_envs, *([1] * v.dim())).clone() for k, v in defaults.items()})
    sim = SimpleNamespace(model=model, get_default_field=lambda f: defaults[f], per_world_default_fields=set())
    idx = torch.arange(3, dtype=torch.int)  # mjlab's index tensors are int32
    robot = SimpleNamespace(indexing=SimpleNamespace(body_ids=idx, joint_ids=idx, joint_v_adr=idx))
    env = SimpleNamespace(num_envs=num_envs, device="cpu", sim=sim, scene={"robot": robot})
    cfg = lambda **kw: SimpleNamespace(name="robot", **kw)
    return env, defaults, cfg(body_ids=[1, 2]), cfg(joint_ids=[1]), cfg(joint_ids=[2])


def test_the_saber_randomization_scales_the_whole_saber_and_each_wobble_and_keeps_the_damping_ratio():
    from fencing.saber.mjlab_flex_task import (
        EDGEWISE_HZ_RANGE,
        FLATWISE_HZ_RANGE,
        FLEX_SABER,
        randomize_bending_saber,
    )
    from fencing.saber.mjlab_task import saber_alpha_range
    from fencing.saber.spec import MASS_RANDOMIZATION_KG

    torch.manual_seed(0)
    n = 4000
    env, d, saber_cfg, flat_cfg, edge_cfg = _fake_randomization_env(n)
    flat_scale = tuple(f / FLEX_SABER.flex_f1_flatwise_hz for f in FLATWISE_HZ_RANGE)
    edge_scale = tuple(f / FLEX_SABER.flex_f1_edgewise_hz for f in EDGEWISE_HZ_RANGE)
    randomize_bending_saber(env, None, saber_alpha_range(FLEX_SABER), flat_scale, edge_scale,
                            saber_cfg, flat_cfg, edge_cfg)
    m = env.sim.model

    a = m.body_mass[:, 1] / d["body_mass"][1]
    torch.testing.assert_close(m.body_mass[:, 2] / d["body_mass"][2], a)  # one scale for the whole saber
    torch.testing.assert_close(m.body_inertia[:, 1:] / d["body_inertia"][1:], a[:, None, None].expand(n, 2, 3))
    assert torch.all(m.body_mass[:, 0] == d["body_mass"][0])  # the wrist is not the saber
    total = m.body_mass[:, 1:].sum(-1)
    lo, hi = MASS_RANDOMIZATION_KG
    assert total.min() >= lo - 1e-6 and total.max() <= hi + 1e-6
    assert total.min() < lo + 0.01 and total.max() > hi - 0.01  # the range is covered

    for j, (f0, (f_lo, f_hi)) in {1: (FLEX_SABER.flex_f1_flatwise_hz, FLATWISE_HZ_RANGE),
                                  2: (FLEX_SABER.flex_f1_edgewise_hz, EDGEWISE_HZ_RANGE)}.items():
        s = torch.sqrt(m.jnt_stiffness[:, j] / d["jnt_stiffness"][j])  # frequency scale at the nominal mass
        assert s.min() * f0 >= f_lo - 1e-4 and s.max() * f0 <= f_hi + 1e-4
        assert s.min() * f0 < f_lo + 0.05 and s.max() * f0 > f_hi - 0.05
        # damping ratio ~ c / sqrt(k * mass) stays the nominal one
        ratio = (m.dof_damping[:, j] / d["dof_damping"][j]) / torch.sqrt((s * s) * a)
        torch.testing.assert_close(ratio, torch.ones(n))
    assert torch.all(m.jnt_stiffness[:, 0] == 0) and torch.all(m.dof_damping[:, 0] == d["dof_damping"][0])
    # the two directions are drawn independently
    s_flat = torch.sqrt(m.jnt_stiffness[:, 1] / d["jnt_stiffness"][1])
    s_edge = torch.sqrt(m.jnt_stiffness[:, 2] / d["jnt_stiffness"][2])
    assert abs(float(torch.corrcoef(torch.stack([s_flat, s_edge]))[0, 1])) < 0.1
    assert abs(float(torch.corrcoef(torch.stack([s_flat, a]))[0, 1])) < 0.1


def test_the_randomization_event_marks_its_fields_for_per_env_copies():
    from fencing.saber.mjlab_flex_task import saber_randomization_event_cfg

    fields = saber_randomization_event_cfg().func.model_fields
    assert {"body_mass", "body_inertia", "jnt_stiffness", "dof_damping"} <= set(fields)


# --- motion command -------------------------------------------------------------------------------

def test_the_motion_command_compares_and_resets_only_the_robot_joints():
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    from fencing.saber.mjlab_flex_task import bending_motion_command_cfg

    base = MotionCommandCfg(motion_file="x.npz", anchor_body_name="torso_link", body_names=("torso_link",),
                            entity_name="robot", resampling_time_range=(1.0e9, 1.0e9))
    cfg = bending_motion_command_cfg(base)
    assert isinstance(cfg, MotionCommandCfg) and cfg.viz.mode == "frames"
    assert all(getattr(cfg, f) == getattr(base, f) for f in ("motion_file", "anchor_body_name", "body_names"))

    # An instance without the env: 3 joints, the last one a blade hinge.
    written = {}
    robot = SimpleNamespace(
        num_joints=3,
        data=SimpleNamespace(joint_pos=torch.tensor([[1.0, 2.0, 0.3]]), joint_vel=torch.tensor([[4.0, 5.0, 6.0]]),
                             soft_joint_pos_limits=torch.tensor([[[-9.0, 9.0], [-9.0, 9.0],
                                                                  [-math.inf, math.inf]]])),
        write_joint_state_to_sim=lambda p, v, env_ids: written.update(pos=p, vel=v),
        write_root_state_to_sim=lambda s, env_ids: None, reset=lambda env_ids: None,
    )
    command_cls = _bending_command_class(cfg)
    assert command_cls.__name__ == "BendingMotionCommand"
    command = object.__new__(command_cls)
    command.robot, command.robot_joint_ids = robot, torch.tensor([0, 1])
    torch.testing.assert_close(command.robot_joint_pos, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(command.robot_joint_vel, torch.tensor([[4.0, 5.0]]))
    zero = torch.zeros(1, 3)
    command._write_reference_state_to_sim(torch.tensor([0]), zero, torch.tensor([[1.0, 0, 0, 0]]), zero, zero,
                                          torch.tensor([[0.5, -0.5]]), torch.tensor([[1.5, -1.5]]))
    torch.testing.assert_close(written["pos"], torch.tensor([[0.5, -0.5, 0.0]]))  # blade straight
    torch.testing.assert_close(written["vel"], torch.tensor([[1.5, -1.5, 0.0]]))  # and still


def _bending_command_class(cfg):
    """The command class cfg.build() instantiates, without building an env."""
    captured = {}

    class Probe(Exception):
        pass

    def init(self, cfg, env):
        captured["cls"] = type(self)
        raise Probe

    from unittest import mock

    from mjlab.tasks.tracking.mdp import MotionCommand

    with mock.patch.object(MotionCommand, "__init__", init), pytest.raises(Probe):
        cfg.build(None)
    return captured["cls"]


# --- export ---------------------------------------------------------------------------------------

def test_the_onnx_metadata_keeps_only_the_robot_joints(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from fencing.saber.mjlab_flex_task import robot_joints_only

    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph)
    for key, value in {"joint_names": "a_joint,b_joint,saber_flex_0_y,saber_flex_0_z",
                       "default_joint_pos": "0.100,0.200,0.000,0.000", "action_scale": "0.5,0.25",
                       "joint_stiffness": "40.000,14.000"}.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    path = tmp_path / "policy.onnx"
    onnx.save(model, str(path))

    robot_joints_only(path)

    meta = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
    assert meta == {"joint_names": "a_joint,b_joint", "default_joint_pos": "0.100,0.200",
                    "action_scale": "0.5,0.25", "joint_stiffness": "40.000,14.000"}


def test_exported_metadata_passes_the_sim2sim_contract_with_the_blade_hinges_dropped():
    """What `fencing train export` does for the bending task: drop the hinges, then make the full-precision
    copy, then read it back with sim2sim's contract parser."""
    from mjlab.rl.exporter_utils import list_to_csv_str

    from fencing.saber.mjlab_flex_task import drop_blade_joints
    from fencing.sim2sim.contract import (
        FLOAT_LIST_KEYS,
        FULL_PRECISION_KEY,
        full_precision_metadata,
        parse_metadata,
    )

    motors = [f"joint_{i}" for i in range(29)]
    base = {  # shaped like mjlab's get_base_metadata for the bending robot
        "joint_names": [*motors, "saber_flex_0_y", "saber_flex_0_z"],
        "default_joint_pos": [0.01 * i for i in range(29)] + [0.0, 0.0],
        "joint_stiffness": [40.0] * 29, "joint_damping": [2.0] * 29, "action_scale": [0.25] * 29,
        "observation_names": ["command"], "observation_terms_scale": [1.0], "observation_terms_clip": [[-1, 1]],
        "observation_terms_flatten_history_dim": [True], "observation_terms_history_length": [0],
        "command_names": ["motion"], "anchor_body_name": "torso_link", "body_names": ["torso_link"], "run_path": "x",
    }
    metadata = drop_blade_joints(base)
    assert metadata["joint_names"] == motors and len(metadata["default_joint_pos"]) == 29
    assert base["joint_names"][-1] == "saber_flex_0_z"  # the input is not modified
    metadata[FULL_PRECISION_KEY] = full_precision_metadata(
        {k: metadata[k] for k in FLOAT_LIST_KEYS if k in metadata})
    props = {k: list_to_csv_str(v) if isinstance(v, list) else str(v) for k, v in metadata.items()}  # as attached
    contract = parse_metadata(props)
    assert contract.joint_names == tuple(motors)
    # In the other order the copy keeps 31 default positions for 29 joints, which the parser refuses.
    stale = full_precision_metadata({k: base[k] for k in FLOAT_LIST_KEYS})
    with pytest.raises(ValueError, match="31 values for 29 joints"):
        parse_metadata({**props, FULL_PRECISION_KEY: stale})


def test_the_saber_task_refusal_points_to_the_bending_task():
    from fencing.saber.mjlab_flex_task import FLEX_SABER, FLEX_TASK_ID
    from fencing.saber.mjlab_task import saber_g1_robot_cfg

    with pytest.raises(NotImplementedError, match=FLEX_TASK_ID):
        saber_g1_robot_cfg(FLEX_SABER)
