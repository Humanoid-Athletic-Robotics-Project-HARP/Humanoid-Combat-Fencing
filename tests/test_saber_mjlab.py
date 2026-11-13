"""mjlab task registration. Skipped in the fencing venv (no mjlab/torch); run in the mjlab venv:

    PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python -m pytest tests/test_saber_mjlab.py
"""

from __future__ import annotations

import pytest

_HOW = ("run in the mjlab venv: PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python -m pytest -q "
        "-p no:cacheprovider tests/test_saber_mjlab.py (docs/evaluation.md, Tests)")
pytest.importorskip("torch", reason=f"no torch; {_HOW}")
pytest.importorskip("mjlab", reason=f"no mjlab; {_HOW}")


def test_register_is_idempotent_and_lists_both_tasks():
    from mjlab.tasks.registry import list_tasks

    from fencing.saber.mjlab_task import ENTRY_TASK_ID, TASK_ID, TASK_ID_NO_STATE_ESTIMATION, register

    first = register()
    second = register()
    # The entry joint variants follow (test_register_lists_the_entry_joint_variants_after_the_existing_tasks).
    assert first == second and first[:3] == [TASK_ID, TASK_ID_NO_STATE_ESTIMATION, ENTRY_TASK_ID]
    assert {TASK_ID, TASK_ID_NO_STATE_ESTIMATION, ENTRY_TASK_ID} <= set(list_tasks())


def test_saber_task_trains_with_mass_randomization_and_plays_without():
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_task import SABER_MASS_EVENT, TASK_ID, register

    register()
    assert SABER_MASS_EVENT in load_env_cfg(TASK_ID).events
    assert SABER_MASS_EVENT not in load_env_cfg(TASK_ID, play=True).events


def test_saber_robot_spec_has_saber_body_and_sites():
    from fencing.saber.mjlab_task import saber_g1_spec

    model = saber_g1_spec().compile()
    assert model.body("saber").id == model.nbody - 1
    assert model.site("blade_tip") is not None and model.site("guard") is not None


def test_saber_task_uses_the_wider_right_wrist_action_scale():
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_task import SABER_WRIST_ACTION_SCALE, TASK_ID, register

    register()
    for play in (False, True):
        scale = load_env_cfg(TASK_ID, play=play).actions["joint_pos"].scale
        assert scale["right_wrist_pitch_joint"] == scale["right_wrist_yaw_joint"] == SABER_WRIST_ACTION_SCALE


def test_eval_configures_the_saber_task_in_a_fresh_process():
    """`fencing train eval` on a saber run used to KeyError: configure_pass loaded the env
    config before our task was registered. A fresh interpreter has an empty registry."""
    import subprocess
    import sys

    code = ("from fencing.train.evaluate import configure_pass\n"
            "from fencing.saber.mjlab_task import TASK_ID\n"
            "configure, cfg = configure_pass('saber_mass', TASK_ID, 0, None)\n"
            "assert 'saber_mass' in cfg.events\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_an_old_saber_run_keeps_the_wrist_scale_it_trained_with(tmp_path):
    """A checkpoint trained before the wrist scale widened must be evaluated and exported
    with 0.0745, not the task's new 0.25 default."""
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_task import TASK_ID, register
    from fencing.train.mjlab_env import apply_trained_action_scale

    register()
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text(
        "actions:\n  joint_pos:\n    scale:\n      .*_wrist_pitch_joint: 0.0745\n      .*_wrist_yaw_joint: 0.0745\n"
        "    offset: 0.0\n")
    env_cfg = load_env_cfg(TASK_ID, play=True)
    assert apply_trained_action_scale(env_cfg, tmp_path, TASK_ID)
    assert env_cfg.actions["joint_pos"].scale == {".*_wrist_pitch_joint": 0.0745, ".*_wrist_yaw_joint": 0.0745}
    assert not apply_trained_action_scale(load_env_cfg(TASK_ID, play=True), tmp_path / "no_such_run", TASK_ID)


def test_blade_tip_reward_is_in_train_and_play_cfgs_for_both_tasks():
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_task import (
        BLADE_TIP_REWARD, BLADE_TIP_REWARD_STD_M, BLADE_TIP_REWARD_WEIGHT, TASK_ID, TASK_ID_NO_STATE_ESTIMATION,
        blade_tip_position_error_exp, register,
    )

    register()
    for task_id in (TASK_ID, TASK_ID_NO_STATE_ESTIMATION):
        for play in (False, True):
            term = load_env_cfg(task_id, play=play).rewards[BLADE_TIP_REWARD]
            assert term.func is blade_tip_position_error_exp
            assert term.weight == BLADE_TIP_REWARD_WEIGHT
            assert term.params["std"] == BLADE_TIP_REWARD_STD_M
            assert term.params["command_name"] == "motion"


def test_blade_tip_reward_uses_the_eval_wrist_body_and_tip_offset():
    """Reward and the blade_tip_error_m gate must measure the same point: eval takes the
    blade_tip site relative to WRIST_BODY from the compiled model."""
    import numpy as np

    from fencing.saber.mjlab_task import BLADE_TIP_REWARD, saber_g1_spec, saber_tracking_env_cfg
    from fencing.sim2sim.observation import resolve_id
    from fencing.sim2sim.rollout import find_site, tip_offset_in_body
    from fencing.train.evaluate import TIP_SITE, WRIST_BODY

    import mujoco

    cfg = saber_tracking_env_cfg()
    params = cfg.rewards[BLADE_TIP_REWARD].params
    assert params["wrist_body"] == WRIST_BODY
    assert WRIST_BODY in cfg.commands["motion"].body_names
    model = saber_g1_spec().compile()
    expected = tip_offset_in_body(model, find_site(model, TIP_SITE),
                                  resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY))
    np.testing.assert_allclose(params["tip_in_wrist"], expected, atol=1e-6)


def _fake_tip_env(ref_pos, ref_quat, robot_pos, robot_quat):
    """Just enough of a ManagerBasedRlEnv for the reward: a motion command with 2 bodies."""
    from types import SimpleNamespace

    import torch

    def bodies(wrist, other_dim):
        wrist = torch.as_tensor(wrist, dtype=torch.float32)
        filler = torch.full((wrist.shape[0], 1, other_dim), 7.0)  # other body: must be ignored
        return torch.cat([filler, wrist[:, None]], dim=1)

    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=("pelvis", "right_wrist_yaw_link")),
        body_pos_relative_w=bodies(ref_pos, 3), body_quat_relative_w=bodies(ref_quat, 4),
        robot_body_pos_w=bodies(robot_pos, 3), robot_body_quat_w=bodies(robot_quat, 4),
    )
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: {"motion": command}[name]))


def _yaw_quat(angle):
    import math

    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def test_blade_tip_reward_is_one_when_the_wrist_matches_and_drops_for_an_offset():
    import math

    import numpy as np

    from fencing.saber.mjlab_task import blade_tip_position_error_exp

    tip = (0.8, 0.0, 0.0)
    std = 0.15
    pos = [[0.1, 0.2, 1.0]] * 3
    ref_quat = [_yaw_quat(0.0)] * 3
    # env 0 matches; env 1 is shifted 5 cm; env 2 has the 0.3 rad wrist yaw error from extension_v2
    robot_pos = [[0.1, 0.2, 1.0], [0.15, 0.2, 1.0], [0.1, 0.2, 1.0]]
    robot_quat = [_yaw_quat(0.0), _yaw_quat(0.0), _yaw_quat(0.3)]
    env = _fake_tip_env(pos, ref_quat, robot_pos, robot_quat)

    reward = blade_tip_position_error_exp(env, "motion", std, "right_wrist_yaw_link", tip).numpy()

    chord = 2 * 0.8 * math.sin(0.3 / 2)  # tip displacement for a pure rotation about the wrist
    np.testing.assert_allclose(reward, [1.0, math.exp(-(0.05 / std) ** 2), math.exp(-(chord / std) ** 2)],
                               rtol=1e-5)
    assert reward.shape == (3,)


def test_blade_tip_reward_agrees_with_the_eval_metric_for_arbitrary_poses():
    import numpy as np
    import torch

    from fencing.saber.mjlab_task import blade_tip_position_error_exp
    from fencing.train.metrics import blade_tip_from_wrist

    rng = np.random.default_rng(0)
    tip = np.array([0.7, 0.05, 0.3])
    std = 0.15
    ref_pos, robot_pos = rng.normal(size=(2, 5, 3)) * 0.05 + [0.0, 0.0, 1.0]
    ref_quat, robot_quat = (q / np.linalg.norm(q, axis=-1, keepdims=True) for q in rng.normal(size=(2, 5, 4)))
    env = _fake_tip_env(ref_pos, ref_quat, robot_pos, robot_quat)

    reward = blade_tip_position_error_exp(env, "motion", std, "right_wrist_yaw_link", tuple(tip))

    error = np.linalg.norm(blade_tip_from_wrist(ref_pos, ref_quat, tip) - blade_tip_from_wrist(robot_pos, robot_quat, tip),
                           axis=-1)
    assert isinstance(reward, torch.Tensor)
    np.testing.assert_allclose(reward.numpy(), np.exp(-error ** 2 / std ** 2), rtol=1e-4, atol=1e-6)


def test_blade_tip_reward_fails_loudly_when_the_wrist_is_not_tracked():
    from fencing.saber.mjlab_task import blade_tip_position_error_exp

    env = _fake_tip_env([[0, 0, 1]], [_yaw_quat(0)], [[0, 0, 1]], [_yaw_quat(0)])
    with pytest.raises(KeyError, match="left_wrist_yaw_link"):
        blade_tip_position_error_exp(env, "motion", 0.15, "left_wrist_yaw_link", (0.8, 0, 0))


# --- The entry task (Fencing-Tracking-Flat-G1-Saber-Entry) is purely additive ----------------
# tests/data/saber_task_cfgs.json describes the two existing saber tasks as registered by the
# committed (HEAD) src/fencing/saber/mjlab_task.py, before the entry task existed. Regenerate only
# on purpose, from a copy of that committed file, never from the working tree:
#   git -C <repo> show HEAD:src/fencing/saber/mjlab_task.py > /tmp/x/head_mjlab_task.py
#   PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python tests/test_saber_mjlab.py /tmp/x/head_mjlab_task.py
SNAPSHOT = __import__("pathlib").Path(__file__).parent / "data" / "saber_task_cfgs.json"
# Keyed term groups: one entry per term, so a diff names the term that changed.
TERM_GROUPS = ("rewards", "events", "terminations", "observations", "commands", "curriculum", "metrics")


def _strip_addresses(text: str) -> str:
    import re

    return re.sub(r" at 0x[0-9a-f]+", "", text)


def _robot_model_digest(cfg) -> dict:
    """The compiled robot (G1 + saber): sizes and a hash of its physical parameters, so a
    change to the saber spec, actuators, joint ranges or masses shows up."""
    import hashlib

    import numpy as np

    model = cfg.scene.entities["robot"].spec_fn().compile()
    fields = ("body_mass", "body_inertia", "body_ipos", "body_pos", "jnt_range", "dof_damping", "dof_armature",
              "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
              "geom_size", "geom_friction", "geom_pos", "site_pos")
    digest = hashlib.sha256()
    for name in fields:
        digest.update(np.round(np.asarray(getattr(model, name), dtype=np.float64), 9).tobytes())
    return {"nbody": model.nbody, "njnt": model.njnt, "nu": model.nu, "ngeom": model.ngeom, "nsite": model.nsite,
            "physical_parameters_sha256": digest.hexdigest()}


def describe_task_cfg(task_id: str, play: bool) -> dict:
    """Every field of the task's env config as text (term groups per term), plus the compiled
    robot model: what a policy trained on `task_id` sees, is paid for, is perturbed by, and runs on."""
    import dataclasses

    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=play)
    description = {}
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if field.name in TERM_GROUPS:
            description[field.name] = {name: _strip_addresses(repr(term)) for name, term in value.items()}
        else:
            description[field.name] = _strip_addresses(repr(value))
    description["robot_model"] = _robot_model_digest(cfg)
    return description


def _existing_tasks() -> dict:
    task_ids = ("Fencing-Tracking-Flat-G1-Saber", "Fencing-Tracking-Flat-G1-Saber-No-State-Estimation")
    return {f"{task}{':play' if play else ''}": describe_task_cfg(task, play)
            for task in task_ids for play in (False, True)}


def test_the_existing_saber_tasks_are_exactly_as_committed_before_the_entry_task():
    import json

    from fencing.saber.mjlab_task import register

    register()
    expected = json.loads(SNAPSHOT.read_text())
    actual = json.loads(json.dumps(_existing_tasks()))
    assert actual.keys() == expected.keys()
    for task in expected:
        assert actual[task].keys() == expected[task].keys(), task
        for field in expected[task]:
            assert actual[task][field] == expected[task][field], f"{task}: {field}"


def test_entry_task_is_the_saber_task_plus_one_feet_reward_term():
    from fencing.saber.mjlab_task import ENTRY_TASK_ID, FEET_REWARD, TASK_ID, register

    assert ENTRY_TASK_ID in register()
    for play in (False, True):
        saber, entry = describe_task_cfg(TASK_ID, play), describe_task_cfg(ENTRY_TASK_ID, play)
        # Same observations, actions, robot, events, sim and episode: the ONNX runs in the
        # unmodified C++ controller and the policy is trained under the same perturbations.
        assert {k: v for k, v in entry.items() if k != "rewards"} == {k: v for k, v in saber.items() if k != "rewards"}
        assert {k: v for k, v in entry["rewards"].items() if k != FEET_REWARD} == saber["rewards"]
        assert FEET_REWARD in entry["rewards"] and FEET_REWARD not in saber["rewards"]


# tests/data/saber_entry_task_cfgs.json describes the entry task as registered before the joint
# variants existed (the working-tree mjlab_task.py of 2026-11-12, sha256 e1e9cbed...). Regenerate
# only on purpose, from a copy of a mjlab_task.py whose entry task is the one to pin:
#   PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python tests/test_saber_mjlab.py --entry /tmp/x/mjlab_task.py
ENTRY_SNAPSHOT = SNAPSHOT.parent / "saber_entry_task_cfgs.json"


def _entry_tasks() -> dict:
    task = "Fencing-Tracking-Flat-G1-Saber-Entry"
    return {f"{task}{':play' if play else ''}": describe_task_cfg(task, play) for play in (False, True)}


def test_the_entry_task_is_exactly_as_before_the_joint_variants():
    import json

    from fencing.saber.mjlab_task import register

    register()
    expected = json.loads(ENTRY_SNAPSHOT.read_text())
    actual = json.loads(json.dumps(_entry_tasks()))
    assert actual.keys() == expected.keys()
    for task in expected:
        assert actual[task].keys() == expected[task].keys(), task
        for field in expected[task]:
            assert actual[task][field] == expected[task][field], f"{task}: {field}"


def test_register_lists_the_entry_joint_variants_after_the_existing_tasks():
    from fencing.saber.mjlab_task import (
        ENTRY_JOINT_TASK_IDS, ENTRY_TASK_ID, TASK_ID, TASK_ID_NO_STATE_ESTIMATION, register,
    )

    assert register() == [TASK_ID, TASK_ID_NO_STATE_ESTIMATION, ENTRY_TASK_ID, *ENTRY_JOINT_TASK_IDS]
    assert ENTRY_JOINT_TASK_IDS == ("Fencing-Tracking-Flat-G1-Saber-Entry-Joint",
                                    "Fencing-Tracking-Flat-G1-Saber-Entry-Joint05")


def test_entry_joint_variants_are_the_entry_task_plus_one_joint_reward_term():
    from fencing.saber.mjlab_task import (
        ENTRY_JOINT_TASK_IDS, ENTRY_JOINT_WEIGHTS, ENTRY_TASK_ID, JOINT_REWARD, JOINT_REWARD_STD_RAD,
        joint_position_error_exp, register,
    )
    from mjlab.tasks.registry import load_env_cfg

    register()
    for task_id in ENTRY_JOINT_TASK_IDS:
        for play in (False, True):
            entry, variant = describe_task_cfg(ENTRY_TASK_ID, play), describe_task_cfg(task_id, play)
            assert {k: v for k, v in variant.items() if k != "rewards"} == {k: v for k, v in entry.items() if k != "rewards"}
            assert {k: v for k, v in variant["rewards"].items() if k != JOINT_REWARD} == entry["rewards"]
            term = load_env_cfg(task_id, play=play).rewards[JOINT_REWARD]
            assert term.func is joint_position_error_exp
            assert term.weight == ENTRY_JOINT_WEIGHTS[task_id]
            assert term.params == {"command_name": "motion", "std": JOINT_REWARD_STD_RAD}
    assert ENTRY_JOINT_WEIGHTS == {ENTRY_JOINT_TASK_IDS[0]: 1.0, ENTRY_JOINT_TASK_IDS[1]: 0.5}


def test_joint_reward_is_exp_of_the_mean_squared_joint_error():
    import math
    from types import SimpleNamespace

    import numpy as np
    import torch

    from fencing.saber.mjlab_task import joint_position_error_exp

    ref = torch.zeros(3, 4)
    robot = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.2, -0.2, 0.2, -0.2]])
    command = SimpleNamespace(joint_pos=ref, robot_joint_pos=robot)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: {"motion": command}[name]))

    reward = joint_position_error_exp(env, "motion", 0.15).numpy()

    np.testing.assert_allclose(reward, [1.0, math.exp(-(0.01 / 4) / 0.15**2), math.exp(-0.04 / 0.15**2)], rtol=1e-6)


def test_feet_reward_tracks_only_the_two_ankle_roll_links_with_a_tight_std():
    from mjlab.tasks.registry import load_env_cfg

    from fencing.saber.mjlab_task import (
        ENTRY_TASK_ID, FEET_BODIES, FEET_REWARD, FEET_REWARD_STD_M, FEET_REWARD_WEIGHT, register,
    )

    register()
    cfg = load_env_cfg(ENTRY_TASK_ID)
    term = cfg.rewards[FEET_REWARD]
    assert term.func.__name__ == "motion_relative_body_position_error_exp"
    assert term.params["body_names"] == FEET_BODIES == ("left_ankle_roll_link", "right_ankle_roll_link")
    assert set(FEET_BODIES) <= set(cfg.commands["motion"].body_names)
    assert term.weight == FEET_REWARD_WEIGHT and term.params["std"] == FEET_REWARD_STD_M < 0.3


if __name__ == "__main__":
    # Write SNAPSHOT from the mjlab_task.py given on the command line (a copy of the committed one),
    # or with --entry, ENTRY_SNAPSHOT (the entry task) from the given copy.
    import importlib.util
    import json
    import sys

    entry = sys.argv[1] == "--entry"
    spec = importlib.util.spec_from_file_location("committed_mjlab_task", sys.argv[-1])
    committed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(committed)
    print("registered", committed.register())
    target, tasks = (ENTRY_SNAPSHOT, _entry_tasks()) if entry else (SNAPSHOT, _existing_tasks())
    target.write_text(json.dumps(tasks, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
