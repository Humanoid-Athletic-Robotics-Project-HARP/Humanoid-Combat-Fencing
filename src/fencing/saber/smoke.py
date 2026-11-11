"""GPU smoke test of the saber tracking task (mjlab venv).

    PYTHONPATH=REPO/src ~/hcf-data/venvs/mjlab/bin/python -m fencing.saber.smoke --num-envs 32

Checks, in order, and fails loudly on the first that does not hold:
  1. register() is idempotent;
  2. the saber robot's bodies are the plain robot's bodies plus "saber" at the end, and the
     NPZ has as many bodies as the plain robot (so plain-robot NPZs index correctly);
  3. posing the saber robot from NPZ frames puts its tracked bodies exactly on the NPZ
     bodies -- the end-to-end proof that body indexing survived the extra body;
  4. the training env builds, steps with zero actions, stays finite, the saber mass is
     randomized across envs within the configured range, and the blade tip stays rigidly
     at the expected offset from the right wrist while the wrist moves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fencing.paths import EVAL, NPZ
from fencing.saber.spec import MASS_RANDOMIZATION_KG, NOMINAL_SABER, PARENT_BODY, SABER_BODY, \
    TIP_SITE, tip_in_hand

DEFAULT_MOTION = NPZ / "dance1_subject2_mjlab.npz"


def _fail(message: str) -> None:
    raise SystemExit(f"[saber smoke] FAILED: {message}")


def check_body_order(motion_file: Path) -> dict:
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    from fencing.saber.mjlab_task import saber_g1_robot_cfg

    plain = Entity(get_g1_robot_cfg()).body_names
    saber = Entity(saber_g1_robot_cfg()).body_names
    if saber[:-1] != plain or saber[-1] != SABER_BODY:
        _fail(f"saber robot bodies are not plain bodies + '{SABER_BODY}': {saber}")
    npz_bodies = int(np.load(motion_file)["body_pos_w"].shape[1])
    if npz_bodies != len(plain):
        _fail(f"{motion_file} has {npz_bodies} bodies, plain robot has {len(plain)}")
    return {"plain_bodies": len(plain), "saber_bodies": len(saber), "npz_bodies": npz_bodies}


def _make_env(task_id: str, motion_file: Path, num_envs: int, device: str, play: bool):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = str(motion_file)
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def check_npz_kinematics(task_id: str, motion_file: Path, device: str) -> float:
    """Pose the saber robot from NPZ frames; its tracked bodies must land on the NPZ bodies.

    Kinematic, not a reset: a reset is followed by a physics step and a command time
    advance, which leaves several cm of honest error even on the upstream task.
    """
    import torch

    env = _make_env(task_id, motion_file, num_envs=4, device=device, play=True)
    try:
        env.reset()
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        motion = command.motion
        last = motion.joint_pos.shape[0] - 1
        frames = torch.tensor([0, last // 3, 2 * last // 3, last], device=device)
        origins = env.scene.env_origins
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = motion.body_pos_w[frames, 0] + origins
        root[:, 3:7] = motion.body_quat_w[frames, 0]
        root[:, 7:] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(motion.joint_pos[frames],
                                       torch.zeros_like(motion.joint_pos[frames]))
        env.sim.forward()
        robot_bodies = robot.data.body_link_pos_w[:, command.body_indexes]
        reference = motion.body_pos_w[frames] + origins[:, None]
        worst = float((robot_bodies - reference).norm(dim=-1).max())
    finally:
        env.close()
    if worst > 1e-4:
        _fail(f"posed from the NPZ, the saber robot's tracked bodies are {worst:.4f} m off "
              "the NPZ bodies (body indexing mismatch?)")
    return worst


def check_training_env(task_id: str, motion_file: Path, num_envs: int, steps: int,
                       device: str) -> dict:
    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    env = _make_env(task_id, motion_file, num_envs=num_envs, device=device, play=False)
    try:
        robot = env.scene["robot"]
        tip_index = robot.find_sites((TIP_SITE,), preserve_order=True)[0][0]
        wrist_index = robot.find_bodies((PARENT_BODY,), preserve_order=True)[0][0]
        saber_index = robot.find_bodies((SABER_BODY,), preserve_order=True)[0][0]
        saber_model_id = int(robot.indexing.body_ids[saber_index])
        masses = env.sim.model.body_mass[:, saber_model_id].detach().cpu().numpy()
        low, high = MASS_RANDOMIZATION_KG
        if masses.min() < low - 1e-4 or masses.max() > high + 1e-4 or np.ptp(masses) < 1e-3:
            _fail(f"saber mass not randomized within {MASS_RANDOMIZATION_KG}: "
                  f"min {masses.min():.4f} max {masses.max():.4f}")

        env.reset()
        expected = torch.tensor(tip_in_hand(NOMINAL_SABER), dtype=torch.float32, device=device)
        zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                                  device=device)
        tips, wrists, offset_errors = [], [], []
        resets = 0
        for _ in range(steps):
            _, reward, terminated, truncated, _ = env.step(zero_action)
            if not torch.isfinite(reward).all():
                _fail("non-finite reward")
            resets += int((terminated | truncated).sum())
            tip = robot.data.site_pos_w[:, tip_index]
            wrist_pos = robot.data.body_link_pos_w[:, wrist_index]
            wrist_quat = robot.data.body_link_quat_w[:, wrist_index]
            offset = quat_apply_inverse(wrist_quat, tip - wrist_pos)
            offset_errors.append(float((offset - expected).norm(dim=-1).max()))
            tips.append(tip[0].cpu().numpy())
            wrists.append(wrist_pos[0].cpu().numpy())
    finally:
        env.close()

    tips_np, wrists_np = np.array(tips), np.array(wrists)
    max_offset_error = max(offset_errors)
    if max_offset_error > 1e-3:
        _fail(f"blade tip drifted {max_offset_error:.4f} m from its mount offset")
    wrist_travel = float(np.linalg.norm(np.ptp(wrists_np, axis=0)))
    tip_travel = float(np.linalg.norm(np.ptp(tips_np, axis=0)))
    if wrist_travel < 1e-3:
        _fail("right wrist did not move during the smoke run; cannot show the tip follows it")
    return {
        "num_envs": num_envs, "steps": steps, "episode_ends": resets,
        "saber_mass_kg": {"min": round(float(masses.min()), 4),
                          "max": round(float(masses.max()), 4),
                          "mean": round(float(masses.mean()), 4)},
        "tip_offset_from_wrist_max_error_m": max_offset_error,
        "env0_wrist_travel_m": round(wrist_travel, 4),
        "env0_tip_travel_m": round(tip_travel, 4),
        "env0_tip_first_last": [np.round(tips_np[0], 3).tolist(),
                                np.round(tips_np[-1], 3).tolist()],
    }


def run(num_envs: int = 32, steps: int = 100, device: str = "cuda:0",
        motion_file: Path = DEFAULT_MOTION) -> dict:
    from fencing.saber.mjlab_task import TASK_ID, register

    motion_file = Path(motion_file).expanduser()
    if not motion_file.exists():
        _fail(f"motion NPZ not found: {motion_file} (build one with `fencing train` first)")
    first, second = register(), register()
    if first != second:
        _fail(f"register() returned different ids on the second call: {first} vs {second}")
    report = {"task_id": TASK_ID, "motion_file": str(motion_file), "device": device}
    report["body_order"] = check_body_order(motion_file)
    report["npz_pose_max_body_error_m"] = check_npz_kinematics(TASK_ID, motion_file, device)
    report["training_env"] = check_training_env(TASK_ID, motion_file, num_envs, steps, device)
    report["passed"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fencing saber smoke", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--out", type=Path, default=EVAL / "saber" / "smoke.json")
    args = parser.parse_args(argv)
    report = run(args.num_envs, args.steps, args.device, args.motion_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[saber smoke] PASSED, report in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
