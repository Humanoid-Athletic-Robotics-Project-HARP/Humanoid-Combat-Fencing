"""`fencing train record-obs`: record mjlab actor observations with the state behind them.

Produces the fixture for tests/test_sim2sim_observation.py: for a few policy steps of a
real rollout we save the robot qpos/qvel, the reference frame the command was on, the
last action and mjlab's actor observation vector. The sim2sim observation builder must
reproduce that vector from the state alone. Observation noise and the encoder-bias
randomization are switched off so the vector is a deterministic function of the state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_STEPS = (3, 25, 60)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="run dir, run dir name or run name")
    parser.add_argument("--checkpoint", help="iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS),
                        help="policy steps after reset at which to record")
    parser.add_argument("--device", help="default: cuda:0 if available")


def _deterministic_obs(env_cfg) -> None:
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("encoder_bias", None)  # biased joint_pos would depend on a random draw


def record(run: str, checkpoint: str | None, out: Path, steps: list[int], device: str | None) -> Path:
    import torch

    from fencing.train import mjlab_env

    run_dir, checkpoint_path = mjlab_env.resolve(run, checkpoint)
    device = device or mjlab_env.default_device()
    task = mjlab_env.run_task(run_dir)
    env, _, agent_cfg = mjlab_env.make_env(task, mjlab_env.run_motion_file(run_dir), 1, play=True,
                                           device=device, configure=_deterministic_obs, run_dir=run_dir)
    try:
        policy = mjlab_env.load_runner(env, task, agent_cfg, checkpoint_path, device).get_inference_policy(device=device)
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        robot = base.scene["robot"]
        model = base.sim.mj_model
        term_names = list(base.observation_manager.active_terms["actor"])
        term_dims = [int(np.prod(d)) for d in base.observation_manager.group_obs_term_dim["actor"]]
        records: dict[str, list] = {k: [] for k in (
            "step", "qpos", "qvel", "time_step", "last_action", "obs", "ref_joint_pos", "ref_joint_vel",
            "ref_anchor_pos_w", "ref_anchor_quat_w", "env_origin", "imu_lin_vel")}
        obs = env.get_observations()
        for step in range(1, max(steps) + 1):
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            if bool(dones.any()):
                raise RuntimeError(f"episode ended at step {step}; record earlier steps or use a better checkpoint")
            if step not in steps:
                continue
            records["step"].append(step)
            records["qpos"].append(base.sim.data.qpos[0].cpu().numpy().astype(np.float64))
            records["qvel"].append(base.sim.data.qvel[0].cpu().numpy().astype(np.float64))
            records["time_step"].append(int(command.time_steps[0]))
            records["last_action"].append(base.action_manager.action[0].cpu().numpy())
            records["obs"].append(obs["actor"][0].cpu().numpy())
            records["ref_joint_pos"].append(command.joint_pos[0].cpu().numpy())
            records["ref_joint_vel"].append(command.joint_vel[0].cpu().numpy())
            records["ref_anchor_pos_w"].append(command.anchor_pos_w[0].cpu().numpy())
            records["ref_anchor_quat_w"].append(command.anchor_quat_w[0].cpu().numpy())
            records["env_origin"].append(base.scene.env_origins[0].cpu().numpy())
            records["imu_lin_vel"].append(base.scene["robot/imu_lin_vel"].data[0].cpu().numpy())
        joint_names = [model.joint(i).name for i in range(model.njnt)]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            **{k: np.asarray(v) for k, v in records.items()},
            term_names=np.asarray(term_names),
            term_dims=np.asarray(term_dims),
            mj_joint_names=np.asarray(joint_names),
            policy_joint_names=np.asarray(list(robot.joint_names)),
            default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy(),
            anchor_body_name=np.asarray(command.cfg.anchor_body_name),
            body_names=np.asarray(list(command.cfg.body_names)),
            checkpoint=np.asarray(f"{run_dir.name}/{checkpoint_path.name}"),
        )
    finally:
        env.close()
    return out


def run(args: argparse.Namespace) -> int:
    out = record(args.run, args.checkpoint, args.out.expanduser(), sorted(set(args.steps)), args.device)
    print(f"[record-obs] wrote {out}")
    return 0
