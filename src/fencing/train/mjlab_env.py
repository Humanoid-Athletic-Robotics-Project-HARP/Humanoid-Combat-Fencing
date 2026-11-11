"""Shared mjlab plumbing for eval/export/play/record-obs. Imports mjlab and torch, so only
import this inside the mjlab venv (the subcommand modules import it lazily in run())."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import torch

from fencing import paths
from fencing.train import naming, provenance

SABER_TASK_MODULE = "fencing.saber.mjlab_task"


def register_tasks() -> list[str]:
    """Register mjlab's tasks plus this repo's saber task (if it exists yet)."""
    import mjlab.tasks  # noqa: F401  (populates the registry)
    from mjlab.tasks.registry import list_tasks

    try:
        import importlib

        saber_task = importlib.import_module(SABER_TASK_MODULE)
    except ModuleNotFoundError as error:
        print(f"[train] saber task not available ({error.name} not importable); "
              "only upstream mjlab tasks are registered", flush=True)
    else:
        saber_task.register()
        print(f"[train] registered saber task(s) from {SABER_TASK_MODULE}", flush=True)
    return list_tasks()


def require_task(task: str) -> None:
    tasks = register_tasks()
    if task not in tasks:
        tracking = [t for t in tasks if "Tracking" in t]
        raise ValueError(f"unknown task {task!r}; tracking tasks: {tracking}")


def run_task(run_dir: Path) -> str:
    record = provenance.read_run_json(run_dir)
    if record and record.get("task"):
        return record["task"]
    return paths.TRACKING_TASK


def run_motion_file(run_dir: Path) -> Path:
    """The NPZ a run trained on: run.json if we launched it, else mjlab's params/env.yaml."""
    record = provenance.read_run_json(run_dir)
    if record and record.get("motion_npz"):
        return Path(record["motion_npz"])
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        raise FileNotFoundError(f"neither run.json nor params/env.yaml in {run_dir}")
    # A regex instead of yaml.load: env.yaml holds python-object tags we do not want to construct.
    match = re.search(r"^\s*motion_file:\s*(\S+)\s*$", env_yaml.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"no motion_file in {env_yaml}")
    motion = Path(match.group(1))
    if not motion.is_file():
        raise FileNotFoundError(f"run {run_dir.name} trained on {motion}, which no longer exists")
    return motion


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def apply_trained_action_scale(env_cfg, run_dir: Path, task: str) -> bool:
    """Set env_cfg's joint_pos scale to the one `run_dir` trained with. True if it changed."""
    trained_scale = provenance.trained_action_scale(run_dir)
    if trained_scale is None:
        return False
    action = env_cfg.actions["joint_pos"]
    changed = not isinstance(action.scale, dict) or dict(action.scale) != trained_scale
    if changed:
        print(f"[train] {run_dir.name}: using the action scale it trained with, which differs from "
              f"{task}'s current default", flush=True)
    action.scale = trained_scale
    return changed


def make_env(
    task: str,
    motion_file: Path,
    num_envs: int,
    *,
    play: bool,
    device: str,
    configure=None,
    render_mode: str | None = None,
    run_dir: Path | None = None,
):
    """Build a wrapped mjlab env. `configure(env_cfg)` may edit the config first.

    With `run_dir`, the joint_pos action scale is the one that run trained with (its
    params/env.yaml), not the task's current default: a checkpoint's actions are only
    meaningful under its own scale, and export writes that scale into the ONNX metadata.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    require_task(task)
    env_cfg = load_env_cfg(task, play=play)
    agent_cfg = load_rl_cfg(task)
    motion_cmd = env_cfg.commands.get("motion")
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise ValueError(f"task {task} is not a tracking task (no motion command)")
    motion_cmd.motion_file = str(motion_file)
    env_cfg.scene.num_envs = num_envs
    if run_dir is not None:
        apply_trained_action_scale(env_cfg, run_dir, task)
    if configure is not None:
        configure(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), env_cfg, agent_cfg


def load_runner(env, task: str, agent_cfg, checkpoint: Path, device: str):
    """Runner with the actor weights of `checkpoint` (critic/optimizer not needed)."""
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_runner_cls

    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner


def resolve(run: str, checkpoint: str | None) -> tuple[Path, Path]:
    run_dir = naming.resolve_run_dir(run)
    return run_dir, naming.resolve_checkpoint(run_dir, checkpoint)


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return path
