"""`fencing train run`: launch mjlab tracking training locally, with provenance.

By default the training is started as a detached background process that logs to
paths.LOGS/train_<run name>_<timestamp>.log, and this command returns immediately
(`--foreground` runs it in this process; that is also what the background child does).

The run dir is mjlab's own (paths.RUNS_LOG_ROOT/<experiment>/<timestamp>_<run name>);
we compute it the same way mjlab's launch_training does so run.json can be written
into it before the first iteration.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from fencing import paths
from fencing.train import naming, npz, provenance


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("motion", type=Path, help="motion CSV (converted first) or an mjlab NPZ")
    parser.add_argument("--name", required=True,
                        help="version, e.g. v1 -> run name <motion>_v1 (a full <motion>_... name is kept)")
    parser.add_argument("--iterations", type=int, help="PPO iterations (default: the task's max_iterations)")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--task", default=paths.TRACKING_TASK)
    parser.add_argument("--resume", help="run dir/name to resume from (loads its latest or --checkpoint)")
    parser.add_argument("--checkpoint", help="checkpoint to resume from (iteration or model_<N>.pt)")
    parser.add_argument("--input-fps", type=float, default=30.0, help="CSV frame rate (CSV input only)")
    parser.add_argument("--foreground", action="store_true", help="train in this process")
    parser.add_argument("--log-root", type=Path, default=paths.RUNS_LOG_ROOT)


def resolve_run_name(motion: str, name: str) -> str:
    return name if name.startswith(f"{motion}_") else naming.run_name(motion, name)


def prepare_motion(motion: Path, input_fps: float) -> tuple[Path, Path | None]:
    """(npz, csv or None). A CSV is (re)converted unless its NPZ sidecar has the same hash."""
    motion = motion.expanduser().resolve()
    if not motion.is_file():
        raise FileNotFoundError(f"motion file not found: {motion}")
    if motion.suffix == ".npz":
        return motion, provenance.find_csv_for_npz(motion)
    if motion.suffix != ".csv":
        raise ValueError(f"motion must be a .csv or .npz file, got {motion}")
    target = naming.npz_path(naming.motion_name_from_path(motion))
    if target.is_file() and provenance.find_csv_for_npz(target) == motion:
        sidecar = provenance.read_json_file(target.with_suffix(".json"))
        if sidecar.get("source_csv_sha256") == provenance.sha256_file(motion):
            print(f"[run] reusing {target} (same CSV hash)", flush=True)
            return target, motion
    return npz.convert_csv(motion, input_fps=input_fps), motion


def spawn_background(argv: list[str], run_name: str) -> int:
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = paths.LOGS / f"train_{run_name}_{stamp}.log"
    cmd = [sys.executable, "-m", "fencing.train.cli", "run", *argv, "--foreground"]
    with open(log_path, "ab") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, env=dict(os.environ))
    print(f"[run] started training in the background: pid {process.pid}")
    print(f"[run] log: {log_path}")
    print("[run] follow with: tail -f " + str(log_path) + "   |  progress: fencing train status")
    return 0


def train_foreground(args: argparse.Namespace, motion_npz: Path, motion_csv: Path | None, run_name: str) -> int:
    from mjlab.scripts.train import TrainConfig, run_train
    from mjlab.utils.gpu import select_gpus

    from fencing.train import mjlab_env

    mjlab_env.require_task(args.task)
    cfg = TrainConfig.from_task(args.task)
    cfg = replace(cfg, log_root=str(args.log_root))
    cfg.env.commands["motion"].motion_file = str(motion_npz)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.run_name = run_name
    cfg.agent.logger = "tensorboard"
    if args.iterations:
        cfg.agent.max_iterations = args.iterations

    resume_from = None
    if args.resume:
        resume_dir = naming.resolve_run_dir(args.resume, args.log_root)
        resume_from = naming.resolve_checkpoint(resume_dir, args.checkpoint)
        cfg.agent.resume = True
        cfg.agent.load_run = re.escape(resume_dir.name)
        cfg.agent.load_checkpoint = re.escape(resume_from.name)
        trained_scale = provenance.trained_action_scale(resume_dir)
        if trained_scale is not None:  # keep the policy's action mapping across the resume
            cfg.env.actions["joint_pos"].scale = trained_scale

    # Same naming as mjlab.scripts.train.launch_training, so run.json lands in mjlab's run dir.
    log_dir = (Path(cfg.log_root) / cfg.agent.experiment_name).resolve() / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{run_name}"
    )
    record = provenance.build_run_record(
        run_name=run_name, task=args.task, motion_npz=motion_npz, motion_csv=motion_csv,
        iterations=cfg.agent.max_iterations, num_envs=args.num_envs, resume_from=resume_from,
    )
    print(f"[run] run.json -> {provenance.write_run_json(log_dir, record)}", flush=True)

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    if num_gpus > 1:
        raise ValueError("multi-GPU training is not supported by `fencing train run`; use mjlab's train")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    run_train(args.task, cfg, log_dir)
    return 0


def run(args: argparse.Namespace) -> int:
    motion_npz, motion_csv = prepare_motion(args.motion, args.input_fps)
    run_name = resolve_run_name(naming.motion_name_from_path(motion_npz), args.name)
    if args.num_envs <= 0:
        raise ValueError(f"--num-envs must be positive, got {args.num_envs}")
    if not args.foreground:
        passthrough = [str(motion_npz), "--name", run_name, "--num-envs", str(args.num_envs),
                       "--task", args.task, "--log-root", str(args.log_root)]
        if args.iterations:
            passthrough += ["--iterations", str(args.iterations)]
        if args.resume:
            passthrough += ["--resume", args.resume]
        if args.checkpoint:
            passthrough += ["--checkpoint", args.checkpoint]
        return spawn_background(passthrough, run_name)
    return train_foreground(args, motion_npz, motion_csv, run_name)
