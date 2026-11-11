"""`fencing train status`: one line per run under paths.RUNS_LOG_ROOT.

Latest iteration, mean reward and episode length come from the tensorboard event file
(the same numbers rsl_rl prints). No mjlab/torch needed, only the tensorboard package.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from fencing import paths
from fencing.train import naming, provenance

TAGS = {
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "episode_length",
    "Metrics/motion/error_body_pos": "error_body_pos",
}
ACTIVE_WINDOW_S = 300


@dataclass
class RunStatus:
    run_dir: Path
    iteration: int | None
    values: dict[str, float]
    checkpoints: list[str]
    is_active: bool
    motion: str | None


def read_latest_scalars(event_files: list[Path]) -> tuple[int | None, dict[str, float]]:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    latest_step: int | None = None
    values: dict[str, float] = {}
    for event_file in sorted(event_files):
        for event in EventFileLoader(str(event_file)).Load():
            for value in event.summary.value:
                if value.tag not in TAGS:
                    continue
                scalar = value.simple_value if value.HasField("simple_value") else _tensor_scalar(value)
                values[TAGS[value.tag]] = scalar
                latest_step = event.step if latest_step is None else max(latest_step, event.step)
    return latest_step, values


def _tensor_scalar(value) -> float:
    from tensorboard.util import tensor_util

    return float(tensor_util.make_ndarray(value.tensor))


def run_status(run_dir: Path) -> RunStatus:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    iteration, values = read_latest_scalars(event_files) if event_files else (None, {})
    newest = max((f.stat().st_mtime for f in event_files), default=0.0)
    record = provenance.read_run_json(run_dir) or {}
    motion = record.get("motion_csv") or record.get("motion_npz")
    return RunStatus(
        run_dir=run_dir,
        iteration=iteration,
        values=values,
        checkpoints=[p.name for p in naming.list_checkpoints(run_dir)],
        is_active=time.time() - newest < ACTIVE_WINDOW_S,
        motion=Path(motion).name if motion else None,
    )


def format_status(status: RunStatus) -> str:
    def fmt(key: str, spec: str) -> str:
        value = status.values.get(key)
        return format(value, spec) if value is not None else "-"

    ckpts = status.checkpoints
    ckpt_text = f"{len(ckpts)} ({ckpts[-1]})" if ckpts else "0"
    state = "ACTIVE" if status.is_active else "idle"
    return (
        f"{status.run_dir.parent.name}/{status.run_dir.name}  [{state}]  "
        f"iter={status.iteration if status.iteration is not None else '-'}  "
        f"reward={fmt('reward', '.2f')}  ep_len={fmt('episode_length', '.1f')}  "
        f"body_pos_err={fmt('error_body_pos', '.3f')}  ckpts={ckpt_text}"
        + (f"  motion={status.motion}" if status.motion else "")
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-root", type=Path, default=paths.RUNS_LOG_ROOT)


def run(args: argparse.Namespace) -> int:
    run_dirs = sorted(d for d in args.log_root.glob("*/*") if d.is_dir())
    if not run_dirs:
        print(f"no runs under {args.log_root}")
        return 0
    for run_dir in run_dirs:
        print(format_status(run_status(run_dir)), flush=True)
    return 0
