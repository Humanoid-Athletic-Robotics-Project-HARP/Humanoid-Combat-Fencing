"""`fencing <command> ...` dispatcher.

Each command lives in its own module exposing `main(argv: list[str] | None) -> int`.
Modules are imported lazily so a command only needs its own dependencies.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    # name: (module, one-line help)
    "scrape": ("fencing.scrape.cli", "Search, filter and download saber fencing videos; sync data/video_log.md"),
    "screen": ("fencing.pose.screen", "Detect/track people in videos, score clips, pick the fencer track"),
    "pose": ("fencing.pose.cli", "Video -> human motion (GVHMR if SMPL-X models exist, else fallback)"),
    "retarget": ("fencing.retarget.cli", "Human motion -> G1 motion CSV with GMR"),
    "motion": ("fencing.motion.cli", "Clean, segment, mirror, check and publish G1 motion CSVs"),
    "saber": ("fencing.saber.cli", "Saber model, blade-tip frame and wrist-reach analysis (Phase 1.3)"),
    "train": ("fencing.train.cli", "CSV->NPZ, train, play, evaluate and export with mjlab (local, no W&B)"),
    "sim2sim": ("fencing.sim2sim.cli", "Run an exported ONNX policy in plain CPU MuJoCo"),
    "pipeline": ("fencing.pipeline", "End-to-end: videos -> cleaned motions -> trained, evaluated policies"),
}


def _usage() -> str:
    width = max(len(n) for n in COMMANDS)
    lines = ["usage: fencing <command> [args...]", "", "commands:"]
    lines += [f"  {n.ljust(width)}  {h}" for n, (_, h) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
