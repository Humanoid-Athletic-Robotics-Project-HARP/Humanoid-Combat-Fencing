"""`fencing train <subcommand>`: local mjlab training, evaluation and export (no W&B).

Runs inside the mjlab venv. When invoked from the fencing venv (`scripts/hcf train ...`)
it re-executes itself with the mjlab venv's python and this repo's src/ on PYTHONPATH,
so callers never have to pick the interpreter.

    fencing train npz <csv> [--name NAME] [--input-fps 30]
    fencing train run <csv-or-npz> --name VERSION [--iterations N] [--num-envs N] [--task T]
                      [--resume RUN_DIR] [--foreground]
    fencing train status
    fencing train eval <run-dir> [--checkpoint N] [--num-envs 1024] [--no-robustness] [--seconds S]
    fencing train export <run-dir> [--checkpoint N]
    fencing train play <run-dir> --video [--checkpoint N] [--seconds 10]
    fencing train record-obs <run-dir> --out FILE   (test fixture for sim2sim)

Metrics, thresholds and passes: docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys

from fencing import paths

REEXEC_MARKER = "HCF_TRAIN_REEXEC"

# subcommand -> (module, help). Modules import mjlab/torch lazily, only when run.
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "npz": ("fencing.train.npz", "motion CSV -> paths.NPZ/<name>_mjlab.npz (lock-safe)"),
    "run": ("fencing.train.launch", "train a tracking policy locally (tensorboard, run.json)"),
    "status": ("fencing.train.status", "list runs: iteration, reward, episode length, checkpoints"),
    "eval": ("fencing.train.evaluate", "Goal 1 metrics for a checkpoint -> paths.EVAL/.../metrics.json"),
    "export": ("fencing.train.export", "ONNX with metadata + the exact MuJoCo scene, verified in onnxruntime"),
    "play": ("fencing.train.play", "render an MP4 of the policy offscreen (MUJOCO_GL=egl)"),
    "record-obs": ("fencing.train.record_obs", "record mjlab observations for the sim2sim test fixture"),
}


def _mjlab_importable() -> bool:
    return importlib.util.find_spec("mjlab") is not None


def _reexec_in_mjlab_venv(argv: list[str]) -> None:
    python = paths.venv_python("mjlab")
    if not python.exists():
        raise FileNotFoundError(f"mjlab venv python not found at {python}; run scripts/setup_linux_gpu.sh")
    if os.environ.get(REEXEC_MARKER):
        raise RuntimeError(f"already re-executed into {python} but mjlab is still not importable")
    env = dict(os.environ)
    src = str(paths.REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env[REEXEC_MARKER] = "1"
    # The fencing venv's uv settings must not leak into the other interpreter.
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME"):
        env.pop(key, None)
    os.execve(str(python), [str(python), "-m", "fencing.train.cli", *argv], env)


def set_runtime_defaults() -> None:
    """W&B is not available on this machine; rendering is offscreen."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MUJOCO_GL", "egl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, (module_name, help_text) in SUBCOMMANDS.items():
        module = importlib.import_module(module_name)
        sub_parser = sub.add_parser(name, help=help_text, description=help_text)
        module.add_arguments(sub_parser)
        sub_parser.set_defaults(_module=module)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not _mjlab_importable() and argv and argv[0] not in {"-h", "--help"}:
        _reexec_in_mjlab_venv(argv)
    set_runtime_defaults()
    args = build_parser().parse_args(argv)
    return int(args._module.run(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
