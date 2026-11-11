"""Names and paths for motion NPZ files, runs and evaluation outputs.

Pure functions only (no mjlab/torch imports) so they are testable in the fencing venv.

Conventions (docs/pipeline.md, ADR 0001):
    NPZ:      paths.NPZ/<motion>_mjlab.npz       (engine-specific, hence the suffix)
    Run name: <motion>_<version>                  e.g. en_garde_v1, dance1_subject2_p12
    Run dir:  paths.RUNS_LOG_ROOT/<experiment>/<YYYY-mm-dd_HH-MM-SS>_<run name>  (mjlab's own)
    Eval:     paths.EVAL/<run dir name>/<checkpoint stem>/metrics.json
"""

from __future__ import annotations

import re
from pathlib import Path

from fencing import paths

NPZ_SUFFIX = "_mjlab"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")
# mjlab prefixes run dirs with a timestamp: 2026-11-09_04-33-56_<run name>
_RUN_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_(?P<run_name>.+))?$")


def validate_name(name: str, what: str = "name") -> str:
    """Reject names that would escape a directory or break shell use."""
    if not name or not _SAFE_NAME.match(name):
        raise ValueError(
            f"invalid {what} {name!r}: use letters, digits, '_', '-', '.' and start with a "
            "letter or digit"
        )
    return name


def motion_name_from_path(path: Path | str) -> str:
    """`.../dance1_subject2.csv` -> `dance1_subject2`; `x_mjlab.npz` -> `x`."""
    stem = Path(path).stem
    if stem.endswith(NPZ_SUFFIX):
        stem = stem[: -len(NPZ_SUFFIX)]
    return validate_name(stem, "motion name")


def npz_path(motion_name: str, npz_dir: Path | None = None) -> Path:
    """Where `fencing train npz` writes the NPZ for a motion (idempotent in the suffix)."""
    name = validate_name(motion_name, "motion name")
    if name.endswith(NPZ_SUFFIX):
        name = name[: -len(NPZ_SUFFIX)]
    return (npz_dir or paths.NPZ) / f"{name}{NPZ_SUFFIX}.npz"


def run_name(motion: str, version: str) -> str:
    """Run-name convention `<motion>_<version>`; version must not repeat the motion."""
    validate_name(motion, "motion name")
    validate_name(version, "version")
    if version.startswith(motion):
        raise ValueError(f"version {version!r} already contains the motion name {motion!r}")
    return f"{motion}_{version}"


def run_name_from_dir(run_dir: Path | str) -> str:
    """Run name part of an mjlab run dir name (falls back to the whole dir name)."""
    name = Path(run_dir).name
    match = _RUN_DIR.match(name)
    if match and match.group("run_name"):
        return match.group("run_name")
    return name


def checkpoint_iteration(checkpoint: Path | str) -> int:
    match = _CHECKPOINT.match(Path(checkpoint).name)
    if not match:
        raise ValueError(f"not an rsl_rl checkpoint name (model_<N>.pt): {checkpoint}")
    return int(match.group(1))


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoints in a run dir, sorted by iteration (not lexically: model_500 < model_1000)."""
    found = [p for p in run_dir.glob("model_*.pt") if _CHECKPOINT.match(p.name)]
    return sorted(found, key=checkpoint_iteration)


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    """`None` -> latest; `1500` or `model_1500.pt` or a path -> that file. Fails loudly."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir does not exist: {run_dir}")
    if checkpoint is None:
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            raise FileNotFoundError(f"no model_<N>.pt checkpoints in {run_dir}")
        return checkpoints[-1]
    candidate = Path(checkpoint)
    if candidate.is_file():
        return candidate
    if checkpoint.isdigit():
        candidate = run_dir / f"model_{checkpoint}.pt"
    else:
        candidate = run_dir / checkpoint
    if not candidate.is_file():
        available = ", ".join(p.name for p in list_checkpoints(run_dir)) or "none"
        raise FileNotFoundError(f"checkpoint {checkpoint!r} not found in {run_dir} (have: {available})")
    return candidate


def eval_dir(run_dir: Path, checkpoint: Path, eval_root: Path | None = None) -> Path:
    """paths.EVAL/<run dir name>/<checkpoint stem>/ -- the run dir name is unique, the run
    name alone is not (two runs can share a name)."""
    return (eval_root or paths.EVAL) / Path(run_dir).name / Path(checkpoint).stem


def resolve_run_dir(run: str, log_root: Path | None = None) -> Path:
    """Accept a run dir path, a run dir name, or a run name (latest match wins)."""
    candidate = Path(run).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    root = log_root or paths.RUNS_LOG_ROOT
    matches = sorted(
        d for d in root.glob("*/*") if d.is_dir() and (d.name == run or run_name_from_dir(d) == run)
    )
    if not matches:
        raise FileNotFoundError(f"no run dir matching {run!r} under {root}")
    return matches[-1]
