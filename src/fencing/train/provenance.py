"""run.json: what a training run was made from, written into the run dir at launch.

mjlab already stores params/env.yaml, params/agent.yaml and a git diff of mjlab itself.
What it cannot know is which *motion CSV* (and which commit of this repo) the NPZ came
from; run.json records that so a policy can be traced back to its data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fencing import paths

RUN_JSON = "run.json"
SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit(repo_root: Path | None = None) -> dict[str, str | bool | None]:
    """Short SHA plus a dirty flag: a dirty tree means the commit alone does not reproduce."""
    root = repo_root or paths.REPO_ROOT

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": sha, "dirty": bool(status) if status is not None else None}


@dataclass
class RunRecord:
    run_name: str
    task: str
    motion_npz: str
    motion_npz_sha256: str
    motion_csv: str | None
    motion_csv_sha256: str | None
    repo_commit: str | None
    repo_dirty: bool | None
    iterations: int
    num_envs: int
    started_at: str
    resume_from: str | None = None
    extra: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


def build_run_record(
    *,
    run_name: str,
    task: str,
    motion_npz: Path,
    motion_csv: Path | None,
    iterations: int,
    num_envs: int,
    resume_from: Path | None = None,
    started_at: datetime | None = None,
    repo_root: Path | None = None,
    extra: dict | None = None,
) -> RunRecord:
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if not motion_npz.is_file():
        raise FileNotFoundError(f"motion NPZ not found: {motion_npz}")
    if motion_csv is not None and not motion_csv.is_file():
        raise FileNotFoundError(f"motion CSV not found: {motion_csv}")
    commit = repo_commit(repo_root)
    return RunRecord(
        run_name=run_name,
        task=task,
        motion_npz=str(motion_npz.resolve()),
        motion_npz_sha256=sha256_file(motion_npz),
        motion_csv=str(motion_csv.resolve()) if motion_csv else None,
        motion_csv_sha256=sha256_file(motion_csv) if motion_csv else None,
        repo_commit=commit["commit"],  # type: ignore[arg-type]
        repo_dirty=commit["dirty"],  # type: ignore[arg-type]
        iterations=iterations,
        num_envs=num_envs,
        started_at=(started_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        resume_from=str(resume_from) if resume_from else None,
        extra=dict(extra or {}),
    )


def write_run_json(run_dir: Path, record: RunRecord) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / RUN_JSON
    out.write_text(json.dumps(asdict(record), indent=2) + "\n")
    return out


def read_run_json(run_dir: Path) -> dict | None:
    path = run_dir / RUN_JSON
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def trained_action_scale(run_dir: Path) -> dict[str, float] | None:
    """The joint_pos action scale a run trained with, from mjlab's params/env.yaml.

    The policy's outputs only mean something under this scale, so eval/export/play must use
    it even if the task's default has changed since (the saber wrist scale did, 2026-11-09).
    None when there is no env.yaml or the scale is not a per-joint mapping. Parsed by indent
    rather than yaml.load: env.yaml holds python-object tags we do not want to construct.
    """
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        return None
    section, scale_indent, scale = None, None, {}
    for line in env_yaml.read_text().splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            section = line.strip()
            scale_indent = None
            continue
        if section != "actions:":
            continue
        if scale_indent is None:
            if line.strip() == "scale:":
                scale_indent = indent
            continue
        if indent <= scale_indent:
            break
        key, _, value = line.strip().rpartition(":")
        scale[key] = float(value)
    return scale or None


def read_json_file(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def find_csv_for_npz(npz: Path) -> Path | None:
    """Best effort: an NPZ written by `fencing train npz` has a sidecar naming its CSV."""
    source = read_json_file(npz.with_suffix(".json")).get("source_csv")
    if source and Path(source).is_file():
        return Path(source)
    return None
