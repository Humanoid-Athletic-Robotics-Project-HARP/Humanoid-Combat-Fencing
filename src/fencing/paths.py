"""Where everything lives.

The repo (often on a slow Windows drive under WSL) holds only source, docs and small
motion CSVs. Everything heavy -- venvs, videos, pose outputs, NPZ files, training runs --
lives under HCF_DATA on the Linux filesystem (default ~/hcf-data).

Layout under HCF_DATA:
    venvs/<name>/            uv venvs: fencing, mjlab, gmr, gvhmr, ...
    videos/raw/<yt_id>.mp4   downloaded source videos (never committed)
    videos/manifest.jsonl    one JSON record per scraped video
    clips/<clip_id>.mp4      cut single-motion segments
    pose/<clip_id>/          pose-estimation outputs (GVHMR or fallback)
    retarget/<clip_id>.pkl   GMR output; retarget/csv/<clip_id>.csv raw G1 CSV (30 fps)
    npz/<name>.npz           mjlab motion NPZ files
    runs/                    mjlab training runs (log root: runs/rsl_rl)
    eval/                    evaluation and sim2sim reports
    lafan1/g1/*.csv          LAFAN1 G1 reference motions (CC BY-NC-ND)
    logs/                    stage logs
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"
MOTIONS_DIR = REPO_ROOT / "motions"
DOCS_DIR = REPO_ROOT / "docs"
VIDEO_LOG = REPO_ROOT / "data" / "video_log.md"

DATA = Path(os.environ.get("HCF_DATA", Path.home() / "hcf-data")).expanduser()

VENVS = DATA / "venvs"
VIDEOS_RAW = DATA / "videos" / "raw"
VIDEO_MANIFEST = DATA / "videos" / "manifest.jsonl"
CLIPS = DATA / "clips"
POSE = DATA / "pose"
RETARGET = DATA / "retarget"
NPZ = DATA / "npz"
RUNS = DATA / "runs"
RUNS_LOG_ROOT = RUNS / "rsl_rl"
EVAL = DATA / "eval"
LAFAN1_G1 = DATA / "lafan1" / "g1"
LOGS = DATA / "logs"

# Upstream robot model used for FK, cleanup and sim2sim (same file mjlab trains on).
G1_MJCF = THIRD_PARTY / "mjlab" / "src" / "mjlab" / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"

TRACKING_TASK = "Mjlab-Tracking-Flat-Unitree-G1"


def venv_dir(name: str) -> Path:
    return VENVS / name


def venv_python(name: str) -> Path:
    """Python interpreter of one of the per-stage venvs (fencing, mjlab, gmr, gvhmr)."""
    return venv_dir(name) / "bin" / "python"


def ensure_dirs() -> None:
    for d in (VIDEOS_RAW, CLIPS, POSE, RETARGET, NPZ, RUNS_LOG_ROOT, EVAL, LOGS):
        d.mkdir(parents=True, exist_ok=True)
