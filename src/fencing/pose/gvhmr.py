"""GVHMR branch of `fencing pose`: used only when the licence-gated SMPL / SMPL-X models exist.

UNTESTED on this machine: the body models are not downloaded (they need a registration at
smpl-x.is.tue.mpg.de / smpl.is.tue.mpg.de, see docs/setup.md), so this code path has never
run end to end. It follows GVHMR's own demo (third_party/GVHMR/tools/demo/demo.py):

    1. cut the clip to clips/<clip_id>.mp4 at a true 30 fps (GVHMR assumes 30 fps input);
    2. run tools/demo/demo.py -s (static camera) in the `gvhmr` venv with output under
       pose/<clip_id>/gvhmr/, then copy hmr4d_results.pt to pose/<clip_id>/hmr4d_results.pt.

Known gap: GVHMR's demo tracks the largest person with its own YOLO tracker, not the fencer
track `fencing screen` chose. On two-person footage check the overlay video it renders.

GVHMR resolves checkpoints as `inputs/checkpoints/...` both relative to the working directory
and to its package root, so we run with cwd = HCF_DATA/gvhmr and link
third_party/GVHMR/inputs -> HCF_DATA/gvhmr/inputs if that link does not exist yet.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fencing import paths
from fencing.pose.venvs import run_in_venv

GVHMR_ROOT = paths.DATA / "gvhmr"
CHECKPOINTS = GVHMR_ROOT / "inputs" / "checkpoints"
BODY_MODELS = CHECKPOINTS / "body_models"
REQUIRED_BODY_MODELS = (
    BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz",
    BODY_MODELS / "smpl" / "SMPL_NEUTRAL.pkl",
)
GVHMR_SRC = paths.THIRD_PARTY / "GVHMR"


def missing_body_models() -> list[Path]:
    return [p for p in REQUIRED_BODY_MODELS if not p.exists()]


def gvhmr_available() -> bool:
    return not missing_body_models() and (CHECKPOINTS / "gvhmr").exists()


def cut_clip_30fps(video: Path, start_s: float, end_s: float, out_mp4: Path) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video),
        "-an", "-r", "30", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out_mp4),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out_mp4


def run_gvhmr(clip_id: str, video: Path, start_s: float, end_s: float, out_dir: Path, log: Path) -> Path:
    """Run GVHMR on one clip; returns pose/<clip_id>/hmr4d_results.pt. Untested, see module doc."""
    link = GVHMR_SRC / "inputs"
    if not link.exists():
        link.symlink_to(GVHMR_ROOT / "inputs", target_is_directory=True)
    clip_mp4 = cut_clip_30fps(video, start_s, end_s, paths.CLIPS / f"{clip_id}.mp4")
    work = out_dir / "gvhmr"
    run_in_venv(
        "gvhmr",
        [str(GVHMR_SRC / "tools" / "demo" / "demo.py"), f"--video={clip_mp4}", f"--output_root={work}", "-s"],
        log,
        cwd=GVHMR_ROOT,
        extra_pythonpath=[GVHMR_SRC],
    )
    produced = work / clip_mp4.stem / "hmr4d_results.pt"
    if not produced.exists():
        raise FileNotFoundError(f"GVHMR finished but {produced} is missing (log {log})")
    target = out_dir / "hmr4d_results.pt"
    shutil.copy2(produced, target)
    return target
