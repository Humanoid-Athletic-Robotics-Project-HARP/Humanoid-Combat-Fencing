"""GVHMR branch of `fencing pose`, in two phases so the GPU work can start before the
licence-gated body models are downloaded.

    phase 1  `fencing pose ... --gvhmr-preprocess`   needs no body model
             cut the clip to clips/<clip_id>.mp4 at a true 30 fps (GVHMR assumes 30 fps), then
             fencing.pose.gvhmr_worker: YOLOv8x detections, the pipeline's fencer box per frame,
             ViTPose-H keypoints, HMR2 features, and the fencer check. Cached under
             pose/<clip_id>/gvhmr/<clip_id>/preprocess/.
    phase 2  `fencing pose ...` once body_models/smplx/SMPLX_NEUTRAL.npz exists
             the same worker with --predict: reuses the phase 1 cache (or makes it) and runs only
             the GVHMR network -> pose/<clip_id>/hmr4d_results.pt.

GVHMR's demo tracks the largest person; the worker gives it the fencer `fencing screen` chose
instead (see fencing.pose.gvhmr_boxes), and writes pose/<clip_id>/gvhmr/fencer_check.json and
fencer_check.png as the evidence. A clip whose GVHMR keypoints are not on MediaPipe's fencer
keeps the MediaPipe fallback.

GVHMR resolves checkpoints as `inputs/checkpoints/...` both relative to the working directory
and to its package root, so the worker runs with cwd = HCF_DATA/gvhmr and
third_party/GVHMR/inputs links to HCF_DATA/gvhmr/inputs (made here if missing).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from fencing import paths
from fencing.pose.clips import ClipSpec
from fencing.pose.venvs import WorkerError, eprint, run_in_venv

GVHMR_ROOT = paths.DATA / "gvhmr"
CHECKPOINTS = GVHMR_ROOT / "inputs" / "checkpoints"
BODY_MODELS = CHECKPOINTS / "body_models"
SMPLX_MODEL = BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"  # GVHMR network and GMR retarget
SMPL_MODEL = BODY_MODELS / "smpl" / "SMPL_NEUTRAL.pkl"  # GVHMR's render videos only
REQUIRED_BODY_MODELS = (SMPLX_MODEL, SMPL_MODEL)
GVHMR_CHECKPOINT = CHECKPOINTS / "gvhmr" / "gvhmr_siga24_release.ckpt"
PREPROCESS_CHECKPOINTS = (
    CHECKPOINTS / "yolo" / "yolov8x.pt",
    CHECKPOINTS / "vitpose" / "vitpose-h-multi-coco.pth",
    CHECKPOINTS / "hmr2" / "epoch=10-step=25000.ckpt",
)
PREPROCESS_FILES = ("bbx.pt", "vitpose.pt", "vit_features.pt")
GVHMR_SRC = paths.THIRD_PARTY / "GVHMR"
GVHMR_FPS = 30
MIN_YOLO_FRAC = 0.5  # fencer check: share of frames whose box is a YOLO person detection
MTIME_SLACK_S = 2.0  # file systems under /mnt/c round mtimes
THREAD_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}


def missing_body_models() -> list[Path]:
    return [p for p in REQUIRED_BODY_MODELS if not p.exists()]


def gvhmr_available() -> bool:
    """The GVHMR network can run (SMPL_NEUTRAL.pkl is only needed for --gvhmr-render)."""
    return SMPLX_MODEL.exists() and GVHMR_CHECKPOINT.exists()


def missing_preprocess_checkpoints() -> list[Path]:
    return [p for p in PREPROCESS_CHECKPOINTS if not p.exists()]


def work_root(clip_id: str) -> Path:
    """pose/<clip_id>/gvhmr: the worker's --output-root and the fencer check."""
    return paths.POSE / clip_id / "gvhmr"


def output_dir(clip_id: str) -> Path:
    """GVHMR's own per-video folder (demo.yaml: output_root/video_name)."""
    return work_root(clip_id) / clip_id


def results_file(clip_id: str) -> Path:
    return output_dir(clip_id) / "hmr4d_results.pt"


def preprocess_done(clip_id: str) -> bool:
    pre = output_dir(clip_id) / "preprocess"
    return all((pre / name).exists() for name in PREPROCESS_FILES) and (work_root(clip_id) / "fencer_check.json").exists()


def read_fencer_check(clip_id: str) -> dict:
    path = work_root(clip_id) / "fencer_check.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fencer_check_reason(check: dict) -> str | None:
    """Why GVHMR must not be used for this clip, or None. Fails only on evidence: GVHMR's
    keypoints on someone other than MediaPipe's fencer, or the person detector backing the
    fencer box on under MIN_YOLO_FRAC of the frames (a logo or graphic the screen tracked)."""
    kp = check.get("keypoints") or {}
    if kp.get("same_person") is False:
        return (
            f"its keypoints are not on MediaPipe's fencer (centroid distance median "
            f"{kp.get('centroid_dist_median')} box heights, {kp.get('frac_within', 0):.0%} of frames close)"
        )
    yolo_frac = (check.get("boxes") or {}).get("yolo_frac")
    if yolo_frac is not None and yolo_frac < MIN_YOLO_FRAC:
        return f"YOLO found a person in the fencer box on only {yolo_frac:.0%} of frames (needs {MIN_YOLO_FRAC:.0%})"
    return None


def fencer_check_failed(check: dict) -> bool:
    return fencer_check_reason(check) is not None


def cut_clip_30fps(video: Path, start_s: float, end_s: float, out_mp4: Path) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video),
        "-an", "-r", str(GVHMR_FPS), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out_mp4),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        out_mp4.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not cut {video} [{start_s}, {end_s}] s: {proc.stderr.strip()[-500:]}")
    return out_mp4


def ensure_inputs_link() -> None:
    link = GVHMR_SRC / "inputs"
    if not link.exists():
        link.symlink_to(GVHMR_ROOT / "inputs", target_is_directory=True)


def results_written_since(clip_id: str, started: float) -> bool:
    """hmr4d_results.pt exists, is not empty and was written by the run that began at `started`
    (a result from an earlier run says nothing about why this worker died)."""
    produced = results_file(clip_id)
    if not produced.exists():
        return False
    info = produced.stat()
    return info.st_size > 0 and info.st_mtime >= started - MTIME_SLACK_S


def run_clip(
    clip: ClipSpec,
    video: Path,
    log: Path,
    predict: bool,
    render: bool = False,
    redo_preprocess: bool = False,
    redo_results: bool = False,
) -> dict:
    """One clip through the GVHMR worker; returns its fencer check. With predict, also copies
    hmr4d_results.pt to pose/<clip_id>/hmr4d_results.pt."""
    ensure_inputs_link()
    if redo_preprocess and output_dir(clip.clip_id).exists():
        shutil.rmtree(output_dir(clip.clip_id))
    if redo_results:
        results_file(clip.clip_id).unlink(missing_ok=True)
    clip_mp4 = paths.CLIPS / f"{clip.clip_id}.mp4"
    if redo_preprocess or not clip_mp4.exists() or clip_mp4.stat().st_size == 0:
        cut_clip_30fps(video, clip.start_s, clip.end_s, clip_mp4)
    cmd = [
        "-m", "fencing.pose.gvhmr_worker", "--video", str(clip_mp4), "--output-root", str(work_root(clip.clip_id)),
        "--start", f"{clip.start_s:.3f}", "--track-npz", str(paths.POSE / clip.yt_id / "track.npz"),
        "--track-id", str(clip.track_id),
    ]  # fmt: skip
    raw_npz = paths.POSE / clip.clip_id / "mediapipe_raw.npz"
    if raw_npz.exists():
        cmd += ["--mediapipe-npz", str(raw_npz)]
    if predict:
        cmd.append("--predict")
    if render:
        cmd.append("--render")
    started = time.time()
    try:
        run_in_venv("gvhmr", cmd, log, cwd=GVHMR_ROOT, extra_env=THREAD_ENV, extra_pythonpath=[GVHMR_SRC])
    except WorkerError as exc:
        # The worker can die (killed, out of memory) in rendering, after the network result is saved.
        if not (predict and render and results_written_since(clip.clip_id, started)):
            raise
        eprint(f"[gvhmr] {clip.clip_id}: WARNING: the worker died after saving hmr4d_results.pt (rendering failed); keeping the result. {str(exc).splitlines()[0]}")
    check = read_fencer_check(clip.clip_id)
    if predict:
        produced = results_file(clip.clip_id)
        if not produced.exists():
            raise FileNotFoundError(f"GVHMR finished but {produced} is missing (log {log})")
        shutil.copy2(produced, paths.POSE / clip.clip_id / "hmr4d_results.pt")
    return check
