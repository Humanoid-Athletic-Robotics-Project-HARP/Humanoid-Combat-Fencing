"""`fencing pose`: screened video ranges -> 3D human keypoints per clip (Phase 1.4c).

    fencing pose IpM7citb5bs                  # every usable range of a screened video
    fencing pose --all                        # every video with a pose/<yt_id>/screen.json
    fencing pose IpM7citb5bs --start 12 --end 19.5
    fencing pose IpM7citb5bs_12000-19500      # one clip by clip_id (re-run with --force)
    fencing pose --all --reprocess            # new post-processing, cached MediaPipe output
    fencing pose --all --max-seconds 10 --max-clips 2 --height 1.80

Usable ranges from screen.json are cut into chunks of at most --max-seconds; each chunk is a
clip `<yt_id>_<start_ms>-<end_ms>` with outputs in pose/<clip_id>/:

    mediapipe_raw.npz     MediaPipe Pose Landmarker on the fencer track (pose venv)
    keypoints_world.npz   keypoints (T,33,3) z-up metres at 30 fps, times_s, valid, visibility
    pose.json             method, fps, height, handedness_guess, camera, quality stats
    overlay.png           contact sheet: landmarks on the video + the lifted 3D skeleton
    hmr4d_results.pt      only with GVHMR (needs the licence-gated SMPL/SMPL-X files)

Method: GVHMR if HCF_DATA/gvhmr/inputs/checkpoints/body_models/{smplx/SMPLX_NEUTRAL.npz,
smpl/SMPL_NEUTRAL.pkl} exist (untested here, see fencing.pose.gvhmr), else the licence-free
`fallback_mediapipe` path. The MediaPipe keypoints are produced in both cases (handedness,
overlay, and a retarget fallback). Clips with a pose.json are skipped unless --force; one
failing clip never stops the batch. Logs: logs/pose/<clip_id>.log.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fencing import paths
from fencing.pose import gvhmr
from fencing.pose.clips import DEFAULT_MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, ClipSpec, parse_clip_id, plan_clips
from fencing.pose.postprocess import OUTPUT_FPS, PoseResult, process_mediapipe
from fencing.pose.root_trajectory import DEFAULT_HUMAN_HEIGHT_M
from fencing.pose.venvs import WorkerError, eprint, run_in_venv

POSE_VERSION = 2  # 2: MediaPipe world landmarks rotated from the hip-ray frame to camera axes
MEDIAPIPE_MODEL = paths.DATA / "models" / "pose_landmarker_heavy.task"
FALLBACK_METHOD = "fallback_mediapipe"


def load_screen(yt_id: str) -> dict:
    path = paths.POSE / yt_id / "screen.json"
    if not path.exists():
        raise FileNotFoundError(f"{yt_id} is not screened yet ({path} missing): run `fencing screen {yt_id}`")
    return json.loads(path.read_text())


def screened_videos() -> list[str]:
    return sorted(p.parent.name for p in paths.POSE.glob("*/screen.json"))


def write_keypoints(path: Path, result: PoseResult) -> None:
    np.savez_compressed(
        path,
        keypoints=result.keypoints_world.astype(np.float32),
        times_s=result.times_s,
        valid=result.valid,
        visibility=result.visibility.astype(np.float32),
        image_xy=result.image_xy.astype(np.float32),
        root_cam=result.root_cam,
        fps=np.array(OUTPUT_FPS),
    )


def pose_summary(clip: ClipSpec, screen: dict, result: PoseResult, method: str, args, seconds: float) -> dict:
    notes = []
    if method == FALLBACK_METHOD:
        notes.append("monocular MediaPipe + pinhole root solve; no body model, wrist twist approximate")
    if result.handedness is None:
        notes.append("handedness undecided (cues weak or disagree)")
    return {
        "clip_id": clip.clip_id,
        "yt_id": clip.yt_id,
        "source_start_s": clip.start_s,
        "source_end_s": clip.end_s,
        "track_id": clip.track_id,
        "video": screen.get("video"),
        "pose_method": method,
        "pose_version": POSE_VERSION,
        "fps": OUTPUT_FPS,
        "human_height_m": args.height,
        "handedness_guess": result.handedness,
        "handedness_score": round(result.handedness_score, 3),
        "camera": {
            "width": screen.get("width"),
            "height": screen.get("height"),
            "focal_px": round(result.focal_px, 1),
            "hfov_deg": args.hfov,
            "pitch_down_deg": round(float(np.rad2deg(result.pitch_down_rad)), 2),
            "view": screen.get("view"),
        },
        "body_scale": round(result.scale, 4),
        "quality": result.stats,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "notes": "; ".join(notes),
    }


def pose_one(clip: ClipSpec, screen: dict, args, use_gvhmr: bool) -> dict:
    out_dir = paths.POSE / clip.clip_id
    pose_json = out_dir / "pose.json"
    log = paths.LOGS / "pose" / f"{clip.clip_id}.log"
    if pose_json.exists() and not (args.force or args.reprocess):
        eprint(f"[pose] {clip.clip_id}: done already ({pose_json})")
        return json.loads(pose_json.read_text())
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(screen.get("video") or paths.VIDEOS_RAW / f"{clip.yt_id}.mp4")
    raw_npz = out_dir / "mediapipe_raw.npz"
    if args.force or not raw_npz.exists():
        eprint(f"[pose] {clip.clip_id}: MediaPipe on track {clip.track_id} ({clip.duration_s:.1f}s, log {log})")
        run_in_venv(
            "pose",
            [
                "-m", "fencing.pose.mediapipe_worker", "--video", str(video),
                "--start", f"{clip.start_s:.3f}", "--end", f"{clip.end_s:.3f}",
                "--track-npz", str(paths.POSE / clip.yt_id / "track.npz"), "--track-id", str(clip.track_id),
                "--out", str(raw_npz), "--model", str(MEDIAPIPE_MODEL),
            ],  # fmt: skip
            log,
            extra_env={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
        )
    with np.load(raw_npz) as data:
        raw = {k: data[k] for k in data.files}
    result = process_mediapipe(raw, args.height, args.hfov, args.pitch_deg)
    write_keypoints(out_dir / "keypoints_world.npz", result)

    method = FALLBACK_METHOD
    if use_gvhmr:
        eprint(f"[pose] {clip.clip_id}: GVHMR (untested path, log {log})")
        gvhmr.run_gvhmr(clip.clip_id, video, clip.start_s, clip.end_s, out_dir, log)
        method = "gvhmr"

    if not args.no_overlay:
        from fencing.pose.overlay import contact_sheet

        contact_sheet(
            video, result.times_s, result.image_xy, result.keypoints_world, out_dir / "overlay.png",
            title=f"{clip.clip_id}  hand={result.handedness}  valid={result.stats['valid_frac']:.0%}",
        )  # fmt: skip
    summary = pose_summary(clip, screen, result, method, args, time.time() - start)
    pose_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _short(summary: dict) -> str:
    q = summary["quality"]
    return (
        f"{summary['clip_id']}: {summary['pose_method']}, {q['frames']} frames, valid {q['valid_frac']:.0%}, "
        f"vis {q['median_visibility']:.2f}, hand {summary['handedness_guess']}, "
        f"root path {q['root_path_m']:.2f} m, pitch {summary['camera']['pitch_down_deg']:.0f} deg"
    )


def collect_clips(args, parser: argparse.ArgumentParser) -> list[tuple[ClipSpec, dict]]:
    tokens = list(args.videos)
    if args.all:
        tokens += screened_videos()
    if not tokens:
        parser.error(f"name videos / clip_ids or pass --all (screened videos live in {paths.POSE}/<yt_id>/screen.json)")
    jobs: list[tuple[ClipSpec, dict]] = []
    for token in dict.fromkeys(tokens):
        yt_id, start, end = parse_clip_id(token)
        if args.start is not None or args.end is not None:
            start, end = args.start, args.end
        try:
            screen = load_screen(yt_id)
            clips = plan_clips(screen, args.max_seconds, args.min_seconds, start, end)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            eprint(f"[pose] {token}: SKIPPED: {exc}")
            continue
        if not clips:
            eprint(f"[pose] {yt_id}: no usable range of at least {args.min_seconds}s")
        if args.max_clips:
            clips = clips[: args.max_clips]
        jobs += [(c, screen) for c in clips]
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing pose", description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="*", help="YouTube ids or clip_ids (<yt_id>_<start_ms>-<end_ms>)")
    parser.add_argument("--all", action="store_true", help="every screened video")
    parser.add_argument("--start", type=float, help="override: start of the range, source seconds")
    parser.add_argument("--end", type=float, help="override: end of the range, source seconds")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS, help="split ranges into clips of at most this length (default %(default)s)")
    parser.add_argument("--min-seconds", type=float, default=MIN_CLIP_SECONDS, help="skip ranges shorter than this (default %(default)s)")
    parser.add_argument("--max-clips", type=int, default=0, help="at most this many clips per video (0 = all)")
    parser.add_argument("--height", type=float, default=DEFAULT_HUMAN_HEIGHT_M, help="fencer height in metres (default %(default)s)")
    parser.add_argument("--hfov", type=float, help="camera horizontal field of view, deg (default: focal = image diagonal)")
    parser.add_argument("--pitch-deg", type=float, help="camera pitch down, deg (default: estimated from the torso)")
    parser.add_argument("--fallback", action="store_true", help="never use GVHMR, even if the SMPL-X models exist")
    parser.add_argument("--no-overlay", action="store_true", help="skip the overlay.png contact sheet")
    parser.add_argument("--force", action="store_true", help="redo clips that already have a pose.json")
    parser.add_argument("--reprocess", action="store_true", help="redo only the post-processing from a cached mediapipe_raw.npz")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths.ensure_dirs()
    jobs = collect_clips(args, parser)

    use_gvhmr = not args.fallback and gvhmr.gvhmr_available()
    if not use_gvhmr and not args.fallback:
        missing = ", ".join(str(p) for p in gvhmr.missing_body_models()) or str(gvhmr.CHECKPOINTS / "gvhmr")
        eprint(f"[pose] GVHMR unavailable (missing {missing}); using the licence-free {FALLBACK_METHOD} path")

    failures = 0
    for clip, screen in jobs:
        start = time.time()
        try:
            summary = pose_one(clip, screen, args, use_gvhmr)
        except (FileNotFoundError, WorkerError, ValueError, KeyError, RuntimeError, OSError) as exc:
            failures += 1
            eprint(f"[pose] {clip.clip_id}: FAILED: {exc}")
            continue
        print(f"{_short(summary)}  [{time.time() - start:.0f}s]", flush=True)
    if jobs:
        eprint(f"[pose] {len(jobs) - failures}/{len(jobs)} clips ok")
    return 1 if failures else 0
