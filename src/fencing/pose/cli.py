"""`fencing pose`: screened video ranges -> 3D human keypoints per clip (Phase 1.4c).

    fencing pose IpM7citb5bs                  # every usable range of a screened video
    fencing pose --all                        # every video with a pose/<yt_id>/screen.json
    fencing pose IpM7citb5bs --start 12 --end 19.5
    fencing pose IpM7citb5bs_12000-19500      # one clip by clip_id (re-run with --force)
    fencing pose --all --reprocess            # new post-processing, cached MediaPipe output
    fencing pose --all --max-seconds 10 --max-clips 2 --height 1.80
    fencing pose --all --existing --gvhmr-preprocess   # GVHMR phase 1 on every clip made so far

Usable ranges from screen.json are cut into chunks of at most --max-seconds; each chunk is a
clip `<yt_id>_<start_ms>-<end_ms>` with outputs in pose/<clip_id>/. A bare yt_id also takes
the clips of that video that already have a pose/<clip_id>/ folder (made with other
--max-seconds or --start/--end); --existing takes only those.

    mediapipe_raw.npz     MediaPipe Pose Landmarker on the fencer track (pose venv)
    keypoints_world.npz   keypoints (T,33,3) z-up metres at 30 fps, times_s, valid, visibility
    pose.json             method, fps, height, handedness_guess, camera, quality stats
    overlay.png           contact sheet: landmarks on the video + the lifted 3D skeleton
    hmr4d_results.pt      only with GVHMR (needs the licence-gated SMPL-X file)
    gvhmr/                GVHMR work folder, fencer_check.json + fencer_check.png

Method: GVHMR if HCF_DATA/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz exists
(see fencing.pose.gvhmr), else the licence-free `fallback_mediapipe` path. The MediaPipe
keypoints are produced in both cases (handedness, overlay, the GVHMR fencer check, and a
retarget fallback). Clips whose pose.json already has the method this run would use are
skipped unless --force, so a fallback clip is upgraded to GVHMR by a plain re-run once the
model exists. --gvhmr-preprocess runs only GVHMR's body-model-free preprocessing (cached,
pose.json untouched). One failing clip never stops the batch. Logs: logs/pose/<clip_id>.log.
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
from fencing.pose.clips import (
    DEFAULT_MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    ClipSpec,
    parse_clip_id,
    plan_clips,
)
from fencing.pose.postprocess import OUTPUT_FPS, PoseResult, process_mediapipe
from fencing.pose.root_trajectory import DEFAULT_HUMAN_HEIGHT_M
from fencing.pose.venvs import WorkerError, eprint, run_in_venv

POSE_VERSION = 2  # 2: MediaPipe world landmarks rotated from the hip-ray frame to camera axes
MEDIAPIPE_MODEL = paths.DATA / "models" / "pose_landmarker_heavy.task"
FALLBACK_METHOD = "fallback_mediapipe"
GVHMR_METHOD = "gvhmr"


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


def pose_summary(
    clip: ClipSpec, screen: dict, result: PoseResult, method: str, args, seconds: float, gvhmr_info: dict | None = None
) -> dict:
    notes = []
    if method == FALLBACK_METHOD:
        notes.append("monocular MediaPipe + pinhole root solve; no body model, wrist twist approximate")
    else:
        notes.append("GVHMR (SMPL-X) on the screen's fencer track; handedness, body_scale and quality are from MediaPipe on the same track")
    if gvhmr_info and gvhmr_info.get("note"):
        notes.append(gvhmr_info["note"])
    if result.handedness is None:
        notes.append("handedness undecided (cues weak or disagree)")
    summary = {
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
    if gvhmr_info:
        summary["gvhmr"] = gvhmr_info
    return summary


def pose_is_current(summary: dict, use_gvhmr: bool) -> bool:
    """An existing pose.json is reused unless this run would upgrade it to GVHMR (never the
    other way: a GVHMR clip stays GVHMR when the model is missing later)."""
    return summary.get("pose_method") == GVHMR_METHOD or not use_gvhmr


def run_gvhmr_branch(clip: ClipSpec, video: Path, log: Path, args) -> tuple[str, dict]:
    """(method, pose.json `gvhmr` block). Keeps the fallback when the fencer check fails
    (GVHMR's keypoints on someone other than MediaPipe's fencer, or no detected person)."""
    eprint(f"[pose] {clip.clip_id}: GVHMR (log {log})")
    check = gvhmr.run_clip(clip, video, log, predict=True, render=args.gvhmr_render, redo_results=args.force)
    info = {"fps": gvhmr.GVHMR_FPS, "results": "hmr4d_results.pt", "fencer_check": check}
    reason = gvhmr.fencer_check_reason(check)
    if reason:
        info["note"] = f"GVHMR not used: {reason}; see gvhmr/fencer_check.png"
        eprint(f"[pose] {clip.clip_id}: WARNING: {info['note']}")
        return FALLBACK_METHOD, info
    return GVHMR_METHOD, info


def pose_one(clip: ClipSpec, screen: dict, args, use_gvhmr: bool) -> dict:
    out_dir = paths.POSE / clip.clip_id
    pose_json = out_dir / "pose.json"
    log = paths.LOGS / "pose" / f"{clip.clip_id}.log"
    if pose_json.exists() and not (args.force or args.reprocess):
        existing = json.loads(pose_json.read_text())
        if pose_is_current(existing, use_gvhmr):
            eprint(f"[pose] {clip.clip_id}: done already ({pose_json})")
            return existing
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

    method, gvhmr_info = FALLBACK_METHOD, None
    if use_gvhmr:
        method, gvhmr_info = run_gvhmr_branch(clip, video, log, args)

    if not args.no_overlay:
        from fencing.pose.overlay import contact_sheet

        contact_sheet(
            video, result.times_s, result.image_xy, result.keypoints_world, out_dir / "overlay.png",
            title=f"{clip.clip_id}  hand={result.handedness}  valid={result.stats['valid_frac']:.0%}",
        )  # fmt: skip
    summary = pose_summary(clip, screen, result, method, args, time.time() - start, gvhmr_info)
    pose_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def gvhmr_preprocess_one(clip: ClipSpec, screen: dict, args) -> str:
    """GVHMR phase 1 for one clip (no body model); returns a one-line status."""
    log = paths.LOGS / "pose" / f"{clip.clip_id}.log"
    if gvhmr.preprocess_done(clip.clip_id) and not args.force:
        eprint(f"[pose] {clip.clip_id}: GVHMR preprocessing cached ({gvhmr.output_dir(clip.clip_id)})")
        check = gvhmr.read_fencer_check(clip.clip_id)
    else:
        video = Path(screen.get("video") or paths.VIDEOS_RAW / f"{clip.yt_id}.mp4")
        eprint(f"[pose] {clip.clip_id}: GVHMR preprocessing on track {clip.track_id} ({clip.duration_s:.1f}s, log {log})")
        check = gvhmr.run_clip(clip, video, log, predict=False, redo_preprocess=args.force)
    boxes, kp = check.get("boxes", {}), check.get("keypoints", {})
    reason = gvhmr.fencer_check_reason(check)
    flag = f"  WARNING: fencer check failed, GVHMR will not be used: {reason}" if reason else ""
    return (
        f"{clip.clip_id}: GVHMR preprocessed, {boxes.get('frames')} frames, YOLO boxes {boxes.get('yolo_frac', 0):.0%}, "
        f"same person {kp.get('same_person')} (centroid {kp.get('centroid_dist_median')}){flag}"
    )


def _short(summary: dict) -> str:
    q = summary["quality"]
    return (
        f"{summary['clip_id']}: {summary['pose_method']}, {q['frames']} frames, valid {q['valid_frac']:.0%}, "
        f"vis {q['median_visibility']:.2f}, hand {summary['handedness_guess']}, "
        f"root path {q['root_path_m']:.2f} m, pitch {summary['camera']['pitch_down_deg']:.0f} deg"
    )


def with_recorded_track(clip: ClipSpec) -> ClipSpec:
    """The clip with the track its existing pose.json was made from: MediaPipe's landmarks and
    GVHMR's boxes must be the same person even if the screen ranges changed since."""
    pose_json = paths.POSE / clip.clip_id / "pose.json"
    if not pose_json.exists():
        return clip
    track = json.loads(pose_json.read_text()).get("track_id")
    if track is None or int(track) == clip.track_id:
        return clip
    return ClipSpec(clip.clip_id, clip.yt_id, clip.start_s, clip.end_s, int(track))


def existing_clips(yt_id: str, screen: dict) -> list[ClipSpec]:
    """Clips of one video that already have a pose/<clip_id>/ folder with MediaPipe output."""
    clips = []
    for folder in sorted(paths.POSE.glob(f"{yt_id}_*")):
        clip_yt, start, end = parse_clip_id(folder.name)
        if clip_yt != yt_id or start is None:
            continue
        if not ((folder / "pose.json").exists() or (folder / "mediapipe_raw.npz").exists()):
            continue
        clips += plan_clips(screen, max_seconds=end - start + 1.0, min_seconds=0.0, start_s=start, end_s=end)
    return clips


def collect_clips(args, parser: argparse.ArgumentParser) -> list[tuple[ClipSpec, dict]]:
    tokens = list(args.videos)
    if args.all:
        tokens += screened_videos()
    if not tokens:
        parser.error(f"name videos / clip_ids or pass --all (screened videos live in {paths.POSE}/<yt_id>/screen.json)")
    jobs: list[tuple[ClipSpec, dict]] = []
    for token in dict.fromkeys(tokens):
        yt_id, start, end = parse_clip_id(token)
        explicit = start is not None or args.start is not None or args.end is not None
        if args.start is not None or args.end is not None:
            start, end = args.start, args.end
        try:
            screen = load_screen(yt_id)
            planned = [] if args.existing and not explicit else plan_clips(screen, args.max_seconds, args.min_seconds, start, end)
            if not explicit:
                known = existing_clips(yt_id, screen)
                planned_ids = {c.clip_id for c in known}
                clips = known + [c for c in planned if c.clip_id not in planned_ids]
            else:
                clips = planned
        except (FileNotFoundError, ValueError, KeyError) as exc:
            eprint(f"[pose] {token}: SKIPPED: {exc}")
            continue
        if not clips and not args.existing:
            eprint(f"[pose] {yt_id}: no usable range of at least {args.min_seconds}s")
        if args.max_clips:
            clips = clips[: args.max_clips]
        jobs += [(with_recorded_track(c), screen) for c in clips]
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
    parser.add_argument("--existing", action="store_true", help="for bare yt_ids: only clips that already have a pose/<clip_id>/ folder")
    parser.add_argument("--fallback", action="store_true", help="never use GVHMR, even if the SMPL-X models exist")
    parser.add_argument("--gvhmr-preprocess", action="store_true", help="only GVHMR's body-model-free preprocessing (phase 1), cached; pose.json untouched")
    parser.add_argument("--gvhmr-render", action="store_true", help="also render GVHMR's mesh videos (needs SMPL_NEUTRAL.pkl; QA only)")
    parser.add_argument("--no-overlay", action="store_true", help="skip the overlay.png contact sheet")
    parser.add_argument("--force", action="store_true", help="redo clips that already have a pose.json (with --gvhmr-preprocess: redo the preprocessing)")
    parser.add_argument("--reprocess", action="store_true", help="redo only the post-processing from a cached mediapipe_raw.npz")
    return parser


def gvhmr_preprocess_ready() -> list[str]:
    """Problems that stop GVHMR phase 1, empty when it can run."""
    problems = [f"missing {p}" for p in gvhmr.missing_preprocess_checkpoints()]
    if not paths.venv_python("gvhmr").exists():
        problems.append(f"venv missing ({paths.venv_python('gvhmr')}; run scripts/setup_linux_gpu.sh gvhmr)")
    return problems


def run_gvhmr_preprocess(jobs: list[tuple[ClipSpec, dict]], args) -> int:
    problems = gvhmr_preprocess_ready()
    if problems:
        eprint(f"[pose] GVHMR preprocessing cannot run: {'; '.join(problems)}")
        return 1
    failures = 0
    for i, (clip, screen) in enumerate(jobs, 1):
        start = time.time()
        try:
            line = gvhmr_preprocess_one(clip, screen, args)
        except (FileNotFoundError, WorkerError, ValueError, KeyError, RuntimeError, OSError) as exc:
            failures += 1
            eprint(f"[pose] {clip.clip_id}: FAILED: {exc}")
            continue
        print(f"[{i}/{len(jobs)}] {line}  [{time.time() - start:.0f}s]", flush=True)
    if jobs:
        eprint(f"[pose] GVHMR preprocessing: {len(jobs) - failures}/{len(jobs)} clips ok")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths.ensure_dirs()
    jobs = collect_clips(args, parser)
    if args.gvhmr_preprocess:
        return run_gvhmr_preprocess(jobs, args)

    use_gvhmr = not args.fallback and gvhmr.gvhmr_available()
    if not use_gvhmr and not args.fallback:
        missing = ", ".join(str(p) for p in (gvhmr.SMPLX_MODEL, gvhmr.GVHMR_CHECKPOINT) if not p.exists())
        eprint(f"[pose] GVHMR unavailable (missing {missing}); using the licence-free {FALLBACK_METHOD} path")
    if use_gvhmr and args.gvhmr_render and not gvhmr.SMPL_MODEL.exists():
        eprint(f"[pose] --gvhmr-render needs {gvhmr.SMPL_MODEL}; the render step will only print a warning")

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
