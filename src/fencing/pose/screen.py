"""`fencing screen`: detect and track people, pick the fencer, score each video (Phase 1.4b).

    fencing screen --all                 # every videos/raw/*.mp4 without a screen.json yet
    fencing screen IpM7citb5bs other_id  # by YouTube id (or a path to an .mp4)
    fencing screen --all --force         # redo

Writes pose/<yt_id>/screen.json (scores + usable time ranges), pose/<yt_id>/track.npz (all
tracks, for `fencing pose`) and keeps pose/<yt_id>/detections.npz (raw detector output, so a
re-score with new thresholds needs no GPU: `--rescore`).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fencing import paths
from fencing.pose.screening import Detections, screen_video, tracks_to_arrays
from fencing.pose.venvs import WorkerError, eprint, run_in_venv

SCREEN_VERSION = 1


def resolve_video(token: str) -> tuple[str, Path]:
    """A YouTube id or an .mp4 path -> (yt_id, path)."""
    as_path = Path(token).expanduser()
    if as_path.suffix == ".mp4" and as_path.exists():
        return as_path.stem, as_path
    candidate = paths.VIDEOS_RAW / f"{token}.mp4"
    if not candidate.exists():
        raise FileNotFoundError(f"no video for '{token}': expected {candidate} or an existing .mp4 path")
    return token, candidate


class ScreenBusy(RuntimeError):
    """Another `fencing screen` process is working on this video right now."""


@contextmanager
def video_lock(out_dir: Path) -> Iterator[None]:
    """Exclusive per-video lock, so two screen runs (or agents) never write the same files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / ".screen.lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ScreenBusy(f"{out_dir.name} is being screened by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def screen_one(yt_id: str, video: Path, force: bool, rescore: bool, device: str) -> dict | None:
    with video_lock(paths.POSE / yt_id):
        return _screen_one_locked(yt_id, video, force, rescore, device)


def _screen_one_locked(yt_id: str, video: Path, force: bool, rescore: bool, device: str) -> dict | None:
    out_dir = paths.POSE / yt_id
    screen_json = out_dir / "screen.json"
    detections_npz = out_dir / "detections.npz"
    log = paths.LOGS / "screen" / f"{yt_id}.log"
    if rescore and not detections_npz.exists():
        eprint(f"[screen] {yt_id}: no cached detections, skipping re-score (run without --rescore)")
        return None
    if screen_json.exists() and not (force or rescore):
        eprint(f"[screen] {yt_id}: done already ({screen_json})")
        return json.loads(screen_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    if force and not rescore and detections_npz.exists():
        detections_npz.unlink()
    detect_s = 0.0
    if not detections_npz.exists():
        eprint(f"[screen] {yt_id}: detecting people (log {log})")
        detect_s = run_in_venv(
            "pose",
            ["-m", "fencing.pose.detect_worker", "--video", str(video), "--out", str(detections_npz), "--device", device],
            log,
        )
    with np.load(detections_npz) as data:
        det = Detections.from_npz(data)
    report, tracks = screen_video(det)
    report = {
        "yt_id": yt_id,
        "video": str(video),
        "screen_version": SCREEN_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detect_seconds": round(detect_s, 1),
        **report,
    }
    np.savez_compressed(out_dir / "track.npz", frame_times=det.frame_times, **tracks_to_arrays(tracks))
    write_json_atomic(screen_json, report)
    return report


def summarize(yt_id: str, report: dict) -> str:
    usable = report["usable_ranges"]
    total = sum(e - s for s, e in usable)
    return (
        f"{yt_id}: people~{report['people_count_median']} (max {report['people_count_max']}), "
        f"fencer track {report['chosen_track_id']} full-body {report['full_body_frac']:.0%}, "
        f"camera {report['camera']}, view {report['view']}, "
        f"{len(usable)} usable range(s) / {total:.1f}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fencing screen", description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="*", help="YouTube ids or .mp4 paths")
    parser.add_argument("--all", action="store_true", help=f"every .mp4 in {paths.VIDEOS_RAW}")
    parser.add_argument("--force", action="store_true", help="re-run detection and scoring")
    parser.add_argument("--rescore", action="store_true", help="re-score from cached detections.npz")
    parser.add_argument("--device", default="cuda", help="torch device for the detector (default cuda)")
    args = parser.parse_args(argv)

    tokens = list(args.videos)
    if args.all:
        tokens += sorted(p.stem for p in paths.VIDEOS_RAW.glob("*.mp4"))
    if not tokens:
        parser.error(f"name videos or pass --all (looked in {paths.VIDEOS_RAW})")
    paths.ensure_dirs()

    failures = 0
    for token in dict.fromkeys(tokens):
        start = time.time()
        try:
            yt_id, video = resolve_video(token)
            report = screen_one(yt_id, video, args.force, args.rescore, args.device)
        except ScreenBusy as exc:
            eprint(f"[screen] {token}: skipped: {exc}")
            continue
        except (FileNotFoundError, WorkerError, ValueError, KeyError) as exc:
            failures += 1
            eprint(f"[screen] {token}: FAILED: {exc}")
            continue
        if report is not None:
            print(f"{summarize(yt_id, report)}  [{time.time() - start:.0f}s]", flush=True)
    return 1 if failures else 0
