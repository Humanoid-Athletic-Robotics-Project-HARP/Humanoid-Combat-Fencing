"""Rank the screened time ranges of downloaded videos into a per-motion shortlist.

Joins two things: the scrape manifest (what the video is about: motion guess from its title)
and `fencing screen`'s pose/<yt_id>/screen.json (what is actually on screen: people, view,
full-body visibility, camera motion, usable time ranges). Pure functions apart from
`load_screen_reports`; the CLI lives in fencing.scrape.cli (`scrape shortlist`).

The ideal range: one fencer, side-on, large in frame, whole body in frame for all of it, static
camera, and long enough to hold several repetitions. A range's `rank` (0..1) multiplies those factors, with
the manifest score as a small tie-break, so a lesson with a coach in frame or a front-on shot
still makes the list, just lower.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from fencing.scrape.scoring import MOTIONS

VIEW_WEIGHT = {"side": 1.0, "oblique": 0.75, "unknown": 0.5, "front": 0.35}
# Confident person detections in frame (screen's people_median). 2 is usually coach + pupil;
# 0 means the detector lost everyone; a full club hall makes overlaps likely.
PEOPLE_WEIGHT = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.35, 5: 0.35}
CROWD_WEIGHT = 0.2
# A fencer this small in frame (median box area / image area) leaves pose estimation few pixels;
# smaller ones are scaled down by sqrt(area / FULL_SIZE_AREA). Ranges whose track is not in
# the report's top-10 track list get UNKNOWN_SIZE_WEIGHT.
FULL_SIZE_AREA = 0.04
UNKNOWN_SIZE_WEIGHT = 0.8
FULL_LENGTH_SECONDS = 8.0  # ranges at least this long get full marks for length
MIN_SHORTLIST_SECONDS = 2.0
MAX_RANGES_PER_VIDEO = 3  # per motion list, so one long drill video cannot fill the top 10
SCORE_CAP = 16.0  # manifest scores above this all count the same

# Screen views -> the video log's View vocabulary (side / front / angled).
LOG_VIEW = {"side": "side", "front": "front", "oblique": "angled"}
VIEW_CONSENSUS = 0.7  # share of usable seconds that must agree before the log gets a View


def load_screen_reports(pose_dir: Path, yt_ids: list[str]) -> dict[str, dict]:
    """screen.json of every id that has one; unreadable files are skipped (being rewritten)."""
    reports = {}
    for yt_id in yt_ids:
        path = pose_dir / yt_id / "screen.json"
        if not path.exists():
            continue
        try:
            reports[yt_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def people_weight(people: int) -> float:
    return PEOPLE_WEIGHT.get(people, CROWD_WEIGHT if people > 5 else 0.0)


def size_weight(area_frac: float | None) -> float:
    if area_frac is None:
        return UNKNOWN_SIZE_WEIGHT
    return min(1.0, (max(area_frac, 0.0) / FULL_SIZE_AREA) ** 0.5)


def range_rank(rng: dict, manifest_score: float = 0.0, area_frac: float | None = None) -> float:
    duration = rng["end_s"] - rng["start_s"]
    length = min(duration / FULL_LENGTH_SECONDS, 1.0)
    quality = (VIEW_WEIGHT.get(rng.get("view") or "unknown", 0.5) * people_weight(int(rng.get("people_median") or 0))
               * float(rng.get("full_body_frac") or 0.0) * (0.5 + 0.5 * length) * size_weight(area_frac))
    tie_break = 0.85 + 0.15 * min(max(manifest_score, 0.0), SCORE_CAP) / SCORE_CAP
    return round(quality * tie_break, 4)


def usable_entries(record: dict, report: dict) -> list[dict]:
    """Shortlist entries for every usable range of one screened video, best first."""
    entries = []
    area_by_track = {t["track_id"]: t.get("area_frac") for t in report.get("tracks") or []}
    for rng in report.get("ranges") or []:
        duration = rng["end_s"] - rng["start_s"]
        if not rng.get("usable") or duration < MIN_SHORTLIST_SECONDS:
            continue
        entries.append({
            "yt_id": record["id"],
            "start": rng["start_s"],
            "end": rng["end_s"],
            "duration": round(duration, 3),
            "motion_guess": record.get("motion_guess") or "footwork",
            "motion_tags": record.get("motion_tags") or [],
            "people": int(rng.get("people_median") or 0),
            "view": rng.get("view") or "unknown",
            "view_ratio": rng.get("view_ratio"),
            "full_body_frac": rng.get("full_body_frac"),
            "static_camera": rng.get("camera") == "static",
            "camera_motion_width_per_s": rng.get("camera_motion_width_per_s"),
            "track_id": rng.get("track_id"),
            "person_area_frac": area_by_track.get(rng.get("track_id")),
            "rank": range_rank(rng, float(record.get("score") or 0.0), area_by_track.get(rng.get("track_id"))),
            "title": record.get("title") or "",
            "channel": record.get("channel") or "",
            "url": record.get("url") or f"https://www.youtube.com/watch?v={record['id']}",
        })
    return sorted(entries, key=lambda e: -e["rank"])


def build_shortlist(records: dict[str, dict], reports: dict[str, dict]) -> dict:
    """The shortlist.json document: usable ranges per motion guess, best first."""
    usable: list[dict] = []
    screened = no_usable = 0
    for yt_id, report in reports.items():
        record = records.get(yt_id)
        if record is None or record.get("status") != "downloaded":
            continue  # excluded/broken videos or files the scraper never saw
        screened += 1
        entries = usable_entries(record, report)
        no_usable += not entries
        usable += entries
    usable.sort(key=lambda e: -e["rank"])

    by_motion: dict[str, list[dict]] = defaultdict(list)
    per_video: dict[tuple[str, str], int] = defaultdict(int)
    for entry in usable:
        key = (entry["motion_guess"], entry["yt_id"])
        if per_video[key] >= MAX_RANGES_PER_VIDEO:
            continue
        per_video[key] += 1
        by_motion[entry["motion_guess"]].append(entry)

    views: dict[str, int] = defaultdict(int)
    for entry in usable:
        views[entry["view"]] += 1
    order = [m for m in MOTIONS if m in by_motion] + sorted(m for m in by_motion if m not in MOTIONS)
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "screened_videos": screened,
        "videos_without_usable_range": no_usable,
        "usable_ranges": len(usable),
        "usable_seconds": round(sum(e["duration"] for e in usable), 1),
        "usable_ranges_by_view": dict(views),
        "max_ranges_per_video_per_motion": MAX_RANGES_PER_VIDEO,
        "by_motion": {m: by_motion[m] for m in order},
        "ranges": usable,
    }


def _mmss(seconds: float, round_up: bool = False) -> str:
    whole = int(seconds) + (1 if round_up and seconds % 1 else 0)
    return f"{whole // 60}:{whole % 60:02d}"


def consensus_view(entries: list[dict]) -> str | None:
    """The log's View for a video's usable ranges, or None unless VIEW_CONSENSUS of their
    seconds agree on one view the log has a word for."""
    seconds_by_view: dict[str, float] = defaultdict(float)
    for entry in entries:
        seconds_by_view[entry["view"]] += entry["end"] - entry["start"]
    if not seconds_by_view:
        return None
    view, seconds = max(seconds_by_view.items(), key=lambda kv: kv[1])
    if view in LOG_VIEW and seconds >= VIEW_CONSENSUS * sum(seconds_by_view.values()):
        return LOG_VIEW[view]
    return None


def no_range_note(report: dict) -> str:
    """Notes text for a screened video without a usable range, with the screen's reasons."""
    reasons = {reason.split(" (")[0] for rng in report.get("ranges") or [] for reason in rng.get("reasons") or []}
    if not report.get("ranges"):
        # No stretch of >= 1.5 s with the whole body in frame, so no per-range reasons: fall back
        # on the video-level figures for the track the screen picked.
        if report.get("chosen_track_id") is None:
            reasons.add("no fencer tracked")
        else:
            reasons.add(f"full body visible in only {float(report.get('full_body_frac') or 0.0):.0%} of frames")
        if report.get("camera") == "moving":
            reasons.add("camera moving")
    return "screen: no usable range" + (f" ({'; '.join(sorted(reasons))})" if reasons else "")


def log_fill(record: dict, report: dict) -> dict[str, str]:
    """Video-log cells the screen answers confidently: best range, its people count, the view.

    Only offered for empty cells (see video_log.sync_video_log), so hand edits always win.
    Handedness is filled only if the screen report carries one ('R' or 'L').
    """
    fill: dict[str, str] = {}
    entries = usable_entries(record, report)
    if not entries:
        fill["Notes"] = no_range_note(report)
        return fill
    best = entries[0]
    fill.update(Start=_mmss(best["start"]), End=_mmss(best["end"], round_up=True), Fencers=str(best["people"]))
    view = consensus_view(entries)
    if view:
        fill["View"] = view
    if report.get("handedness") in {"R", "L"}:
        fill["Hand"] = report["handedness"]
    return fill
