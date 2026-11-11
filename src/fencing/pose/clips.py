"""Clip identifiers and how a screened video's usable ranges become pose clips.

Pure Python; used by `fencing pose` and `fencing retarget`.

A clip_id is `<yt_id>_<start_ms>-<end_ms>` (milliseconds in the source video, see
docs/pipeline.md). Long usable ranges are split into equal chunks of at most `max_seconds`,
so one bad stretch costs one chunk, and the retarget/cleanup stages work on short files.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DEFAULT_MAX_CLIP_SECONDS = 20.0
MIN_CLIP_SECONDS = 1.5

_CLIP_ID_RE = re.compile(r"^(?P<yt>.+)_(?P<start>\d+)-(?P<end>\d+)$")


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    yt_id: str
    start_s: float
    end_s: float
    track_id: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def make_clip_id(yt_id: str, start_s: float, end_s: float) -> str:
    if not 0.0 <= start_s < end_s:
        raise ValueError(f"bad clip range [{start_s}, {end_s}] s for {yt_id}")
    return f"{yt_id}_{int(round(start_s * 1000))}-{int(round(end_s * 1000))}"


def parse_clip_id(clip_id: str) -> tuple[str, float | None, float | None]:
    """(yt_id, start_s, end_s); start/end are None for a whole-video clip_id."""
    match = _CLIP_ID_RE.match(clip_id)
    if not match:
        return clip_id, None, None
    start, end = int(match["start"]) / 1000.0, int(match["end"]) / 1000.0
    if end <= start:
        raise ValueError(f"clip_id {clip_id!r} ends before it starts")
    return match["yt"], start, end


def split_range(
    start_s: float, end_s: float, max_seconds: float = DEFAULT_MAX_CLIP_SECONDS, min_seconds: float = MIN_CLIP_SECONDS
) -> list[tuple[float, float]]:
    """Equal chunks of at most max_seconds covering [start_s, end_s]; [] if shorter than min_seconds."""
    if max_seconds <= 0:
        raise ValueError(f"max_seconds must be positive, got {max_seconds}")
    duration = end_s - start_s
    if duration < min_seconds:
        return []
    n = max(1, math.ceil(duration / max_seconds - 1e-9))
    step = duration / n
    # Round to whole milliseconds so the clip_id reproduces the range exactly.
    edges = [round(start_s + i * step, 3) for i in range(n)] + [round(end_s, 3)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _track_for_range(screen: dict, start_s: float, end_s: float) -> int:
    """Track id of the screen range overlapping [start_s, end_s] most, else the chosen track."""
    best, best_overlap = None, 0.0
    for r in screen.get("ranges", []):
        overlap = min(end_s, r["end_s"]) - max(start_s, r["start_s"])
        if overlap > best_overlap:
            best, best_overlap = r.get("track_id"), overlap
    track = best if best is not None else screen.get("chosen_track_id")
    if track is None:
        raise ValueError(f"screen report for {screen.get('yt_id')} has no fencer track")
    return int(track)


def plan_clips(
    screen: dict,
    max_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    min_seconds: float = MIN_CLIP_SECONDS,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[ClipSpec]:
    """Clips for one screened video: its usable ranges, or one explicit [start_s, end_s] range."""
    yt_id = screen["yt_id"]
    if start_s is not None or end_s is not None:
        start = 0.0 if start_s is None else float(start_s)
        end = float(screen.get("duration_s", 0.0)) if end_s is None else float(end_s)
        if end <= start:
            raise ValueError(f"--end ({end}) must be after --start ({start})")
        ranges = [(start, end)]
    else:
        ranges = [(float(s), float(e)) for s, e in screen.get("usable_ranges", [])]
    clips = []
    for range_start, range_end in ranges:
        for s, e in split_range(range_start, range_end, max_seconds, min_seconds):
            clips.append(ClipSpec(make_clip_id(yt_id, s, e), yt_id, s, e, _track_for_range(screen, s, e)))
    return clips
