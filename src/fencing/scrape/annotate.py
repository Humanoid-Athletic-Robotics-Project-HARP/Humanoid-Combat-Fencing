"""Tag video-log rows from what the pipeline already wrote (`fencing scrape log --annotate`).

The sync (video_log.sync_video_log) only knows the manifest and the screen reports. Later
stages leave more behind, and this module turns it into hand-edited cells for rows that are
still empty, so nobody has to copy it over by hand:

- `videos/shortlist.json`: the best-ranked usable range of a video gives Start / End / Fencers
  (none if the detector saw nobody, people 0). View takes the same agreement rule as the sync
  (shortlist.consensus_view: 70% of the usable seconds on one view), mapped side -> side,
  front -> front, oblique -> angled; `unknown` or disagreeing ranges give no View.
- `pose/<yt_id>/screen.json`: a screened video with no usable range is `rejected`, with the
  screen's reasons as the note (video-level figures when it found no range at all).
- `pose/<clip_id>/pose.json` and `retarget/csv/<clip_id>.json`: `handedness_guess` per processed
  clip; a retarget sidecar means the video reached `retargeted`.
- `motions/*.json` (published motions): the source YouTube id, and for video-derived motions
  the source fencer's hand (top-level `hand`, the hand before any mirroring, or the provenance
  hand) when it came from the pose sidecar or a human config, never from detection. Bladework motions name their base clip too, but their
  provenance hand is the robot's keyframed saber arm, so it never counts as the fencer's.

Hand is filled only when every decided source agrees. Quality is never filled: A/B/C is a
human judgement. Pure functions apart from the two loaders; the CLI is in fencing.scrape.cli.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from fencing.scrape.shortlist import _mmss, consensus_view, no_range_note, usable_entries

SOURCE_ID = re.compile(r"youtube ([\w-]{11})")
CLIP_ID = re.compile(r"([\w-]{11})_\d+-\d+")  # <yt_id>_<start ms>-<end ms>; yt ids may hold "_"
# Where the cleaner's hand came from (motion/pipeline.py resolve_hand). "detected (confidence
# x)" is not evidence: detect_handedness answers R or L even at confidence 0.
TRUSTED_HAND_SOURCES = ("sidecar", "config")
# Provenance spelling of the same: "R (sidecar)", "L (sidecar) -> mirrored to R", "R (config)",
# and the hand-published "R (source hand per sidecar; mirrored if L)".
SOURCE_HAND = re.compile(r"^([RL]) \((?:sidecar|config|source hand per sidecar)\b")
VERSIONED = re.compile(r"^(.*)_v(\d+)$")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # being rewritten by another stage, or not a sidecar: the next run sees it


def _source_id(sidecar: dict) -> str | None:
    provenance = sidecar.get("provenance") or {}
    for text in (provenance.get("source_clip"), (sidecar.get("base") or {}).get("source")):
        match = SOURCE_ID.search(text or "")
        if match:
            return match.group(1)
    match = CLIP_ID.search(Path(str(sidecar.get("input") or "")).name)
    return match.group(1) if match else None


def _source_hand(sidecar: dict) -> str | None:
    if sidecar.get("hand") in {"R", "L"}:
        return sidecar["hand"] if sidecar.get("hand_source") in TRUSTED_HAND_SOURCES else None
    match = SOURCE_HAND.match((sidecar.get("provenance") or {}).get("hand") or "")
    return match.group(1) if match else None


def published_sources(motions_dir: Path) -> dict[str, dict]:
    """yt_id -> {"direct": motions cleaned from it, "base": bladework motions built on one of
    those, "hands": the source fencer's hand as the cleaned motions recorded it}."""
    sources: dict[str, dict] = defaultdict(lambda: {"direct": [], "base": [], "hands": set()})
    for path in sorted(motions_dir.glob("*.json")):
        sidecar = _read_json(path)
        if not sidecar:
            continue
        yt_id = _source_id(sidecar)
        if yt_id is None:
            continue  # synthetic motions have no video behind them
        name = Path(sidecar.get("file") or path.name).stem
        if sidecar.get("base"):
            sources[yt_id]["base"].append(name)
            continue
        sources[yt_id]["direct"].append(name)
        hand = _source_hand(sidecar)
        if hand:
            sources[yt_id]["hands"].add(hand)
    return dict(sources)


def clip_evidence(pose_dir: Path, retarget_dir: Path) -> dict[str, dict]:
    """yt_id -> {"hands": decided handedness guesses of its clips, "retargeted": (start_s, end_s)
    of every clip with a retarget sidecar}."""
    clips: dict[str, dict] = defaultdict(lambda: {"hands": set(), "retargeted": []})
    for path in sorted(pose_dir.glob("*/pose.json")) + sorted((retarget_dir / "csv").glob("*.json")):
        sidecar = _read_json(path)
        if not sidecar or not sidecar.get("yt_id"):
            continue
        entry = clips[sidecar["yt_id"]]
        if sidecar.get("handedness_guess") in {"R", "L"}:
            entry["hands"].add(sidecar["handedness_guess"])
        if path.parent.name == "csv":
            entry["retargeted"].append((sidecar.get("source_start_s"), sidecar.get("source_end_s")))
    return dict(clips)


def _versions(names: list[str]) -> str:
    """["lunge_v1", "lunge_v2", "lunge_v3", "retreat_v1"] -> "lunge v1-3, retreat v1"."""
    grouped: dict[str, list[int]] = defaultdict(list)
    plain = []
    for name in names:
        match = VERSIONED.match(name)
        if match:
            grouped[match.group(1)].append(int(match.group(2)))
        else:
            plain.append(name)
    parts = []
    for stem, numbers in sorted(grouped.items()):
        runs: list[list[int]] = []
        for n in sorted(set(numbers)):
            if runs and n == runs[-1][-1] + 1:
                runs[-1].append(n)
            else:
                runs.append([n])
        parts.append(f"{stem} " + ", ".join(f"v{r[0]}-{r[-1]}" if len(r) > 1 else f"v{r[0]}" for r in runs))
    return ", ".join(parts + sorted(plain))


def _published_note(source: dict) -> str:
    direct, base = source["direct"], source["base"]
    if not direct:
        return f"published as base: {_versions(base)}"
    return f"published: {_versions(direct)}" + (f"; as base: {_versions(base)}" if base else "")


def _best_ranges(shortlist: dict) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for entry in shortlist.get("ranges") or []:
        if entry["yt_id"] not in best or entry["rank"] > best[entry["yt_id"]]["rank"]:
            best[entry["yt_id"]] = entry
    return best


def video_annotations(records: dict[str, dict], shortlist: dict, reports: dict[str, dict],
                      sources: dict[str, dict], clips: dict[str, dict]) -> dict[str, dict]:
    """yt_id -> {"fill": {column: value} for empty cells, "status": an upgrade for the default
    `new`, or None}. Videos nothing is known about are left out."""
    best = _best_ranges(shortlist)
    out = {}
    for yt_id in sorted(set(records) | set(best) | set(reports) | set(sources) | set(clips)):
        fill: dict[str, str] = {}
        status = None
        if yt_id in best:
            entry = best[yt_id]
            fill.update(Start=_mmss(entry["start"]), End=_mmss(entry["end"], round_up=True))
            view = consensus_view([e for e in shortlist.get("ranges") or [] if e["yt_id"] == yt_id])
            if view:
                fill["View"] = view
            if entry.get("people"):
                fill["Fencers"] = str(entry["people"])

        source = sources.get(yt_id) or {"direct": [], "base": [], "hands": set()}
        clip = clips.get(yt_id) or {"hands": set(), "retargeted": []}
        hands = source["hands"] | clip["hands"]
        if len(hands) == 1:
            fill["Hand"] = next(iter(hands))

        if source["direct"] or source["base"]:
            status, fill["Notes"] = "retargeted", _published_note(source)
        elif clip["retargeted"]:
            spans = ", ".join(f"{_mmss(s)}-{_mmss(e, round_up=True)}" for s, e in sorted(clip["retargeted"]))
            status, fill["Notes"] = "retargeted", f"retargeted, not published: {spans}"
        elif yt_id in reports and yt_id in records and not usable_entries(records[yt_id], reports[yt_id]):
            status, fill["Notes"] = "rejected", no_range_note(reports[yt_id])

        if fill or status:
            out[yt_id] = {"fill": fill, "status": status}
    return out
