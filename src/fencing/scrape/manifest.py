"""The scrape manifest: HCF_DATA/videos/manifest.jsonl, one JSON record per video id.

Status lifecycle:
    candidate -> downloaded            file on disk, verified with OpenCV
              -> failed                retried by later runs until it has MAX_ATTEMPTS attempts
              -> unavailable           private, removed, age-gated...: never retried
    downloaded -> excluded             on disk but unusable (broken, vertical, or the scoring
                                       rules now reject its metadata); kept for the record
    rejected                           filtered out at search time from metadata alone
The file is small (a few thousand lines at most), so it is rewritten whole on every
save via a temp file + os.replace, which never leaves a half-written manifest behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from fencing.scrape.scoring import ScrapeFilter, evaluate

# Fields copied from a yt-dlp flat search entry into a new record.
SEARCH_FIELDS = ("id", "url", "title", "channel", "channel_url", "duration", "view_count", "live_status")
DESCRIPTION_CHARS = 300  # snippet kept so records can be re-scored without hitting the network


def load_manifest(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"corrupt manifest line {path}:{line_number}: {error}") from error
            records[record["id"]] = record
    return records


def save_manifest(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records.values():
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def upsert_hit(records: dict[str, dict], entry: dict, query: str, query_motion: str,
               flt: ScrapeFilter) -> dict:
    """Add or refresh one search hit; dedupes by id and re-scores undecided records."""
    record = records.get(entry["id"])
    if record is None:
        record = {key: entry.get(key) for key in SEARCH_FIELDS}
        record["description"] = (entry.get("description") or "")[:DESCRIPTION_CHARS]
        record.update(query=query, query_motion=query_motion, queries=[], status="candidate")
        records[entry["id"]] = record
    if query not in record["queries"]:
        record["queries"].append(query)

    rescore_record(record, flt)
    return record


MAX_ATTEMPTS = 3

# yt-dlp error text that means this video will never download for us (we use no cookies or
# login), so retrying only wastes requests.
PERMANENT_ERROR_MARKERS = (
    "Video unavailable", "Private video", "This video has been removed", "account associated",
    "copyright", "members-only", "Join this channel", "Sign in to confirm your age",
    "age-restricted", "inappropriate for some users", "not available in your country",
    "who has blocked it", "This live event", "Premieres in", "is not a valid URL",
)


def record_failure(record: dict, error: str, permanent: bool = False) -> str:
    """Count one failed attempt; the status becomes 'unavailable' for errors retrying can't fix."""
    attempts = int(record.get("attempts") or 0) + 1
    if not permanent:
        permanent = any(marker.lower() in error.lower() for marker in PERMANENT_ERROR_MARKERS)
    record.update(status="unavailable" if permanent else "failed", error=error[:500], attempts=attempts)
    return record["status"]


def exclude_record(record: dict, reason: str) -> None:
    """A downloaded video we keep on disk but never use (the shortlist and log skip it)."""
    record.update(status="excluded", exclude_reason=reason)


def rescore_record(record: dict, flt: ScrapeFilter) -> bool:
    """Re-apply the scoring rules; returns True if the status flipped.

    Search-time decisions (candidate/rejected) are revisited both ways. Downloaded videos get
    a fresh score and motion guess, and are excluded if the rules now reject them (a blocklist
    entry added after the download) -- and un-excluded if a rule change clears them again.
    Failed/unavailable records and videos excluded for other reasons keep their state.
    """
    metadata_excluded = record["status"] == "excluded" and str(record.get("exclude_reason")).startswith("metadata:")
    if record["status"] not in {"candidate", "rejected", "downloaded"} and not metadata_excluded:
        return False
    hits = len(record.get("queries") or []) or 1
    verdict = evaluate({**record, "hits": hits}, record.get("query_motion") or "footwork", flt)
    record.update(score=verdict.score, motion_guess=verdict.motion_guess, motion_tags=verdict.motion_tags)
    if record["status"] in {"downloaded", "excluded"}:
        # Duration/shorts limits were applied before downloading; only content rules count now.
        content_reject = verdict.reject_reason not in {None, "duration", "short", "live"}
        if content_reject and record["status"] == "downloaded":
            exclude_record(record, f"metadata: {verdict.reject_reason}")
            return True
        if not content_reject and record["status"] == "excluded":
            record.update(status="downloaded", exclude_reason=None)
            return True
        return False
    new_status = "rejected" if verdict.reject_reason else "candidate"
    changed = new_status != record["status"]
    record.update(reject_reason=verdict.reject_reason, status=new_status)
    return changed


def manifest_stats(records: dict[str, dict]) -> dict:
    downloaded = [r for r in records.values() if r.get("status") == "downloaded"]
    return {
        "by_status": dict(Counter(r.get("status") for r in records.values())),
        "downloaded_by_motion": dict(Counter(r.get("motion_guess") for r in downloaded)),
        "downloaded_bytes": sum(r.get("filesize") or 0 for r in downloaded),
        "downloaded_seconds": sum(r.get("duration") or 0 for r in downloaded),
        "reject_reasons": dict(Counter(r.get("reject_reason") for r in records.values()
                                       if r.get("status") == "rejected")),
        "exclude_reasons": dict(Counter(r.get("exclude_reason") for r in records.values()
                                        if r.get("status") == "excluded")),
    }
