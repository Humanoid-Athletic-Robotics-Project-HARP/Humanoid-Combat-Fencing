"""Manifest storage and dedupe for `fencing scrape` (no network)."""

from __future__ import annotations

import json

import pytest

from fencing.scrape.manifest import (
    load_manifest,
    manifest_stats,
    record_failure,
    rescore_record,
    save_manifest,
    upsert_hit,
)
from fencing.scrape.scoring import ScrapeFilter


def hit(vid: str = "abcdefghijk", title: str = "Sabre lunge drill", **extra) -> dict:
    return {
        "id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": title,
        "description": "",
        "duration": 120,
        "channel": "Club",
        **extra,
    }


def test_missing_manifest_loads_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.jsonl") == {}


def test_roundtrip_and_no_temp_file_left(tmp_path):
    path = tmp_path / "videos" / "manifest.jsonl"
    records = {"a": {"id": "a", "status": "candidate"}, "b": {"id": "b", "status": "rejected"}}
    save_manifest(path, records)
    assert load_manifest(path) == records
    assert [p.name for p in path.parent.iterdir()] == ["manifest.jsonl"]
    assert len(path.read_text().splitlines()) == 2


def test_corrupt_line_raises_with_line_number(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n{broken\n")
    with pytest.raises(ValueError, match=":2"):
        load_manifest(path)


def test_upsert_creates_scored_candidate():
    records: dict = {}
    upsert_hit(records, hit(), query="sabre lunge drill", query_motion="lunge", flt=ScrapeFilter())
    record = records["abcdefghijk"]
    assert record["status"] == "candidate"
    assert record["motion_guess"] == "lunge"
    assert record["query"] == "sabre lunge drill"
    assert record["queries"] == ["sabre lunge drill"]
    assert record["score"] > 0


def test_upsert_marks_rejections():
    records: dict = {}
    upsert_hit(records, hit(title="Foil lunge drill"), "q", "lunge", ScrapeFilter())
    assert records["abcdefghijk"]["status"] == "rejected"
    assert records["abcdefghijk"]["reject_reason"] == "foil_or_epee_only"


def test_upsert_dedupes_and_counts_distinct_queries():
    records: dict = {}
    upsert_hit(records, hit(), "q1", "lunge", ScrapeFilter())
    first_score = records["abcdefghijk"]["score"]
    upsert_hit(records, hit(), "q2", "lunge", ScrapeFilter())
    upsert_hit(records, hit(), "q2", "lunge", ScrapeFilter())
    assert len(records) == 1
    assert records["abcdefghijk"]["queries"] == ["q1", "q2"]
    assert records["abcdefghijk"]["query"] == "q1"
    assert records["abcdefghijk"]["score"] > first_score


def test_upsert_never_downgrades_downloaded_record():
    records = {"abcdefghijk": {**hit(), "status": "downloaded", "path": "/x.mp4", "queries": ["q1"],
                               "query": "q1", "score": 3.0, "motion_guess": "lunge"}}
    upsert_hit(records, hit(), "q2", "lunge", ScrapeFilter())
    assert records["abcdefghijk"]["status"] == "downloaded"
    assert records["abcdefghijk"]["path"] == "/x.mp4"


def test_stats_counts_by_status_and_motion():
    records = {
        "a": {"id": "a", "status": "downloaded", "motion_guess": "lunge", "filesize": 1000, "duration": 60},
        "b": {"id": "b", "status": "downloaded", "motion_guess": "lunge", "filesize": 500, "duration": 30},
        "c": {"id": "c", "status": "candidate", "motion_guess": "parry"},
        "d": {"id": "d", "status": "rejected", "reject_reason": "foil_or_epee_only"},
    }
    stats = manifest_stats(records)
    assert stats["by_status"] == {"downloaded": 2, "candidate": 1, "rejected": 1}
    assert stats["downloaded_by_motion"] == {"lunge": 2}
    assert stats["downloaded_bytes"] == 1500
    assert stats["downloaded_seconds"] == 90
    assert stats["reject_reasons"] == {"foil_or_epee_only": 1}


def test_transient_failure_counts_attempts():
    record = {**hit(), "status": "candidate"}
    assert record_failure(record, "HTTP Error 503: Service Unavailable") == "failed"
    assert record_failure(record, "timed out") == "failed"
    assert record["attempts"] == 2 and record["error"] == "timed out"


@pytest.mark.parametrize("error", ["ERROR: [youtube] x: Private video. Sign in if you've been granted access",
                                   "ERROR: [youtube] x: Video unavailable. This video has been removed",
                                   "Sign in to confirm your age. This video may be inappropriate for some users."])
def test_permanent_failure_is_never_retried(error):
    record = {**hit(), "status": "candidate"}
    assert record_failure(record, error) == "unavailable"


def test_rescore_excludes_a_downloaded_video_the_rules_now_reject_and_can_undo_it():
    records = {"abcdefghijk": {**hit(title="Seated lunge drill sabre"), "status": "downloaded", "queries": ["q"],
                               "query_motion": "lunge"}}
    record = records["abcdefghijk"]
    assert rescore_record(record, ScrapeFilter()) is True
    assert record["status"] == "excluded" and record["exclude_reason"] == "metadata: blocklisted"
    record["title"] = "Sabre lunge drill"  # e.g. the blocklist entry was a mistake and got removed
    assert rescore_record(record, ScrapeFilter()) is True
    assert record["status"] == "downloaded" and record["exclude_reason"] is None


def test_rescore_ignores_duration_limits_for_downloaded_videos():
    record = {**hit(duration=5000), "status": "downloaded", "queries": ["q"], "query_motion": "lunge"}
    assert rescore_record(record, ScrapeFilter()) is False
    assert record["status"] == "downloaded"


def test_rescore_leaves_other_exclusions_alone():
    record = {**hit(), "status": "excluded", "exclude_reason": "vertical video (720x1280)", "queries": ["q"]}
    assert rescore_record(record, ScrapeFilter()) is False
    assert record["status"] == "excluded"
