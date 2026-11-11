"""The automatic `fencing scrape run` path end to end, with yt-dlp faked out (no network).

Covers OpenCV verification of downloads, resume/idempotency, bounded retries, search-state
skipping, the shortlist built from screen reports and the screen-derived video-log cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fencing import paths
from fencing.scrape import cli
from fencing.scrape.manifest import load_manifest, save_manifest
from fencing.scrape.shortlist import build_shortlist, log_fill, range_rank
from fencing.scrape.video_log import parse_table
from fencing.scrape.youtube import BrokenVideoError, verify_video


def write_video(path: Path, seconds: float = 2.0, size: tuple[int, int] = (160, 96), fps: int = 10) -> Path:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for i in range(int(seconds * fps)):
        frame = np.full((height, width, 3), (i * 7) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# --- verify_video -----------------------------------------------------------------------

def test_verify_reads_frames_fps_and_size(tmp_path):
    info = verify_video(write_video(tmp_path / "v.mp4"), expected_duration=2)
    assert info["cv_frames"] == 20 and info["cv_fps"] == 10
    assert (info["cv_width"], info["cv_height"]) == (160, 96)


def test_verify_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    with pytest.raises(BrokenVideoError):
        verify_video(bad)


def test_verify_rejects_truncated_download(tmp_path):
    with pytest.raises(BrokenVideoError, match="truncated"):
        verify_video(write_video(tmp_path / "v.mp4", seconds=2), expected_duration=60)


# --- the run command ----------------------------------------------------------------------

def candidate(vid: str, motion: str = "footwork", score: float = 10.0, **extra) -> dict:
    return {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": f"Sabre {motion} drill {vid}",
            "channel": f"Club {vid}", "duration": 12, "status": "candidate", "score": score,
            "motion_guess": motion, "queries": ["q"], "query_motion": motion} | extra


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A manifest, a videos dir and a video log in tmp_path; yt-dlp replaced by fakes."""
    monkeypatch.setattr(paths, "POSE", tmp_path / "pose")
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    calls: list[str] = []

    def fake_download(url, out_dir, max_height):
        vid = url.rsplit("=", 1)[1]
        calls.append(vid)
        out_dir.mkdir(parents=True, exist_ok=True)
        if vid.startswith("broken"):
            (out_dir / f"{vid}.mp4").write_bytes(b"\0" * 1000)
        elif vid.startswith("gone"):
            raise cli.DownloadError("ERROR: [youtube] gone: Private video")
        elif vid.startswith("flaky"):
            raise cli.DownloadError("ERROR: unable to download video data: HTTP Error 503")
        else:
            write_video(out_dir / f"{vid}.mp4", seconds=12, size=(96, 160) if vid.startswith("tall") else (160, 96))
        return out_dir / f"{vid}.mp4", {"title": None, "duration": 12}

    monkeypatch.setattr(cli, "download_video", fake_download)
    monkeypatch.setattr(cli, "probe_video", lambda path: {"filesize": path.stat().st_size})
    monkeypatch.setattr(cli, "search", lambda target: [])
    manifest = tmp_path / "videos" / "manifest.jsonl"
    video_log = tmp_path / "video_log.md"
    video_log.write_text("# Video log\n\n| # | Video | Motion |\n|---|---|---|\n")

    def run(*extra: str) -> int:
        argv = ["--manifest", str(manifest), "--videos-dir", str(tmp_path / "videos" / "raw"), "run",
                "--queries", str(write_queries(tmp_path)), "--video-log", str(video_log), "--min-score", "0", *extra]
        return cli.main(argv)

    return {"manifest": manifest, "video_log": video_log, "raw": tmp_path / "videos" / "raw", "calls": calls,
            "run": run, "tmp": tmp_path}


def write_queries(tmp_path: Path) -> Path:
    path = tmp_path / "queries.yaml"
    path.write_text("motions:\n  footwork:\n    - sabre footwork drill\n")
    return path


def test_run_downloads_verifies_and_logs(workspace):
    ids = ("good1xxxxxx", "broken1xxxx", "tall1xxxxxx", "gone1xxxxxx")
    save_manifest(workspace["manifest"], {v: candidate(v) for v in ids})
    assert workspace["run"]() == 0
    records = load_manifest(workspace["manifest"])
    assert records["good1xxxxxx"]["status"] == "downloaded" and records["good1xxxxxx"]["cv_frames"] == 120
    assert records["broken1xxxx"]["status"] == "failed" and records["broken1xxxx"]["attempts"] == 1
    assert not (workspace["raw"] / "broken1xxxx.mp4").exists()  # deleted, so the retry starts clean
    assert records["tall1xxxxxx"]["status"] == "excluded" and "vertical" in records["tall1xxxxxx"]["exclude_reason"]
    assert records["gone1xxxxxx"]["status"] == "unavailable"
    _, rows = parse_table(workspace["video_log"].read_text())
    assert [r["_id"] for r in rows] == ["good1xxxxxx"]


def test_rerun_is_idempotent_and_retries_are_bounded(workspace):
    save_manifest(workspace["manifest"], {v: candidate(v) for v in ("good1xxxxxx", "flaky1xxxxx", "gone1xxxxxx")})
    for _ in range(5):
        workspace["run"]()
    assert workspace["calls"].count("good1xxxxxx") == 1  # never downloaded twice
    assert workspace["calls"].count("gone1xxxxxx") == 1  # permanent errors are not retried
    assert workspace["calls"].count("flaky1xxxxx") == 3  # MAX_ATTEMPTS
    assert load_manifest(workspace["manifest"])["flaky1xxxxx"]["attempts"] == 3


def test_file_left_by_an_interrupted_run_is_adopted_not_redownloaded(workspace):
    save_manifest(workspace["manifest"], {"good1xxxxxx": candidate("good1xxxxxx")})
    workspace["raw"].mkdir(parents=True)
    write_video(workspace["raw"] / "good1xxxxxx.mp4", seconds=12)
    workspace["run"]()
    assert workspace["calls"] == []
    assert load_manifest(workspace["manifest"])["good1xxxxxx"]["status"] == "downloaded"


def test_recent_searches_are_skipped(workspace, monkeypatch):
    searched: list[str] = []
    monkeypatch.setattr(cli, "search", lambda target: searched.append(target) or [])
    workspace["run"]()
    workspace["run"]()
    assert len(searched) == 1
    workspace["run"]("--search-max-age-days", "0")
    assert len(searched) == 2


def test_verify_command_requeues_missing_and_broken_files(workspace):
    save_manifest(workspace["manifest"], {"good1xxxxxx": candidate("good1xxxxxx"), "good2xxxxxx": candidate("good2xxxxxx")})
    workspace["run"]()
    (workspace["raw"] / "good1xxxxxx.mp4").unlink()
    (workspace["raw"] / "good2xxxxxx.mp4").write_bytes(b"\0" * 10)
    argv = ["--manifest", str(workspace["manifest"]), "--videos-dir", str(workspace["raw"]), "verify"]
    assert cli.main(argv) == 1  # a broken file is worth a non-zero exit
    records = load_manifest(workspace["manifest"])
    assert records["good1xxxxxx"]["status"] == "candidate"
    assert records["good2xxxxxx"]["status"] == "failed" and records["good2xxxxxx"]["attempts"] == 1


def test_run_with_screen_screens_new_downloads_and_writes_shortlist(workspace, monkeypatch):
    from fencing.pose import screen

    screened: list[list[str]] = []

    def fake_screen(ids):
        screened.append(list(ids))
        for vid in ids:
            out = paths.POSE / vid
            out.mkdir(parents=True)
            (out / "screen.json").write_text(json.dumps(report([rng(0, 10)])))
        return 0

    monkeypatch.setattr(screen, "main", fake_screen)
    save_manifest(workspace["manifest"], {"good1xxxxxx": candidate("good1xxxxxx")})
    assert workspace["run"]("--screen") == 0
    assert screened == [["good1xxxxxx"]]
    shortlist = json.loads((workspace["manifest"].parent / "shortlist.json").read_text())
    assert shortlist["by_motion"]["footwork"][0]["yt_id"] == "good1xxxxxx"
    row = {r["_id"]: r for r in parse_table(workspace["video_log"].read_text())[1]}["good1xxxxxx"]
    assert (row["Start"], row["End"], row["View"], row["Fencers"]) == ("0:00", "0:10", "side", "1")
    workspace["run"]("--screen")
    assert screened == [["good1xxxxxx"]]  # already screened: not run again


# --- shortlist ----------------------------------------------------------------------------

def rng(start: float, end: float, view: str = "side", people: int = 1, full: float = 1.0, usable: bool = True) -> dict:
    return {"start_s": start, "end_s": end, "view": view, "view_ratio": 0.2, "people_median": people,
            "full_body_frac": full, "camera": "static" if usable else "moving", "usable": usable,
            "camera_motion_width_per_s": 0.0, "track_id": 1, "reasons": [] if usable else ["camera moving (0.1)"]}


def report(ranges: list[dict]) -> dict:
    return {"ranges": ranges, "usable_ranges": [[r["start_s"], r["end_s"]] for r in ranges if r["usable"]]}


def test_rank_prefers_one_fencer_side_on_full_body_and_longer():
    best = range_rank(rng(0, 10))
    assert best > range_rank(rng(0, 10, people=2))
    assert best > range_rank(rng(0, 10, view="front"))
    assert best > range_rank(rng(0, 10, full=0.85))
    assert best > range_rank(rng(0, 3))
    assert range_rank(rng(0, 10), area_frac=0.08) > range_rank(rng(0, 10), area_frac=0.005)


def test_shortlist_groups_by_motion_caps_ranges_per_video_and_skips_unusable():
    records = {"a": candidate("a", "lunge", status="downloaded"), "b": candidate("b", "lunge", status="downloaded"),
               "x": candidate("x", status="excluded")}
    reports = {
        "a": report([rng(i * 20, i * 20 + 10) for i in range(5)] + [rng(200, 210, usable=False)]),
        "b": report([rng(0, 5, view="front")]),
        "x": report([rng(0, 10)]),
    }
    shortlist = build_shortlist(records, reports)
    lunge = shortlist["by_motion"]["lunge"]
    assert [e["yt_id"] for e in lunge] == ["a", "a", "a", "b"]
    assert shortlist["usable_ranges"] == 6 and shortlist["screened_videos"] == 2
    assert all(e["static_camera"] for e in shortlist["ranges"])


def test_log_fill_needs_a_view_consensus_and_explains_empty_screens():
    record = candidate("a")
    mixed = log_fill(record, report([rng(0, 10, view="side"), rng(20, 30, view="front")]))
    assert "View" not in mixed and mixed["Start"] == "0:00"
    angled = log_fill(record, report([rng(0, 10, view="oblique"), rng(20, 22, view="front")]))
    assert angled["View"] == "angled"
    empty = log_fill(record, report([rng(0, 10, usable=False)]))
    assert empty == {"Notes": "screen: no usable range (camera moving)"}
