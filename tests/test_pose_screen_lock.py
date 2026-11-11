"""`fencing screen` must never let two processes work on the same video at once."""

from __future__ import annotations

import json

import pytest

from fencing import paths
from fencing.pose import screen


def test_second_screen_of_the_same_video_is_refused_while_the_first_runs(tmp_path):
    with screen.video_lock(tmp_path / "abc"):
        with pytest.raises(screen.ScreenBusy):
            with screen.video_lock(tmp_path / "abc"):
                pass
    with screen.video_lock(tmp_path / "abc"):  # released again afterwards
        pass


def test_busy_video_is_skipped_not_counted_as_a_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "POSE", tmp_path / "pose")
    monkeypatch.setattr(paths, "VIDEOS_RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "vid12345678.mp4").write_bytes(b"")
    with screen.video_lock(tmp_path / "pose" / "vid12345678"):
        assert screen.main(["vid12345678"]) == 0
    assert "being screened by another process" in capsys.readouterr().err


def test_rescore_without_cached_detections_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "POSE", tmp_path / "pose")
    assert screen.screen_one("vid12345678", tmp_path / "v.mp4", force=False, rescore=True, device="cpu") is None


def test_atomic_json_write_leaves_no_temp_file(tmp_path):
    screen.write_json_atomic(tmp_path / "screen.json", {"a": 1})
    assert json.loads((tmp_path / "screen.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["screen.json"]
