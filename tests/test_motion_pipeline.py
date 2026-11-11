import json
from functools import lru_cache

import numpy as np
import pytest

from fencing.motion import cli
from fencing.motion.io import read_csv, write_csv, write_sidecar
from fencing.motion.kinematics import forward_kinematics
from fencing.motion.normalize import fencing_line, mirror_motion, rotate_about_z
from fencing.motion.pipeline import CleanConfig, clean_motion
from fencing.motion.synthetic import add_root_slide, add_vertical_drift, synthesize


@lru_cache(maxsize=1)
def raw_session():
    motion, spans = synthesize([("en_garde", 0.8), ("lunge", 1.2), ("en_garde", 0.8), ("advance", 0.6), ("en_garde", 0.8)])
    corrupted = add_root_slide(add_vertical_drift(motion, 0.04, 3.0, 0.05), 0.08)
    return rotate_about_z(corrupted, -0.7, np.array([1.0, 2.0])), spans


def test_auto_segment_clean_left_handed_lunge():
    raw, spans = raw_session()
    left = mirror_motion(raw)
    result = clean_motion(left, CleanConfig(), label="lunge", auto_segment=True)
    truth = next(s for s in spans if s[2] == "lunge")
    assert result.segment["label"] == "lunge"
    assert result.segment["start"] <= truth[0] and result.segment["end"] >= truth[1]
    assert result.hand == "L" and result.mirrored
    assert result.report["verdict"] == "pass", result.report["reasons"]
    out = result.motion
    np.testing.assert_allclose(out.root_pos[0, :2], 0, atol=1e-9)
    assert abs(fencing_line(out, forward_kinematics(out))[0]) < 0.05  # fencing line along +x
    # holds at both ends
    assert np.abs(np.diff(out.joint_pos[:10], axis=0)).max() == 0



def test_end_hold_can_be_longer_than_the_start_hold():
    """A clip cut mid-recovery needs a long end hold so the policy learns to stabilise it."""
    raw, _ = raw_session()
    frames = (40, 100)
    base = clean_motion(raw, CleanConfig(), label="lunge", frames=frames).motion
    long_end = clean_motion(raw, CleanConfig(end_hold_s=2.0), label="lunge", frames=frames).motion
    assert long_end.n_frames - base.n_frames == round(1.5 * base.fps)
    np.testing.assert_allclose(long_end.joint_pos[:15], base.joint_pos[:15])  # start hold unchanged
    assert np.abs(np.diff(long_end.joint_pos[-60:], axis=0)).max() == 0

def test_sidecar_handedness_is_used(tmp_path):
    raw, _ = raw_session()
    path = write_csv(mirror_motion(raw), tmp_path / "clip.csv")
    write_sidecar(tmp_path / "clip.json", {"clip_id": "clip", "fps": 30, "handedness_guess": "L"})
    result = clean_motion(read_csv(path), CleanConfig(hold_s=0))
    assert result.hand_source == "sidecar" and result.mirrored


def test_cli_clean_segment_and_check(tmp_path, capsys):
    raw, _ = raw_session()
    src = write_csv(raw, tmp_path / "abc_0-5000.csv")
    out = tmp_path / "out.csv"
    assert cli.main(["clean", str(src), "--label", "lunge", "--segment", "auto", "--out", str(out)]) == 0
    assert read_csv(out).n_frames > 30
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["quality"]["verdict"] == "pass"
    assert [s["step"] for s in report["steps"]][:2] == ["segment", "canonicalize"]
    capsys.readouterr()

    assert cli.main(["segment", str(src)]) == 0
    labels = [s["label"] for s in json.loads(capsys.readouterr().out)["segments"]]
    assert "lunge" in labels and "advance" in labels

    assert cli.main(["check", str(out)]) == 0
    assert cli.main(["check", str(src)]) == 1  # raw: floating/sliding feet fail


def test_cli_rejects_bad_segment_spec(tmp_path):
    raw, _ = raw_session()
    src = write_csv(raw, tmp_path / "x.csv")
    assert cli.main(["clean", str(src), "--label", "lunge", "--segment", "5:2"]) == 2


def test_cli_mirror_round_trip(tmp_path):
    raw, _ = raw_session()
    src = write_csv(raw, tmp_path / "a.csv")
    assert cli.main(["mirror", str(src), str(tmp_path / "b.csv")]) == 0
    assert cli.main(["mirror", str(tmp_path / "b.csv"), str(tmp_path / "c.csv")]) == 0
    np.testing.assert_allclose(read_csv(tmp_path / "c.csv").to_array(), read_csv(src).to_array(), atol=2e-6)


@pytest.mark.slow
def test_render_contact_sheet(tmp_path):
    import cv2

    raw, _ = raw_session()
    src = write_csv(raw.slice(0, 20), tmp_path / "a.csv")
    png = tmp_path / "a.png"
    assert cli.main(["render", str(src), "--png", str(png), "--columns", "2"]) == 0
    image = cv2.imread(str(png))
    assert image is not None and image.shape[1] == 2 * 320 and image.std() > 10
