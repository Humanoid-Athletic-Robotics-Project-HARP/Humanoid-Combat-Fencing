import json
import re

import numpy as np
import pytest

from fencing import paths
from fencing.motion.io import (
    JOINT_NAMES,
    Motion,
    MotionFormatError,
    read_csv,
    read_sidecar,
    validate_motion,
    write_csv,
)


def make_motion(n=20, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return Motion(rng.normal(size=(n, 3)), quat, rng.uniform(-1, 1, size=(n, 29)), fps)


def test_csv_round_trip_preserves_values(tmp_path):
    motion = make_motion()
    path = write_csv(motion, tmp_path / "m.csv")
    back = read_csv(path)
    assert back.n_frames == motion.n_frames
    np.testing.assert_allclose(back.to_array(), motion.to_array(), atol=1e-6)
    assert back.fps == 30.0


def test_csv_is_headerless_36_columns(tmp_path):
    path = write_csv(make_motion(5), tmp_path / "m.csv")
    lines = path.read_text().splitlines()
    assert len(lines) == 5
    assert all(len(line.split(",")) == 36 for line in lines)


def test_wrong_column_count_names_the_count(tmp_path):
    path = tmp_path / "bad.csv"
    np.savetxt(path, np.zeros((4, 35)), delimiter=",")
    with pytest.raises(MotionFormatError, match="got 35"):
        read_csv(path)


def test_nan_is_reported_with_its_frame(tmp_path):
    array = make_motion(10).to_array()
    array[7, 12] = np.nan
    path = tmp_path / "nan.csv"
    np.savetxt(path, array, delimiter=",")
    with pytest.raises(MotionFormatError, match="frame 7"):
        read_csv(path)


def test_non_unit_quaternion_is_rejected():
    motion = make_motion(10)
    motion.root_quat[3] *= 2.0
    problems = validate_motion(motion)
    assert any("frame 3" in p for p in problems)


def test_exclusive_write_refuses_to_overwrite(tmp_path):
    path = write_csv(make_motion(), tmp_path / "m.csv")
    with pytest.raises(FileExistsError):
        write_csv(make_motion(), path, exclusive=True)


def test_sidecar_fps_is_used(tmp_path):
    path = write_csv(make_motion(), tmp_path / "clip.csv")
    (tmp_path / "clip.json").write_text(json.dumps({"clip_id": "clip", "fps": 50, "handedness_guess": "L"}))
    motion = read_csv(path)
    assert motion.fps == 50
    assert motion.meta["sidecar"]["handedness_guess"] == "L"


def test_sidecar_with_bad_handedness_fails_loudly(tmp_path):
    path = write_csv(make_motion(), tmp_path / "clip.csv")
    (tmp_path / "clip.json").write_text(json.dumps({"handedness_guess": "left"}))
    with pytest.raises(MotionFormatError, match="'left'"):
        read_sidecar(path)


def test_joint_order_matches_mjlab_csv_to_npz():
    """The CSV contract: our column order is exactly mjlab's joint_names list."""
    script = paths.THIRD_PARTY / "mjlab" / "src" / "mjlab" / "scripts" / "csv_to_npz.py"
    if not script.exists():
        pytest.skip("mjlab submodule not checked out")
    block = script.read_text().split("joint_names=[", 1)[1].split("]", 1)[0]
    assert tuple(re.findall(r'"([a-z_]+_joint)"', block)) == JOINT_NAMES
