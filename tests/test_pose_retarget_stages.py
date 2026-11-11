"""Pure parts of `fencing pose` and `fencing retarget`: clip planning, MediaPipe clean-up,
qpos -> CSV conversion, fps resampling and the sidecar contract with `fencing motion`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fencing.motion.io import N_COLS, read_csv
from fencing.pose import landmarks as lm
from fencing.pose.clips import ClipSpec, make_clip_id, parse_clip_id, plan_clips, split_range
from fencing.pose.postprocess import clean_landmarks, resample_valid, uniform_times
from fencing.retarget.cli import choose_method
from fencing.retarget.convert import (
    build_sidecar,
    csv_array_to_pickle_dict,
    despike_joints,
    make_continuous,
    qpos_to_csv_array,
    resample_csv_array,
)

# ---- clip ids and planning -----------------------------------------------------------------


@pytest.mark.parametrize("yt_id", ["IpM7citb5bs", "53FZTK_8qOI", "YyiWe-ZakzE"])
def test_clip_id_round_trips_for_ids_with_underscores_and_dashes(yt_id):
    clip_id = make_clip_id(yt_id, 12.0, 19.5)
    assert clip_id == f"{yt_id}_12000-19500"
    assert parse_clip_id(clip_id) == (yt_id, 12.0, 19.5)


def test_parse_clip_id_of_a_bare_video_id_has_no_range():
    assert parse_clip_id("53FZTK_8qOI") == ("53FZTK_8qOI", None, None)


def test_make_clip_id_rejects_empty_range():
    with pytest.raises(ValueError, match="bad clip range"):
        make_clip_id("abc", 5.0, 5.0)


def test_split_range_makes_equal_chunks_no_longer_than_the_cap():
    chunks = split_range(3.837, 48.9, max_seconds=20.0)
    assert len(chunks) == 3
    assert chunks[0][0] == 3.837 and chunks[-1][1] == 48.9
    lengths = [e - s for s, e in chunks]
    assert max(lengths) <= 20.0 and np.ptp(lengths) < 0.01
    assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:]))


def test_split_range_drops_ranges_shorter_than_the_minimum():
    assert split_range(1.0, 2.0, max_seconds=20.0, min_seconds=1.5) == []
    assert split_range(1.0, 10.0, max_seconds=20.0) == [(1.0, 10.0)]


SCREEN = {
    "yt_id": "vid",
    "duration_s": 60.0,
    "chosen_track_id": 1,
    "ranges": [
        {"start_s": 0.0, "end_s": 10.0, "track_id": 1, "usable": True},
        {"start_s": 20.0, "end_s": 45.0, "track_id": 7, "usable": True},
        {"start_s": 50.0, "end_s": 51.0, "track_id": 9, "usable": False},
    ],
    "usable_ranges": [[0.0, 10.0], [20.0, 45.0]],
}


def test_plan_clips_splits_usable_ranges_and_keeps_each_shots_track():
    clips = plan_clips(SCREEN, max_seconds=20.0)
    assert [c.clip_id for c in clips] == ["vid_0-10000", "vid_20000-32500", "vid_32500-45000"]
    assert [c.track_id for c in clips] == [1, 7, 7]


def test_plan_clips_with_explicit_range_uses_overlapping_track():
    (clip,) = plan_clips(SCREEN, start_s=22.0, end_s=30.0)
    assert clip == ClipSpec("vid_22000-30000", "vid", 22.0, 30.0, 7)


def test_plan_clips_rejects_reversed_override():
    with pytest.raises(ValueError, match="must be after"):
        plan_clips(SCREEN, start_s=30.0, end_s=20.0)


# ---- MediaPipe clean-up ----------------------------------------------------------------------


def test_uniform_times_are_30fps_and_stay_inside_the_range():
    t = uniform_times(10.0, 11.0)
    assert len(t) == 31 and np.isclose(t[-1], 11.0)
    assert np.allclose(np.diff(t), 1 / 30)


def test_resample_valid_takes_the_nearest_source_frame():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    valid = np.array([True, False, True, True])
    assert resample_valid(times, valid, np.array([0.0, 0.04, 0.09, 0.16, 0.3])).tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]


def raw_worker_output(num_frames: int = 60, fps: float = 50.0, invalid: tuple[int, ...] = ()) -> dict:
    rng = np.random.default_rng(1)
    base_xy = rng.uniform(300, 600, (lm.NUM_LANDMARKS, 2))
    base_world = rng.uniform(-0.5, 0.5, (lm.NUM_LANDMARKS, 3))
    image_xy = np.stack([base_xy + [i, 0.0] for i in range(num_frames)])
    world = np.stack([base_world] * num_frames)
    visibility = np.full((num_frames, lm.NUM_LANDMARKS), 0.9)
    valid = np.ones(num_frames, dtype=bool)
    for i in invalid:
        image_xy[i], world[i], visibility[i], valid[i] = np.nan, np.nan, 0.0, False
    return {
        "times_s": 5.0 + np.arange(num_frames) / fps,
        "image_xy": image_xy,
        "world": world,
        "visibility": visibility,
        "valid": valid,
        "width": 1280,
        "height": 720,
    }


def test_clean_landmarks_resamples_to_30fps_and_fills_dropouts():
    out = clean_landmarks(raw_worker_output(invalid=(0, 10, 11)))
    assert np.allclose(np.diff(out["times_s"]), 1 / 30)
    assert np.isclose(out["times_s"][0], 5.0)
    for key in ("image_xy", "world", "visibility"):
        assert np.isfinite(out[key]).all(), key
    assert not out["valid"][0] and out["valid"][1:5].all()
    assert out["visibility"].min() > 0.5, "dropout visibility is interpolated, not zero"


def test_clean_landmarks_refuses_clips_mostly_without_a_pose():
    with pytest.raises(ValueError, match="only"):
        clean_landmarks(raw_worker_output(invalid=tuple(range(40))))


# ---- retarget conversion --------------------------------------------------------------------


def test_qpos_to_csv_array_reorders_root_quaternion_to_xyzw():
    rot = Rotation.from_euler("z", 30, degrees=True)
    xyzw = rot.as_quat()
    qpos = np.zeros((2, N_COLS))
    qpos[:, :3] = [0.1, 0.2, 0.75]
    qpos[:, 3:7] = [xyzw[3], *xyzw[:3]]  # MuJoCo wxyz
    qpos[:, 7:] = np.arange(29) * 0.01
    out = qpos_to_csv_array(qpos)
    assert np.allclose(out[:, 3:7], xyzw)
    assert np.allclose(out[:, 7:], qpos[:, 7:]) and np.allclose(out[:, :3], qpos[:, :3])


def test_make_continuous_removes_quaternion_sign_flips():
    q = Rotation.from_euler("z", np.linspace(0, 90, 5)[:, None], degrees=True).as_quat()
    q[2] *= -1
    out = make_continuous(q)
    assert np.all(np.sum(out[1:] * out[:-1], axis=1) > 0)


def test_resample_csv_array_60_to_30_fps_keeps_motion_and_unit_quaternions():
    n = 61
    t = np.arange(n) / 60.0
    array = np.zeros((n, N_COLS))
    array[:, 0] = t  # 1 m/s along x
    array[:, 3:7] = Rotation.from_euler("z", 90 * t[:, None], degrees=True).as_quat()
    array[:, 7] = np.sin(t)
    out = resample_csv_array(array, 60.0, 30.0)
    assert out.shape == (31, N_COLS)
    assert np.allclose(out[:, 0], np.arange(31) / 30.0)
    assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0)
    yaw = Rotation.from_quat(out[:, 3:7]).as_euler("zyx", degrees=True)[:, 0]
    assert np.allclose(yaw, 90 * np.arange(31) / 30.0, atol=1e-6)


def test_resample_csv_array_is_identity_at_30fps():
    array = np.random.default_rng(0).normal(size=(10, N_COLS))
    assert np.array_equal(resample_csv_array(array, 30.0), array)


def test_despike_removes_a_two_frame_ik_glitch_but_keeps_a_fast_real_move():
    t = np.arange(60) / 30.0
    joints = np.stack([np.sin(2 * np.pi * t), np.clip((t - 1.0) * 6.0, 0.0, 1.5)], axis=1)  # 1 Hz swing, fast ramp
    glitched = joints.copy()
    glitched[20:22, 0] += 1.4
    out, n = despike_joints(glitched)
    assert n == 2
    assert np.allclose(out[20:22, 0], joints[20:22, 0], atol=0.05)
    assert np.allclose(out[:, 1], joints[:, 1]), "a monotonic 6 rad/s ramp is not a spike"


def test_pickle_dict_has_gmr_keys():
    array = np.zeros((3, N_COLS))
    d = csv_array_to_pickle_dict(array, 30.0)
    assert set(d) == {"fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"}
    assert d["dof_pos"].shape == (3, 29) and d["root_rot"].shape == (3, 4)


POSE_SUMMARY = {
    "clip_id": "abc123_12000-19500",
    "yt_id": "abc123",
    "source_start_s": 12.0,
    "source_end_s": 19.5,
    "pose_method": "fallback_mediapipe",
    "track_id": 1,
    "human_height_m": 1.75,
    "handedness_guess": "L",
    "notes": "monocular",
}


def test_sidecar_has_every_pipeline_field_and_is_read_by_fencing_motion(tmp_path):
    sidecar = build_sidecar(POSE_SUMMARY, {"frames": 3}, notes="gmr")
    for key in ("clip_id", "yt_id", "source_start_s", "source_end_s", "fps", "pose_method", "track_id", "human_height_m", "handedness_guess", "notes"):
        assert key in sidecar, key
    assert sidecar["fps"] == 30 and sidecar["notes"] == "monocular; gmr"
    csv = tmp_path / "abc123_12000-19500.csv"
    row = np.zeros(N_COLS)
    row[6] = 1.0  # identity quaternion, xyzw
    np.savetxt(csv, np.stack([row] * 3), delimiter=",", fmt="%.6f")
    csv.with_suffix(".json").write_text(json.dumps(sidecar))
    motion = read_csv(csv)
    assert motion.fps == 30.0 and motion.meta["sidecar"]["handedness_guess"] == "L"


def test_sidecar_requires_pose_fields():
    with pytest.raises(KeyError, match="handedness_guess"):
        build_sidecar({k: v for k, v in POSE_SUMMARY.items() if k != "handedness_guess"})


def test_choose_method_follows_pose_json_unless_forced():
    assert choose_method({"pose_method": "gvhmr"}, "auto") == "gvhmr"
    assert choose_method({"pose_method": "fallback_mediapipe"}, "auto") == "fallback"
    assert choose_method({"pose_method": "gvhmr"}, "fallback") == "fallback"
