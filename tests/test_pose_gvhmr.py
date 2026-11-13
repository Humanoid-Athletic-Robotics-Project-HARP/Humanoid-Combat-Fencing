"""GVHMR branch of `fencing pose` / `fencing retarget` without the body models: the fencer
box selection, the same-person check, the two-phase CLI flow and GVHMR output -> GMR input."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fencing.pose import gvhmr_boxes as gb

# ---- fencer boxes ----------------------------------------------------------------------------


def _scene(n=40):
    """Two fencers walking apart and a referee; the pipeline track is fencer A (10 fps, lagging)."""
    t = np.arange(n)
    fencer_a = np.stack([100 + 3 * t, 0 * t + 200, 200 + 3 * t, 0 * t + 500], axis=1).astype(float)
    fencer_b = np.stack([600 - 3 * t, 0 * t + 190, 720 - 3 * t, 0 * t + 520], axis=1).astype(float)
    referee = np.tile([900.0, 100.0, 1100.0, 600.0], (n, 1))  # biggest box: GVHMR's own pick
    rows = [(f, b) for f in range(n) for b in (fencer_b[f], referee[f], fencer_a[f])]
    det_frame = np.array([r[0] for r in rows])
    det_box = np.array([r[1] for r in rows])
    lagging = fencer_a.copy()
    lagging[:, [0, 2]] -= 6.0  # the interpolated 10 fps track trails the fencer a little
    return det_frame, det_box, lagging, fencer_a


def test_fencer_boxes_follow_the_pipeline_track_not_the_largest_person():
    det_frame, det_box, pipeline, fencer_a = _scene()
    boxes, source, iou = gb.select_fencer_boxes(det_frame, det_box, pipeline)
    assert (source == gb.SOURCE_YOLO).all() and (iou > 0.8).all()
    # The smoothed result stays on fencer A (constant velocity survives a moving average; two
    # 5-frame passes reach 4 frames in from each edge).
    assert np.abs(boxes[4:-4] - fencer_a[4:-4]).max() < 1e-6
    assert np.abs(boxes - fencer_a).max() <= 3.0 + 1e-9  # edge replication: at most 1 frame of motion


def test_short_detection_gaps_are_bridged_and_long_ones_use_the_pipeline_box():
    det_frame, det_box, pipeline, fencer_a = _scene(60)
    keep = ~(((det_frame >= 10) & (det_frame < 15)) | (det_frame >= 30))
    # Fencer A missed by YOLO in frames 10-14 and from 30 on (only B and the referee seen there).
    a_rows = np.flatnonzero(np.all(np.isclose(det_box, fencer_a[det_frame]), axis=1))
    mask = np.ones(len(det_frame), bool)
    mask[a_rows[~keep[a_rows]]] = False
    _, source, _ = gb.select_fencer_boxes(det_frame[mask], det_box[mask], pipeline, max_bridge=15)
    assert (source[10:15] == gb.SOURCE_BRIDGED).all()
    assert (source[30:] == gb.SOURCE_PIPELINE).all()
    assert (source[:10] == gb.SOURCE_YOLO).all()


def test_a_detection_that_barely_overlaps_the_fencer_is_not_taken():
    pipeline = np.array([[0.0, 0.0, 100.0, 300.0]])
    other = np.array([[90.0, 0.0, 190.0, 300.0]])  # neighbour, IoU ~0.05
    boxes, source, _ = gb.select_fencer_boxes(np.array([0]), other, pipeline)
    assert source[0] == gb.SOURCE_PIPELINE and np.allclose(boxes[0], pipeline[0])


def test_moving_average_keeps_length_and_replicates_edges():
    x = np.arange(10, dtype=float)[:, None]
    y = gb.moving_average(x, 5)
    assert y.shape == x.shape
    assert np.allclose(y[2:-2, 0], x[2:-2, 0])
    assert y[0, 0] == pytest.approx((0 + 0 + 0 + 1 + 2) / 5)


def test_clip_frame_times_are_source_seconds_at_30_fps():
    assert np.allclose(gb.clip_frame_times(12.0, 4), [12.0, 12 + 1 / 30, 12 + 2 / 30, 12.1])


# ---- same-person check -------------------------------------------------------------------------


def _skeletons(n, offset_px):
    """ViTPose (n,17,3) and MediaPipe (n,33,2) skeletons of a 300 px tall person; MediaPipe is
    shifted by offset_px (0 = the same person, large = someone else)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(100, 400, size=(17, 2))
    vit = np.zeros((n, 17, 3))
    vit[:, :, :2] = base
    vit[:, :, 2] = 0.9
    mp = np.full((n, 33, 2), 250.0)
    mp[:, gb.MEDIAPIPE_LIMBS] = base[gb.COCO17_LIMBS][None] + [offset_px, 0.0]
    boxes = np.tile([100.0, 100.0, 400.0, 400.0], (n, 1))
    return vit, mp, boxes


def test_keypoint_agreement_says_same_person_when_the_skeletons_coincide():
    vit, mp, boxes = _skeletons(30, offset_px=5.0)
    times = gb.clip_frame_times(2.0, 30)
    out = gb.keypoint_agreement(vit, times, times.copy(), mp, np.ones(30, bool), boxes)
    assert out["same_person"] is True and out["frames_compared"] == 30
    assert out["centroid_dist_median"] < 0.05


def test_keypoint_agreement_flags_a_different_person():
    vit, mp, boxes = _skeletons(30, offset_px=250.0)
    times = gb.clip_frame_times(2.0, 30)
    out = gb.keypoint_agreement(vit, times, times.copy(), mp, np.ones(30, bool), boxes)
    assert out["same_person"] is False and out["centroid_dist_median"] > 0.5


def test_keypoint_agreement_ignores_left_right_swaps():
    vit, mp, boxes = _skeletons(10, offset_px=0.0)
    swapped = mp.copy()
    swapped[:, [11, 12]] = mp[:, [12, 11]]
    times = gb.clip_frame_times(0.0, 10)
    out = gb.keypoint_agreement(vit, times, times.copy(), swapped, np.ones(10, bool), boxes)
    assert out["centroid_dist_median"] == pytest.approx(0.0, abs=1e-9)


def test_keypoint_agreement_skips_frames_without_a_mediapipe_pose_nearby():
    vit, mp, boxes = _skeletons(10, offset_px=0.0)
    clip_times = gb.clip_frame_times(0.0, 10)
    valid = np.ones(10, bool)
    valid[:4] = False
    out = gb.keypoint_agreement(vit, clip_times, clip_times + 0.0, mp, valid, boxes)
    assert out["frames_compared"] == 6
    far = gb.keypoint_agreement(vit, clip_times, clip_times + 1.0, mp, np.ones(10, bool), boxes)
    assert far == {"frames_compared": 0, "same_person": None}


def test_box_summary_fractions_add_up():
    source = np.array([0, 0, 1, 2], dtype=np.int8)
    out = gb.box_summary(source, np.array([0.8, 0.6, 0.0, 0.0]))
    assert out["yolo_frac"] + out["bridged_frac"] + out["pipeline_frac"] == pytest.approx(1.0)
    assert out["matched_iou_median"] == pytest.approx(0.7)


# ---- `fencing pose`: which clips, and when GVHMR runs ---------------------------------------

from fencing import paths
from fencing.pose import cli as pose_cli
from fencing.pose import gvhmr
from fencing.pose.clips import ClipSpec

FALLBACK_POSE = {"pose_method": "fallback_mediapipe", "track_id": 7}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A screened video `vidAAAAAAAA` (usable 0-30 s on track 2) with one older clip folder
    vidAAAAAAAA_5000-12000 whose pose.json was made from track 7."""
    pose = tmp_path / "pose"
    monkeypatch.setattr(paths, "POSE", pose)
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(paths, "CLIPS", tmp_path / "clips")
    monkeypatch.setattr(paths, "VIDEOS_RAW", tmp_path / "videos")
    screen = {
        "yt_id": "vidAAAAAAAA", "video": str(tmp_path / "videos" / "vidAAAAAAAA.mp4"), "duration_s": 40.0,
        "usable_ranges": [[0.0, 30.0]], "ranges": [{"start_s": 0.0, "end_s": 30.0, "track_id": 2}], "chosen_track_id": 2,
    }  # fmt: skip
    (pose / "vidAAAAAAAA").mkdir(parents=True)
    (pose / "vidAAAAAAAA" / "screen.json").write_text(json.dumps(screen))
    old = pose / "vidAAAAAAAA_5000-12000"
    old.mkdir()
    (old / "pose.json").write_text(json.dumps(FALLBACK_POSE))
    np.savez(old / "mediapipe_raw.npz", times_s=np.arange(3.0))
    return tmp_path


def _collect(*argv):
    parser = pose_cli.build_parser()
    return [c for c, _ in pose_cli.collect_clips(parser.parse_args(list(argv)), parser)]


def test_a_bare_video_id_takes_its_existing_clip_folders_and_the_planned_clips(data_dir):
    clips = {c.clip_id: c.track_id for c in _collect("vidAAAAAAAA")}
    assert clips == {"vidAAAAAAAA_5000-12000": 7, "vidAAAAAAAA_0-15000": 2, "vidAAAAAAAA_15000-30000": 2}


def test_existing_takes_only_clips_that_already_have_a_folder(data_dir):
    assert [c.clip_id for c in _collect("vidAAAAAAAA", "--existing")] == ["vidAAAAAAAA_5000-12000"]


def test_an_explicit_clip_id_keeps_the_track_its_pose_json_was_made_from(data_dir):
    (clip,) = _collect("vidAAAAAAAA_5000-12000")
    assert clip.track_id == 7
    (new,) = _collect("vidAAAAAAAA_20000-25000")
    assert new.track_id == 2


@pytest.mark.parametrize(
    ("method", "use_gvhmr", "current"),
    [("fallback_mediapipe", False, True), ("fallback_mediapipe", True, False), ("gvhmr", True, True), ("gvhmr", False, True)],
)
def test_pose_json_is_redone_only_to_upgrade_it_to_gvhmr(method, use_gvhmr, current):
    assert pose_cli.pose_is_current({"pose_method": method}, use_gvhmr) is current


def _fake_pose_result(n=4):
    return SimpleNamespace(
        times_s=np.arange(n) / 30.0, keypoints_world=np.zeros((n, 33, 3)), image_xy=np.zeros((n, 33, 2)),
        visibility=np.ones((n, 33)), valid=np.ones(n, bool), root_cam=np.zeros((n, 3)), scale=1.0,
        pitch_down_rad=0.1, focal_px=1000.0, handedness="R", handedness_score=0.8,
        stats={"frames": n, "valid_frac": 1.0, "median_visibility": 0.9, "root_path_m": 0.0},
    )  # fmt: skip


def _patch_pose_one(monkeypatch, check):
    calls = []
    monkeypatch.setattr(pose_cli, "process_mediapipe", lambda *a, **k: _fake_pose_result())
    monkeypatch.setattr(pose_cli, "write_keypoints", lambda *a, **k: None)
    monkeypatch.setattr(pose_cli, "run_in_venv", lambda *a, **k: pytest.fail("MediaPipe must come from the cache"))

    def fake_run_clip(clip, video, log, predict, render=False, redo_preprocess=False, redo_results=False):
        calls.append({"clip": clip, "predict": predict, "render": render, "redo_results": redo_results})
        return check

    monkeypatch.setattr(pose_cli.gvhmr, "run_clip", fake_run_clip)
    return calls


def _pose_args(*extra):
    return pose_cli.build_parser().parse_args(["vidAAAAAAAA", "--no-overlay", *extra])


def test_a_fallback_clip_is_upgraded_to_gvhmr_from_cached_mediapipe(data_dir, monkeypatch):
    check = {"keypoints": {"same_person": True, "centroid_dist_median": 0.01}, "boxes": {"yolo_frac": 1.0}}
    calls = _patch_pose_one(monkeypatch, check)
    clip = ClipSpec("vidAAAAAAAA_5000-12000", "vidAAAAAAAA", 5.0, 12.0, 7)
    summary = pose_cli.pose_one(clip, {"video": "v.mp4"}, _pose_args(), use_gvhmr=True)
    assert summary["pose_method"] == "gvhmr" and summary["track_id"] == 7
    assert summary["gvhmr"]["fencer_check"] == check and summary["gvhmr"]["fps"] == 30
    assert calls == [{"clip": clip, "predict": True, "render": False, "redo_results": False}]
    written = json.loads((paths.POSE / clip.clip_id / "pose.json").read_text())
    assert written["pose_method"] == "gvhmr" and written["handedness_guess"] == "R"
    # Run again: now current, nothing is redone.
    assert pose_cli.pose_one(clip, {"video": "v.mp4"}, _pose_args(), use_gvhmr=True)["pose_method"] == "gvhmr"
    assert len(calls) == 1


def test_gvhmr_on_the_wrong_person_keeps_the_fallback(data_dir, monkeypatch):
    check = {"keypoints": {"same_person": False, "centroid_dist_median": 0.6, "frac_within": 0.0}}
    _patch_pose_one(monkeypatch, check)
    clip = ClipSpec("vidAAAAAAAA_5000-12000", "vidAAAAAAAA", 5.0, 12.0, 7)
    summary = pose_cli.pose_one(clip, {"video": "v.mp4"}, _pose_args(), use_gvhmr=True)
    assert summary["pose_method"] == "fallback_mediapipe"
    assert "GVHMR not used" in summary["notes"] and summary["gvhmr"]["fencer_check"] == check


def test_without_the_body_model_an_existing_clip_is_left_alone(data_dir, monkeypatch):
    calls = _patch_pose_one(monkeypatch, {})
    clip = ClipSpec("vidAAAAAAAA_5000-12000", "vidAAAAAAAA", 5.0, 12.0, 7)
    assert pose_cli.pose_one(clip, {}, _pose_args(), use_gvhmr=False) == FALLBACK_POSE
    assert calls == []


def test_gvhmr_preprocess_runs_phase_one_only_and_reuses_its_cache(data_dir, monkeypatch):
    check = {"boxes": {"frames": 210, "yolo_frac": 0.98}, "keypoints": {"same_person": True, "centroid_dist_median": 0.01}}
    calls = _patch_pose_one(monkeypatch, check)
    done = {"value": False}
    monkeypatch.setattr(pose_cli.gvhmr, "preprocess_done", lambda clip_id: done["value"])
    monkeypatch.setattr(pose_cli.gvhmr, "read_fencer_check", lambda clip_id: check)
    clip = ClipSpec("vidAAAAAAAA_5000-12000", "vidAAAAAAAA", 5.0, 12.0, 7)
    line = pose_cli.gvhmr_preprocess_one(clip, {"video": "v.mp4"}, _pose_args("--gvhmr-preprocess"))
    assert "210 frames" in line and "same person True" in line
    assert calls[0]["predict"] is False
    done["value"] = True
    pose_cli.gvhmr_preprocess_one(clip, {"video": "v.mp4"}, _pose_args("--gvhmr-preprocess"))
    assert len(calls) == 1
    assert json.loads((paths.POSE / clip.clip_id / "pose.json").read_text()) == FALLBACK_POSE


def test_gvhmr_preprocess_refuses_to_start_without_its_checkpoints(data_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(pose_cli.gvhmr, "missing_preprocess_checkpoints", lambda: [tmp_path / "yolov8x.pt"])
    assert pose_cli.main(["vidAAAAAAAA", "--existing", "--gvhmr-preprocess"]) == 1


def test_run_clip_drives_the_worker_on_the_fencer_track_and_collects_the_results(data_dir, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(gvhmr, "ensure_inputs_link", lambda: None)
    monkeypatch.setattr(gvhmr, "cut_clip_30fps", lambda video, s, e, out: out.parent.mkdir(parents=True, exist_ok=True) or out.write_bytes(b"x"))

    def fake_worker(venv, cmd, log, cwd=None, extra_env=None, extra_pythonpath=None):
        seen.update(venv=venv, cmd=cmd, cwd=cwd)
        gvhmr.output_dir("vidAAAAAAAA_5000-12000").mkdir(parents=True, exist_ok=True)
        gvhmr.results_file("vidAAAAAAAA_5000-12000").write_bytes(b"pt")
        (gvhmr.work_root("vidAAAAAAAA_5000-12000") / "fencer_check.json").write_text(json.dumps({"keypoints": {"same_person": True}}))
        return 1.0

    monkeypatch.setattr(gvhmr, "run_in_venv", fake_worker)
    clip = ClipSpec("vidAAAAAAAA_5000-12000", "vidAAAAAAAA", 5.0, 12.0, 7)
    check = gvhmr.run_clip(clip, tmp_path / "v.mp4", tmp_path / "log", predict=True)
    cmd = seen["cmd"]
    assert seen["venv"] == "gvhmr" and seen["cwd"] == gvhmr.GVHMR_ROOT
    assert cmd[cmd.index("--track-id") + 1] == "7" and cmd[cmd.index("--start") + 1] == "5.000"
    assert "--predict" in cmd and "--mediapipe-npz" in cmd and "--render" not in cmd
    assert cmd[cmd.index("--track-npz") + 1].endswith("vidAAAAAAAA/track.npz")
    assert (paths.POSE / clip.clip_id / "hmr4d_results.pt").read_bytes() == b"pt"
    assert check["keypoints"]["same_person"] is True


def test_gvhmr_needs_only_the_smplx_model_and_the_network_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(gvhmr, "SMPLX_MODEL", tmp_path / "SMPLX_NEUTRAL.npz")
    monkeypatch.setattr(gvhmr, "GVHMR_CHECKPOINT", tmp_path / "gvhmr_siga24_release.ckpt")
    assert not gvhmr.gvhmr_available()
    gvhmr.GVHMR_CHECKPOINT.write_bytes(b"")
    assert not gvhmr.gvhmr_available()
    gvhmr.SMPLX_MODEL.write_bytes(b"")
    assert gvhmr.gvhmr_available()  # SMPL_NEUTRAL.pkl is for the render videos only


def test_body_model_paths_are_the_ones_gvhmr_and_smplx_create_look_for():
    # GVHMR: inputs/checkpoints/body_models/{smplx/SMPLX_NEUTRAL.npz, smpl/SMPL_NEUTRAL.pkl};
    # smplx.create(BODY_MODELS, "smplx", gender="neutral") opens BODY_MODELS/smplx/SMPLX_NEUTRAL.npz.
    assert gvhmr.SMPLX_MODEL == gvhmr.BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"
    assert gvhmr.SMPL_MODEL == gvhmr.BODY_MODELS / "smpl" / "SMPL_NEUTRAL.pkl"
    assert gvhmr.BODY_MODELS == paths.DATA / "gvhmr" / "inputs" / "checkpoints" / "body_models"


def test_fencer_check_fails_only_when_it_ran_and_disagreed():
    assert gvhmr.fencer_check_failed({"keypoints": {"same_person": False}})
    assert not gvhmr.fencer_check_failed({"keypoints": {"same_person": None}})
    assert not gvhmr.fencer_check_failed({})


def test_fencer_check_fails_when_the_person_detector_rarely_backs_the_box():
    # A logo or graphic the screen tracked as a person: keypoints agree, YOLO never sees anyone.
    logo = {"boxes": {"yolo_frac": 0.0}, "keypoints": {"same_person": True}}
    assert gvhmr.fencer_check_failed(logo)
    assert "YOLO" in gvhmr.fencer_check_reason(logo)
    ok = {"boxes": {"yolo_frac": gvhmr.MIN_YOLO_FRAC}, "keypoints": {"same_person": True}}
    assert gvhmr.fencer_check_reason(ok) is None
    assert "MediaPipe" in gvhmr.fencer_check_reason({"keypoints": {"same_person": False, "centroid_dist_median": 0.3}})


# ---- GVHMR output -> GMR input (gmr_worker, no body model) ----------------------------------

from fencing.retarget import gmr_worker


def _gvhmr_params(n=12):
    rng = np.random.default_rng(1)
    return {
        "body_pose": rng.normal(0, 0.1, (n, 63)),
        "global_orient": rng.normal(0, 0.1, (n, 3)),
        "transl": rng.normal(0, 0.1, (n, 3)),
        "betas": np.tile(np.linspace(0.5, -0.4, 10), (n, 1)) + rng.normal(0, 0.01, (n, 10)),
    }


def test_gvhmr_params_become_gmr_smplx_data_with_unpadded_mean_betas():
    raw = _gvhmr_params()
    data = gmr_worker.gvhmr_smplx_params(raw)
    assert data["pose_body"].shape == (12, 63) and data["root_orient"].shape == (12, 3) and data["trans"].shape == (12, 3)
    # As many betas as GVHMR wrote (10): the model is built with num_betas=len(betas). GMR's
    # loader pads to 16 against a 10-beta model, which fails inside smplx's lbs().
    assert data["betas"].shape == (10,)
    assert np.allclose(data["betas"], raw["betas"].mean(axis=0))
    assert data["mocap_frame_rate"].item() == 30.0


def test_gvhmr_params_with_a_wrong_shape_or_key_are_refused():
    raw = _gvhmr_params()
    with pytest.raises(ValueError, match="body_pose"):
        gmr_worker.gvhmr_smplx_params({**raw, "body_pose": raw["body_pose"][:, :60]})
    with pytest.raises(KeyError, match="transl"):
        gmr_worker.gvhmr_smplx_params({k: v for k, v in raw.items() if k != "transl"})


def test_height_from_betas_is_gmrs_rule():
    assert gmr_worker.height_from_betas(np.array([1.0] + [0.0] * 9)) == pytest.approx(1.76)


def test_ground_frames_puts_the_floor_at_zero_and_the_pelvis_at_the_origin():
    rng = np.random.default_rng(2)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    frames = []
    for t in range(60):
        lift = 0.2 if 20 <= t < 30 else 0.0  # left foot steps
        frames.append({
            "pelvis": (np.array([3.0 + 0.01 * t, -2.0, 1.85]), q),
            "left_foot": (np.array([3.0, -2.1, 0.93 + lift + rng.normal(0, 0.002)]), q),
            "right_foot": (np.array([3.3, -1.9, 0.93 + rng.normal(0, 0.002)]), q),
        })  # fmt: skip
    grounded, shift = gmr_worker.ground_frames(frames)
    feet = np.array([[f[j][0][2] for j in gmr_worker.FOOT_JOINTS] for f in grounded])
    assert np.median(feet.min(axis=1)) == pytest.approx(gmr_worker.BALL_OF_FOOT_HEIGHT_M, abs=0.005)
    assert np.allclose(grounded[0]["pelvis"][0][:2], 0.0)
    assert shift.shape == (60, 3)
    assert grounded[0]["pelvis"][0][2] == pytest.approx(1.85 - shift[0, 2])
    assert grounded[5]["left_foot"][1] is q  # orientations untouched


def _stepping_frames(n, height_at, jump=range(0)):
    """Feet alternate steps (one always down) while the whole body sits at height_at(t)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    frames = []
    for t in range(n):
        base = height_at(t)
        up = 0.15 if t in jump else 0.0
        left = 0.93 + (0.1 if (t // 15) % 2 else 0.0) + up
        right = 0.93 + (0.0 if (t // 15) % 2 else 0.1) + up
        frames.append({
            "pelvis": (np.array([0.01 * t, 0.0, 1.85 + base + up]), q),
            "left_foot": (np.array([0.0, 0.1, left + base]), q),
            "right_foot": (np.array([0.3, -0.1, right + base]), q),
        })  # fmt: skip
    return frames


def test_ground_frames_follows_a_drifting_floor_so_no_stretch_of_the_clip_floats():
    # GVHMR's global height drifts 0.14-0.29 m over a broadcast clip (measured on six clips,
    # 2026-11-12); a single floor left the G1's planted sole 7-21 cm up for most of a clip.
    # 0.3 m over 10 s is the fastest of those drifts (the pilot's 0.2 m in 7.6 s).
    frames = _stepping_frames(300, lambda t: 0.3 * t / 299)
    grounded, _ = gmr_worker.ground_frames(frames)
    lower = np.array([[f[j][0][2] for j in gmr_worker.FOOT_JOINTS] for f in grounded]).min(axis=1)
    assert np.abs(lower - gmr_worker.BALL_OF_FOOT_HEIGHT_M).max() < 0.02


def test_ground_frames_keeps_a_short_jump_off_the_floor():
    frames = _stepping_frames(150, lambda t: 0.0, jump=range(70, 79))
    grounded, _ = gmr_worker.ground_frames(frames)
    lower = np.array([[f[j][0][2] for j in gmr_worker.FOOT_JOINTS] for f in grounded]).min(axis=1)
    assert lower[72:77].min() > gmr_worker.BALL_OF_FOOT_HEIGHT_M + 0.1
    assert np.abs(lower[:60] - gmr_worker.BALL_OF_FOOT_HEIGHT_M).max() < 0.01


# ---- `fencing retarget`: upgrading a clip to GVHMR -------------------------------------------

from fencing.retarget import cli as retarget_cli

GVHMR_POSE = {
    "clip_id": "vidAAAAAAAA_5000-12000", "yt_id": "vidAAAAAAAA", "source_start_s": 5.0, "source_end_s": 12.0,
    "pose_method": "gvhmr", "track_id": 7, "human_height_m": 1.75, "handedness_guess": "L", "notes": "",
}  # fmt: skip


@pytest.fixture
def retarget_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "POSE", tmp_path / "pose")
    monkeypatch.setattr(paths, "RETARGET", tmp_path / "retarget")
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(retarget_cli, "CSV_DIR", tmp_path / "retarget" / "csv")
    monkeypatch.setattr(gvhmr, "SMPLX_MODEL", tmp_path / "SMPLX_NEUTRAL.npz")
    pose = tmp_path / "pose" / GVHMR_POSE["clip_id"]
    pose.mkdir(parents=True)
    (pose / "pose.json").write_text(json.dumps(GVHMR_POSE))
    csv_dir = tmp_path / "retarget" / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / f"{GVHMR_POSE['clip_id']}.csv").write_text("old fallback csv\n")
    (csv_dir / f"{GVHMR_POSE['clip_id']}.json").write_text(json.dumps({**GVHMR_POSE, "pose_method": "fallback_mediapipe"}))
    calls = []

    def fake_worker(venv, cmd, log, extra_env=None):
        calls.append(cmd)
        out_csv = cmd[cmd.index("--out-csv") + 1]
        (tmp_path / "retarget").joinpath("csv", out_csv.rsplit("/", 1)[-1]).write_text("new gvhmr csv\n")
        stats = cmd[cmd.index("--out-stats") + 1]
        with open(stats, "w") as fh:
            json.dump({"frames": 210, "human_height_m": 1.71}, fh)
        return 1.0

    monkeypatch.setattr(retarget_cli, "run_in_venv", fake_worker)
    return SimpleNamespace(root=tmp_path, pose=pose, csv_dir=csv_dir, calls=calls)


def _retarget_args(*extra):
    return SimpleNamespace(method="auto", height=None, force="--force" in extra)


def test_a_clip_upgraded_to_gvhmr_is_retargeted_again_and_the_old_csv_kept(retarget_dir):
    (retarget_dir.pose / "hmr4d_results.pt").write_bytes(b"pt")
    gvhmr.SMPLX_MODEL.write_bytes(b"")
    sidecar = retarget_cli.retarget_one(GVHMR_POSE["clip_id"], _retarget_args())
    cmd = retarget_dir.calls[0]
    assert cmd[cmd.index("--method") + 1] == "gvhmr"
    assert cmd[cmd.index("--body-models") + 1] == str(gvhmr.BODY_MODELS) and "--height" not in cmd
    assert sidecar["pose_method"] == "gvhmr" and sidecar["human_height_m"] == 1.71 and sidecar["handedness_guess"] == "L"
    archived = retarget_dir.csv_dir / "previous" / f"{GVHMR_POSE['clip_id']}.fallback_mediapipe.csv"
    assert archived.read_text() == "old fallback csv\n"
    assert (retarget_dir.csv_dir / f"{GVHMR_POSE['clip_id']}.csv").read_text() == "new gvhmr csv\n"
    # Now current: skipped without --force.
    retarget_cli.retarget_one(GVHMR_POSE["clip_id"], _retarget_args())
    assert len(retarget_dir.calls) == 1


def test_gvhmr_retarget_without_its_inputs_fails_before_touching_the_old_csv(retarget_dir):
    with pytest.raises(FileNotFoundError, match="hmr4d_results.pt"):
        retarget_cli.retarget_one(GVHMR_POSE["clip_id"], _retarget_args())
    assert not (retarget_dir.csv_dir / "previous").exists() and retarget_dir.calls == []
    assert (retarget_dir.csv_dir / f"{GVHMR_POSE['clip_id']}.csv").read_text() == "old fallback csv\n"


def test_a_second_overwrite_keeps_both_backups(retarget_dir):
    (retarget_dir.pose / "hmr4d_results.pt").write_bytes(b"pt")
    gvhmr.SMPLX_MODEL.write_bytes(b"")
    clip = GVHMR_POSE["clip_id"]
    retarget_cli.retarget_one(clip, _retarget_args())  # fallback -> gvhmr
    retarget_cli.retarget_one(clip, _retarget_args("--force"))  # gvhmr -> gvhmr
    previous = retarget_dir.csv_dir / "previous"
    assert (previous / f"{clip}.fallback_mediapipe.csv").read_text() == "old fallback csv\n"
    assert (previous / f"{clip}.gvhmr.csv").read_text() == "new gvhmr csv\n"
    retarget_cli.retarget_one(clip, _retarget_args("--force"))  # a second gvhmr backup must not replace the first
    assert sorted(p.name for p in previous.glob("*.csv")) == [
        f"{clip}.fallback_mediapipe.csv", f"{clip}.gvhmr.1.csv", f"{clip}.gvhmr.csv",
    ]  # fmt: skip
    assert (previous / f"{clip}.gvhmr.1.json").exists()


def test_forced_gvhmr_on_a_fallback_clip_leaves_a_backup_and_a_gvhmr_sidecar(retarget_dir):
    clip = GVHMR_POSE["clip_id"]
    (retarget_dir.pose / "pose.json").write_text(json.dumps({**GVHMR_POSE, "pose_method": "fallback_mediapipe"}))
    (retarget_dir.csv_dir / f"{clip}.json").write_text(json.dumps({**GVHMR_POSE, "pose_method": "fallback_mediapipe"}))
    (retarget_dir.pose / "hmr4d_results.pt").write_bytes(b"pt")
    gvhmr.SMPLX_MODEL.write_bytes(b"")
    args = SimpleNamespace(method="gvhmr", height=None, force=True)
    sidecar = retarget_cli.retarget_one(clip, args)
    assert (retarget_dir.csv_dir / "previous" / f"{clip}.fallback_mediapipe.csv").read_text() == "old fallback csv\n"
    assert sidecar["pose_method"] == "gvhmr"
    assert json.loads((retarget_dir.csv_dir / f"{clip}.json").read_text())["pose_method"] == "gvhmr"


def test_an_existing_csv_without_a_sidecar_is_backed_up_as_unknown(retarget_dir):
    clip = GVHMR_POSE["clip_id"]
    (retarget_dir.csv_dir / f"{clip}.json").unlink()
    (retarget_dir.pose / "hmr4d_results.pt").write_bytes(b"pt")
    gvhmr.SMPLX_MODEL.write_bytes(b"")
    retarget_cli.retarget_one(clip, _retarget_args())
    assert (retarget_dir.csv_dir / "previous" / f"{clip}.unknown.csv").read_text() == "old fallback csv\n"


# ---- GVHMR worker: preprocessing cache key; wrapper: a render crash after the results ----------

import os
import time

from fencing.pose import gvhmr_worker


def _guide(offset=0.0, n=8):
    return np.tile(np.array([10.0, 20.0, 110.0, 220.0]), (n, 1)) + offset


def _worker_args(tmp_path):
    return SimpleNamespace(track_npz=tmp_path / "track.npz", track_id=3, start=12.0)


def test_same_track_path_and_id_with_different_boxes_gives_a_different_key(tmp_path):
    args = _worker_args(tmp_path)
    assert gvhmr_worker.fencer_key(args, 8, _guide()) == gvhmr_worker.fencer_key(args, 8, _guide())
    assert gvhmr_worker.fencer_key(args, 8, _guide()) != gvhmr_worker.fencer_key(args, 8, _guide(0.5))
    assert gvhmr_worker.guide_digest(_guide(n=8)) != gvhmr_worker.guide_digest(_guide(n=4))  # shape counts


def _write_cache(tmp_path, key, guide):
    pre = tmp_path / "pre"
    pre.mkdir(exist_ok=True)
    (pre / "bbx.pt").write_bytes(b"x")
    (pre / "fencer_boxes.json").write_text(json.dumps({"key": key, "yolo_frac": 0.9}))
    np.savez_compressed(pre / "fencer_boxes.npz", boxes=guide, pipeline=guide)
    return pre / "fencer_boxes.json", pre / "bbx.pt", pre / "fencer_boxes.npz"


def test_a_cache_made_for_other_track_contents_is_not_reused(tmp_path):
    args = _worker_args(tmp_path)
    meta_path, bbx, npz = _write_cache(tmp_path, gvhmr_worker.fencer_key(args, 8, _guide()), _guide())
    assert gvhmr_worker.cached_boxes_meta(meta_path, bbx, gvhmr_worker.fencer_key(args, 8, _guide()), npz)["yolo_frac"] == 0.9
    assert gvhmr_worker.cached_boxes_meta(meta_path, bbx, gvhmr_worker.fencer_key(args, 8, _guide(0.5)), npz) is None


def test_an_old_format_cache_stays_valid_and_is_upgraded_in_place_when_its_guide_matches(tmp_path):
    args = _worker_args(tmp_path)
    key = gvhmr_worker.fencer_key(args, 8, _guide())
    old_key = {k: v for k, v in key.items() if k != "guide_sha256"}
    meta_path, bbx, npz = _write_cache(tmp_path, old_key, _guide())
    assert gvhmr_worker.cached_boxes_meta(meta_path, bbx, key, npz)["yolo_frac"] == 0.9
    assert json.loads(meta_path.read_text())["key"] == key
    # An old-format cache whose stored guide differs from the track now is stale.
    meta_path, bbx, npz = _write_cache(tmp_path, old_key, _guide(0.5))
    assert gvhmr_worker.cached_boxes_meta(meta_path, bbx, key, npz) is None
    # Other key fields still matter.
    meta_path, bbx, npz = _write_cache(tmp_path, {**old_key, "track_id": 4}, _guide())
    assert gvhmr_worker.cached_boxes_meta(meta_path, bbx, key, npz) is None


def _render_crash_setup(data_dir, monkeypatch, tmp_path, results_age_s):
    clip_id = "vidAAAAAAAA_5000-12000"
    monkeypatch.setattr(gvhmr, "ensure_inputs_link", lambda: None)
    monkeypatch.setattr(gvhmr, "cut_clip_30fps", lambda video, s, e, out: out.parent.mkdir(parents=True, exist_ok=True) or out.write_bytes(b"x"))

    def dying_worker(venv, cmd, log, cwd=None, extra_env=None, extra_pythonpath=None):
        gvhmr.output_dir(clip_id).mkdir(parents=True, exist_ok=True)
        results = gvhmr.results_file(clip_id)
        results.write_bytes(b"pt")
        stamp = time.time() - results_age_s
        os.utime(results, (stamp, stamp))
        (gvhmr.work_root(clip_id) / "fencer_check.json").write_text(json.dumps({"keypoints": {"same_person": True}}))
        raise WorkerError("gvhmr worker failed (exit -9); log x:\nKilled")

    monkeypatch.setattr(gvhmr, "run_in_venv", dying_worker)
    return ClipSpec(clip_id, "vidAAAAAAAA", 5.0, 12.0, 7)


from fencing.pose.venvs import WorkerError


def test_a_worker_killed_while_rendering_after_the_results_were_saved_still_succeeds(data_dir, monkeypatch, tmp_path, capsys):
    clip = _render_crash_setup(data_dir, monkeypatch, tmp_path, results_age_s=0)
    check = gvhmr.run_clip(clip, tmp_path / "v.mp4", tmp_path / "log", predict=True, render=True)
    assert check["keypoints"]["same_person"] is True
    assert (paths.POSE / clip.clip_id / "hmr4d_results.pt").read_bytes() == b"pt"
    assert "rendering failed" in capsys.readouterr().err


def test_a_worker_failure_is_still_an_error_without_a_fresh_result(data_dir, monkeypatch, tmp_path):
    clip = _render_crash_setup(data_dir, monkeypatch, tmp_path, results_age_s=3600)  # left by an earlier run
    with pytest.raises(WorkerError):
        gvhmr.run_clip(clip, tmp_path / "v.mp4", tmp_path / "log", predict=True, render=True)
    clip = _render_crash_setup(data_dir, monkeypatch, tmp_path, results_age_s=0)
    with pytest.raises(WorkerError):  # no render requested: a crash is not a render failure
        gvhmr.run_clip(clip, tmp_path / "v.mp4", tmp_path / "log", predict=True)
