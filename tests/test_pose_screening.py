"""Tracking, fencer selection and screening scores on synthetic detections."""

from __future__ import annotations

import numpy as np

from fencing.pose.screening import (
    COCO_LEFT_ANKLE,
    COCO_LEFT_HIP,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_ANKLE,
    COCO_RIGHT_HIP,
    COCO_RIGHT_SHOULDER,
    Detections,
    classify_view,
    screen_video,
    shot_cuts,
    track_from_arrays,
    tracks_to_arrays,
)
from fencing.pose.tracking import box_iou, interpolate_track_boxes, square_crop_box, track_detections

WIDTH, HEIGHT = 1280, 720


def person(x_center: float, height_px: float, shoulder_px: float, ankles_visible: bool = True):
    box = np.array([x_center - 0.2 * height_px, 600 - height_px, x_center + 0.2 * height_px, 600.0])
    kps = np.zeros((17, 3))
    kps[:, 2] = 5.0
    kps[COCO_LEFT_SHOULDER, :2] = [x_center + shoulder_px / 2, 600 - 0.8 * height_px]
    kps[COCO_RIGHT_SHOULDER, :2] = [x_center - shoulder_px / 2, 600 - 0.8 * height_px]
    kps[COCO_LEFT_HIP, :2] = [x_center + 0.05 * height_px, 600 - 0.5 * height_px]
    kps[COCO_RIGHT_HIP, :2] = [x_center - 0.05 * height_px, 600 - 0.5 * height_px]
    kps[[COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE], :2] = [x_center, 595]
    if not ankles_visible:
        kps[[COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE], 2] = -1.0
    return box, kps


def make_detections(num_frames=90, cut_at=None, pan=False, fencer_ankles=True) -> Detections:
    rows = []
    for f in range(num_frames):
        # Fencer: big, central, moving right 3 px/frame, side-on (narrow shoulders).
        rows.append((f, *person(500 + 3 * f, 400, 20, fencer_ankles), 0.99))
        # Coach: small, at the edge, standing still, facing the camera.
        rows.append((f, *person(1150, 200, 60), 0.95))
    hist = np.zeros(num_frames)
    if cut_at is not None:
        hist[cut_at] = 0.9
    motion = np.zeros((num_frames, 2))
    if pan:
        motion[:, 0] = 6.0
    return Detections(
        frame_times=np.arange(num_frames) / 30.0,
        width=WIDTH,
        height=HEIGHT,
        hist_diff=hist,
        cam_motion=motion,
        det_frame=np.array([r[0] for r in rows]),
        det_box=np.stack([r[1] for r in rows]),
        det_score=np.array([r[3] for r in rows]),
        det_kps=np.stack([r[2] for r in rows]),
    )


def test_box_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    assert np.allclose(box_iou(a, b), [[1.0, 0.0, 1 / 3]])


def test_tracker_keeps_two_people_apart_and_bridges_a_dropout():
    det = make_detections(60)
    keep = ~((det.det_frame == 30) & (det.det_box[:, 0] < 1000))  # fencer missed in frame 30
    tracks = track_detections(det.det_frame[keep], det.det_box[keep], det.det_score[keep], 60)
    assert len(tracks) == 2
    assert sorted(t.length for t in tracks) == [59, 60]


def test_tracks_end_at_shot_cuts():
    det = make_detections(60)
    tracks = track_detections(det.det_frame, det.det_box, det.det_score, 60, cut_frames={30})
    assert len(tracks) == 4


def test_screen_picks_the_large_central_fencer_and_reports_side_view():
    report, _ = screen_video(make_detections())
    fencer = report["tracks"][0]
    assert report["chosen_track_id"] == fencer["track_id"]
    assert fencer["area_frac"] > report["tracks"][1]["area_frac"]
    assert report["people_count_median"] == 2
    assert report["camera"] == "static"
    assert report["view"] == "side"
    assert report["usable_ranges"] == [[0.0, round(89 / 30, 3)]]


def test_screen_flags_panning_camera_as_unusable():
    report, _ = screen_video(make_detections(pan=True))
    assert report["camera"] == "moving"
    assert report["ranges"] and not report["ranges"][0]["usable"]
    assert report["usable_ranges"] == []


def test_screen_needs_feet_visible_for_a_usable_range():
    report, _ = screen_video(make_detections(fencer_ankles=False))
    assert report["usable_ranges"] == []


def test_screen_splits_ranges_at_shot_cuts():
    report, _ = screen_video(make_detections(num_frames=120, cut_at=60))
    assert len(report["usable_ranges"]) == 2
    assert report["shot_cuts_s"] == [2.0]


def test_track_arrays_round_trip():
    _, tracks = screen_video(make_detections(60))
    arrays = tracks_to_arrays(tracks)
    restored = track_from_arrays(arrays, tracks[0].track_id)
    assert restored.frames == tracks[0].frames
    assert np.allclose(np.stack(restored.boxes), np.stack(tracks[0].boxes))


def test_interpolated_boxes_fill_gaps_linearly():
    det = make_detections(10)
    keep = det.det_box[:, 0] < 1000
    tracks = track_detections(det.det_frame[keep], det.det_box[keep], det.det_score[keep], 10)
    boxes = interpolate_track_boxes(tracks[0], np.array([0.0, 4.5, 9.0]))
    assert np.isclose(boxes[1, 0], (boxes[0, 0] + boxes[2, 0]) / 2)


def test_square_crop_is_square_and_centered():
    crop = square_crop_box(np.array([100, 100, 200, 400]), 1.2, WIDTH, HEIGHT)
    assert np.isclose(crop[2] - crop[0], crop[3] - crop[1])
    assert np.isclose(crop[2] - crop[0], 360)
    assert np.allclose([(crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2], [150, 250])


def test_shot_cuts_and_view_classes():
    assert shot_cuts(np.array([0.0, 0.1, 0.9, 0.2])) == [2]
    assert classify_view(0.1) == "side"
    assert classify_view(0.9) == "front"
    assert classify_view(0.45) == "oblique"
    assert classify_view(float("nan")) == "unknown"
