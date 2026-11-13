"""Which person GVHMR sees: the pipeline's fencer, not the largest person. Pure numpy.

Used by `fencing.pose.gvhmr_worker` (gvhmr venv) and by the tests (fencing venv).

GVHMR's demo picks the person with the largest summed box area from its own YOLOv8x tracker.
In a fencing clip that can be the other fencer or the referee. `fencing screen` already chose
the fencer track (Keypoint R-CNN boxes at <= 10 fps, see fencing.pose.detect_worker), and
MediaPipe ran on that track. Here, per clip frame (30 fps):

    1. the pipeline box: the fencer track interpolated to the frame's source time;
    2. the YOLO detection that overlaps it most (IoU >= MIN_MATCH_IOU) is the fencer's box:
       YOLO runs on every frame, so its boxes follow a lunge where the 10 fps track lags;
    3. short runs with no match are bridged between the YOLO boxes on either side, longer ones
       use the pipeline box;
    4. the same double 5-frame moving average GVHMR applies to its own track.

`keypoint_agreement` is the evidence that GVHMR's 2D keypoints (ViTPose) land on the same
person MediaPipe tracked: the distance between the two skeletons' limb centroids, in units of
the fencer box height.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from fencing.pose.tracking import box_iou

GVHMR_FPS = 30.0
MIN_MATCH_IOU = 0.3
MAX_BRIDGE_FRAMES = 15  # 0.5 s: longer gaps fall back to the pipeline box
SMOOTH_WINDOW = 5
SOURCE_YOLO, SOURCE_BRIDGED, SOURCE_PIPELINE = 0, 1, 2

# Limb joints present in both skeletons: shoulders, elbows, wrists, hips, knees, ankles.
COCO17_LIMBS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MEDIAPIPE_LIMBS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
VITPOSE_MIN_CONF = 0.5
# Same person: limb centroids within this many box heights. Two fencers side by side in the
# image are at least a body width (~0.25 box heights) apart, usually much more.
SAME_PERSON_MAX_DIST = 0.15
SAME_PERSON_MIN_FRAC = 0.9


def clip_frame_times(start_s: float, n_frames: int, fps: float = GVHMR_FPS) -> np.ndarray:
    """Source-video time of each frame of a clip cut at `fps` from `start_s`."""
    return start_s + np.arange(n_frames) / fps


def moving_average(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centred moving average along axis 0 with edge replication (GVHMR's moving_average_smooth)."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    rad = window // 2
    padded = np.concatenate([np.repeat(x[:1], rad, axis=0), x, np.repeat(x[-1:], rad, axis=0)])
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(x.shape[1])], axis=1)


def select_fencer_boxes(
    det_frame: np.ndarray,
    det_box: np.ndarray,
    pipeline_boxes: np.ndarray,
    min_iou: float = MIN_MATCH_IOU,
    max_bridge: int = MAX_BRIDGE_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame fencer box (L, 4) xyxy, its source (L,) SOURCE_*, and the matched IoU (L,).

    det_frame (N,) clip frame index of each detection, det_box (N, 4) xyxy, pipeline_boxes
    (L, 4) the screen track's fencer box at each clip frame.
    """
    pipeline_boxes = np.asarray(pipeline_boxes, dtype=np.float64)
    det_frame = np.asarray(det_frame, dtype=np.int64)
    det_box = np.asarray(det_box, dtype=np.float64).reshape(-1, 4)
    n = len(pipeline_boxes)
    boxes = pipeline_boxes.copy()
    source = np.full(n, SOURCE_PIPELINE, dtype=np.int8)
    matched_iou = np.zeros(n)
    for f in range(n):
        rows = np.flatnonzero(det_frame == f)
        if not len(rows):
            continue
        iou = box_iou(pipeline_boxes[f], det_box[rows])[0]
        best = int(np.argmax(iou))
        if iou[best] >= min_iou:
            boxes[f] = det_box[rows[best]]
            source[f] = SOURCE_YOLO
            matched_iou[f] = iou[best]

    matched = np.flatnonzero(source == SOURCE_YOLO)
    for a, b in pairwise(matched):
        if 1 < b - a <= max_bridge + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            boxes[a + 1 : b] = (1 - w)[:, None] * boxes[a] + w[:, None] * boxes[b]
            source[a + 1 : b] = SOURCE_BRIDGED
    return moving_average(moving_average(boxes)), source, matched_iou


def _nearest(times: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(times, queries), 1, len(times) - 1)
    left_closer = np.abs(queries - times[idx - 1]) <= np.abs(times[idx] - queries)
    return np.where(left_closer, idx - 1, idx)


def keypoint_agreement(
    vitpose: np.ndarray,
    clip_times: np.ndarray,
    mp_times: np.ndarray,
    mp_image_xy: np.ndarray,
    mp_valid: np.ndarray,
    boxes: np.ndarray,
    max_dt: float = 0.5 / GVHMR_FPS + 1e-3,
) -> dict:
    """How well GVHMR's ViTPose keypoints (L, 17, 3: x, y, conf) match MediaPipe's landmarks
    (M, 33, 2 at mp_times) of the pipeline's fencer. Limb centroids are used, so a left/right
    swap in either skeleton does not count as a different person."""
    vitpose = np.asarray(vitpose, dtype=np.float64)
    mp_times = np.asarray(mp_times, dtype=np.float64)
    mp_image_xy = np.asarray(mp_image_xy, dtype=np.float64)
    nearest = _nearest(mp_times, np.asarray(clip_times, dtype=np.float64))
    dists = []
    for f, j in enumerate(nearest):
        if abs(mp_times[j] - clip_times[f]) > max_dt or not mp_valid[j]:
            continue
        vit = vitpose[f, COCO17_LIMBS]
        confident = vit[:, 2] >= VITPOSE_MIN_CONF
        mp = mp_image_xy[j, MEDIAPIPE_LIMBS]
        if confident.sum() < 4 or not np.isfinite(mp).all():
            continue
        height = max(boxes[f, 3] - boxes[f, 1], 1.0)
        dists.append(np.linalg.norm(vit[confident, :2].mean(axis=0) - mp.mean(axis=0)) / height)
    if not dists:
        return {"frames_compared": 0, "same_person": None}
    d = np.asarray(dists)
    within = float((d <= SAME_PERSON_MAX_DIST).mean())
    return {
        "frames_compared": len(d),
        "centroid_dist_median": round(float(np.median(d)), 4),
        "centroid_dist_p95": round(float(np.percentile(d, 95)), 4),
        "frac_within": round(within, 4),
        "same_person": bool(within >= SAME_PERSON_MIN_FRAC),
    }


def box_summary(source: np.ndarray, matched_iou: np.ndarray) -> dict:
    """Where the fencer boxes came from, for fencer_check.json."""
    n = max(len(source), 1)
    yolo = source == SOURCE_YOLO
    return {
        "frames": len(source),
        "yolo_frac": round(float(yolo.mean()) if len(source) else 0.0, 4),
        "bridged_frac": round(float((source == SOURCE_BRIDGED).sum() / n), 4),
        "pipeline_frac": round(float((source == SOURCE_PIPELINE).sum() / n), 4),
        "matched_iou_median": round(float(np.median(matched_iou[yolo])), 4) if yolo.any() else None,
    }
