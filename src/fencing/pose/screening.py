"""Turn raw detections of one video into the screen report: shots, the fencer track, quality
scores and usable time ranges. Pure numpy; the detections come from detect_worker.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fencing.pose.tracking import Track, track_detections

# COCO-17 keypoint indices (Keypoint R-CNN output order).
COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER = 5, 6
COCO_LEFT_HIP, COCO_RIGHT_HIP = 11, 12
COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE = 15, 16

# Keypoint R-CNN keypoint scores are heatmap logits; ~2 separates visible joints from guesses
# on cropped or occluded limbs in our clips.
KEYPOINT_VISIBLE_SCORE = 2.0
SHOT_CUT_BHATTACHARYYA = 0.45
EDGE_MARGIN_FRACTION = 0.005  # a box this close to the image edge is cut off
MIN_TRACK_SECONDS = 1.0
MIN_RANGE_SECONDS = 1.5
MAX_GAP_SECONDS = 0.35  # bridge short detector dropouts inside a usable range
STATIC_CAMERA_MAX_WIDTH_PER_S = 0.015  # background drift as a fraction of image width per second
MIN_FULL_BODY_FRACTION = 0.8
SIDE_VIEW_MAX_RATIO = 0.35  # shoulder width / torso height in the image
FRONT_VIEW_MIN_RATIO = 0.6
PEOPLE_SCORE = 0.8


@dataclass
class Detections:
    frame_times: np.ndarray
    width: int
    height: int
    hist_diff: np.ndarray
    cam_motion: np.ndarray  # (F, 2) px/frame, full resolution, NaN when unmeasurable
    det_frame: np.ndarray
    det_box: np.ndarray
    det_score: np.ndarray
    det_kps: np.ndarray  # (N, 17, 3) x, y, score

    @classmethod
    def from_npz(cls, data) -> Detections:
        return cls(
            frame_times=np.asarray(data["frame_times"], dtype=np.float64),
            width=int(data["width"]),
            height=int(data["height"]),
            hist_diff=np.asarray(data["hist_diff"], dtype=np.float64),
            cam_motion=np.asarray(data["cam_motion"], dtype=np.float64),
            det_frame=np.asarray(data["det_frame"], dtype=np.int64),
            det_box=np.asarray(data["det_box"], dtype=np.float64),
            det_score=np.asarray(data["det_score"], dtype=np.float64),
            det_kps=np.asarray(data["det_kps"], dtype=np.float64),
        )

    @property
    def num_frames(self) -> int:
        return len(self.frame_times)

    @property
    def fps(self) -> float:
        if self.num_frames < 2:
            return 30.0
        return float((self.num_frames - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-6))


def shot_cuts(hist_diff: np.ndarray, threshold: float = SHOT_CUT_BHATTACHARYYA) -> list[int]:
    """Frames that start a new shot (hard cuts, title cards)."""
    return [int(i) for i in np.flatnonzero(hist_diff > threshold) if i > 0]


def shot_spans(num_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    edges = [0] + sorted(c for c in cuts if 0 < c < num_frames) + [num_frames]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def full_body_visible(det: Detections, det_index: int) -> bool:
    """Both ankles confidently seen and the box not cut by the image border."""
    kps = det.det_kps[det_index]
    if min(kps[COCO_LEFT_ANKLE, 2], kps[COCO_RIGHT_ANKLE, 2]) < KEYPOINT_VISIBLE_SCORE:
        return False
    x0, y0, x1, y1 = det.det_box[det_index]
    mx, my = EDGE_MARGIN_FRACTION * det.width, EDGE_MARGIN_FRACTION * det.height
    return x0 > mx and y0 > my and x1 < det.width - mx and y1 < det.height - my


def view_ratio(det: Detections, det_index: int) -> float:
    """Shoulder width over torso height in the image: ~0.8 facing the camera, ~0.1 side-on."""
    kps = det.det_kps[det_index]
    used = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
    if kps[used, 2].min() < KEYPOINT_VISIBLE_SCORE:
        return float("nan")
    shoulder_width = np.linalg.norm(kps[COCO_LEFT_SHOULDER, :2] - kps[COCO_RIGHT_SHOULDER, :2])
    mid_sh = (kps[COCO_LEFT_SHOULDER, :2] + kps[COCO_RIGHT_SHOULDER, :2]) / 2
    mid_hip = (kps[COCO_LEFT_HIP, :2] + kps[COCO_RIGHT_HIP, :2]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip)
    return float(shoulder_width / torso) if torso > 1e-6 else float("nan")


def classify_view(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unknown"
    if ratio < SIDE_VIEW_MAX_RATIO:
        return "side"
    if ratio > FRONT_VIEW_MIN_RATIO:
        return "front"
    return "oblique"


def camera_speed_width_per_s(det: Detections, frames: np.ndarray) -> float:
    """Median background motion over `frames`, as a fraction of image width per second."""
    motion = det.cam_motion[frames]
    magnitude = np.linalg.norm(motion, axis=1)
    magnitude = magnitude[np.isfinite(magnitude)]
    if magnitude.size == 0:
        return float("nan")
    return float(np.median(magnitude) * det.fps / det.width)


def track_score(det: Detections, track: Track) -> dict:
    """The fencer is the long, large, central, fully visible track."""
    boxes = np.stack(track.boxes)
    area = np.median((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (det.width * det.height)
    cx = np.median((boxes[:, 0] + boxes[:, 2]) / 2)
    centrality = 1.0 - min(abs(cx - det.width / 2) / (det.width / 2), 1.0)
    full = float(np.mean([full_body_visible(det, i) for i in track.det_indices]))
    score = track.length * np.sqrt(area) * (0.5 + full) * (0.5 + 0.5 * centrality)
    return {
        "track_id": track.track_id,
        "start_s": round(float(det.frame_times[track.frames[0]]), 3),
        "end_s": round(float(det.frame_times[track.frames[-1]]), 3),
        "frames": track.length,
        "area_frac": round(float(area), 4),
        "centrality": round(float(centrality), 3),
        "full_body_frac": round(full, 3),
        "score": round(float(score), 3),
    }


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Half-open runs of True, with gaps of up to `max_gap` False frames bridged."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((int(start), int(prev) + 1))
            start = i
        prev = i
    runs.append((int(start), int(prev) + 1))
    return runs


def people_counts(det: Detections) -> np.ndarray:
    counts = np.zeros(det.num_frames, dtype=np.int64)
    confident = det.det_score >= PEOPLE_SCORE
    np.add.at(counts, det.det_frame[confident], 1)
    return counts


def _range_report(det: Detections, track: Track, start: int, end: int, counts: np.ndarray) -> dict:
    frame_to_det = dict(zip(track.frames, track.det_indices, strict=True))
    frames = np.arange(start, end)
    dets = [frame_to_det[f] for f in frames if f in frame_to_det]
    full = float(np.mean([full_body_visible(det, i) for i in dets])) if dets else 0.0
    ratios = np.array([view_ratio(det, i) for i in dets])
    ratio = float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else float("nan")
    speed = camera_speed_width_per_s(det, frames)
    camera = "static" if np.isfinite(speed) and speed <= STATIC_CAMERA_MAX_WIDTH_PER_S else "moving"
    reasons = []
    if camera != "static":
        reasons.append(f"camera moving ({speed:.3f} image widths/s)")
    if full < MIN_FULL_BODY_FRACTION:
        reasons.append(f"full body visible in only {full:.0%} of frames")
    last = min(end, det.num_frames - 1)
    return {
        "start_s": round(float(det.frame_times[start]), 3),
        "end_s": round(float(det.frame_times[last]), 3),
        "track_id": track.track_id,
        "full_body_frac": round(full, 3),
        "camera_motion_width_per_s": None if not np.isfinite(speed) else round(speed, 4),
        "camera": camera,
        "view_ratio": None if not np.isfinite(ratio) else round(ratio, 3),
        "view": classify_view(ratio),
        "people_median": int(np.median(counts[start:end])),
        "usable": not reasons,
        "reasons": reasons,
    }


def screen_video(det: Detections) -> tuple[dict, list[Track]]:
    """Screen report (JSON-ready) and the tracks it refers to."""
    cuts = shot_cuts(det.hist_diff)
    tracks = track_detections(det.det_frame, det.det_box, det.det_score, det.num_frames, set(cuts))
    min_track_frames = int(MIN_TRACK_SECONDS * det.fps)
    tracks = [t for t in tracks if t.length >= min_track_frames]
    scored = sorted((track_score(det, t) for t in tracks), key=lambda s: -s["score"])
    by_id = {t.track_id: t for t in tracks}
    counts = people_counts(det)

    ranges = []
    for shot_start, shot_end in shot_spans(det.num_frames, cuts):
        in_shot = [s for s in scored if shot_start <= by_id[s["track_id"]].frames[0] < shot_end]
        if not in_shot:
            continue
        fencer = by_id[in_shot[0]["track_id"]]
        visible = np.zeros(det.num_frames, dtype=bool)
        for frame, det_index in zip(fencer.frames, fencer.det_indices, strict=True):
            visible[frame] = full_body_visible(det, det_index)
        visible[:shot_start] = visible[shot_end:] = False
        for start, end in _runs(visible, int(MAX_GAP_SECONDS * det.fps)):
            if (end - start) / det.fps >= MIN_RANGE_SECONDS:
                ranges.append(_range_report(det, fencer, start, end, counts))

    chosen = scored[0] if scored else None
    all_frames = np.arange(det.num_frames)
    speed = camera_speed_width_per_s(det, all_frames)
    view_ratios = [r["view_ratio"] for r in ranges if r["view_ratio"] is not None]
    report = {
        "width": det.width,
        "height": det.height,
        "processed_fps": round(det.fps, 3),
        "num_frames": det.num_frames,
        "duration_s": round(float(det.frame_times[-1] - det.frame_times[0]), 3) if det.num_frames else 0.0,
        "shot_cuts_s": [round(float(det.frame_times[c]), 3) for c in cuts],
        "people_count_median": int(np.median(counts)) if det.num_frames else 0,
        "people_count_max": int(counts.max()) if det.num_frames else 0,
        "chosen_track_id": chosen["track_id"] if chosen else None,
        "full_body_frac": chosen["full_body_frac"] if chosen else 0.0,
        "camera_motion_width_per_s": None if not np.isfinite(speed) else round(speed, 4),
        "camera": "static" if np.isfinite(speed) and speed <= STATIC_CAMERA_MAX_WIDTH_PER_S else "moving",
        "view_ratio_median": round(float(np.median(view_ratios)), 3) if view_ratios else None,
        "view": classify_view(float(np.median(view_ratios))) if view_ratios else "unknown",
        "tracks": scored[:10],
        "ranges": ranges,
        "usable_ranges": [[r["start_s"], r["end_s"]] for r in ranges if r["usable"]],
    }
    return report, tracks


def tracks_to_arrays(tracks: list[Track]) -> dict[str, np.ndarray]:
    """Flat arrays for track.npz (no object arrays, so np.load needs no pickle)."""
    rows = [(t.track_id, f, d, b) for t in tracks for f, d, b in zip(t.frames, t.det_indices, t.boxes, strict=True)]
    if not rows:
        return {
            "trk_id": np.zeros(0, np.int64),
            "trk_frame": np.zeros(0, np.int64),
            "trk_det": np.zeros(0, np.int64),
            "trk_box": np.zeros((0, 4)),
        }
    ids, frames, dets, boxes = zip(*rows, strict=True)
    return {
        "trk_id": np.asarray(ids, np.int64),
        "trk_frame": np.asarray(frames, np.int64),
        "trk_det": np.asarray(dets, np.int64),
        "trk_box": np.asarray(boxes, np.float64),
    }


def track_from_arrays(arrays, track_id: int) -> Track:
    mask = np.asarray(arrays["trk_id"]) == track_id
    if not mask.any():
        raise KeyError(f"track {track_id} not in track.npz (have {sorted(set(np.asarray(arrays['trk_id']).tolist()))})")
    order = np.argsort(np.asarray(arrays["trk_frame"])[mask])
    return Track(
        track_id,
        np.asarray(arrays["trk_frame"])[mask][order].tolist(),
        np.asarray(arrays["trk_det"])[mask][order].tolist(),
        list(np.asarray(arrays["trk_box"])[mask][order]),
    )
