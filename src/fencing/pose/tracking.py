"""Greedy IoU tracker over per-frame person detections, and box helpers. Pure numpy.

Fencing clips have few people who rarely occlude each other for long, so a constant-velocity
IoU tracker (SORT without the Kalman filter) is enough and keeps the dependency list empty.
Tracks never continue across a shot cut: the caller passes `cut_frames`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MATCH_IOU = 0.3
MAX_MISSED_FRAMES = 15  # 0.5 s at 30 fps: covers a fencer briefly hidden behind another
MIN_START_SCORE = 0.8


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N, 4) and b (M, 4), xyxy."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    det_indices: list[int] = field(default_factory=list)  # row in the detections arrays
    boxes: list[np.ndarray] = field(default_factory=list)

    def predicted_box(self, frame: int) -> np.ndarray:
        last = self.boxes[-1]
        if len(self.boxes) < 2:
            return last
        gap = max(self.frames[-1] - self.frames[-2], 1)
        velocity = (self.boxes[-1] - self.boxes[-2]) / gap
        return last + velocity * (frame - self.frames[-1])

    @property
    def length(self) -> int:
        return len(self.frames)


def track_detections(
    det_frame: np.ndarray,
    det_box: np.ndarray,
    det_score: np.ndarray,
    num_frames: int,
    cut_frames: set[int] | None = None,
) -> list[Track]:
    """Link detections into tracks. `cut_frames`: frames that start a new shot."""
    cut_frames = cut_frames or set()
    order = np.argsort(det_frame, kind="stable")
    by_frame: dict[int, list[int]] = {}
    for i in order:
        by_frame.setdefault(int(det_frame[i]), []).append(int(i))

    finished: list[Track] = []
    active: list[Track] = []
    next_id = 1
    for frame in range(num_frames):
        if frame in cut_frames:
            finished.extend(active)
            active = []
        active_still = [t for t in active if frame - t.frames[-1] <= MAX_MISSED_FRAMES]
        finished.extend(t for t in active if t not in active_still)
        active = active_still

        dets = by_frame.get(frame, [])
        if active and dets:
            predicted = np.stack([t.predicted_box(frame) for t in active])
            iou = box_iou(predicted, det_box[dets])
            matched_tracks, matched_dets = set(), set()
            for flat in np.argsort(-iou, axis=None):
                ti, di = np.unravel_index(flat, iou.shape)
                if iou[ti, di] < MATCH_IOU:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                matched_tracks.add(ti)
                matched_dets.add(di)
                track, det = active[ti], dets[di]
                track.frames.append(frame)
                track.det_indices.append(det)
                track.boxes.append(np.asarray(det_box[det], dtype=np.float64))
            unmatched = [d for j, d in enumerate(dets) if j not in matched_dets]
        else:
            unmatched = dets
        for det in unmatched:
            if det_score[det] < MIN_START_SCORE:
                continue
            active.append(Track(next_id, [frame], [det], [np.asarray(det_box[det], dtype=np.float64)]))
            next_id += 1
    finished.extend(active)
    return sorted(finished, key=lambda t: t.track_id)


def interpolate_track_boxes(track: Track, frames: np.ndarray) -> np.ndarray:
    """Boxes (len(frames), 4) at the requested frames, linearly interpolated inside the track's
    span and held constant outside it."""
    known = np.asarray(track.frames, dtype=np.float64)
    boxes = np.stack(track.boxes)
    return np.stack([np.interp(frames, known, boxes[:, k]) for k in range(4)], axis=1)


def square_crop_box(box: np.ndarray, enlarge: float, width: int, height: int) -> np.ndarray:
    """Square xyxy crop around a person box, `enlarge` x its longer side. May extend past the
    image; the caller pads. Square because the pose model resizes to a square input."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    side = enlarge * max(box[2] - box[0], box[3] - box[1], 1.0)
    side = min(side, 2.0 * max(width, height))
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
