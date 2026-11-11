"""Quick-look contact sheet for a pose clip: landmarks on the video, and the lifted 3D skeleton.

Top row: video frames cropped around the fencer with the cleaned 2D landmarks drawn on.
Bottom row: the world keypoints seen from the camera side (x right, z up) over a floor line,
so a floating foot, a sinking hip or a flipped leg is obvious at a glance. Runs in the
`fencing` venv (OpenCV only).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fencing.pose import landmarks as lm

PANEL_PX = 300
SKELETON = [
    (lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
    (lm.LEFT_HIP, lm.RIGHT_HIP),
    (lm.LEFT_SHOULDER, lm.LEFT_HIP),
    (lm.RIGHT_SHOULDER, lm.RIGHT_HIP),
    (lm.LEFT_SHOULDER, lm.LEFT_ELBOW),
    (lm.LEFT_ELBOW, lm.LEFT_WRIST),
    (lm.RIGHT_SHOULDER, lm.RIGHT_ELBOW),
    (lm.RIGHT_ELBOW, lm.RIGHT_WRIST),
    (lm.LEFT_HIP, lm.LEFT_KNEE),
    (lm.LEFT_KNEE, lm.LEFT_ANKLE),
    (lm.LEFT_ANKLE, lm.LEFT_HEEL),
    (lm.LEFT_HEEL, lm.LEFT_FOOT_INDEX),
    (lm.LEFT_ANKLE, lm.LEFT_FOOT_INDEX),
    (lm.RIGHT_HIP, lm.RIGHT_KNEE),
    (lm.RIGHT_KNEE, lm.RIGHT_ANKLE),
    (lm.RIGHT_ANKLE, lm.RIGHT_HEEL),
    (lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX),
    (lm.RIGHT_ANKLE, lm.RIGHT_FOOT_INDEX),
]
LEFT_BGR, RIGHT_BGR, CENTRE_BGR = (255, 160, 0), (0, 80, 255), (230, 230, 230)


def _colour(a: int, b: int) -> tuple[int, int, int]:
    left = {i for i in range(lm.NUM_LANDMARKS) if i % 2 == 1 and i >= lm.LEFT_SHOULDER}
    if a in left and b in left:
        return LEFT_BGR
    if a not in left and b not in left and a >= lm.LEFT_SHOULDER and b >= lm.LEFT_SHOULDER:
        return RIGHT_BGR
    return CENTRE_BGR


def _draw(image: np.ndarray, points: np.ndarray, thickness: int = 2) -> None:
    for a, b in SKELETON:
        pa, pb = points[a], points[b]
        if np.all(np.isfinite([*pa, *pb])):
            cv2.line(image, tuple(int(v) for v in pa), tuple(int(v) for v in pb), _colour(a, b), thickness, cv2.LINE_AA)


def _video_panel(frame: np.ndarray, image_xy: np.ndarray) -> np.ndarray:
    _draw(frame, image_xy)
    pts = image_xy[lm.BODY]
    x0, y0 = np.nanmin(pts, axis=0)
    x1, y1 = np.nanmax(pts, axis=0)
    side = 1.25 * max(x1 - x0, y1 - y0, 50.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    h, w = frame.shape[:2]
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        crop = frame
    scale = PANEL_PX / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    panel = np.zeros((PANEL_PX, PANEL_PX, 3), np.uint8)
    panel[: crop.shape[0], : crop.shape[1]] = crop
    return panel


def _world_panel(keypoints: np.ndarray, centre_x: float) -> np.ndarray:
    """Side view (world x right, z up), 2.2 m wide, floor 20 px above the bottom."""
    panel = np.full((PANEL_PX, PANEL_PX, 3), 40, np.uint8)
    px_per_m = PANEL_PX / 2.2
    floor_y = PANEL_PX - 20
    cv2.line(panel, (0, floor_y), (PANEL_PX, floor_y), (120, 120, 120), 1)
    pts = np.stack(
        [(keypoints[:, 0] - centre_x) * px_per_m + PANEL_PX / 2.0, floor_y - keypoints[:, 2] * px_per_m], axis=-1
    )
    _draw(panel, pts)
    return panel


def contact_sheet(
    video: Path,
    times_s: np.ndarray,
    image_xy: np.ndarray,
    keypoints_world: np.ndarray,
    out_png: Path,
    columns: int = 8,
    title: str = "",
) -> Path:
    """Write the two-row contact sheet for `columns` evenly spaced frames of the clip."""
    picks = np.linspace(0, len(times_s) - 1, min(columns, len(times_s))).round().astype(int)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    tops, bottoms = [], []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(times_s[i]) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((PANEL_PX, PANEL_PX, 3), np.uint8)
        top = _video_panel(frame, image_xy[i])
        cv2.putText(top, f"{times_s[i]:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tops.append(top)
        mid_hip_x = float((keypoints_world[i, lm.LEFT_HIP, 0] + keypoints_world[i, lm.RIGHT_HIP, 0]) / 2.0)
        bottoms.append(_world_panel(keypoints_world[i], mid_hip_x))
    cap.release()
    sheet = np.concatenate([np.concatenate(tops, axis=1), np.concatenate(bottoms, axis=1)], axis=0)
    if title:
        cv2.putText(sheet, title, (6, sheet.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png
