"""MediaPipe Pose landmark layout and the temporal clean-up applied to raw landmarks.

Pure numpy/scipy: runs in the `fencing` venv and in the `pose` venv alike.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe Pose Landmarker, 33 points. "left" is the person's left.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32
NUM_LANDMARKS = 33

# Left/right pairs swapped by the leg and arm identity fix. The torso (shoulders, hips) is
# not swapped: a torso swap means the whole body is mirrored, which a per-limb check cannot
# tell apart from the fencer turning round.
LEG_PAIRS = [
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL),
    (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
]
ARM_PAIRS = [
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_PINKY, RIGHT_PINKY),
    (LEFT_INDEX, RIGHT_INDEX),
    (LEFT_THUMB, RIGHT_THUMB),
]
FEET = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]
BODY = list(range(LEFT_SHOULDER, NUM_LANDMARKS))  # everything but the face


def _swap_pairs(frame: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    for a, b in pairs:
        out[a], out[b] = frame[b], frame[a]
    return out


def fix_left_right_swaps(
    image_xy: np.ndarray,
    arrays_to_fix: list[np.ndarray],
    body_scale_px: np.ndarray,
    swap_ratio: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Undo frame-to-frame left/right label flips of the legs and of the arms.

    In side views MediaPipe sometimes swaps which leg is "left". A swap shows up as both
    legs jumping to where the other leg was a frame ago. For each frame we compare the
    motion cost of keeping the labels with the cost of swapping them, against the previous
    (already corrected) frame, and swap when swapping is clearly cheaper. Legs crossing in a
    cross-over step give similar costs and are left alone by the `swap_ratio` margin.

    image_xy: (T, 33, 2) pixel coordinates used to decide.
    arrays_to_fix: arrays of shape (T, 33, ...) that receive the same swaps.
    body_scale_px: (T,) body size in pixels; costs are normalised by it.
    Returns the fixed arrays (copies) and a (T, 2) bool array of [legs_swapped, arms_swapped].
    """
    num_frames = image_xy.shape[0]
    fixed_xy = image_xy.copy()
    fixed = [a.copy() for a in arrays_to_fix]
    swapped = np.zeros((num_frames, 2), dtype=bool)
    for t in range(1, num_frames):
        scale = max(float(body_scale_px[t]), 1e-6)
        for group, pairs in enumerate((LEG_PAIRS, ARM_PAIRS)):
            left = [a for a, _ in pairs]
            right = [b for _, b in pairs]
            prev, cur = fixed_xy[t - 1], fixed_xy[t]
            keep = np.linalg.norm(cur[left] - prev[left], axis=-1).sum() + np.linalg.norm(
                cur[right] - prev[right], axis=-1
            ).sum()
            swap = np.linalg.norm(cur[left] - prev[right], axis=-1).sum() + np.linalg.norm(
                cur[right] - prev[left], axis=-1
            ).sum()
            if swap / scale < swap_ratio * keep / scale:
                fixed_xy[t] = _swap_pairs(fixed_xy[t], pairs)
                for arr in fixed:
                    arr[t] = _swap_pairs(arr[t], pairs)
                swapped[t, group] = True
    return fixed, swapped


def resample_uniform(times_s: np.ndarray, values: np.ndarray, out_times_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate (T, ...) samples at irregular `times_s` onto `out_times_s`.

    Video frame timestamps are not always uniform (VFR uploads, 29.97 vs 30 fps), so the
    30 fps output is built from timestamps, never from frame counts.
    """
    if times_s.ndim != 1 or len(times_s) != len(values):
        raise ValueError(f"times ({times_s.shape}) and values ({values.shape}) disagree")
    if len(times_s) < 2:
        raise ValueError("need at least two samples to resample")
    flat = values.reshape(len(values), -1)
    out = np.empty((len(out_times_s), flat.shape[1]), dtype=np.float64)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(out_times_s, times_s, flat[:, k])
    return out.reshape((len(out_times_s),) + values.shape[1:])


def smooth_time(values: np.ndarray, fps: float, window_s: float, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0; keeps peaks of fast actions better than a box filter."""
    window = int(round(window_s * fps)) | 1  # odd
    if window <= polyorder + 1 or len(values) < window:
        return values.copy()
    return savgol_filter(values, window, polyorder, axis=0, mode="interp")


def fill_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate (T, ...) values over frames where `valid` is False."""
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return values.copy()
    if not valid.any():
        raise ValueError("no valid frames to interpolate from")
    idx = np.arange(len(values))
    flat = values.reshape(len(values), -1).astype(np.float64)
    out = flat.copy()
    for k in range(flat.shape[1]):
        out[~valid, k] = np.interp(idx[~valid], idx[valid], flat[valid, k])
    return out.reshape(values.shape)
