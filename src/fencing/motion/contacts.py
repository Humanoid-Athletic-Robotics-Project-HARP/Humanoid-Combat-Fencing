"""Per-frame foot contact detection.

A foot is in contact when its lowest sole point is near the ground AND its sole centre is
nearly still horizontally. Separate enter/exit thresholds (hysteresis) stop the flag from
flickering on noisy video data; runs shorter than `min_duration_s` are dropped and gaps
shorter than `max_gap_s` are bridged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from fencing.motion.filters import lowpass, runs
from fencing.motion.kinematics import FKResult


@dataclass(frozen=True)
class ContactParams:
    # Tuned on LAFAN1 walking: a 0.6 m/s exit let low shuffling glides (foot 1 cm up,
    # 0.5 m/s) count as planted, so the slide fix dragged the body back by 15 cm.
    height_enter_m: float = 0.03
    height_exit_m: float = 0.05
    speed_enter_mps: float = 0.20
    speed_exit_mps: float = 0.35
    min_duration_s: float = 0.10
    max_gap_s: float = 0.07
    # Foot speed is differentiated position; a light low-pass keeps jitter from reading as motion.
    speed_smoothing_hz: float = 6.0


def foot_speeds(fk: FKResult, fps: float, smoothing_hz: float = 6.0) -> np.ndarray:
    """(T, 2) horizontal speed of each sole centre in m/s."""
    xy = fk.sole_center[..., :2]
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[:2])
    xy = lowpass(xy.reshape(xy.shape[0], -1), smoothing_hz, fps).reshape(xy.shape)
    velocity = np.gradient(xy, 1.0 / fps, axis=0)
    return np.linalg.norm(velocity, axis=-1)


def estimate_ground_track(foot_lowest_z: np.ndarray, fps: float, window_s: float = 1.0) -> np.ndarray:
    """Rough per-frame floor height before contacts are known.

    Over any ~1 s window of walking or fencing at least one foot touches the floor, so the
    window minimum of the lower foot tracks the floor even when pose estimation makes the
    whole body drift vertically. Smoothed so it carries drift but not step-to-step detail.
    """
    lower_foot = foot_lowest_z.min(axis=1)
    size = max(3, int(round(window_s * fps)) | 1)
    track = minimum_filter1d(lower_foot, size=size, mode="nearest")
    return lowpass(track, cutoff_hz=0.5, fps=fps)


def hysteresis(enter: np.ndarray, stay: np.ndarray) -> np.ndarray:
    """Turns on where `enter` holds, stays on while `stay` holds."""
    state = np.zeros(enter.shape[0], dtype=bool)
    on = False
    for t in range(enter.shape[0]):
        on = bool(stay[t]) if on else bool(enter[t])
        state[t] = on
    return state


def clean_runs(mask: np.ndarray, min_length: int, max_gap: int) -> np.ndarray:
    """Bridge gaps of <= max_gap frames, then drop runs shorter than min_length."""
    mask = np.asarray(mask, dtype=bool).copy()
    spans = runs(mask)
    for (_, end_a), (start_b, _) in zip(spans, spans[1:]):
        if start_b - end_a <= max_gap:
            mask[end_a:start_b] = True
    for start, end in runs(mask):
        if end - start < min_length:
            mask[start:end] = False
    return mask


def detect_contacts(
    heights: np.ndarray,
    speeds: np.ndarray,
    fps: float,
    params: ContactParams = ContactParams(),
) -> np.ndarray:
    """(T, 2) bool contact flags from heights above the floor and horizontal speeds."""
    if heights.shape != speeds.shape or heights.ndim != 2:
        raise ValueError(f"heights {heights.shape} and speeds {speeds.shape} must both be (T, 2)")
    min_length = max(1, int(round(params.min_duration_s * fps)))
    max_gap = int(round(params.max_gap_s * fps))
    contacts = np.zeros(heights.shape, dtype=bool)
    for f in range(heights.shape[1]):
        enter = (heights[:, f] < params.height_enter_m) & (speeds[:, f] < params.speed_enter_mps)
        stay = (heights[:, f] < params.height_exit_m) & (speeds[:, f] < params.speed_exit_mps)
        contacts[:, f] = clean_runs(hysteresis(enter, stay), min_length, max_gap)
    return contacts


def contacts_from_fk(
    fk: FKResult,
    fps: float,
    params: ContactParams = ContactParams(),
    floor_known: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Contacts plus the per-frame floor height used for them.

    floor_known=True means the motion is already ground-aligned (floor at z = 0).
    """
    if floor_known:
        floor = np.zeros(fk.foot_lowest_z.shape[0])
    else:
        floor = estimate_ground_track(fk.foot_lowest_z, fps)
    heights = fk.foot_lowest_z - floor[:, None]
    speeds = foot_speeds(fk, fps, params.speed_smoothing_hz)
    return detect_contacts(heights, speeds, fps, params), floor


def contact_phases(contact: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges of one foot's contact flag."""
    return runs(contact)


def contact_pattern(contacts: np.ndarray, width: int = 60) -> dict[str, str]:
    """Compact text strip per foot ('#' = contact, '.' = air), resampled to `width` chars."""
    n = contacts.shape[0]
    idx = np.linspace(0, n - 1, min(width, n)).round().astype(int)
    return {side: "".join("#" if c else "." for c in contacts[idx, f]) for f, side in enumerate(("left", "right"))}
