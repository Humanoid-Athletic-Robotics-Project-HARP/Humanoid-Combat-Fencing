"""Small 1-D signal helpers shared by the cleanup steps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt


def lowpass(signal: np.ndarray, cutoff_hz: float, fps: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (no lag, so contacts stay aligned).

    Falls back to the input when the clip is too short to pad, and to the mean when the
    cutoff is below what the clip length can resolve.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = fps / 2.0
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if cutoff_hz >= nyquist:
        return signal.copy()
    b, a = butter(order, cutoff_hz / nyquist)
    n = signal.shape[0]
    padlen = min(3 * max(len(a), len(b)) * 4, n - 1)
    if n < 4 or padlen < 1:
        return signal.copy()
    # Odd padding reflects the trend at the ends instead of pulling them toward zero.
    return filtfilt(b, a, signal, axis=0, padtype="odd", padlen=padlen)


def interpolate_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear fill of 1-D `values` where `valid` is False; ends hold the nearest valid value."""
    values = np.asarray(values, dtype=np.float64)
    if valid.all():
        return values.copy()
    if not valid.any():
        raise ValueError("interpolate_gaps: no valid samples to interpolate from")
    idx = np.arange(values.shape[0])
    return np.interp(idx, idx[valid], values[valid])


def covering_smooth_max(signal: np.ndarray, half_width: int) -> np.ndarray:
    """Smooth envelope that is >= signal everywhere.

    A max filter of width 2w+1 followed by a moving average of the same width never drops
    below the original: every sample inside the averaging window is already >= signal(t).
    Used to lift the robot out of the floor without introducing steps.
    """
    if half_width <= 0:
        return np.asarray(signal, dtype=np.float64).copy()
    size = 2 * half_width + 1
    widened = maximum_filter1d(signal, size=size, mode="nearest")
    return uniform_filter1d(widened, size=size, mode="nearest")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where a 1-D boolean mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]


def smoothstep(x: np.ndarray) -> np.ndarray:
    """C1 ramp 0 -> 1 on [0, 1]; zero slope at both ends so blends add no velocity jump."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
