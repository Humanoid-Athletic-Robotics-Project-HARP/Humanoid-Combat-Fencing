"""Quaternion helpers in numpy, (w, x, y, z) like MuJoCo and mjlab.

Each function mirrors the mjlab.utils.lab_api.math function of the same name, batched
over any leading axes, so observations and metrics rebuilt here match mjlab's torch code.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return quat_conjugate(q) / np.maximum((q * q).sum(axis=-1, keepdims=True), 1e-12)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by quaternions q (..., 4)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conjugate(q), v)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """(..., 4) -> (..., 3, 3) rotation matrices (normalises like mjlab)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    two_s = 2.0 / (q * q).sum(axis=-1)
    m = np.stack([
        1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
        two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
        two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y),
    ], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """The yaw-only part of q (rotation about world z)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = np.zeros_like(q)
    out[..., 0] = np.cos(yaw / 2)
    out[..., 3] = np.sin(yaw / 2)
    return normalize(out)


def subtract_frame_transforms(t01, q01, t02, q02) -> tuple[np.ndarray, np.ndarray]:
    """T12 = T01^-1 T02: pose of frame 2 expressed in frame 1."""
    q10 = quat_inv(q01)
    return quat_apply(q10, np.asarray(t02) - np.asarray(t01)), quat_mul(q10, q02)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation angle (rad, in [0, pi]) between q1 and q2."""
    diff = quat_mul(q1, quat_conjugate(q2))
    return 2.0 * np.arctan2(np.linalg.norm(diff[..., 1:], axis=-1), np.abs(diff[..., 0]))
