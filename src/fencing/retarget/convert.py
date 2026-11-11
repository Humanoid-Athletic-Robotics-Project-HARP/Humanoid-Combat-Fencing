"""GMR qpos <-> the raw G1 CSV / GMR pickle layouts, fps resampling and the CSV sidecar.

Pure numpy/scipy so it runs in the `gmr` venv (inside the worker) and in the `fencing` venv
(tests, CLI). Conventions (docs/pipeline.md):

    MuJoCo qpos   x, y, z, qw, qx, qy, qz, 29 joints   (root quaternion wxyz)
    GMR pickle    root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29), fps
    raw G1 CSV    x, y, z, qx, qy, qz, qw, 29 joints   (root quaternion xyzw), 30 fps
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CSV_FPS = 30.0
N_DOF = 29


def qpos_to_csv_array(qpos: np.ndarray) -> np.ndarray:
    """(T, 36) MuJoCo qpos (root quat wxyz) -> (T, 36) CSV rows (root quat xyzw, normalised)."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 7 + N_DOF:
        raise ValueError(f"expected qpos of shape (T, {7 + N_DOF}), got {qpos.shape}")
    quat_xyzw = qpos[:, [4, 5, 6, 3]]
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    return np.concatenate([qpos[:, :3], quat_xyzw, qpos[:, 7:]], axis=1)


# A joint more than this far from its running median for a few frames is an IK glitch (a
# branch hop and back, or one bad keypoint frame), not motion: real fencing joint motion at
# 30 fps moves < 0.3 rad between neighbouring frames and does not snap back.
DESPIKE_WINDOW = 7
DESPIKE_THRESHOLD_RAD = 0.5


def despike_joints(
    joints: np.ndarray, window: int = DESPIKE_WINDOW, threshold: float = DESPIKE_THRESHOLD_RAD
) -> tuple[np.ndarray, int]:
    """Bridge short spikes in (T, J) joint angles by interpolation; returns (joints, n spiked samples)."""
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < window:
        return joints.copy(), 0
    half = window // 2
    padded = np.pad(joints, ((half, half), (0, 0)), mode="edge")
    median = np.median(np.lib.stride_tricks.sliding_window_view(padded, window, axis=0), axis=-1)
    spikes = np.abs(joints - median) > threshold
    out = joints.copy()
    frames = np.arange(len(joints))
    for j in np.flatnonzero(spikes.any(axis=0)):
        good = ~spikes[:, j]
        if good.sum() >= 2:
            out[:, j] = np.interp(frames, frames[good], joints[good, j])
    return out, int(spikes.sum())


def make_continuous(quat_xyzw: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie in the same hemisphere (q and -q are equal)."""
    out = np.asarray(quat_xyzw, dtype=np.float64).copy()
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def resample_csv_array(array: np.ndarray, src_fps: float, dst_fps: float = CSV_FPS) -> np.ndarray:
    """Resample CSV rows in time: linear for positions and joints, slerp for the root quaternion."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"fps must be positive, got {src_fps} -> {dst_fps}")
    array = np.asarray(array, dtype=np.float64)
    if abs(src_fps - dst_fps) < 1e-6 or len(array) < 2:
        return array.copy()
    t_src = np.arange(len(array)) / src_fps
    n_dst = int(np.floor(t_src[-1] * dst_fps + 1e-6)) + 1
    t_dst = np.arange(n_dst) / dst_fps
    out = np.empty((n_dst, array.shape[1]))
    for col in list(range(3)) + list(range(7, array.shape[1])):
        out[:, col] = np.interp(t_dst, t_src, array[:, col])
    slerp = Slerp(t_src, Rotation.from_quat(array[:, 3:7]))
    out[:, 3:7] = make_continuous(slerp(t_dst).as_quat())
    return out


def csv_array_to_pickle_dict(array: np.ndarray, fps: float) -> dict:
    """The dict GMR's own scripts pickle (scripts/gvhmr_to_robot.py, batch_gmr_pkl_to_csv.py)."""
    return {
        "fps": float(fps),
        "root_pos": array[:, :3].copy(),
        "root_rot": array[:, 3:7].copy(),
        "dof_pos": array[:, 7:].copy(),
        "local_body_pos": None,
        "link_body_list": None,
    }


SIDECAR_FROM_POSE = (
    "clip_id",
    "yt_id",
    "source_start_s",
    "source_end_s",
    "pose_method",
    "track_id",
    "human_height_m",
    "handedness_guess",
)


def build_sidecar(pose_summary: dict, retarget_stats: dict | None = None, notes: str = "") -> dict:
    """The `<clip_id>.json` next to the raw CSV (docs/pipeline.md) from pose.json + GMR stats."""
    missing = [k for k in SIDECAR_FROM_POSE if k not in pose_summary]
    if missing:
        raise KeyError(f"pose summary lacks {missing}")
    sidecar = {k: pose_summary[k] for k in SIDECAR_FROM_POSE}
    sidecar["fps"] = int(CSV_FPS)
    sidecar["notes"] = "; ".join(n for n in (pose_summary.get("notes", ""), notes) if n)
    if retarget_stats:
        sidecar["retarget"] = retarget_stats
    return sidecar
