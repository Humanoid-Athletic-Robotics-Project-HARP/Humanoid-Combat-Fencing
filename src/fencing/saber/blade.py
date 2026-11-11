"""Blade-tip positions from G1 motion CSVs, for the Goal 1 "blade-tip error" metric.

Pure numpy + mujoco (no torch), so evaluation code in the fencing venv can import it:

    from fencing.saber.blade import blade_tip_positions, load_motion_csv
    tips = blade_tip_positions(load_motion_csv("motions/parry_quinte_v1.csv"))  # (T, 3)

Input rows use the repo's motion CSV format (motions/README.md): 36 columns,
x, y, z, qx, qy, qz, qw (root quaternion xyzw), then the 29 joint angles in G1 order.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from fencing.saber.spec import GUARD_SITE, NOMINAL_SABER, TIP_SITE, SaberSpec, g1_saber_model

CSV_COLUMNS = 36
NUM_JOINTS = 29


@dataclass(frozen=True)
class BladeTrajectory:
    """Per-frame blade state in world coordinates (metres)."""

    tip: np.ndarray  # (T, 3) blade_tip site
    guard: np.ndarray  # (T, 3) guard site (base of the blade)
    direction: np.ndarray  # (T, 3) unit vector guard -> tip


@lru_cache(maxsize=8)
def _cached_model(saber: SaberSpec) -> mujoco.MjModel:
    # Compiling the G1 costs ~0.1 s; evaluation calls this once per motion.
    return g1_saber_model(saber)


def load_motion_csv(path: str | Path) -> np.ndarray:
    """Read a headerless 36-column motion CSV into a (T, 36) float array."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    _check_motion_shape(rows, source=str(path))
    return rows


def _check_motion_shape(motion: np.ndarray, source: str = "motion array") -> None:
    if motion.ndim != 2 or motion.shape[1] != CSV_COLUMNS:
        raise ValueError(
            f"{source}: expected shape (T, {CSV_COLUMNS}) "
            f"[x,y,z,qx,qy,qz,qw + {NUM_JOINTS} joints], got {motion.shape}"
        )
    if not np.all(np.isfinite(motion)):
        bad_row = int(np.argwhere(~np.isfinite(motion))[0, 0])
        raise ValueError(f"{source}: non-finite value in row {bad_row}")


def csv_rows_to_qpos(motion: np.ndarray) -> np.ndarray:
    """(T, 36) CSV rows -> (T, 36) MuJoCo qpos: root quaternion xyzw -> wxyz, normalised."""
    motion = np.asarray(motion, dtype=float)
    _check_motion_shape(motion)
    quat_xyzw = motion[:, 3:7]
    norms = np.linalg.norm(quat_xyzw, axis=1)
    if np.any(norms < 1e-6):
        bad_row = int(np.argmin(norms))
        raise ValueError(f"root quaternion has ~zero norm in row {bad_row}: {quat_xyzw[bad_row]}")
    qpos = np.empty_like(motion)
    qpos[:, 0:3] = motion[:, 0:3]
    qpos[:, 3] = quat_xyzw[:, 3] / norms
    qpos[:, 4:7] = quat_xyzw[:, 0:3] / norms[:, None]
    qpos[:, 7:] = motion[:, 7:]
    return qpos


def blade_trajectory(motion: np.ndarray, saber: SaberSpec = NOMINAL_SABER) -> BladeTrajectory:
    """Forward kinematics of the saber-equipped G1 for every frame of a motion."""
    qpos = csv_rows_to_qpos(motion)
    model = _cached_model(saber)
    if model.nq != CSV_COLUMNS:
        raise RuntimeError(f"saber G1 model has nq={model.nq}, expected {CSV_COLUMNS}")
    data = mujoco.MjData(model)
    tip_id = model.site(TIP_SITE).id
    guard_id = model.site(GUARD_SITE).id
    tips = np.empty((len(qpos), 3))
    guards = np.empty((len(qpos), 3))
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        tips[frame] = data.site_xpos[tip_id]
        guards[frame] = data.site_xpos[guard_id]
    direction = tips - guards
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return BladeTrajectory(tip=tips, guard=guards, direction=direction)


def blade_tip_positions(motion: np.ndarray, saber: SaberSpec = NOMINAL_SABER) -> np.ndarray:
    """(T, 36) motion CSV rows -> (T, 3) world blade-tip positions in metres."""
    return blade_trajectory(motion, saber).tip


def blade_tip_error(robot_motion: np.ndarray, reference_motion: np.ndarray,
                    saber: SaberSpec = NOMINAL_SABER) -> np.ndarray:
    """Per-frame distance (m) between the robot's and the reference's blade tips.

    Both motions must have the same number of frames; pick the key frame(s) afterwards.
    """
    if len(robot_motion) != len(reference_motion):
        raise ValueError(
            f"frame count mismatch: robot {len(robot_motion)} vs reference {len(reference_motion)}"
        )
    robot_tips = blade_tip_positions(robot_motion, saber)
    reference_tips = blade_tip_positions(reference_motion, saber)
    return np.linalg.norm(robot_tips - reference_tips, axis=1)
