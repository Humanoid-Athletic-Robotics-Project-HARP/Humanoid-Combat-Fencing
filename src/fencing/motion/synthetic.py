"""Synthetic right-handed fencing footwork on the G1, built with leg IK.

Ground truth for tests and demos: feet are exactly planted (zero slip, soles on z = 0),
so any corruption added later has a known correct answer. The fencing line is +x; the
right (front) foot points down the line, the left (back) foot is turned out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from fencing.motion.filters import smoothstep
from fencing.motion.ik import LegIK
from fencing.motion.io import JOINT_NAMES, Motion
from fencing.motion.kinematics import G1Model, load_g1

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class Stance:
    pelvis_x: float = 0.0
    pelvis_height: float = 0.68
    pelvis_yaw: float = np.radians(30)
    front_x: float = 0.20  # right foot, relative to pelvis_x
    back_x: float = -0.20  # left foot, relative to pelvis_x
    front_y: float = -0.10
    back_y: float = 0.10
    front_yaw: float = 0.0
    back_yaw: float = np.radians(60)


@dataclass
class _Keys:
    """Per-frame targets the IK tracks."""

    pelvis: np.ndarray  # (T, 3)
    pelvis_yaw: np.ndarray  # (T,)
    feet: np.ndarray  # (T, 2, 3) sole centre targets
    foot_yaw: np.ndarray  # (T, 2)


# Arm pose: sword arm forward and bent, back arm raised behind -- only cosmetic for footwork.
_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.6,
    "right_shoulder_roll_joint": -0.15,
    "right_elbow_joint": 0.6,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.5,
    "left_elbow_joint": 1.2,
}
_LEG_GUESS = {"hip_pitch_joint": -0.4, "knee_joint": 0.9, "ankle_pitch_joint": -0.45}


def _stance_keys(stance: Stance, n: int) -> _Keys:
    pelvis = np.tile([stance.pelvis_x, 0.0, stance.pelvis_height], (n, 1))
    feet = np.zeros((n, 2, 3))
    feet[:, RIGHT, :2] = [stance.pelvis_x + stance.front_x, stance.front_y]
    feet[:, LEFT, :2] = [stance.pelvis_x + stance.back_x, stance.back_y]
    return _Keys(pelvis, np.full(n, stance.pelvis_yaw), feet, np.tile([stance.back_yaw, stance.front_yaw], (n, 1)))


def _step(keys: _Keys, foot: int, start: int, end: int, dx: float, lift: float) -> None:
    """Move one foot by dx along the line between frames [start, end), with a lift arc."""
    span = end - start
    s = smoothstep(np.arange(1, span + 1) / span)
    keys.feet[start:end, foot, 0] += dx * s
    keys.feet[end:, foot, 0] += dx
    keys.feet[start:end, foot, 2] += lift * np.sin(np.pi * np.arange(1, span + 1) / span)


def _shift_pelvis(keys: _Keys, start: int, end: int, dx: float, dz: float = 0.0) -> None:
    span = max(1, end - start)
    s = smoothstep(np.arange(1, span + 1) / span)
    keys.pelvis[start:end, 0] += dx * s
    keys.pelvis[end:, 0] += dx
    keys.pelvis[start:end, 2] += dz * s
    keys.pelvis[end:, 2] += dz


def footwork_keys(script: list[tuple[str, float]], fps: float = 30.0, stance: Stance = Stance()) -> tuple[_Keys, list]:
    """Targets for a script like [("en_garde", 1.0), ("advance", 0.6), ("lunge", 1.2)].

    Distances: advance/retreat 0.25 m; lunge: front foot +0.32 m, pelvis +0.14 m and
    -0.06 m in the extension (first 30 %), held, then recovered over the last 45 %.
    Returns the keys and [(start, end, label)] ground-truth spans.
    """
    lengths = [max(2, int(round(d * fps))) for _, d in script]
    keys = _stance_keys(stance, sum(lengths))
    spans, t = [], 0
    for (label, _), n in zip(script, lengths):
        a, half = t, n // 2
        if label == "en_garde":
            pass
        elif label in ("advance", "retreat"):
            sign = 1.0 if label == "advance" else -1.0
            first, second = (RIGHT, LEFT) if label == "advance" else (LEFT, RIGHT)
            _step(keys, first, a, a + half, sign * 0.25, 0.04)
            _step(keys, second, a + half, a + n, sign * 0.25, 0.04)
            _shift_pelvis(keys, a, a + n, sign * 0.25)
        elif label == "lunge":
            ext, hold = int(0.30 * n), int(0.25 * n)
            _step(keys, RIGHT, a, a + ext, 0.32, 0.06)
            _shift_pelvis(keys, a, a + ext, 0.14, -0.06)
            _step(keys, RIGHT, a + ext + hold, a + n, -0.32, 0.04)
            _shift_pelvis(keys, a + ext + hold, a + n, -0.14, 0.06)
        else:
            raise ValueError(f"unknown synthetic phase {label!r}; use en_garde/advance/retreat/lunge")
        spans.append((a, a + n, label))
        t += n
    return keys, spans


def solve_keys(keys: _Keys, fps: float = 30.0, g1: G1Model | None = None) -> Motion:
    g1 = g1 or load_g1()
    solver = LegIK(g1, damping=0.01, rotation_weight=0.5)
    n = keys.pelvis.shape[0]
    joints0 = np.zeros(len(JOINT_NAMES))
    for name, value in _ARM_POSE.items():
        joints0[JOINT_NAMES.index(name)] = value
    for side in ("left", "right"):
        for suffix, value in _LEG_GUESS.items():
            joints0[JOINT_NAMES.index(f"{side}_{suffix}")] = value
    quats = Rotation.from_euler("z", keys.pelvis_yaw[:, None]).as_quat()
    motion = Motion(keys.pelvis.copy(), quats, np.tile(joints0, (n, 1)), fps, {"synthetic": True})
    qpos_all = g1.qpos_from_motion(motion)
    previous = qpos_all[0].copy()
    for t in range(n):
        qpos = qpos_all[t].copy()
        leg_q = np.concatenate([g1.qpos_index[g1.leg_columns[s]] for s in ("left", "right")])
        qpos[leg_q] = previous[leg_q]  # warm start from the last frame keeps the solution branch
        for f in (LEFT, RIGHT):
            target_rot = Rotation.from_euler("z", keys.foot_yaw[t, f]).as_matrix()
            qpos, err = solver.solve(qpos, f, keys.feet[t, f], target_rot, iterations=60 if t == 0 else 25)
            if err > 0.01:
                raise RuntimeError(f"synthetic IK could not reach foot {f} target at frame {t} (error {err:.3f} m)")
        qpos_all[t] = qpos
        previous = qpos
    motion.joint_pos = qpos_all[:, g1.qpos_index]
    return motion


def synthesize(script: list[tuple[str, float]], fps: float = 30.0, stance: Stance = Stance()) -> tuple[Motion, list]:
    keys, spans = footwork_keys(script, fps, stance)
    return solve_keys(keys, fps), spans


def stance_with(**changes) -> Stance:
    return replace(Stance(), **changes)


# ---------------------------------------------------------------- corruption


def add_vertical_drift(motion: Motion, amplitude_m: float = 0.06, period_s: float = 4.0, offset_m: float = 0.0) -> Motion:
    """Slow pelvis bob + constant offset, as monocular depth errors produce."""
    t = np.arange(motion.n_frames) / motion.fps
    root = motion.root_pos.copy()
    root[:, 2] += offset_m + amplitude_m * np.sin(2 * np.pi * t / period_s)
    return motion.replace(root_pos=root)


def add_root_slide(motion: Motion, speed_mps: float = 0.10, direction_deg: float = 30.0) -> Motion:
    """Whole body glides at constant speed: planted feet skate."""
    t = np.arange(motion.n_frames) / motion.fps
    root = motion.root_pos.copy()
    direction = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    root[:, :2] += speed_mps * t[:, None] * direction
    return motion.replace(root_pos=root)


def add_jitter(motion: Motion, joint_sigma_rad: float = 0.02, root_sigma_m: float = 0.005, seed: int = 0) -> Motion:
    rng = np.random.default_rng(seed)
    return motion.replace(
        joint_pos=motion.joint_pos + rng.normal(0, joint_sigma_rad, motion.joint_pos.shape),
        root_pos=motion.root_pos + rng.normal(0, root_sigma_m, motion.root_pos.shape),
    )


def add_yaw_drift(motion: Motion, total_deg: float = 40.0) -> Motion:
    """The whole world estimate turning slowly about the start point (monocular heading drift):
    the body yaws by up to `total_deg` and the path bends into an arc with it."""
    t = np.arange(motion.n_frames) / max(motion.n_frames - 1, 1)
    angle = np.radians(total_deg) * smoothstep(t)
    rot = Rotation.from_euler("z", angle[:, None])
    root = motion.root_pos.copy()
    start = root[0, :2].copy()
    offset = np.c_[root[:, :2] - start, np.zeros(motion.n_frames)]
    root[:, :2] = start + rot.apply(offset)[:, :2]
    quat = (rot * Rotation.from_quat(motion.root_quat)).as_quat()
    return motion.replace(root_pos=root, root_quat=quat)
