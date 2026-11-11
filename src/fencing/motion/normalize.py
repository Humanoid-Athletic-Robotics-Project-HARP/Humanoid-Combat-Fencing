"""Canonical placement, left/right mirroring and handedness detection.

Mirroring reflects the world across the x-z plane (y -> -y). For a motion facing +x that
is the robot's sagittal plane. The joint sign rules are DERIVED from the MJCF, not typed in:
reflecting a rotation R(a, q) by M = diag(1, -1, 1) gives R(-M a, q) (M has det -1), so the
partner joint with axis b gets angle  q' = -dot(M a, b) * q.  For the G1 that makes pitch
joints keep their sign and roll/yaw joints flip -- tests check the result with FK.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from fencing.motion.io import JOINT_NAMES, Motion
from fencing.motion.kinematics import FKResult, G1Model, load_g1

REFLECT_Y = np.diag([1.0, -1.0, 1.0])


# ---------------------------------------------------------------- placement


def yaw_of(quat_xyzw: np.ndarray) -> np.ndarray:
    """Heading (rad) of the body x-axis projected on the ground, per frame."""
    forward = Rotation.from_quat(quat_xyzw).apply([1.0, 0.0, 0.0])
    return np.arctan2(forward[..., 1], forward[..., 0])


def mean_angle(angles: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def rotate_about_z(motion: Motion, angle: float, pivot_xy: np.ndarray) -> Motion:
    rot = Rotation.from_euler("z", angle)
    root = motion.root_pos.copy()
    root[:, :2] -= pivot_xy
    root = rot.apply(root)
    quat = (rot * Rotation.from_quat(motion.root_quat)).as_quat()
    return motion.replace(root_pos=root, root_quat=quat)


def fencing_line(motion: Motion, fk: FKResult, min_travel_m: float = 0.3, min_linearity: float = 3.0) -> tuple[float, str]:
    """Direction (rad) of the fencing line, pointing from the back foot to the front foot
    (right-handed stance: right foot in front), and where it came from.

    "travel": the principal direction of the root's ground path, used when the root really
    travels along a line (extent >= min_travel_m, major/minor spread >= min_linearity). In a
    side view that is the best-observed direction of the whole clip.
    "feet": otherwise (a hold, a lunge on the spot), the median back -> front foot direction
    over the wider-stance frames, where it is least noisy.
    """
    feet = fk.sole_center[:, 1, :2] - fk.sole_center[:, 0, :2]
    width = np.linalg.norm(feet, axis=1)
    wide = width >= np.percentile(width, 25)
    feet_dir = np.median(feet[wide] / np.maximum(width[wide, None], 1e-9), axis=0)
    feet_dir /= max(float(np.linalg.norm(feet_dir)), 1e-9)
    xy = motion.root_pos[:, :2] - motion.root_pos[:, :2].mean(axis=0)
    if motion.n_frames >= 3:
        _, spread, axes = np.linalg.svd(xy, full_matrices=False)
        major = axes[0]
        if np.ptp(xy @ major) >= min_travel_m and spread[0] >= min_linearity * max(spread[1], 1e-9):
            major = major if major @ feet_dir >= 0 else -major
            return float(np.arctan2(major[1], major[0])), "travel"
    return float(np.arctan2(feet_dir[1], feet_dir[0])), "feet"


def initial_heading(motion: Motion, fk: FKResult | None = None, source: str = "pelvis", window_s: float = 0.3) -> float:
    """Heading at the start of the motion, averaged over `window_s`.

    source="pelvis": pelvis yaw.
    source="feet": the fencing line, back foot -> front foot (needs fk and assumes a
    right-handed stance, i.e. right foot in front). In en garde the pelvis is turned
    30-60 deg off the line, so the two conventions differ by that angle.
    source="line": the whole clip's fencing line (`fencing_line`, needs fk): travel
    direction for footwork, the feet otherwise. The pipeline default, so footwork travels
    along +x and the bladework's "+x = fencing line" holds.
    """
    n = max(1, min(motion.n_frames, int(round(window_s * motion.fps))))
    if source == "line":
        if fk is None:
            raise ValueError("initial_heading(source='line') needs forward kinematics")
        return fencing_line(motion, fk)[0]
    if source == "pelvis":
        return mean_angle(yaw_of(motion.root_quat[:n]))
    if source == "feet":
        if fk is None:
            raise ValueError("initial_heading(source='feet') needs forward kinematics")
        line = fk.sole_center[:n, 1, :2] - fk.sole_center[:n, 0, :2]
        return mean_angle(np.arctan2(line[:, 1], line[:, 0]))
    raise ValueError(f"unknown heading source {source!r}; expected 'pelvis', 'feet' or 'line'")


def canonicalize(motion: Motion, heading: float | None = None) -> tuple[Motion, dict]:
    """Start at the origin (xy) facing +x. z is untouched (ground alignment owns it)."""
    if heading is None:
        heading = initial_heading(motion)
    start_xy = motion.root_pos[0, :2].copy()
    out = rotate_about_z(motion, -heading, start_xy)
    return out, {"rotated_deg": float(np.degrees(-heading)), "translated_xy": (-start_xy).tolist()}


# ---------------------------------------------------------------- mirroring


@dataclass(frozen=True)
class MirrorMap:
    partner: np.ndarray  # (29,) CSV column of each joint's mirror partner
    sign: np.ndarray  # (29,) +1 / -1 applied after swapping


def _partner_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


@lru_cache(maxsize=4)
def _mirror_map_cached(mjcf_path: str) -> MirrorMap:
    g1 = load_g1(mjcf_path)
    return derive_mirror_map(g1)


def derive_mirror_map(g1: G1Model) -> MirrorMap:
    partner = np.array([JOINT_NAMES.index(_partner_name(n)) for n in JOINT_NAMES])
    sign = np.empty(len(JOINT_NAMES))
    for j, p in enumerate(partner):
        mirrored_axis = REFLECT_Y @ g1.joint_axes[j]
        alignment = float(np.dot(mirrored_axis, g1.joint_axes[p]))
        if abs(abs(alignment) - 1.0) > 1e-6:
            raise ValueError(
                f"joint {JOINT_NAMES[j]} axis {g1.joint_axes[j]} does not mirror onto "
                f"{JOINT_NAMES[p]} axis {g1.joint_axes[p]}; the model is not left/right symmetric"
            )
        sign[j] = -alignment
    return MirrorMap(partner=partner, sign=sign)


def mirror_map(g1: G1Model | None = None) -> MirrorMap:
    g1 = g1 or load_g1()
    return _mirror_map_cached(str(g1.mjcf_path))


def mirror_motion(motion: Motion, g1: G1Model | None = None) -> Motion:
    """Left-handed <-> right-handed. Mirror of a mirror is the original motion."""
    mmap = mirror_map(g1)
    root = motion.root_pos.copy()
    root[:, 1] *= -1
    # M R M for M = diag(1,-1,1): rotations about x and z flip, about y stay -> (-x, y, -z, w).
    quat = motion.root_quat * np.array([-1.0, 1.0, -1.0, 1.0])
    # Output column j takes its partner's angle; the sign belongs to the source joint.
    joints = motion.joint_pos[:, mmap.partner] * mmap.sign[mmap.partner]
    meta = dict(motion.meta)
    meta["mirrored"] = not meta.get("mirrored", False)
    return Motion(root, quat, joints, motion.fps, meta)


# ---------------------------------------------------------------- handedness


@dataclass(frozen=True)
class Handedness:
    hand: str  # "R" or "L"
    confidence: float  # 0..1
    evidence: dict


def detect_handedness(fk: FKResult, frames: np.ndarray | None = None) -> Handedness:
    """Which hand holds the sword, from the en garde stance.

    In en garde the front foot is on the sword-arm side, its toes point down the fencing
    line, the back foot is turned ~90 deg, and the pelvis is turned so its sword side leads.
    With d = right_foot - left_foot (ground plane):
      * toe cue: right-handed -> right toe parallel to d, left toe ~perpendicular to it;
                 left-handed  -> left toe parallel to -d, right toe ~perpendicular.
      * pelvis cue: the pelvis forward axis has a positive component along d when the
                    right foot leads (right-handed), negative when the left foot leads.
    Both cues are scored in [-1, 1] (positive = right-handed) over `frames` (default: all)
    and combined. Near-zero means a square stance (not en garde) -> low confidence.
    """
    n = fk.sole_center.shape[0]
    frames = np.arange(n) if frames is None else np.asarray(frames)
    d = fk.sole_center[frames, 1, :2] - fk.sole_center[frames, 0, :2]
    d_unit = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    toe_dirs = fk.foot_rot[frames][..., :2, 0]  # (F, 2, 2) toe direction xy per foot
    toe_dirs = toe_dirs / np.maximum(np.linalg.norm(toe_dirs, axis=-1, keepdims=True), 1e-6)
    right_parallel = np.einsum("fi,fi->f", toe_dirs[:, 1], d_unit)  # +1 if right toe points along d
    left_parallel = np.einsum("fi,fi->f", toe_dirs[:, 0], -d_unit)  # +1 if left toe points along -d
    toe_score = float(np.clip(np.mean(right_parallel - left_parallel) / 2.0, -1, 1))
    pelvis_idx = fk.body_names.index("pelvis")
    pelvis_fwd = fk.body_rot[frames, pelvis_idx][:, :2, 0]
    pelvis_fwd = pelvis_fwd / np.maximum(np.linalg.norm(pelvis_fwd, axis=1, keepdims=True), 1e-6)
    pelvis_score = float(np.clip(np.mean(np.einsum("fi,fi->f", pelvis_fwd, d_unit)) / 0.5, -1, 1))
    stance_width = float(np.median(np.linalg.norm(d, axis=1)))
    combined = 0.5 * toe_score + 0.5 * pelvis_score
    # A stance narrower than ~25 cm is not en garde; both cues are then meaningless.
    width_factor = float(np.clip((stance_width - 0.15) / 0.15, 0.0, 1.0))
    return Handedness(
        hand="R" if combined >= 0 else "L",
        confidence=float(min(1.0, abs(combined)) * width_factor),
        evidence={
            "toe_score": toe_score,
            "pelvis_score": pelvis_score,
            "stance_width_m": stance_width,
            "frames_used": int(len(frames)),
        },
    )
