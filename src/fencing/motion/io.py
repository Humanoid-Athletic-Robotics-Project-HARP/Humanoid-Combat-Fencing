"""Read, write and validate G1 motion CSVs and their JSON sidecars.

CSV format (headerless, one row per frame, see REPO/motions/README.md):
    x, y, z,  qx, qy, qz, qw,  <29 joint angles>
Root position in metres (z up), root quaternion xyzw, joint angles in radians, in the
Unitree G1 29-DoF order below. That order is the `joint_names` list of mjlab's
`csv_to_npz.py`; kinematics maps columns to MJCF qpos by these names, never by index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
N_JOINTS = len(JOINT_NAMES)
N_ROOT_COLS = 7
N_COLS = N_ROOT_COLS + N_JOINTS
DEFAULT_FPS = 30.0

# A unit quaternion written with 6 decimals has |q| within ~1e-5 of 1; anything far off
# means the columns are shifted or the producer wrote wxyz/euler instead of xyzw.
QUAT_NORM_TOLERANCE = 1e-2


class MotionFormatError(ValueError):
    """The CSV or sidecar does not follow the G1 motion format."""


@dataclass
class Motion:
    """A G1 trajectory. Arrays are float64 and never shared between Motion instances."""

    root_pos: np.ndarray  # (T, 3) metres, world frame, z up
    root_quat: np.ndarray  # (T, 4) xyzw, world frame
    joint_pos: np.ndarray  # (T, 29) radians, JOINT_NAMES order
    fps: float = DEFAULT_FPS
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.n_frames - 1) / self.fps if self.n_frames else 0.0

    def copy(self) -> Motion:
        return Motion(
            self.root_pos.copy(),
            self.root_quat.copy(),
            self.joint_pos.copy(),
            float(self.fps),
            dict(self.meta),
        )

    def replace(self, **changes) -> Motion:
        """Copy with some fields replaced (arrays not passed are copied, not shared)."""
        new = self.copy()
        return replace(new, **changes)

    def slice(self, start: int, end: int) -> Motion:
        if not 0 <= start < end <= self.n_frames:
            raise ValueError(f"bad frame range [{start}, {end}) for a {self.n_frames}-frame motion")
        return Motion(
            self.root_pos[start:end].copy(),
            self.root_quat[start:end].copy(),
            self.joint_pos[start:end].copy(),
            self.fps,
            dict(self.meta),
        )

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.root_pos, self.root_quat, self.joint_pos], axis=1)

    @classmethod
    def from_array(cls, array: np.ndarray, fps: float = DEFAULT_FPS, meta: dict | None = None) -> Motion:
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != N_COLS:
            raise MotionFormatError(
                f"expected a (frames, {N_COLS}) array (7 root + {N_JOINTS} joints), got shape {array.shape}"
            )
        return cls(
            array[:, :3].copy(),
            array[:, 3:7].copy(),
            array[:, 7:].copy(),
            float(fps),
            dict(meta or {}),
        )


def sidecar_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".json")


def read_sidecar(csv_path: str | Path) -> dict | None:
    """The retarget stage's `<clip_id>.json` next to the CSV, or None if there is none."""
    path = sidecar_path(csv_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MotionFormatError(f"sidecar {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MotionFormatError(f"sidecar {path} must hold a JSON object, got {type(data).__name__}")
    if "fps" in data and not (isinstance(data["fps"], (int, float)) and data["fps"] > 0):
        raise MotionFormatError(f"sidecar {path} has invalid fps {data['fps']!r}")
    hand = data.get("handedness_guess")
    if hand not in (None, "R", "L"):
        raise MotionFormatError(f"sidecar {path} has handedness_guess {hand!r}; expected 'R', 'L' or null")
    return data


def read_csv(path: str | Path, fps: float | None = None, validate_data: bool = True) -> Motion:
    """Load a motion CSV. fps: explicit > sidecar > 30."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"motion CSV not found: {path}")
    try:
        array = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError as exc:
        raise MotionFormatError(f"{path}: not a numeric headerless CSV ({exc})") from exc
    if array.shape[1] != N_COLS:
        raise MotionFormatError(
            f"{path}: expected {N_COLS} columns (7 root + {N_JOINTS} joints), got {array.shape[1]}"
        )
    sidecar = read_sidecar(path)
    if fps is None:
        fps = float(sidecar["fps"]) if sidecar and "fps" in sidecar else DEFAULT_FPS
    meta = {"source_path": str(path)}
    if sidecar:
        meta["sidecar"] = sidecar
    motion = Motion.from_array(array, fps=fps, meta=meta)
    if validate_data:
        check_motion(motion, label=str(path))
    return motion


def write_csv(motion: Motion, path: str | Path, exclusive: bool = False) -> Path:
    """Write the 36-column CSV. exclusive=True refuses to overwrite (used by publish)."""
    check_motion(motion, label=str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quat = normalize_quaternions(motion.root_quat)
    array = np.concatenate([motion.root_pos, quat, motion.joint_pos], axis=1)
    with open(path, "x" if exclusive else "w") as fh:
        np.savetxt(fh, array, delimiter=",", fmt="%.6f")
    return path


def write_sidecar(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        bad = int(np.argmax(norms[:, 0] < 1e-9))
        raise MotionFormatError(f"zero root quaternion at frame {bad}")
    return quat / norms


def validate_motion(motion: Motion) -> list[str]:
    """Every format problem found, as human-readable strings (empty list = valid)."""
    problems: list[str] = []
    n = motion.n_frames
    shapes = {"root_pos": (n, 3), "root_quat": (n, 4), "joint_pos": (n, N_JOINTS)}
    for name, expected in shapes.items():
        actual = getattr(motion, name).shape
        if actual != expected:
            problems.append(f"{name} has shape {actual}, expected {expected}")
    if problems:
        return problems
    if n < 2:
        problems.append(f"motion has {n} frame(s); need at least 2")
    if not (np.isfinite(motion.fps) and motion.fps > 0):
        problems.append(f"fps must be positive, got {motion.fps}")
    array = motion.to_array()
    bad = ~np.isfinite(array)
    if bad.any():
        frame, col = (int(v) for v in np.argwhere(bad)[0])
        problems.append(f"{int(bad.sum())} non-finite values; first at frame {frame}, column {col}")
        return problems
    norms = np.linalg.norm(motion.root_quat, axis=1)
    off = np.abs(norms - 1.0) > QUAT_NORM_TOLERANCE
    if off.any():
        frame = int(np.argmax(off))
        problems.append(
            f"root quaternion norm {norms[frame]:.4f} at frame {frame} (|q| must be 1; "
            f"{int(off.sum())} frames off) -- columns shifted or not xyzw?"
        )
    return problems


def check_motion(motion: Motion, label: str = "motion") -> None:
    problems = validate_motion(motion)
    if problems:
        raise MotionFormatError(f"{label}: " + "; ".join(problems))
