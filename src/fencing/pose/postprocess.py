"""Raw MediaPipe output of one clip -> clean 30 fps world keypoints (the fallback pose path).

Pure numpy/scipy, runs in the `fencing` venv after `fencing.pose.mediapipe_worker`:

    raw = dict(np.load("pose/<clip_id>/mediapipe_raw.npz"))
    result = process_mediapipe(raw, human_height_m=1.75)

Steps: drop frames with no pose and interpolate over them, undo left/right leg and arm label
flips, resample onto a uniform 30 fps clock from the frame timestamps, smooth, lift into a
z-up metric world with `fencing.pose.root_trajectory`, guess the sword hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fencing.pose import landmarks as lm
from fencing.pose.handedness import guess_handedness, handedness_score
from fencing.pose.root_trajectory import (
    DEFAULT_HUMAN_HEIGHT_M,
    PinholeCamera,
    estimate_root_trajectory,
)

OUTPUT_FPS = 30.0
MIN_VALID_FRACTION = 0.6
# MediaPipe world landmarks jitter by ~1 cm frame to frame; 0.2 s keeps a lunge's ~0.3 s
# extension intact while removing it.
WORLD_SMOOTH_S = 0.2
IMAGE_SMOOTH_S = 0.1
VISIBLE = 0.5


@dataclass
class PoseResult:
    times_s: np.ndarray  # (T,) source-video seconds, uniform at OUTPUT_FPS
    keypoints_world: np.ndarray  # (T, 33, 3) z-up metres, feet on z = 0, starts at x = y = 0
    image_xy: np.ndarray  # (T, 33, 2) cleaned full-image pixels (for overlays)
    visibility: np.ndarray  # (T, 33)
    valid: np.ndarray  # (T,) bool: a real MediaPipe detection near this time (else interpolated)
    root_cam: np.ndarray  # (T, 3)
    scale: float
    pitch_down_rad: float
    focal_px: float
    handedness: str | None
    handedness_score: float
    stats: dict = field(default_factory=dict)


def body_scale_px(image_xy: np.ndarray) -> np.ndarray:
    """Hip-centre to shoulder-centre distance in pixels per frame, (T,)."""
    mid_hip = (image_xy[:, lm.LEFT_HIP] + image_xy[:, lm.RIGHT_HIP]) / 2.0
    mid_sh = (image_xy[:, lm.LEFT_SHOULDER] + image_xy[:, lm.RIGHT_SHOULDER]) / 2.0
    return np.linalg.norm(mid_sh - mid_hip, axis=-1)


def uniform_times(start_s: float, end_s: float, fps: float = OUTPUT_FPS) -> np.ndarray:
    """Frame times start_s, start_s + 1/fps, ... not past end_s (at least two frames)."""
    if end_s <= start_s:
        raise ValueError(f"need end > start, got [{start_s}, {end_s}]")
    n = max(2, int(np.floor((end_s - start_s) * fps + 1e-6)) + 1)
    return start_s + np.arange(n) / fps


def resample_valid(times_s: np.ndarray, valid: np.ndarray, out_times_s: np.ndarray) -> np.ndarray:
    """Validity of the nearest source frame at each output time."""
    idx = np.searchsorted(times_s, out_times_s).clip(1, len(times_s) - 1)
    nearest = np.where(out_times_s - times_s[idx - 1] <= times_s[idx] - out_times_s, idx - 1, idx)
    return np.asarray(valid, dtype=bool)[nearest]


def clean_landmarks(raw: dict, fps: float = OUTPUT_FPS) -> dict[str, np.ndarray]:
    """Gap-fill, fix L/R swaps, resample to `fps` and smooth the raw worker arrays."""
    times = np.asarray(raw["times_s"], dtype=np.float64)
    valid = np.asarray(raw["valid"], dtype=bool)
    if valid.mean() < MIN_VALID_FRACTION:
        raise ValueError(f"MediaPipe found a pose in only {valid.mean():.0%} of frames (need {MIN_VALID_FRACTION:.0%})")
    image_xy = lm.fill_gaps(np.asarray(raw["image_xy"], dtype=np.float64), valid)
    world = lm.fill_gaps(np.asarray(raw["world"], dtype=np.float64), valid)
    # Interpolated, not zeroed: the root solve needs weights in every frame; `valid` keeps the truth.
    visibility = lm.fill_gaps(np.asarray(raw["visibility"], dtype=np.float64), valid)

    (world, visibility, image_fixed), swapped = lm.fix_left_right_swaps(
        image_xy, [world, visibility, image_xy], body_scale_px(image_xy)
    )
    out_t = uniform_times(times[0], times[-1], fps)
    image_out = lm.resample_uniform(times, image_fixed, out_t)
    world_out = lm.resample_uniform(times, world, out_t)
    vis_out = lm.resample_uniform(times, visibility, out_t)
    return {
        "times_s": out_t,
        "image_xy": lm.smooth_time(image_out, fps, IMAGE_SMOOTH_S),
        "world": lm.smooth_time(world_out, fps, WORLD_SMOOTH_S),
        "visibility": vis_out,
        "valid": resample_valid(times, valid, out_t),
        "leg_swaps": np.array(int(swapped[:, 0].sum())),
        "arm_swaps": np.array(int(swapped[:, 1].sum())),
    }


def quality_stats(cleaned: dict, keypoints_world: np.ndarray, raw_valid: np.ndarray) -> dict:
    """Numbers for pose.json that say whether a clip is worth retargeting."""
    vis = cleaned["visibility"][:, lm.BODY]
    feet_z = keypoints_world[:, lm.FEET, 2]
    mid_hip = (keypoints_world[:, lm.LEFT_HIP] + keypoints_world[:, lm.RIGHT_HIP]) / 2.0
    step = np.linalg.norm(np.diff(mid_hip[:, :2], axis=0), axis=-1)
    shin = np.linalg.norm(keypoints_world[:, lm.LEFT_KNEE] - keypoints_world[:, lm.LEFT_ANKLE], axis=-1)
    return {
        "frames": int(len(keypoints_world)),
        "raw_frames": int(len(raw_valid)),
        "valid_frac": round(float(np.mean(raw_valid)), 3),
        "median_visibility": round(float(np.median(vis)), 3),
        "visible_landmark_frac": round(float(np.mean(vis > VISIBLE)), 3),
        "leg_swaps_fixed": int(cleaned["leg_swaps"]),
        "arm_swaps_fixed": int(cleaned["arm_swaps"]),
        "root_path_m": round(float(step.sum()), 3),
        "root_max_speed_mps": round(float(step.max() * OUTPUT_FPS) if len(step) else 0.0, 3),
        "pelvis_height_median_m": round(float(np.median(mid_hip[:, 2])), 3),
        "upper_foot_height_median_m": round(float(np.median(np.sort(feet_z, axis=1)[:, -1])), 3),
        "shin_length_cv": round(float(np.std(shin) / max(np.mean(shin), 1e-9)), 3),
    }


def process_mediapipe(
    raw: dict,
    human_height_m: float = DEFAULT_HUMAN_HEIGHT_M,
    hfov_deg: float | None = None,
    pitch_down_deg: float | None = None,
    fps: float = OUTPUT_FPS,
) -> PoseResult:
    cleaned = clean_landmarks(raw, fps)
    camera = PinholeCamera.from_image(int(raw["width"]), int(raw["height"]), hfov_deg)
    traj = estimate_root_trajectory(
        cleaned["image_xy"],
        cleaned["world"],
        cleaned["visibility"],
        camera,
        fps,
        human_height_m=human_height_m,
        pitch_down_rad=None if pitch_down_deg is None else float(np.deg2rad(pitch_down_deg)),
    )
    score, _ = handedness_score(traj.keypoints_world)
    stats = quality_stats(cleaned, traj.keypoints_world, np.asarray(raw["valid"], dtype=bool))
    return PoseResult(
        times_s=cleaned["times_s"],
        keypoints_world=traj.keypoints_world,
        image_xy=cleaned["image_xy"],
        visibility=cleaned["visibility"],
        valid=cleaned["valid"],
        root_cam=traj.root_cam,
        scale=float(traj.scale),
        pitch_down_rad=float(traj.pitch_down_rad),
        focal_px=float(camera.focal_px),
        handedness=guess_handedness(traj.keypoints_world),
        handedness_score=float(score),
        stats=stats,
    )
