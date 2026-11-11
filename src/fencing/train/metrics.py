"""Goal 1 evaluation metrics as pure numpy functions.

Shared by `fencing train eval` (mjlab, many parallel envs) and `fencing sim2sim` (plain
MuJoCo, one env) so both report the same definitions. Arrays carry a leading env axis
`E` (use E=1 for a single rollout); per-step values are accumulated with `MaskedMean`
so an env stops contributing once it has terminated, as in mjlab's evaluate.py.

Definitions (docs/evaluation.md has the rationale):
    success      no termination before the last reference frame
    foot slip    horizontal speed of a robot foot while it should be planted: the
                 reference foot is low and still, *and* the robot foot is low. Requiring
                 both separates slip from a tracking error (foot lifted when it should not).
    recovery     success, no termination during the hold period after the motion, and at
                 the end of the hold the pelvis is near the reference final height and
                 nearly still
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MaskedMean:
    """Per-env running mean over the steps where `mask` is true."""

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.total = np.zeros(num_envs, dtype=np.float64)
        self.count = np.zeros(num_envs, dtype=np.int64)

    def add(self, values: np.ndarray, mask: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.shape != self.total.shape or mask.shape != self.total.shape:
            raise ValueError(
                f"expected values and mask of shape {self.total.shape}, got {values.shape} "
                f"and {mask.shape}"
            )
        self.total += np.where(mask, values, 0.0)
        self.count += mask

    def per_env(self) -> np.ndarray:
        """NaN for envs that never contributed (e.g. a foot that was never planted)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.total / np.maximum(self.count, 1), np.nan)

    def mean(self) -> float:
        """Mean of per-env means (mjlab's convention), ignoring envs with no samples."""
        per_env = self.per_env()
        valid = ~np.isnan(per_env)
        return float(per_env[valid].mean()) if valid.any() else float("nan")

    def pooled(self) -> float:
        """Mean over all contributing samples (weights long-lived envs more)."""
        count = int(self.count.sum())
        return float(self.total.sum() / count) if count else float("nan")


def joint_angle_error(reference_joint_pos: np.ndarray, robot_joint_pos: np.ndarray) -> np.ndarray:
    """Mean absolute joint-angle error over joints, radians. Shapes (E, J) -> (E,)."""
    reference_joint_pos = np.asarray(reference_joint_pos)
    robot_joint_pos = np.asarray(robot_joint_pos)
    if reference_joint_pos.shape != robot_joint_pos.shape:
        raise ValueError(
            f"joint arrays differ in shape: {reference_joint_pos.shape} vs {robot_joint_pos.shape}"
        )
    return np.abs(reference_joint_pos - robot_joint_pos).mean(axis=-1)


def mean_body_position_error(reference_pos: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
    """MPKPE: mean Euclidean distance over bodies. Shapes (E, B, 3) -> (E,)."""
    return np.linalg.norm(np.asarray(reference_pos) - np.asarray(robot_pos), axis=-1).mean(axis=-1)


def estimate_ground_height(reference_foot_z: np.ndarray, percentile: float = 2.0) -> float:
    """Foot-frame height when standing, from the reference itself.

    The foot body's origin sits a few cm above the sole, so contact is "near the lowest
    height this foot reaches in the motion". A low percentile ignores single-frame dips
    from retargeting noise.
    """
    reference_foot_z = np.asarray(reference_foot_z, dtype=np.float64)
    if reference_foot_z.size == 0:
        raise ValueError("reference_foot_z is empty")
    return float(np.percentile(reference_foot_z, percentile))


def reference_planted_mask(
    reference_foot_z: np.ndarray,
    reference_foot_speed_xy: np.ndarray,
    ground_z: float,
    height_margin: float = 0.03,
    speed_threshold: float = 0.15,
) -> np.ndarray:
    """True where the reference foot is on the ground and not moving horizontally."""
    reference_foot_z = np.asarray(reference_foot_z)
    reference_foot_speed_xy = np.asarray(reference_foot_speed_xy)
    if reference_foot_z.shape != reference_foot_speed_xy.shape:
        raise ValueError(
            f"height and speed arrays differ in shape: {reference_foot_z.shape} vs "
            f"{reference_foot_speed_xy.shape}"
        )
    return (reference_foot_z < ground_z + height_margin) & (reference_foot_speed_xy < speed_threshold)


def foot_slip_mask(
    reference_planted: np.ndarray,
    robot_foot_z: np.ndarray,
    ground_z: float,
    height_margin: float = 0.03,
) -> np.ndarray:
    """Frames where slip is measured: reference planted and the robot foot is down too."""
    return np.asarray(reference_planted, dtype=bool) & (np.asarray(robot_foot_z) < ground_z + height_margin)


def horizontal_speed(velocity: np.ndarray) -> np.ndarray:
    """|v_xy| for velocities shaped (..., 3) or (..., 2)."""
    velocity = np.asarray(velocity)
    if velocity.shape[-1] not in (2, 3):
        raise ValueError(f"velocity last axis must be 2 or 3, got shape {velocity.shape}")
    return np.linalg.norm(velocity[..., :2], axis=-1)


def foot_slip_speed(
    robot_foot_vel: np.ndarray, slip_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-env sum of slip speed and number of (foot, step) samples for one step.

    robot_foot_vel: (E, F, 3) world linear velocity of each foot body.
    slip_mask:      (E, F) from `foot_slip_mask`.
    Returns (sum over feet of speed where masked, count of masked feet), each (E,), so a
    caller can accumulate a per-env mean speed over a whole rollout.
    """
    speed = horizontal_speed(robot_foot_vel)
    slip_mask = np.asarray(slip_mask, dtype=bool)
    if speed.shape != slip_mask.shape:
        raise ValueError(f"foot speed {speed.shape} and mask {slip_mask.shape} differ in shape")
    return np.where(slip_mask, speed, 0.0).sum(axis=-1), slip_mask.sum(axis=-1)


class FootSlipAccumulator:
    """Mean horizontal planted-foot speed per env, m/s."""

    def __init__(self, num_envs: int):
        self.total = np.zeros(num_envs, dtype=np.float64)
        self.count = np.zeros(num_envs, dtype=np.int64)

    def add(self, robot_foot_vel: np.ndarray, slip_mask: np.ndarray, active: np.ndarray) -> None:
        speed_sum, count = foot_slip_speed(robot_foot_vel, slip_mask)
        active = np.asarray(active, dtype=bool)
        self.total += np.where(active, speed_sum, 0.0)
        self.count += np.where(active, count, 0)

    def mean(self) -> float:
        count = int(self.count.sum())
        return float(self.total.sum() / count) if count else float("nan")

    def planted_samples(self) -> int:
        return int(self.count.sum())


NOT_TERMINATED = -1


def termination_time_s(terminated_at_step, step_dt: float):
    """Seconds simulated up to a termination at 0-based step k: the env ran k + 1 steps."""
    return (terminated_at_step + 1) * step_dt


@dataclass(frozen=True)
class EpisodeOutcomes:
    num_episodes: int
    success: np.ndarray  # (E,) finished the motion without termination
    survived_window: np.ndarray  # (E,) no termination in the first `window_steps`
    recovered: np.ndarray  # (E,) success + stable at end of hold

    @property
    def success_count(self) -> int:
        return int(self.success.sum())

    @property
    def success_rate(self) -> float:
        return self.success_count / self.num_episodes

    @property
    def window_success_rate(self) -> float:
        return float(self.survived_window.mean())

    @property
    def recovery_rate(self) -> float:
        return float(self.recovered.mean())


def episode_outcomes(
    terminated_at_step: np.ndarray,
    motion_steps: int,
    window_steps: int,
    final_pelvis_height_error: np.ndarray | None = None,
    final_pelvis_speed: np.ndarray | None = None,
    height_tolerance: float = 0.05,
    speed_tolerance: float = 0.2,
) -> EpisodeOutcomes:
    """Classify episodes from the policy step at which each was terminated.

    terminated_at_step: (E,) step index (0-based, counted from the start of the motion) of
        the first termination, or NOT_TERMINATED. Time-outs are not terminations.
    motion_steps: number of policy steps that cover the reference motion. A termination at
        step >= motion_steps happened during the hold, after the motion was finished.
    window_steps: mjlab's evaluation horizon (episode_length_s / step_dt = 500 for 10 s),
        reported so our numbers stay comparable with mjlab's evaluate.py.
    final_pelvis_*: measured at the end of the hold; None -> recovery not evaluated (all False).
    """
    terminated_at_step = np.asarray(terminated_at_step, dtype=np.int64)
    if terminated_at_step.ndim != 1 or terminated_at_step.size == 0:
        raise ValueError(f"terminated_at_step must be a non-empty 1-D array, got {terminated_at_step.shape}")
    if motion_steps <= 0 or window_steps <= 0:
        raise ValueError(f"motion_steps and window_steps must be positive, got {motion_steps}, {window_steps}")
    bad = terminated_at_step[(terminated_at_step < 0) & (terminated_at_step != NOT_TERMINATED)]
    if bad.size:
        raise ValueError(f"invalid termination steps {bad[:5].tolist()} (use {NOT_TERMINATED} for none)")
    never = terminated_at_step == NOT_TERMINATED
    success = never | (terminated_at_step >= motion_steps)
    survived_window = never | (terminated_at_step >= min(window_steps, motion_steps))
    if final_pelvis_height_error is None or final_pelvis_speed is None:
        recovered = np.zeros_like(success)
    else:
        stable = (np.abs(np.asarray(final_pelvis_height_error)) < height_tolerance) & (
            np.asarray(final_pelvis_speed) < speed_tolerance
        )
        recovered = never & stable
    return EpisodeOutcomes(
        num_episodes=int(terminated_at_step.size),
        success=success,
        survived_window=survived_window,
        recovered=recovered,
    )


def nan_to_none(value: float) -> float | None:
    """JSON has no NaN; write null for "not measured"."""
    return None if value is None or not np.isfinite(value) else float(value)


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation angle between (..., 4) wxyz quaternions, radians in [0, pi]."""
    from fencing.sim2sim.quat import quat_error_magnitude

    return quat_error_magnitude(q1, q2)


def blade_tip_from_wrist(wrist_pos: np.ndarray, wrist_quat: np.ndarray, tip_in_wrist: np.ndarray) -> np.ndarray:
    """Blade-tip world position from the right_wrist_yaw_link pose (the saber is welded)."""
    from fencing.sim2sim.quat import quat_apply

    return np.asarray(wrist_pos) + quat_apply(wrist_quat, np.broadcast_to(tip_in_wrist, np.shape(wrist_pos)))


# Half-width, in policy steps, of the window scored around each blade key frame: frames
# key-2..key+2 (0.1 s at 50 Hz). The reference tip is near rest there (min-jerk arrival,
# then a hold), the 30 fps keyframe lands on the 50 Hz grid only to within a frame, and five
# samples damp single-step jitter in the one-episode sim2sim and MTC scores.
KEY_WINDOW_STEPS = 2


def key_window_frames(key_frames, half_width: int = KEY_WINDOW_STEPS) -> frozenset[int]:
    """Reference frames within `half_width` of any key frame (overlapping windows merge)."""
    return frozenset(f for k in key_frames for f in range(int(k) - half_width, int(k) + half_width + 1) if f >= 0)


class TrackingAccumulator:
    """Accumulates the per-step Goal 1 tracking metrics for E envs over the motion.

    Feed one `add()` per policy step with the robot state *after* the step and the
    reference frame that step was scored against (mjlab's evaluate.py pairing). Only
    envs flagged `active` contribute. Positions are world frame; `ref_body_pos_rel_w` is
    the reference re-anchored to the robot's root x/y and heading (mjlab's
    body_pos_relative_w), which removes global drift.

    `key_frames` (reference frames at the policy rate, e.g. the parry or the hit): the
    blade-tip error is also averaged over the window around them, and `add()` then needs the
    `frame` of each step. Without key frames the summary is exactly what it always was.
    """

    def __init__(self, num_envs: int, body_names: list[str] | tuple[str, ...], foot_indices: list[int],
                 ground_z: float, key_frames=None):
        self.key_frames = None if key_frames is None else sorted(int(k) for k in key_frames)
        self.key_window = key_window_frames(self.key_frames or ())
        self.blade_tip_key = MaskedMean(num_envs)
        self.blade_tip_key_global = MaskedMean(num_envs)
        self.num_envs = num_envs
        self.body_names = list(body_names)
        self.foot_indices = list(foot_indices)
        self.ground_z = float(ground_z)
        self.joint_angle = MaskedMean(num_envs)
        self.body_pos_rel = MaskedMean(num_envs)
        self.body_pos_global = MaskedMean(num_envs)
        self.anchor_pos = MaskedMean(num_envs)
        self.anchor_ori = MaskedMean(num_envs)
        self.blade_tip = MaskedMean(num_envs)
        self.blade_tip_global = MaskedMean(num_envs)
        self.blade_tip_max = np.full(num_envs, np.nan)
        self.per_body_total = np.zeros((num_envs, len(self.body_names)))
        self.per_body_count = np.zeros(num_envs, dtype=np.int64)
        self.foot_slip = FootSlipAccumulator(num_envs)
        self.steps = 0

    def add(self, *, active, ref_joint_pos, robot_joint_pos, ref_body_pos_w, ref_body_pos_rel_w,
            robot_body_pos_w, ref_anchor_pos_w, ref_anchor_quat_w, robot_anchor_pos_w, robot_anchor_quat_w,
            ref_body_lin_vel_w, robot_body_lin_vel_w, ref_tip_rel_w=None, ref_tip_w=None, robot_tip_w=None,
            frame: int | None = None) -> None:
        if self.key_frames is not None and frame is None:
            raise ValueError("TrackingAccumulator has key frames: add() needs the reference frame of the step")
        active = np.asarray(active, dtype=bool)
        self.steps += 1
        self.joint_angle.add(joint_angle_error(ref_joint_pos, robot_joint_pos), active)
        per_body = np.linalg.norm(np.asarray(ref_body_pos_rel_w) - np.asarray(robot_body_pos_w), axis=-1)
        self.body_pos_rel.add(per_body.mean(axis=-1), active)
        self.per_body_total += np.where(active[:, None], per_body, 0.0)
        self.per_body_count += active
        self.body_pos_global.add(mean_body_position_error(ref_body_pos_w, robot_body_pos_w), active)
        self.anchor_pos.add(np.linalg.norm(np.asarray(ref_anchor_pos_w) - np.asarray(robot_anchor_pos_w), axis=-1),
                            active)
        self.anchor_ori.add(quat_angle_error(ref_anchor_quat_w, robot_anchor_quat_w), active)

        feet = self.foot_indices
        ref_feet_z = np.asarray(ref_body_pos_w)[:, feet, 2]
        ref_feet_speed = horizontal_speed(np.asarray(ref_body_lin_vel_w)[:, feet])
        planted = reference_planted_mask(ref_feet_z, ref_feet_speed, self.ground_z)
        slip_mask = foot_slip_mask(planted, np.asarray(robot_body_pos_w)[:, feet, 2], self.ground_z)
        self.foot_slip.add(np.asarray(robot_body_lin_vel_w)[:, feet], slip_mask, active)

        if robot_tip_w is not None:
            tip_error = np.linalg.norm(np.asarray(ref_tip_rel_w) - np.asarray(robot_tip_w), axis=-1)
            self.blade_tip.add(tip_error, active)
            self.blade_tip_global.add(np.linalg.norm(np.asarray(ref_tip_w) - np.asarray(robot_tip_w), axis=-1),
                                      active)
            self.blade_tip_max = np.where(active, np.fmax(self.blade_tip_max, tip_error), self.blade_tip_max)
            if frame is not None and int(frame) in self.key_window:
                self.blade_tip_key.add(tip_error, active)
                self.blade_tip_key_global.add(
                    np.linalg.norm(np.asarray(ref_tip_w) - np.asarray(robot_tip_w), axis=-1), active)

    def key_frame_summary(self) -> dict:
        """Key-frame blade metrics; empty without key frames so old reports keep their keys."""
        if self.key_frames is None:
            return {}
        return {
            "blade_key_frames": list(self.key_frames),
            "blade_tip_error_key_m": nan_to_none(self.blade_tip_key.mean()),
            "blade_tip_error_key_global_m": nan_to_none(self.blade_tip_key_global.mean()),
        }

    def summary(self) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_body_env = np.where(self.per_body_count[:, None] > 0,
                                    self.per_body_total / np.maximum(self.per_body_count[:, None], 1), np.nan)
        per_body = {}
        for i, name in enumerate(self.body_names):
            column = per_body_env[:, i]
            per_body[name] = nan_to_none(float(np.nanmean(column))) if np.isfinite(column).any() else None
        has_blade = bool(self.blade_tip.count.any())
        return {
            "joint_angle_error_rad": nan_to_none(self.joint_angle.mean()),
            "body_pos_error_m": nan_to_none(self.body_pos_rel.mean()),
            "body_pos_error_global_m": nan_to_none(self.body_pos_global.mean()),
            "per_body_pos_error_m": per_body,
            "anchor_pos_error_m": nan_to_none(self.anchor_pos.mean()),
            "anchor_ori_error_rad": nan_to_none(self.anchor_ori.mean()),
            "foot_slip_m_s": nan_to_none(self.foot_slip.mean()),
            "foot_slip_samples": self.foot_slip.planted_samples(),
            "blade_tip_error_m": nan_to_none(self.blade_tip.mean()) if has_blade else None,
            "blade_tip_error_global_m": nan_to_none(self.blade_tip_global.mean()) if has_blade else None,
            "blade_tip_error_max_m": (nan_to_none(float(np.nanmax(self.blade_tip_max)))
                                      if has_blade and np.isfinite(self.blade_tip_max).any() else None),
            **self.key_frame_summary(),
            "ground_z_m": self.ground_z,
            "scored_steps": self.steps,
        }


def outcome_summary(outcomes: EpisodeOutcomes, hold_evaluated: bool) -> dict:
    return {
        "num_episodes": outcomes.num_episodes,
        "success_rate": outcomes.success_rate,
        "window_success_rate": outcomes.window_success_rate,
        "recovery_rate": outcomes.recovery_rate if hold_evaluated else None,
    }
