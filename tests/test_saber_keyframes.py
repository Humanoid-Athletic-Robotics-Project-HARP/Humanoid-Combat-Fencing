"""Bladework keyframes: minimum-jerk timing, scheduling, targets and the full IK pipeline."""

from functools import lru_cache

import numpy as np
import pytest

from fencing.motion.fixes import hardware_velocity_limits
from fencing.motion.io import JOINT_NAMES
from fencing.saber.keyframes import (
    ARM_COLUMNS,
    BladeKey,
    SchedulingError,
    apply_blade_script,
    blade_targets,
    min_move_frames,
    minimum_jerk,
    resolve_target,
    schedule_keys,
)

FPS = 30.0


# ---------------------------------------------------------------- pure parts


def test_minimum_jerk_starts_and_ends_at_rest():
    tau = np.linspace(0.0, 1.0, 1001)
    s = minimum_jerk(tau)
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(1.0)
    velocity = np.gradient(s, tau)
    assert abs(velocity[0]) < 1e-2 and abs(velocity[-1]) < 1e-2
    assert np.all(np.diff(s) >= -1e-12)  # monotone: never overshoots the target
    assert velocity.max() == pytest.approx(1.875, rel=1e-3)
    assert np.allclose(s + s[::-1], 1.0)  # symmetric


def test_minimum_jerk_clips_outside_unit_interval():
    assert minimum_jerk(np.array([-0.5, 1.5])).tolist() == [0.0, 1.0]


def test_min_move_frames_respects_speed_cap():
    delta = np.array([1.0, -2.0, 0.1])
    vmax = np.array([10.0, 10.0, 10.0])
    frames = min_move_frames(delta, vmax, FPS)
    # Peak min-jerk speed 1.875 * |dq| / T must not exceed vmax.
    assert 1.875 * 2.0 / (frames / FPS) <= 10.0
    assert min_move_frames(np.zeros(3), vmax, FPS) == 0


def test_schedule_runs_keys_back_to_back():
    keys = [BladeKey("en_garde", move_s=0.0, hold_s=0.5), BladeKey("parry_quinte", 0.3, 0.4),
            BladeKey("en_garde", 0.5, 0.5)]
    spans = schedule_keys(keys, FPS)
    assert spans == [(0, 0, 15), (15, 24, 36), (36, 51, 66)]


def test_schedule_stretches_moves_that_would_be_too_fast():
    keys = [BladeKey("en_garde", 0.0, 0.5), BladeKey("cut_head", 0.1, 0.2)]
    spans = schedule_keys(keys, FPS, min_frames=[0, 9])
    assert spans[1] == (15, 24, 30)


def test_timed_key_lands_exactly_and_shortens_the_previous_hold():
    keys = [BladeKey("en_garde", 0.0, 2.0), BladeKey("cut_head", 0.3, 0.3, at_s=1.0)]
    spans = schedule_keys(keys, FPS)
    assert spans[1] == (21, 30, 39)


def test_timed_key_that_cannot_fit_is_an_error():
    keys = [BladeKey("en_garde", 0.0, 0.0), BladeKey("extension", 0.5, 0.0, at_s=0.6),
            BladeKey("cut_head", 0.4, 0.0, at_s=0.7)]
    with pytest.raises(SchedulingError, match="cut_head"):
        schedule_keys(keys, FPS)


def test_targets_include_aliases_and_reachable_high_lines():
    targets = blade_targets()
    for name in ("en_garde", "extension", "parry_prime", "parry_quinte", "parry_tierce_high",
                 "cut_head", "cut_flank", "cut_chest", "point_thrust"):
        assert name in targets
    assert resolve_target("en_garde_guard").name == "en_garde"
    assert resolve_target("thrust").name == "point_thrust"
    with pytest.raises(KeyError, match="parry_quinte"):
        resolve_target("parry_nonsense")


# ---------------------------------------------------------------- full pipeline


@lru_cache(maxsize=1)
def _quinte():
    from fencing.motion.synthetic import synthesize

    base, _ = synthesize([("en_garde", 1.0)])
    script = [BladeKey("en_garde", 0.0, 0.4), BladeKey("parry_quinte", 0.3, 0.4),
              BladeKey("en_garde", 0.4, 0.4)]
    return base, apply_blade_script(base, script)


def test_keyframes_put_the_blade_tip_on_target():
    _, result = _quinte()
    assert [k.action for k in result.keys] == ["en_garde", "parry_quinte", "en_garde"]
    for key in result.keys:
        assert key.tip_error_m <= 0.02, key
        assert key.angle_error_deg <= 5.0, key
        assert not key.collisions


def test_bladework_leaves_legs_and_root_alone_and_returns_to_guard():
    base, result = _quinte()
    motion = result.motion
    assert motion.n_frames >= result.keys[-1].frame + 1
    n = base.n_frames
    np.testing.assert_allclose(motion.root_pos[:n], base.root_pos)
    np.testing.assert_allclose(motion.root_quat[:n], base.root_quat)
    legs = [j for j, name in enumerate(JOINT_NAMES) if "hip" in name or "knee" in name or "ankle" in name]
    np.testing.assert_allclose(motion.joint_pos[:n, legs], base.joint_pos[:, legs])
    # Starts and ends in the same en garde guard.
    np.testing.assert_allclose(motion.joint_pos[0, ARM_COLUMNS], motion.joint_pos[-1, ARM_COLUMNS], atol=1e-9)


def test_return_to_guard_on_a_changed_stance_stays_in_the_first_guards_solution_family():
    # advance_lunge_cut_v2-v4: the torso under the final guard differs from the first one, so
    # the first pose no longer fits and the guard is re-solved. Warm-started from the cut, that
    # landed in the other redundant arm solution (wrist yaw -1.12 at the start, +1.45 at the
    # end), and the policy averaged the two guards over the static start (wrist yaw error 0.78).
    from fencing.motion.synthetic import synthesize

    base, _ = synthesize([("en_garde", 2.0)])
    waist_yaw = JOINT_NAMES.index("waist_yaw_joint")
    base.joint_pos[:, waist_yaw] += 0.2 * np.clip((np.arange(base.n_frames) - 25) / 10, 0.0, 1.0)
    script = [BladeKey("en_garde", 0.0, 0.0), BladeKey("cut_head", 0.3, 0.3, at_s=1.0),
              BladeKey("en_garde", 0.4, 0.3)]
    result = apply_blade_script(base, script)
    arm = result.motion.joint_pos[:, ARM_COLUMNS]
    first, last = result.keys[0].frame, result.keys[2].frame
    for key in result.keys:
        assert key.tip_error_m <= 0.02 and key.angle_error_deg <= 5.0 and not key.collisions, key
    assert np.abs(arm[last] - arm[first]).max() < 0.5, (arm[first].round(2), arm[last].round(2))


def test_bladework_respects_speed_caps_and_soft_limits():
    _, result = _quinte()
    joints = result.motion.joint_pos
    speeds = np.abs(np.diff(joints, axis=0)) * FPS
    caps = 0.5 * hardware_velocity_limits()
    assert np.all(speeds[:, ARM_COLUMNS] <= caps[ARM_COLUMNS] * 1.001)
    assert result.summary["controlled_joints_beyond_soft_limit"] == []
    assert result.summary["collision_frames"] == 0
