"""Goal 1 metric definitions on synthetic data (fencing.train.metrics, numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from fencing.train import metrics as m


def test_foot_slip_is_zero_for_a_planted_foot_that_does_not_move():
    accumulator = m.FootSlipAccumulator(num_envs=1)
    for _ in range(10):
        accumulator.add(np.zeros((1, 2, 3)), np.ones((1, 2), bool), np.ones(1, bool))
    assert accumulator.mean() == 0.0
    assert accumulator.planted_samples() == 20


def test_foot_slip_measures_horizontal_speed_only_while_planted():
    # Foot 0 slides at 0.3 m/s in x and 0.4 in y (|v_xy| = 0.5) and is planted;
    # foot 1 swings at 2 m/s but is not planted, so it must not count.
    velocity = np.array([[[0.3, 0.4, -1.0], [2.0, 0.0, 0.0]]])
    mask = np.array([[True, False]])
    accumulator = m.FootSlipAccumulator(num_envs=1)
    accumulator.add(velocity, mask, np.ones(1, bool))
    assert accumulator.mean() == pytest.approx(0.5)  # vertical component ignored


def test_foot_slip_ignores_envs_after_they_terminate():
    accumulator = m.FootSlipAccumulator(num_envs=2)
    velocity = np.array([[[0.1, 0, 0]], [[5.0, 0, 0]]])
    accumulator.add(velocity, np.ones((2, 1), bool), np.array([True, False]))
    assert accumulator.mean() == pytest.approx(0.1)


def test_foot_slip_is_nan_when_no_foot_was_ever_planted():
    accumulator = m.FootSlipAccumulator(num_envs=1)
    accumulator.add(np.ones((1, 2, 3)), np.zeros((1, 2), bool), np.ones(1, bool))
    assert np.isnan(accumulator.mean())


def test_slip_mask_requires_reference_planted_and_robot_foot_down():
    ground = 0.03
    ref_z = np.array([0.03, 0.03, 0.20, 0.03])
    ref_speed = np.array([0.0, 0.0, 0.0, 1.0])  # last frame: reference foot sliding/stepping
    planted = m.reference_planted_mask(ref_z, ref_speed, ground)
    assert planted.tolist() == [True, True, False, False]
    robot_z = np.array([0.04, 0.30, 0.04, 0.04])  # second frame: robot lifted the foot
    assert m.foot_slip_mask(planted, robot_z, ground).tolist() == [True, False, False, False]


def test_synthetic_walk_reports_the_injected_slip_speed():
    """A foot that alternates stance/swing, sliding 2 cm/s during stance."""
    steps, dt = 200, 0.02
    phase = (np.arange(steps) // 25) % 2 == 0  # stance for 25 steps, swing for 25
    ref_z = np.where(phase, 0.035, 0.12)
    ref_speed = np.where(phase, 0.0, 1.2)
    ground = m.estimate_ground_height(ref_z)
    planted = m.reference_planted_mask(ref_z, ref_speed, ground)
    accumulator = m.FootSlipAccumulator(num_envs=1)
    for t in range(steps):
        robot_vel = np.array([[[0.02 if phase[t] else 1.2, 0.0, 0.0]]])
        mask = m.foot_slip_mask(planted[t : t + 1], ref_z[t : t + 1], ground)[None]
        accumulator.add(robot_vel, mask, np.ones(1, bool))
    assert accumulator.mean() == pytest.approx(0.02)
    assert accumulator.planted_samples() == int(phase.sum())
    del dt


def test_success_counts_only_episodes_without_termination_before_motion_end():
    outcomes = m.episode_outcomes(
        terminated_at_step=np.array([m.NOT_TERMINATED, 10, 499, 500, 1200, m.NOT_TERMINATED]),
        motion_steps=1000,
        window_steps=500,
    )
    # terminated during the hold (1200 >= 1000) still finished the motion
    assert outcomes.success.tolist() == [True, False, False, False, True, True]
    assert outcomes.success_count == 3
    assert outcomes.success_rate == pytest.approx(0.5)
    # mjlab's 10 s window: termination at step 500 is outside the first 500 steps
    assert outcomes.survived_window.tolist() == [True, False, False, True, True, True]


def test_window_is_capped_at_motion_length_for_short_motions():
    outcomes = m.episode_outcomes(np.array([150, m.NOT_TERMINATED]), motion_steps=150, window_steps=500)
    assert outcomes.survived_window.tolist() == [True, True]


def test_recovery_needs_success_no_hold_termination_and_a_stable_pelvis():
    outcomes = m.episode_outcomes(
        terminated_at_step=np.array([m.NOT_TERMINATED, m.NOT_TERMINATED, m.NOT_TERMINATED, 1100, 3]),
        motion_steps=1000,
        window_steps=500,
        final_pelvis_height_error=np.array([0.01, 0.10, 0.01, 0.0, 0.0]),
        final_pelvis_speed=np.array([0.05, 0.05, 0.50, 0.0, 0.0]),
    )
    assert outcomes.recovered.tolist() == [True, False, False, False, False]
    assert outcomes.recovery_rate == pytest.approx(0.2)


def test_episode_outcomes_rejects_invalid_termination_steps():
    with pytest.raises(ValueError, match="-5"):
        m.episode_outcomes(np.array([-5]), motion_steps=10, window_steps=5)


def test_masked_mean_matches_mjlab_per_env_then_across_env_convention():
    accumulator = m.MaskedMean(num_envs=2)
    accumulator.add(np.array([1.0, 10.0]), np.array([True, True]))
    accumulator.add(np.array([3.0, 99.0]), np.array([True, False]))
    assert accumulator.per_env().tolist() == [2.0, 10.0]
    assert accumulator.mean() == pytest.approx(6.0)
    assert accumulator.pooled() == pytest.approx(14.0 / 3)


def test_joint_angle_error_is_mean_absolute_over_joints():
    reference = np.array([[0.0, 0.0, 0.0, 0.0]])
    robot = np.array([[0.1, -0.1, 0.2, 0.0]])
    assert m.joint_angle_error(reference, robot) == pytest.approx([0.1])


def test_joint_angle_error_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        m.joint_angle_error(np.zeros((1, 29)), np.zeros((1, 28)))
