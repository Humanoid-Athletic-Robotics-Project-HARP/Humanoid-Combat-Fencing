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


def _guard_qpos_and_arm():
    from fencing.motion.kinematics import load_g1
    from fencing.motion.synthetic import synthesize

    base, _ = synthesize([("en_garde", 0.5)])
    return load_g1().qpos_from_motion(base)[0], base.joint_pos[0, ARM_COLUMNS].copy()


def test_move_torque_ratio_falls_as_the_move_slows():
    # Inertial and velocity-product torques scale with 1/T^2, so a slower move demands less.
    from fencing.saber.keyframes import move_torque_ratio

    qpos, arm = _guard_qpos_and_arm()
    swing = arm + np.array([-1.5, -0.3, 1.0, 0.0, -0.9, 0.8, 0.9])
    fast, slower, slow = (max(move_torque_ratio(qpos, arm, swing, t).values()) for t in (0.2, 0.3, 3.0))
    assert fast > slower > slow > 0
    assert set(move_torque_ratio(qpos, arm, swing, 0.2)) == {JOINT_NAMES[c] for c in ARM_COLUMNS}


def test_move_torque_ratio_of_a_held_pose_is_the_static_load_whatever_the_time():
    from fencing.saber.keyframes import move_torque_ratio

    qpos, arm = _guard_qpos_and_arm()
    held_fast, held_slow = move_torque_ratio(qpos, arm, arm, 0.1), move_torque_ratio(qpos, arm, arm, 2.0)
    assert held_fast == pytest.approx(held_slow)
    # Holding the nominal saber at en garde is cheap (docs/saber.md payload: wrist pitch 0.6 of 5 Nm).
    assert max(held_fast.values()) < 0.5


def test_blade_summary_reports_the_peak_torque_demand_of_its_moves():
    _, result = _quinte()
    summary = result.summary
    assert summary["max_torque_ratio"] > 0
    assert summary["max_torque_ratio_joint"] in {JOINT_NAMES[c] for c in ARM_COLUMNS}
    assert summary["max_torque_ratio_key"] in {"parry_quinte", "en_garde"}


def test_blade_path_defaults_to_joint_space_and_rejects_unknown_values():
    from fencing.saber.keyframes import BladeConfig

    assert BladeConfig().blade_path == "joint"  # every motion published before thrust_v5
    with pytest.raises(ValueError, match="blade_path"):
        BladeConfig(blade_path="spline")


@lru_cache(maxsize=2)
def _line_script(target):
    from fencing.saber.keyframes import BladeConfig

    base, _ = _quinte()
    script = [BladeKey("en_garde", 0.0, 0.4), BladeKey(target, 0.3, 0.4), BladeKey("en_garde", 0.4, 0.4)]
    return apply_blade_script(base, script, BladeConfig(blade_path="line"))


def _tip_path(motion):
    from fencing.motion.kinematics import load_g1
    from fencing.saber import keyframes

    ctx = keyframes._context(keyframes.NOMINAL_SABER, keyframes.SOFT_LIMIT_FACTOR)
    return np.array([keyframes._blade_tip(ctx, q)[0] for q in load_g1().qpos_from_motion(motion)])


def test_line_blade_path_falls_back_to_joint_space_where_the_tip_cannot_follow_the_line():
    # parry_quinte from the guard: the blade has to pass over the head, a straight tip line cannot.
    result = _line_script("parry_quinte")
    assert [m["used"] for m in result.summary["line_moves"]] == [False, False]
    assert max(m["tracking_max_m"] for m in result.summary["line_moves"]) > 0.05
    for key in result.keys:
        assert key.tip_error_m <= 0.02 and key.angle_error_deg <= 5.0 and not key.collisions, key
    # Falling back means joint space all the way: the parry key is not solved from where the failed
    # line ended either, so the motion is the joint-space one.
    np.testing.assert_array_equal(result.motion.joint_pos, _quinte()[1].motion.joint_pos)


def test_line_blade_path_falls_back_to_joint_space_on_a_moving_base():
    # The line is solved on the body at the reach frame. On a base that moves during the move (a
    # lunge) the tip would follow that body's line, not the one the base carries it along.
    from fencing.motion.synthetic import synthesize
    from fencing.saber.keyframes import LINE_BODY_SHIFT_TOLERANCE_M, BladeConfig

    base, _ = synthesize([("en_garde", 3.0)])
    base.root_pos[:, 0] += 0.5 * np.arange(base.n_frames) / FPS  # 0.5 m/s forward
    script = [BladeKey("en_garde", 0.0, 0.4), BladeKey("cut_flank", 0.3, 0.4), BladeKey("en_garde", 0.4, 0.4)]
    line = apply_blade_script(base, script, BladeConfig(blade_path="line"))
    moves = line.summary["line_moves"]
    assert [m["used"] for m in moves] == [False, False], moves
    assert min(m["body_shift_m"] for m in moves) > LINE_BODY_SHIFT_TOLERANCE_M
    np.testing.assert_array_equal(line.motion.joint_pos, apply_blade_script(base, script).motion.joint_pos)


def _solve_inputs(base, script):
    """What apply_blade_script's first round hands _solve_all."""
    from fencing.motion.kinematics import load_g1
    from fencing.saber import keyframes

    spans = schedule_keys(script, FPS, [0] * len(script))
    padded = keyframes._pad_base(base, max(base.n_frames, spans[-1][2] + 1, spans[-1][1] + 1))
    return spans, load_g1().qpos_from_motion(padded), padded


def test_a_line_seeded_pose_from_an_earlier_round_is_dropped_once_its_line_falls_back():
    # The speed stretch re-solves at new frames and keeps a previous round's pose that still fits.
    # A pose solved from a blade line must not survive into a round whose move falls back to joint
    # space: there it would be the line's solution family under a joint-space move.
    from fencing.motion.synthetic import synthesize
    from fencing.saber import keyframes
    from fencing.saber.keyframes import BladeConfig

    ctx = keyframes._context(keyframes.NOMINAL_SABER, keyframes.SOFT_LIMIT_FACTOR)
    script = [BladeKey("en_garde", 0.0, 0.4), BladeKey("cut_flank", 0.3, 0.4), BladeKey("en_garde", 0.4, 0.4)]
    line, joint = BladeConfig(blade_path="line"), BladeConfig()
    still, _ = synthesize([("en_garde", 3.0)])
    seeded = keyframes._solve_all(ctx, script, *_solve_inputs(still, script), line, {})[1]
    assert seeded.line_seeded

    moving, _ = synthesize([("en_garde", 3.0)])
    moving.root_pos[:, 0] += 0.5 * np.arange(moving.n_frames) / FPS  # the line now falls back
    spans, qpos, padded = _solve_inputs(moving, script)
    target = resolve_target("cut_flank")
    frame = qpos[spans[1][1]].copy()
    assert keyframes._reuse_ok(ctx, seeded, frame, target, line)  # the pose itself would still do
    solved = keyframes._solve_all(ctx, script, spans, qpos, padded, line, {1: seeded})
    expected = keyframes._solve_all(ctx, script, spans, qpos, padded, joint, {})
    assert not solved[1].line_seeded
    np.testing.assert_array_equal(solved[1].arm_q, expected[1].arm_q)


def test_blade_line_refuses_a_start_pose_outside_the_soft_limits():
    # Clipping it would jump the arm on the move's first frame.
    from fencing.saber import keyframes

    ctx = keyframes._context(keyframes.NOMINAL_SABER, keyframes.SOFT_LIMIT_FACTOR)
    qpos, arm = _guard_qpos_and_arm()
    tip, direction = keyframes._blade_tip(ctx, keyframes._with_arm(qpos, ctx.arm, arm))
    outside = arm.copy()
    outside[3] = ctx.arm.upper[3] + 0.05
    with pytest.raises(ValueError, match=keyframes.ARM_JOINTS[3]):
        keyframes.track_blade_line(ctx, qpos, outside, tip, direction)


def test_line_blade_path_keeps_the_tip_on_the_straight_line_between_keys():
    result = _line_script("cut_flank")
    tips = _tip_path(result.motion)
    for key in result.keys:
        assert key.tip_error_m <= 0.02 and key.angle_error_deg <= 5.0 and not key.collisions, key
    assert [m["used"] for m in result.summary["line_moves"]] == [True, True]
    for i in (1, 2):
        start = result.keys[i].frame - int(round(result.keys[i].move_s_used * FPS))
        a, b = tips[start], tips[result.keys[i].frame]
        segment = tips[start:result.keys[i].frame + 1]
        along = np.clip((segment - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        off_line = np.linalg.norm(segment - (a + along[:, None] * (b - a)), axis=1)
        assert off_line.max() < 0.03, (i, off_line.max())
    # Same guard at both ends, speeds and limits as the joint-space moves.
    arm = result.motion.joint_pos[:, ARM_COLUMNS]
    np.testing.assert_allclose(arm[0], arm[-1], atol=1e-9)
    speeds = np.abs(np.diff(result.motion.joint_pos, axis=0)) * FPS
    assert np.all(speeds[:, ARM_COLUMNS] <= 0.5 * hardware_velocity_limits()[ARM_COLUMNS] * 1.001)
    assert result.summary["controlled_joints_beyond_soft_limit"] == [] and result.summary["collision_frames"] == 0


def test_bladework_respects_speed_caps_and_soft_limits():
    _, result = _quinte()
    joints = result.motion.joint_pos
    speeds = np.abs(np.diff(joints, axis=0)) * FPS
    caps = 0.5 * hardware_velocity_limits()
    assert np.all(speeds[:, ARM_COLUMNS] <= caps[ARM_COLUMNS] * 1.001)
    assert result.summary["controlled_joints_beyond_soft_limit"] == []
    assert result.summary["collision_frames"] == 0
