"""Provisional pass/fail thresholds and the rollout metric accumulator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fencing.train import metrics, thresholds


def _report(**nominal):
    return {"nominal": nominal, "robustness": {"push": {"success_rate": 0.85}}}


def test_check_passes_fails_and_skips():
    results = thresholds.check(
        _report(success_rate=0.97, joint_angle_error_rad=0.2, blade_tip_error_m=None),
        {"nominal.success_rate": (">=", 0.95), "nominal.joint_angle_error_rad": ("<=", 0.1),
         "nominal.blade_tip_error_m": ("<=", 0.1), "robustness.push.success_rate": (">=", 0.8)},
    )
    status = {r.metric: r.status for r in results}
    assert status == {"nominal.success_rate": "pass", "nominal.joint_angle_error_rad": "fail",
                      "nominal.blade_tip_error_m": "skipped", "robustness.push.success_rate": "pass"}
    assert thresholds.verdict(results) == "FAIL"


def test_verdict_needs_at_least_one_check_that_ran():
    skipped = thresholds.check({}, {"nominal.success_rate": (">=", 0.95)})
    assert thresholds.verdict(skipped) == "FAIL"
    passed = thresholds.check(_report(success_rate=1.0), {"nominal.success_rate": (">=", 0.95)})
    assert thresholds.verdict(passed) == "PASS"



_TWO = {"nominal.success_rate": (">=", 0.95), "nominal.blade_tip_error_m": ("<=", 0.1),
        "robustness.saber_mass.success_rate": (">=", 0.9)}


def test_verdict_is_partial_when_an_applicable_check_did_not_run():
    """--no-robustness or a crashed pass must not read as PASS for a hardware gate."""
    report = {"nominal": {"success_rate": 1.0, "has_blade": True, "blade_tip_error_m": 0.05}, "robustness": {}}
    results = thresholds.check(report, _TWO)
    assert thresholds.verdict(results, thresholds.not_applicable(report)) == "PARTIAL"


def test_verdict_ignores_checks_that_do_not_apply_to_the_run():
    """No saber: the blade metric and the saber_mass pass are not applicable, so PASS."""
    report = {"nominal": {"success_rate": 1.0, "has_blade": False, "blade_tip_error_m": None},
              "robustness": {"saber_mass": {"skipped": "task X has no saber_mass randomization"}}}
    results = thresholds.check(report, _TWO)
    assert thresholds.not_applicable(report) == {"nominal.blade_tip_error_m", "robustness.saber_mass.success_rate"}
    assert thresholds.verdict(results, thresholds.not_applicable(report)) == "PASS"


def test_truncated_motion_is_partial_and_a_failure_still_wins():
    report = {"nominal": {"success_rate": 1.0, "has_blade": False, "truncated": True}}
    checks = {"nominal.success_rate": (">=", 0.95)}
    assert thresholds.verdict(thresholds.check(report, checks), thresholds.not_applicable(report),
                              incomplete=thresholds.incomplete_reasons(report)) == "PARTIAL"
    report["nominal"]["success_rate"] = 0.5
    assert thresholds.verdict(thresholds.check(report, checks), thresholds.not_applicable(report),
                              incomplete=thresholds.incomplete_reasons(report)) == "FAIL"

def test_overrides_from_file_and_cli(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"nominal.success_rate": 0.5, "nominal.foot_slip_m_s": None,
                                "nominal.new_metric": ["<", 3]}))
    merged = thresholds.load_thresholds(path, ["nominal.success_rate=0.8", "nominal.body_pos_error_m=<=0.2"])
    assert merged["nominal.success_rate"] == (">=", 0.8)  # CLI wins, keeps the default operator
    assert merged["nominal.body_pos_error_m"] == ("<=", 0.2)
    assert merged["nominal.new_metric"] == ("<", 3.0)
    assert "nominal.foot_slip_m_s" not in merged


def test_new_threshold_without_operator_is_rejected():
    with pytest.raises(ValueError, match="op, value"):
        thresholds.merge_thresholds({"nominal.unknown": 1.0})
    with pytest.raises(ValueError, match="KEY=VALUE"):
        thresholds.parse_cli_overrides(["no_equals_sign"])


def _accumulator_inputs(num_envs=2, offset=0.0):
    body = np.zeros((num_envs, 3, 3))
    body[:, 1, 2] = 0.05  # bodies 1 and 2 are feet on the ground
    body[:, 2, 2] = 0.05
    robot = body.copy()
    robot[:, 0, 0] += offset
    quat = np.tile([1.0, 0, 0, 0], (num_envs, 1))
    return dict(
        ref_joint_pos=np.zeros((num_envs, 4)), robot_joint_pos=np.full((num_envs, 4), 0.1),
        ref_body_pos_w=body, ref_body_pos_rel_w=body, robot_body_pos_w=robot,
        ref_anchor_pos_w=body[:, 0], ref_anchor_quat_w=quat, robot_anchor_pos_w=robot[:, 0],
        robot_anchor_quat_w=quat, ref_body_lin_vel_w=np.zeros((num_envs, 3, 3)),
        robot_body_lin_vel_w=np.full((num_envs, 3, 3), 0.1),
    )


def test_tracking_accumulator_reports_means_over_active_envs():
    acc = metrics.TrackingAccumulator(2, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05)
    acc.add(active=np.array([True, True]), **_accumulator_inputs(offset=0.3))
    acc.add(active=np.array([True, False]), **_accumulator_inputs(offset=0.0))
    summary = acc.summary()
    assert summary["joint_angle_error_rad"] == pytest.approx(0.1)
    # env 0: steps with 0.3/3 and 0 -> 0.05; env 1: one step 0.1 -> mean of per-env means 0.075
    assert summary["body_pos_error_m"] == pytest.approx(0.075)
    assert summary["per_body_pos_error_m"]["pelvis"] == pytest.approx(0.225)
    assert summary["anchor_pos_error_m"] == pytest.approx(0.225)
    assert summary["foot_slip_m_s"] == pytest.approx(np.hypot(0.1, 0.1))
    assert summary["foot_slip_samples"] == 6
    assert summary["blade_tip_error_m"] is None


def test_tracking_accumulator_blade_tip_error():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05)
    tip = np.array([[0.0, 0.0, 1.0]])
    acc.add(active=np.array([True]), ref_tip_rel_w=tip, ref_tip_w=tip + [1.0, 0, 0], robot_tip_w=tip + [0, 0.04, 0],
            **_accumulator_inputs(num_envs=1))
    summary = acc.summary()
    assert summary["blade_tip_error_m"] == pytest.approx(0.04)
    assert summary["blade_tip_error_max_m"] == pytest.approx(0.04)
    assert summary["blade_tip_error_global_m"] == pytest.approx(np.hypot(1.0, 0.04))


def test_blade_tip_from_wrist_rotates_the_offset():
    quarter_turn_z = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    tip = metrics.blade_tip_from_wrist(np.array([1.0, 0, 0]), quarter_turn_z, np.array([0.5, 0, 0]))
    np.testing.assert_allclose(tip, [1.0, 0.5, 0.0], atol=1e-12)


def test_termination_time_counts_the_step_that_terminated():
    """Terminating at 0-based step k means k + 1 steps were simulated."""
    assert metrics.termination_time_s(0, 0.02) == pytest.approx(0.02)
    assert metrics.termination_time_s(np.array([0, 49]), 0.02).tolist() == pytest.approx([0.02, 1.0])


def _blade_step(acc, frame, tip_error, active=True, global_offset=0.0):
    tip = np.array([[0.0, 0.0, 1.0]])
    acc.add(active=np.array([active]), ref_tip_rel_w=tip, ref_tip_w=tip + [global_offset, 0, 0],
            robot_tip_w=tip + [0, tip_error, 0], frame=frame, **_accumulator_inputs(num_envs=1))


def test_key_frame_blade_error_averages_the_window_around_each_key():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05,
                                      key_frames=[10])
    for frame in range(20):
        # 0.10 m at the 5 frames within +-2 of the key, 0.50 m everywhere else
        _blade_step(acc, frame, 0.10 if abs(frame - 10) <= metrics.KEY_WINDOW_STEPS else 0.50, global_offset=0.3)
    summary = acc.summary()
    assert metrics.KEY_WINDOW_STEPS == 2
    assert summary["blade_tip_error_key_m"] == pytest.approx(0.10)
    assert summary["blade_tip_error_key_global_m"] == pytest.approx(np.hypot(0.3, 0.10))
    assert summary["blade_key_frames"] == [10]
    assert summary["blade_tip_error_m"] == pytest.approx((5 * 0.10 + 15 * 0.50) / 20)  # unchanged definition


def test_key_frame_windows_that_overlap_count_each_frame_once():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05,
                                      key_frames=[10, 11])
    for frame in range(20):
        _blade_step(acc, frame, 0.01 * frame)
    # frames 8..13 once each
    assert acc.summary()["blade_tip_error_key_m"] == pytest.approx(0.01 * np.mean(range(8, 14)))


def test_key_frame_error_is_none_when_the_env_died_before_the_key():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05,
                                      key_frames=[10])
    for frame in range(20):
        _blade_step(acc, frame, 0.1, active=frame < 5)
    summary = acc.summary()
    assert summary["blade_tip_error_key_m"] is None and summary["blade_key_frames"] == [10]


def test_without_key_frames_the_summary_has_exactly_the_old_keys():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05)
    _blade_step(acc, 0, 0.1)  # passing the frame is harmless without key frames
    assert not {k for k in acc.summary() if "key" in k}


def test_key_frames_without_a_saber_report_null_errors():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05,
                                      key_frames=[0])
    acc.add(active=np.array([True]), frame=0, **_accumulator_inputs(num_envs=1))
    summary = acc.summary()
    assert summary["blade_tip_error_key_m"] is None and summary["blade_tip_error_key_global_m"] is None


def test_key_frames_need_the_frame_of_each_step():
    acc = metrics.TrackingAccumulator(1, ["pelvis", "left_foot", "right_foot"], [1, 2], ground_z=0.05,
                                      key_frames=[0])
    with pytest.raises(ValueError, match="frame"):
        acc.add(active=np.array([True]), **_accumulator_inputs(num_envs=1))


KEY = "nominal.blade_tip_error_key_m"


def test_key_frame_check_is_a_default_and_gates_runs_with_key_frames():
    assert thresholds.DEFAULT_THRESHOLDS[KEY][0] == "<="
    limit = thresholds.DEFAULT_THRESHOLDS[KEY][1]
    report = {"has_blade": True, "nominal": {"success_rate": 1.0, "blade_key_frames": [40],
                                             "blade_tip_error_key_m": limit + 0.01}}
    status = {r.metric: r.status for r in thresholds.check(report)}
    assert status[KEY] == "fail"
    report["nominal"]["blade_tip_error_key_m"] = limit - 0.01
    assert {r.metric: r.status for r in thresholds.check(report)}[KEY] == "pass"


def test_runs_without_key_frames_get_exactly_the_old_checks_and_verdict():
    """Additive: a run with no key-frame data reports the same check list and verdict as before."""
    old_defaults = {k: v for k, v in thresholds.DEFAULT_THRESHOLDS.items() if k != KEY}
    for report in ({"has_blade": True, "nominal": {"success_rate": 1.0, "blade_tip_error_m": 0.05}},
                   {"has_blade": False, "nominal": {"success_rate": 1.0, "blade_tip_error_m": None}},
                   {"has_blade": True, "nominal": {"success_rate": 1.0, "blade_key_frames": [],
                                                   "blade_tip_error_key_m": None}}):
        new = thresholds.check(report)
        old = thresholds.check(report, old_defaults)
        assert new == old
        assert KEY not in thresholds.not_applicable(report)
        assert (thresholds.verdict(new, thresholds.not_applicable(report))
                == thresholds.verdict(old, thresholds.not_applicable(report)))


def test_key_frame_check_without_a_saber_is_not_applicable():
    """Key frames on the motion but no saber in the scene (MTC's plain G1): skipped, never failing."""
    checks = {"nominal.success_rate": (">=", 0.95), KEY: ("<=", 0.05)}
    report = {"has_blade": False, "nominal": {"success_rate": 1.0, "blade_key_frames": [40],
                                              "blade_tip_error_key_m": None}}
    results = thresholds.check(report, checks)
    assert {r.metric: r.status for r in results}[KEY] == "skipped"
    assert KEY in thresholds.not_applicable(report)
    assert thresholds.verdict(results, thresholds.not_applicable(report)) == "PASS"


def test_key_frame_check_skipped_on_a_saber_run_is_partial():
    """The env died before the key (or --seconds cut it off): the key was never scored."""
    checks = {"nominal.success_rate": (">=", 0.0), KEY: ("<=", 0.05)}
    report = {"has_blade": True, "nominal": {"success_rate": 1.0, "blade_key_frames": [40],
                                             "blade_tip_error_key_m": None}}
    assert thresholds.verdict(thresholds.check(report, checks), thresholds.not_applicable(report)) == "PARTIAL"


def test_sim2sim_layout_with_top_level_flags_is_gated_the_same_way():
    """sim2sim's rollout writes has_blade and truncated next to `nominal`, not inside it."""
    checks = {"nominal.success_rate": (">=", 0.95), "nominal.blade_tip_error_m": ("<=", 0.1)}
    plain = {"has_blade": False, "truncated": False, "nominal": {"success_rate": 1.0, "blade_tip_error_m": None}}
    assert thresholds.verdict(thresholds.check(plain, checks), thresholds.not_applicable(plain),
                              thresholds.incomplete_reasons(plain)) == "PASS"
    cut = {"has_blade": True, "truncated": True, "nominal": {"success_rate": 1.0, "blade_tip_error_m": 0.05}}
    assert thresholds.verdict(thresholds.check(cut, checks), thresholds.not_applicable(cut),
                              thresholds.incomplete_reasons(cut)) == "PARTIAL"
