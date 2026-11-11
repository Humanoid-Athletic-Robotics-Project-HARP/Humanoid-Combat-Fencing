"""The plausibility gate: motions no fencer (or no G1) could do must fail, not warn."""

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from fencing.motion.quality import QualityThresholds, quality_report
from fencing.motion.synthetic import synthesize


@lru_cache(maxsize=1)
def drill():
    motion, _ = synthesize([("en_garde", 0.6), ("advance", 0.6), ("en_garde", 0.4), ("lunge", 1.2), ("en_garde", 0.6)])
    return motion


def implausible(report):
    return [r for r in report["reasons"] if r.startswith("implausible")]


def tilted(motion, start, end, degrees):
    """Whole body pitched about the pelvis over [start, end) (the robot bending backwards)."""
    quat = motion.root_quat.copy()
    quat[start:end] = (Rotation.from_euler("y", -np.radians(degrees)) * Rotation.from_quat(quat[start:end])).as_quat()
    return motion.replace(root_quat=quat)


def test_clean_footwork_is_plausible():
    report = quality_report(drill())
    assert report["verdict"] != "fail" and not implausible(report), report["reasons"]
    p = report["metrics"]["plausibility"]
    assert p["torso_tilt_max_deg"] < 5 and p["retarget_ik_error_median"] is None


def test_sustained_backward_bend_fails():
    report = quality_report(tilted(drill(), 30, 60, 70.0), floor_at_zero=False)
    assert report["verdict"] == "fail"
    assert any("torso tilt" in r for r in implausible(report)), report["reasons"]


def test_a_two_frame_tilt_glitch_only_warns():
    report = quality_report(tilted(drill(), 30, 32, 40.0), floor_at_zero=False)
    assert not implausible(report), report["reasons"]
    assert any("torso tilt" in r for r in report["reasons"])


def test_root_teleport_fails():
    motion = drill().copy()
    motion.root_pos[40:, 0] += 0.4  # 0.4 m in one frame = 12 m/s
    report = quality_report(motion, floor_at_zero=False)
    assert any("root" in r for r in implausible(report)), report["reasons"]


def test_pelvis_far_below_a_stance_fails():
    motion = drill().copy()
    motion.root_pos[20:50, 2] -= 0.25  # pelvis 0.43 m above the floor for 1 s
    report = quality_report(motion)
    assert any("pelvis height" in r for r in implausible(report)), report["reasons"]


def test_retarget_stats_from_the_sidecar_gate_the_verdict():
    bad = drill().copy()
    bad.meta["sidecar"] = {"retarget": {"ik_error_median": 4.7, "joint_limit_frac": 0.30}}
    reasons = implausible(quality_report(bad))
    assert any("IK error" in r for r in reasons) and any("joint limits" in r for r in reasons), reasons
    good = drill().copy()
    good.meta["sidecar"] = {"retarget": {"ik_error_median": 1.3, "joint_limit_frac": 0.04}}
    report = quality_report(good)
    assert not implausible(report), report["reasons"]
    assert report["metrics"]["plausibility"]["retarget_ik_error_median"] == 1.3


def test_thresholds_are_configurable():
    report = quality_report(tilted(drill(), 30, 60, 70.0), QualityThresholds(torso_tilt_fail_deg=80, torso_tilt_warn_deg=80), floor_at_zero=False)
    assert not any("torso tilt" in r for r in report["reasons"])
