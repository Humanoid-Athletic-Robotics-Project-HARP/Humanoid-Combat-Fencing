"""Entry motion: from upstream's standby pose into a motion's first frame (docs/mtc.md, "Entry from standby").

The reference must start exactly where standby_controller leaves the robot (its default_position,
feet flat on the floor) and end exactly on the target motion's first frame, held, so that both
controller switches (standby -> entry, entry -> motion) start the incoming policy on its reference.
In between it has to be something a G1 can do: one foot moves at a time, the swing foot lifts, and
the centre of mass is over the stance foot while the other foot is in the air, except for the last
0.13 s of each step, when it moves toward the landing foot as that foot comes down.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fencing import paths
from fencing.motion import entry
from fencing.motion.io import JOINT_NAMES, read_csv
from fencing.motion.kinematics import forward_kinematics, load_g1
from fencing.motion.quality import quality_report

TARGET = paths.MOTIONS_DIR / "en_garde_v3.csv"


# The published entry, motions/stand_to_en_garde_v1.csv: the saber guard (frame 0 of every static
# bladework motion), built with the default EntryConfig.
PUBLISHED_TARGET = paths.MOTIONS_DIR / "parry_quinte_v3.csv"
PUBLISHED = paths.MOTIONS_DIR / "stand_to_en_garde_v1.csv"


@pytest.fixture(scope="module")
def built():
    return entry.build_entry(read_csv(TARGET))


@pytest.fixture(scope="module")
def published():
    return entry.build_entry(read_csv(PUBLISHED_TARGET))


@pytest.fixture(scope="module", params=["en_garde", "published"])
def either(request, built, published):
    return built if request.param == "en_garde" else published


def test_standby_pose_is_read_from_motion_tracking_controllers_own_config_by_joint_name():
    joints = entry.standby_joint_positions()
    assert joints.shape == (len(JOINT_NAMES),)
    by_name = dict(zip(JOINT_NAMES, joints))
    assert by_name["left_knee_joint"] == pytest.approx(0.669)
    assert by_name["right_ankle_pitch_joint"] == pytest.approx(-0.33)
    assert by_name["right_shoulder_roll_joint"] == pytest.approx(-0.2)
    assert by_name["left_shoulder_roll_joint"] == pytest.approx(0.2)
    assert by_name["waist_yaw_joint"] == 0.0


def test_entry_starts_in_the_standby_pose_with_both_feet_flat_on_the_floor(built):
    motion = built.motion
    np.testing.assert_allclose(motion.joint_pos[0], entry.standby_joint_positions(), atol=1e-9)
    fk = forward_kinematics(motion.slice(0, 1))
    np.testing.assert_allclose(fk.foot_lowest_z[0], 0.0, atol=1e-6)
    for foot in range(2):
        up = fk.foot_rot[0, foot][:, 2]
        assert np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))) < 0.1


def test_entry_starts_facing_the_way_the_target_pelvis_faces(built):
    target = read_csv(TARGET)
    yaw = lambda q: Rotation.from_quat(q).as_euler("ZYX")[0]  # noqa: E731 (intrinsic: heading)
    assert yaw(built.motion.root_quat[0]) == pytest.approx(yaw(target.root_quat[0]), abs=1e-6)


def test_entry_ends_on_the_target_first_frame_held_for_at_least_two_seconds(built):
    target = read_csv(TARGET)
    motion = built.motion
    held = int(round(2.0 * motion.fps))
    np.testing.assert_allclose(motion.joint_pos[-held:], np.tile(target.joint_pos[0], (held, 1)), atol=1e-9)
    np.testing.assert_allclose(motion.root_pos[-held:], np.tile(target.root_pos[0], (held, 1)), atol=1e-9)
    np.testing.assert_allclose(np.abs(np.sum(motion.root_quat[-held:] * target.root_quat[0], axis=1)), 1.0, atol=1e-9)


def test_the_last_moving_frame_joins_the_held_target_without_a_jump(built):
    motion = built.motion
    steps = np.abs(np.diff(motion.joint_pos, axis=0)).max(axis=1)
    end = built.phase("settle")[1]
    assert steps[end - 2:end + 1].max() < 0.01


def test_one_foot_moves_at_a_time_and_each_swing_lifts_off_the_floor(built):
    fk = forward_kinematics(built.motion)
    for foot, label in ((1, "step_right"), (0, "step_left")):
        start, end = built.phase(label)
        other = 1 - foot
        xy = fk.sole_center[:, foot, :2]
        # Planted (sole centre still within 2 mm, on the floor) outside its own step. Not exactly
        # still: near the ankle roll limit the leg IK misses by up to ~1 mm.
        assert np.abs(xy[:start] - xy[0]).max() < 2e-3
        assert np.abs(xy[end:] - xy[-1]).max() < 2e-3
        assert fk.foot_lowest_z[start:end, foot].max() > 0.05
        # The other foot is planted during this step.
        assert np.abs(fk.sole_center[start:end, other, :2] - fk.sole_center[start, other, :2]).max() < 2e-3
        assert np.abs(fk.foot_lowest_z[start:end, other]).max() < 2e-3


def test_centre_of_mass_is_over_the_stance_foot_while_the_other_foot_is_in_the_air(either):
    g1 = load_g1()
    for stance, label in ((0, "step_right"), (1, "step_left")):
        start, end = either.phase(label)
        held = range(start, either.report["com_release_frames"][label])
        margin = entry.support_margin(either.motion, stance, held, g1)
        # The sole capsule axes span +-2.6 cm across the foot: 1 cm inside them is 2 cm inside the sole edge.
        assert margin.min() > 0.01, f"{label}: CoM {margin.min():.3f} m inside the stance sole"


def test_centre_of_mass_leaves_the_stance_foot_only_in_the_last_four_frames_of_each_step(either):
    # Not a static transfer. The CoM is released toward the landing foot over the last quarter of
    # each step, while that foot is still coming down: on the back-foot step it is up to 5 cm
    # outside the stance sole with the foot still 0.6-2.8 cm up, for 4 frames (0.13 s), as in a
    # walking step. Releasing only at touchdown would push the ankle roll further past its limit
    # (docs/mtc.md, "Entry from standby").
    fk = forward_kinematics(either.motion)
    for stance, swing, label in ((0, 1, "step_right"), (1, 0, "step_left")):
        start, end = either.phase(label)
        release = either.report["com_release_frames"][label]
        assert start + 0.7 * (end - start) <= release < end
        frames = np.arange(release, end)
        margin = entry.support_margin(either.motion, stance, frames)
        height = fk.foot_lowest_z[frames, swing]
        outside = frames[margin < 0.0]
        assert height[margin < 0.0].max(initial=0.0) < 0.03, f"{label}: CoM outside the stance sole, foot 3 cm up"
        assert len(outside) <= 4, f"{label}: CoM outside the stance sole for {len(outside)} frames"
        assert margin.min() > -0.055, f"{label}: CoM {-margin.min():.3f} m outside the stance sole"


def test_published_entry_is_this_generator_on_the_saber_guard_with_the_default_config(published):
    # So every test above that takes `either` holds for the motion the entry policy was trained on.
    np.testing.assert_allclose(read_csv(PUBLISHED).joint_pos, published.motion.joint_pos, atol=1e-5)
    np.testing.assert_allclose(read_csv(PUBLISHED).root_pos, published.motion.root_pos, atol=1e-5)


def test_entry_passes_the_motion_quality_check(built):
    report = quality_report(built.motion)
    assert report["verdict"] == "pass", report["reasons"]
    assert report["metrics"]["foot_slip"]["max_deviation_m"] < 0.01


def test_motion_entry_command_writes_the_entry_csv_and_its_report(tmp_path, capsys):
    from fencing.motion import cli

    out = tmp_path / "stand_to_en_garde.csv"
    assert cli.main(["entry", str(TARGET), "--out", str(out)]) == 0
    written = read_csv(out)
    np.testing.assert_allclose(written.joint_pos[0], entry.standby_joint_positions(), atol=1e-6)
    np.testing.assert_allclose(written.joint_pos[-1], read_csv(TARGET).joint_pos[0], atol=1e-6)
    report = __import__("json").loads(out.with_suffix(".json").read_text())
    assert report["entry"]["target"] == TARGET.name and report["quality"]["verdict"] == "pass"
    assert "pass" in capsys.readouterr().out


def test_saber_entry_never_puts_the_arm_or_blade_through_the_body():
    # The saber guard (frame 0 of every static bladework motion): the sword arm blends from the
    # standby pose into it, so the blade must clear the body all the way (same check as the blade
    # keyframer, fencing.saber.keyframes).
    from fencing.saber.reach import build_arm_model, self_collisions

    result = entry.build_entry(read_csv(paths.MOTIONS_DIR / "parry_quinte_v3.csv"))
    arm = build_arm_model()
    qpos = load_g1().qpos_from_motion(result.motion)
    hits = {t: self_collisions(arm, qpos[t]) for t in range(result.motion.n_frames)}
    assert not {t: h for t, h in hits.items() if h}


def _runs(frames):
    """Consecutive frame runs as (first, last) pairs."""
    if len(frames) == 0:
        return []
    parts = np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)
    return [(int(p[0]), int(p[-1])) for p in parts]


def test_published_entry_has_no_joint_beyond_a_limit_but_both_ankle_rolls_sit_on_theirs(published):
    # The quality check counts values beyond a limit, not on it, so it reports 0 violations while
    # the ankle rolls rest at +-0.2618 rad (docs/mtc.md, "Entry from standby"). Pin the frames.
    g1 = load_g1()
    joints = published.motion.joint_pos
    assert np.maximum(g1.joint_lower - joints, joints - g1.joint_upper).max() <= 0.0
    margin = np.minimum(joints - g1.joint_lower, g1.joint_upper - joints)
    at_limit = {name: _runs(np.where(margin[:, i] < 1e-4)[0]) for i, name in enumerate(JOINT_NAMES)}
    assert {k: v for k, v in at_limit.items() if v} == {
        "left_ankle_roll_joint": [(97, 126)],
        "right_ankle_roll_joint": [(69, 71), (114, 120)],
    }
