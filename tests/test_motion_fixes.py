from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fencing import paths
from fencing.motion import fixes
from fencing.motion.contacts import contacts_from_fk
from fencing.motion.io import JOINT_NAMES, read_csv
from fencing.motion.kinematics import forward_kinematics, load_g1
from fencing.motion.normalize import fencing_line, rotate_about_z, yaw_of
from fencing.motion.pipeline import CleanConfig, clean_motion
from fencing.motion.quality import quality_report
from fencing.motion.synthetic import (
    Stance,
    add_jitter,
    add_root_slide,
    add_vertical_drift,
    add_yaw_drift,
    footwork_keys,
    solve_keys,
    synthesize,
)

LAFAN_WALK = paths.LAFAN1_G1 / "walk1_subject1.csv"


@lru_cache(maxsize=1)
def truth():
    motion, _ = synthesize([("en_garde", 1.0), ("advance", 0.6), ("en_garde", 0.6), ("lunge", 1.2), ("en_garde", 0.8)])
    return motion


def planted_heights(motion):
    fk = forward_kinematics(motion)
    contacts, _ = contacts_from_fk(fk, motion.fps, floor_known=False)
    return fk.foot_lowest_z[contacts], contacts


def test_vertical_drift_and_offset_are_removed():
    clean = truth()
    corrupted = add_vertical_drift(clean, amplitude_m=0.05, period_s=3.0, offset_m=0.06)
    fk = forward_kinematics(corrupted)
    contacts, _ = contacts_from_fk(fk, corrupted.fps)
    fixed, _ = fixes.remove_vertical_drift(corrupted, contacts, fk)
    fixed, _ = fixes.align_to_ground(fixed, contacts)
    error = fixed.root_pos[:, 2] - clean.root_pos[:, 2]
    assert np.abs(corrupted.root_pos[:, 2] - clean.root_pos[:, 2]).max() > 0.1
    assert np.abs(error).max() < 0.012
    heights, _ = planted_heights(fixed)
    assert np.percentile(np.abs(heights), 95) < 0.01


def test_floor_track_follows_the_lower_foot_after_the_last_contact():
    """Monocular depth sinks the held lunge 10 cm (the only frames with a planted foot), then
    the clip is cut while the fencer recovers with both feet moving (no contact). Holding the
    sunken contact height through those frames lifted the end en garde 8-16 cm off the floor
    (aY1P3wzvNBM, 53FZTK_8qOI); outside the contact frames the lower foot is the floor."""
    n = 40
    lowest = np.zeros((n, 2))
    lowest[10:30] = [0.04, -0.10]  # front foot planted, sunk 10 cm; back foot sliding
    lowest[30:] = np.linspace([0.02, -0.08], [0.02, 0.01], 10)
    contacts = np.zeros((n, 2), dtype=bool)
    contacts[10:30, 1] = True
    fk = SimpleNamespace(foot_lowest_z=lowest)
    track = fixes.contact_floor_track(fk, contacts)
    np.testing.assert_allclose(track[10:30], -0.10)
    np.testing.assert_allclose(track[30:], lowest[30:].min(axis=1))
    np.testing.assert_allclose(track[:10], 0.0)  # before the first contact: the lower foot too


def test_floor_track_interpolates_through_flight():
    lowest = np.full((30, 2), 0.2)
    lowest[:10, 0] = 0.0
    lowest[20:, 1] = 0.03
    contacts = np.zeros((30, 2), dtype=bool)
    contacts[:10, 0] = contacts[20:, 1] = True
    track = fixes.contact_floor_track(SimpleNamespace(foot_lowest_z=lowest), contacts)
    np.testing.assert_allclose(track[9:21], np.linspace(0.0, 0.03, 12))


def test_lunge_pelvis_drop_survives_drift_removal():
    """A real pelvis drop bends the legs; it is not drift and must not be removed."""
    clean = truth()
    fk = forward_kinematics(clean)
    contacts, _ = contacts_from_fk(fk, clean.fps)
    fixed, _ = fixes.remove_vertical_drift(clean, contacts, fk)
    assert np.ptp(fixed.root_pos[:, 2]) > 0.05
    assert np.abs(fixed.root_pos[:, 2] - clean.root_pos[:, 2]).max() < 0.005


def test_root_slide_is_removed_by_foot_pinning():
    corrupted = add_root_slide(truth(), speed_mps=0.12)
    before = quality_report(corrupted)["metrics"]["foot_slip"]["max_deviation_m"]
    fk = forward_kinematics(corrupted)
    contacts, _ = contacts_from_fk(fk, corrupted.fps, floor_known=True)
    fixed, info = fixes.remove_foot_slide(corrupted, contacts)
    after = quality_report(fixed)["metrics"]["foot_slip"]["max_deviation_m"]
    assert before > 0.05
    assert info["slip_after_m"] < 0.005
    assert after < 0.02


@lru_cache(maxsize=1)
def hovering_lunge():
    """En garde + lunge whose front foot floats 10 cm through the held lunge (the
    aY1P3wzvNBM / 53FZTK_8qOI retargets: front knee over-bent), and the frames of that hold."""
    script = [("en_garde", 1.0), ("lunge", 1.2), ("en_garde", 0.8)]
    keys, spans = footwork_keys(script)
    a, b, _ = spans[1]
    ext, hold = int(0.30 * (b - a)), int(0.25 * (b - a))
    profile = np.zeros(keys.feet.shape[0])
    profile[a + ext - 4 : a + ext + hold + 4] = 1.0
    profile[a + ext - 4 : a + ext] = fixes.smoothstep(np.linspace(0.2, 1.0, 4))
    profile[a + ext + hold : a + ext + hold + 4] = fixes.smoothstep(np.linspace(1.0, 0.2, 4))
    keys.feet[:, 1, 2] += 0.10 * profile
    return solve_keys(keys), (a + ext, a + ext + hold)


def test_still_hovering_foot_is_put_on_the_floor():
    motion, (h0, h1) = hovering_lunge()
    before = forward_kinematics(motion).foot_lowest_z
    fixed, info = fixes.plant_still_feet(motion)
    after = forward_kinematics(fixed).foot_lowest_z
    assert before[h0:h1, 1].min() > 0.09
    assert np.abs(after[h0:h1, 1]).max() < 0.005, after[h0:h1, 1]
    # Blends never push a sole into the floor; the leg IK holds foot orientation only
    # approximately (weighted), which can leave the lowest sole point a mm or two low.
    assert after[:, 1].min() > -0.003
    g1 = load_g1()
    right_leg = g1.leg_columns["right"]
    others = np.setdiff1d(np.arange(motion.joint_pos.shape[1]), right_leg)
    np.testing.assert_array_equal(fixed.joint_pos[:, others], motion.joint_pos[:, others])
    np.testing.assert_array_equal(fixed.root_pos, motion.root_pos)
    assert info["planted_phases"] == [0, 1]



@pytest.mark.parametrize("target_deg", [-45.0, 30.0])
def test_set_foot_heading_turns_the_foot_in_place(target_deg):
    """en_garde_v1's front foot came out of monocular pose 47 deg off the line; fencers point it down it."""
    motion = truth()
    before = forward_kinematics(motion)
    fixed, info = fixes.set_foot_heading(motion, foot_index=1, target_yaw_rad=np.radians(target_deg))
    after = forward_kinematics(fixed)
    np.testing.assert_allclose(np.degrees(after.foot_yaw[:, 1]), target_deg, atol=2.0)
    np.testing.assert_allclose(after.sole_center[:, 1], before.sole_center[:, 1], atol=0.005)
    np.testing.assert_allclose(after.foot_lowest_z[:, 1], before.foot_lowest_z[:, 1], atol=0.005)
    np.testing.assert_allclose(np.degrees(after.foot_yaw[:, 0]), np.degrees(before.foot_yaw[:, 0]))
    others = np.setdiff1d(np.arange(motion.joint_pos.shape[1]), load_g1().leg_columns["right"])
    np.testing.assert_array_equal(fixed.joint_pos[:, others], motion.joint_pos[:, others])
    np.testing.assert_array_equal(fixed.root_pos, motion.root_pos)
    assert info["worst_ik_error_m"] < 0.005 and info["max_turn_deg"] > 25
    assert info["worst_heading_error_deg"] < 2.0


def test_set_foot_heading_wraps_across_180_and_reports_a_short_turn():
    motion = truth()
    turned, _ = fixes.set_foot_heading(motion, foot_index=1, target_yaw_rad=np.radians(170.0))
    # 170 -> -170 is a 20 deg turn through 180, not 340 deg the long way
    _, info = fixes.set_foot_heading(turned, foot_index=1, target_yaw_rad=np.radians(-170.0))
    assert info["max_turn_deg"] < 30
    # far past what the hip can turn: the heading it could not reach is reported
    _, info = fixes.set_foot_heading(motion, foot_index=1, target_yaw_rad=np.radians(150.0))
    assert info["worst_heading_error_deg"] > 2.0


def _arm_torso_depth(motion, side):
    import mujoco
    g1 = load_g1(); data = mujoco.MjData(g1.model); worst = np.zeros(motion.n_frames)
    arm = {b for b in range(g1.model.nbody) if (g1.body_names[b] or "").startswith(f"{side}_") and
           any(k in g1.body_names[b] for k in ("shoulder", "elbow", "wrist", "hand", "rubber"))}
    for t in range(motion.n_frames):
        data.qpos[:] = g1.qpos_from_motion(motion)[t]
        mujoco.mj_forward(g1.model, data)
        for c in data.contact[:data.ncon]:
            bodies = {g1.model.geom_bodyid[c.geom1], g1.model.geom_bodyid[c.geom2]}
            if bodies & arm and not bodies <= arm and 0 not in bodies:
                worst[t] = min(worst[t], c.dist)
    return worst


@pytest.mark.parametrize("side,inward", [("right", 0.6), ("left", -0.6)])
def test_arm_pushed_into_the_torso_is_abducted_just_clear(side, inward):
    """lunge_v2's sword arm hangs adducted inside the torso (up to 3.7 cm) for the whole clip."""
    motion = truth().copy()
    roll = JOINT_NAMES.index(f"{side}_shoulder_roll_joint")
    motion.joint_pos[:, roll] = inward
    assert _arm_torso_depth(motion, side).min() < -0.01
    fixed, info = fixes.clear_arm_torso_collision(motion, side, margin_m=0.005)
    assert _arm_torso_depth(fixed, side).min() >= 0.0
    others = np.setdiff1d(np.arange(motion.joint_pos.shape[1]), [roll])
    np.testing.assert_array_equal(fixed.joint_pos[:, others], motion.joint_pos[:, others])
    outward = -1 if side == "right" else 1
    assert np.all(np.sign(fixed.joint_pos[:, roll] - inward) == outward) and info["frames_fixed"] > 0
    # just clear (5 mm margin): 0.08 rad back toward the body at the worst frame touches again
    worst = int(np.argmax(np.abs(fixed.joint_pos[:, roll] - inward)))
    back = fixed.copy()
    back.joint_pos[:, roll] = fixed.joint_pos[worst, roll] - outward * 0.08
    assert _arm_torso_depth(back, side)[worst] < 0


def test_arm_clearance_is_smoothed_and_stays_inside_the_joint_limit():
    """Smoothing spreads a neighbour's larger push; it must neither re-enter the body nor pass the limit."""
    motion = truth().copy()
    roll = JOINT_NAMES.index("right_shoulder_roll_joint")
    g1 = load_g1()
    motion.joint_pos[:, roll] = 0.6  # inside the torso
    motion.joint_pos[60:64, roll] = np.linspace(0.6, g1.joint_lower[roll] + 0.05, 4)  # dash out to the limit
    fixed, info = fixes.clear_arm_torso_collision(motion, "right", g1=g1)
    assert fixed.joint_pos[:, roll].min() >= g1.joint_lower[roll]
    assert _arm_torso_depth(fixed, "right").min() >= 0.0
    assert info["frames_still_touching"] == 0
    step = np.abs(np.diff(fixed.joint_pos[:, roll]))
    assert step[:55].max() < 0.05  # the eased offset has no jumps where the need is constant

def test_plant_still_feet_leaves_clean_footwork_alone():
    fixed, info = fixes.plant_still_feet(truth())
    assert info["planted_phases"] == [0, 0]
    np.testing.assert_array_equal(fixed.joint_pos, truth().joint_pos)


def test_clean_plants_the_hovering_front_foot():
    motion, (h0, h1) = hovering_lunge()
    result = clean_motion(motion, CleanConfig(hold_s=0))
    lowest = forward_kinematics(result.motion).foot_lowest_z
    assert lowest[h0 + 2 : h1 - 2, 1].max() < 0.01
    assert result.report["verdict"] in ("pass", "warn"), result.report["reasons"]


def test_clip_cut_while_the_front_foot_lands_ends_on_both_feet():
    """aY1P3wzvNBM: the drill flows from the recovery into a retreat, so the clip is cut
    while the front foot is still coming down; the synthesized end hold must stand on it."""
    motion, spans = synthesize([("en_garde", 1.0), ("lunge", 1.2)])
    a, b, _ = spans[1]
    cut = b - int(0.4 * 0.45 * (b - a))  # 60 % into the recovery step: front foot ~4 cm up
    cut_motion = motion.slice(0, cut)
    assert forward_kinematics(cut_motion).foot_lowest_z[-1, 1] > 0.03
    result = clean_motion(cut_motion, CleanConfig())
    last = forward_kinematics(result.motion).foot_lowest_z[-1]
    assert np.abs(last).max() < 0.01, last


def test_floor_penetration_is_lifted_smoothly():
    sunk = truth().replace()
    sunk.root_pos[:, 2] -= 0.03
    fixed, info = fixes.remove_floor_penetration(sunk)
    lowest = forward_kinematics(fixed).foot_lowest_z.min()
    assert info["max_depth_m"] > 0.025
    assert lowest > -1e-6
    assert np.abs(np.diff(fixed.root_pos[:, 2], 2)).max() < 0.01  # no steps


def test_smoothing_reduces_jitter():
    clean = truth()
    noisy = add_jitter(clean, joint_sigma_rad=0.03, root_sigma_m=0.005)
    smooth, _ = fixes.smooth_motion(noisy, cutoff_hz=6.0)
    err_noisy = np.abs(noisy.joint_pos - clean.joint_pos).mean()
    err_smooth = np.abs(smooth.joint_pos - clean.joint_pos).mean()
    # White noise up to Nyquist (15 Hz); a 6 Hz cutoff keeps ~40 % of its power -> ~0.6x amplitude.
    assert err_smooth < 0.7 * err_noisy
    np.testing.assert_allclose(np.linalg.norm(smooth.root_quat, axis=1), 1.0, atol=1e-12)


def test_smoothing_handles_quaternion_sign_flips():
    motion = truth().replace()
    motion.root_quat[::2] *= -1  # same rotations, alternating sign
    smooth, _ = fixes.smooth_motion(motion)
    dots = np.abs(np.sum(smooth.root_quat * truth().root_quat, axis=1))
    assert dots.min() > 0.9999


def test_joint_limits_are_clamped_with_margin():
    g1 = load_g1()
    motion = truth().replace()
    motion.joint_pos[10, 3] = g1.joint_upper[3] + 0.5
    out, info = fixes.clamp_joint_limits(motion, margin_rad=0.02)
    assert out.joint_pos[10, 3] == pytest.approx(g1.joint_upper[3] - 0.02)
    assert info["clamped_values"] >= 1


def test_joint_speed_is_clamped():
    motion = truth().replace()
    motion.joint_pos[40:, 3] += 0.8  # a 24 rad/s step on the left knee
    out, info = fixes.clamp_joint_velocity(motion, hardware_fraction=0.5)
    speed = np.abs(np.diff(out.joint_pos, axis=0)) * motion.fps
    assert info["frames_over"] > 0
    assert (speed <= fixes.hardware_velocity_limits() * 0.5 + 1e-9).all()


def test_global_max_joint_speed():
    out, _ = fixes.clamp_joint_velocity(truth(), hardware_fraction=1.0, max_joint_speed=1.0)
    assert (np.abs(np.diff(out.joint_pos, axis=0)) * 30 <= 1.0 + 1e-9).all()


def test_time_scale_slows_down_and_keeps_endpoints():
    motion = truth()
    slow, info = fixes.time_scale(motion, 2.0)
    assert slow.n_frames == 2 * (motion.n_frames - 1) + 1
    np.testing.assert_allclose(slow.joint_pos[0], motion.joint_pos[0])
    np.testing.assert_allclose(slow.joint_pos[-1], motion.joint_pos[-1], atol=1e-12)
    np.testing.assert_allclose(slow.joint_pos[::2], motion.joint_pos, atol=1e-9)
    speed_ratio = np.abs(np.diff(slow.joint_pos, axis=0)).max() / np.abs(np.diff(motion.joint_pos, axis=0)).max()
    assert speed_ratio < 0.7  # PCHIP keeps local peaks, so ~0.6 rather than exactly 0.5


def test_full_chain_fixes_combined_corruption():
    clean = truth()
    corrupted = add_jitter(add_root_slide(add_vertical_drift(clean, 0.05, 3.0, 0.04), 0.1), 0.01, 0.003)
    raw = quality_report(corrupted)
    result = clean_motion(corrupted, CleanConfig(hold_s=0))
    m = result.report["metrics"]
    assert raw["verdict"] == "fail"
    assert result.report["verdict"] == "pass", result.report["reasons"]
    assert m["foot_slip"]["max_deviation_m"] < 0.02
    assert m["penetration"]["max_depth_m"] < 0.002
    assert m["joint_limits"]["violations"] == 0


@pytest.mark.skipif(not LAFAN_WALK.exists(), reason="LAFAN1 G1 data not downloaded")
def test_clean_lafan1_is_not_materially_altered():
    raw = read_csv(LAFAN_WALK).slice(0, 600)
    result = clean_motion(raw, CleanConfig(hand="R", hold_s=0, velocity_fraction=None))
    change = np.abs(result.motion.joint_pos - raw.joint_pos)
    assert change.mean() < 0.01, change.mean()
    assert np.percentile(change, 99) < 0.06
    # placement: same trajectory up to the canonical rigid transform
    path_raw = np.linalg.norm(np.diff(raw.root_pos[:, :2], axis=0), axis=1).sum()
    path_out = np.linalg.norm(np.diff(result.motion.root_pos[:, :2], axis=0), axis=1).sum()
    assert abs(path_out - path_raw) / path_raw < 0.15
    assert np.abs(result.motion.root_pos[:, 2] - raw.root_pos[:, 2]).max() < 0.03
    assert result.report["verdict"] in ("pass", "warn")


# ---------------------------------------------------------------- fencing line / heading drift


@lru_cache(maxsize=1)
def drill():
    """Three advances: 0.75 m along +x with a constant heading (the truth)."""
    motion, _ = synthesize([("en_garde", 0.6), ("advance", 0.6), ("advance", 0.6), ("advance", 0.6), ("en_garde", 0.6)])
    return motion


def line_frame_stats(motion):
    """Pelvis yaw relative to the fitted fencing line, and lateral root offsets from it."""
    angle, _ = fencing_line(motion, forward_kinematics(motion))
    u, n = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    rel = motion.root_pos[:, :2] - motion.root_pos[0, :2]
    yaw = np.unwrap(yaw_of(motion.root_quat)) - angle
    return np.degrees(yaw), rel @ u, rel @ n


def test_fencing_line_follows_travel_and_points_to_the_front_foot():
    motion = rotate_about_z(drill(), 0.8, np.zeros(2))
    angle, source = fencing_line(motion, forward_kinematics(motion))
    assert source == "travel"
    assert abs(np.degrees(angle - 0.8)) < 2.0
    hold, _ = synthesize([("en_garde", 1.0)])
    hold = rotate_about_z(hold, -2.0, np.zeros(2))
    angle, source = fencing_line(hold, forward_kinematics(hold))
    assert source == "feet"
    stance = Stance()
    feet_angle = np.arctan2(stance.front_y - stance.back_y, stance.front_x - stance.back_x)  # not exactly +x
    assert abs(np.degrees(angle - (feet_angle - 2.0))) < 2.0


def test_straighten_line_removes_heading_drift_and_bent_path():
    truth_yaw, truth_s, _ = line_frame_stats(drill())
    corrupted = add_yaw_drift(drill(), 40.0)
    yaw0, _, lat0 = line_frame_stats(corrupted)
    assert np.ptp(yaw0) > 30.0 and np.ptp(lat0) > 0.08
    fixed, info = fixes.straighten_line(corrupted)
    yaw, s, lat = line_frame_stats(fixed)
    assert info["line_source"] == "travel" and not info.get("skipped")
    # The drift is gone; a constant heading offset (part of the mean drift) may remain, since
    # nothing in the data says which moment of a drifting clip had the right heading.
    assert np.ptp(yaw - truth_yaw) < 6.0, np.ptp(yaw - truth_yaw)
    assert np.abs(yaw - truth_yaw).max() < 25.0
    assert np.ptp(lat) < 0.3 * np.ptp(lat0)
    assert abs(s[-1] - truth_s[-1]) < 0.05


def test_straighten_line_leaves_a_clean_drill_alone():
    fixed, _ = fixes.straighten_line(drill())
    yaw0, _, _ = line_frame_stats(drill())
    yaw, _, _ = line_frame_stats(fixed)
    assert np.abs(yaw - yaw0).max() < 2.0
    assert np.abs(fixed.root_pos - drill().root_pos).max() < 0.01
    np.testing.assert_array_equal(fixed.joint_pos, drill().joint_pos)


def test_straighten_line_keeps_fast_rotations():
    motion = drill().copy()
    t = np.arange(motion.n_frames) / motion.fps
    bump = np.radians(15.0) * np.exp(-0.5 * ((t - 1.5) / 0.08) ** 2)  # a 0.3 s hip turn
    motion = motion.replace(root_quat=(Rotation.from_euler("z", bump[:, None]) * Rotation.from_quat(motion.root_quat)).as_quat())
    fixed, _ = fixes.straighten_line(motion)
    kept = np.unwrap(yaw_of(fixed.root_quat)) - np.unwrap(yaw_of(drill().root_quat))
    assert np.degrees(kept.max()) > 11.0


def test_straighten_line_skips_a_real_turn():
    turned = add_yaw_drift(drill(), 120.0)
    fixed, info = fixes.straighten_line(turned)
    assert info["skipped"]
    np.testing.assert_array_equal(fixed.to_array(), turned.to_array())


def test_full_chain_puts_footwork_on_plus_x():
    corrupted = rotate_about_z(add_yaw_drift(drill(), 35.0), 1.1, np.array([0.5, -0.3]))
    result = clean_motion(corrupted, CleanConfig(hold_s=0))
    out = result.motion
    travel = out.root_pos[-1, :2] - out.root_pos[0, :2]
    assert travel[0] > 0.6 and abs(travel[1]) < 0.05, travel
    assert np.ptp(out.root_pos[:, 1]) < 0.05
    assert result.report["verdict"] == "pass", result.report["reasons"]


def test_unobserved_wrist_angles_are_set_to_the_neutral_pose():
    """Video pose has no wrist rotation, yet the retargeter emits wrist pitch/yaw up to 1.2 rad.
    The wrist motors' action scale is 0.075 rad, so the policy never reaches them
    (advance_lunge_cut_v3: left wrist error 0.27 rad, mean joint gate missed at 0.110)."""
    motion = truth().copy()
    pitch, yaw = (JOINT_NAMES.index(f"left_wrist_{axis}_joint") for axis in ("pitch", "yaw"))
    motion.joint_pos[:, pitch] = np.linspace(-0.95, -0.38, motion.n_frames)
    motion.joint_pos[:, yaw] = -1.2
    fixed, info = fixes.neutralize_wrists(motion, sides=("left",))
    assert np.all(fixed.joint_pos[:, [pitch, yaw]] == 0.0)
    others = np.setdiff1d(np.arange(motion.joint_pos.shape[1]), [pitch, yaw])
    np.testing.assert_array_equal(fixed.joint_pos[:, others], motion.joint_pos[:, others])
    assert info["max_change_rad"] == pytest.approx(1.2) and info["joints"] == ["left_wrist_pitch_joint", "left_wrist_yaw_joint"]


def test_neutralize_wrists_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="middle"):
        fixes.neutralize_wrists(truth().copy(), sides=("middle",))
