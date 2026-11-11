"""Saber model, blade-tip helper and IK solver (fencing venv: numpy + mujoco only)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from fencing.saber import blade, reach, spec
from fencing.saber.mjlab_task import saber_alpha_range
from fencing.saber.spec import NOMINAL_SABER, SABER_BODY, TIP_SITE, SaberSpec


@pytest.fixture(scope="module")
def saber_model() -> mujoco.MjModel:
    return spec.g1_saber_model()


def _standing_motion(frames: int = 5) -> np.ndarray:
    """CSV rows (xyzw root quaternion) of the G1 standing at the origin with a moving arm."""
    rows = np.zeros((frames, blade.CSV_COLUMNS))
    rows[:, 2] = 0.79
    rows[:, 6] = 1.0  # qw (xyzw order)
    right_elbow_column = 7 + 25  # 12 leg + 3 waist + 7 left arm + 3 right shoulder
    rows[:, right_elbow_column] = np.linspace(0.0, 1.2, frames)
    return rows


# --- mass properties --------------------------------------------------------------------


def test_nominal_saber_matches_adr_spec():
    assert NOMINAL_SABER.total_mass == pytest.approx(0.20)
    assert NOMINAL_SABER.total_length == pytest.approx(0.88)
    assert NOMINAL_SABER.blade_length <= 0.88  # FIE blade maximum
    assert NOMINAL_SABER.total_length <= 1.05  # FIE overall maximum


def test_mass_properties_are_physical():
    props = spec.mass_properties()
    assert props.mass == pytest.approx(NOMINAL_SABER.total_mass)
    assert 0 < props.com[0] < NOMINAL_SABER.tip_x  # COM between grip centre and tip
    principal = np.diag(props.inertia)
    assert np.all(principal > 0)
    for i in range(3):  # triangle inequality of principal moments
        assert principal[i] <= principal[(i + 1) % 3] + principal[(i + 2) % 3] + 1e-12
    # A long thin weapon: tiny moment about its own axis, large ones across it.
    assert principal[0] < 0.01 * principal[1]


def test_mass_properties_agree_with_mujoco_inertia_from_geoms():
    """Compile the three parts as massive geoms and let MuJoCo do the composite inertia."""
    world = mujoco.MjSpec()
    body = world.worldbody.add_body(name="saber")
    for name, mass, centre, _ in spec.saber_parts(NOMINAL_SABER):
        s = NOMINAL_SABER
        if name == "guard":
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre,
                                 size=[s.guard_thickness / 2, s.guard_width / 2, s.guard_height / 2])
        else:
            radius, length = ((s.grip_radius, s.grip_length) if name == "grip"
                              else (s.blade_radius, s.blade_length))
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=centre,
                                 size=[radius, length / 2, 0], quat=[math.sqrt(0.5), 0, math.sqrt(0.5), 0])
        geom.mass = mass
    model = world.compile()
    props = spec.mass_properties()
    assert model.body_mass[1] == pytest.approx(props.mass)
    np.testing.assert_allclose(model.body_ipos[1], props.com, atol=1e-9)
    # MuJoCo reports principal moments in its own order; compare as sorted sets.
    np.testing.assert_allclose(np.sort(model.body_inertia[1]), np.sort(np.diag(props.inertia)),
                               rtol=1e-6)


def test_with_total_mass_scales_every_part_and_keeps_com():
    heavy = NOMINAL_SABER.with_total_mass(0.35)
    assert heavy.total_mass == pytest.approx(0.35)
    np.testing.assert_allclose(spec.mass_properties(heavy).com, spec.mass_properties().com)


@pytest.mark.parametrize("field, value", [("blade_length", 0.0), ("blade_mass", -0.1),
                                          ("grip_radius", float("nan")), ("mount_pitch_deg", 95.0)])
def test_invalid_saber_spec_names_the_bad_field(field, value):
    with pytest.raises(ValueError, match=field):
        SaberSpec(**{field: value})


def test_mass_randomization_alpha_range_reproduces_kg_range():
    low, high = saber_alpha_range(NOMINAL_SABER, (0.12, 0.35))
    assert NOMINAL_SABER.total_mass * math.exp(2 * low) == pytest.approx(0.12)
    assert NOMINAL_SABER.total_mass * math.exp(2 * high) == pytest.approx(0.35)


# --- MJCF -------------------------------------------------------------------------------


def test_saber_is_appended_after_every_existing_body(saber_model):
    plain = spec.g1_spec().compile()
    names = [saber_model.body(i).name for i in range(saber_model.nbody)]
    assert names[:-1] == [plain.body(i).name for i in range(plain.nbody)]
    assert names[-1] == SABER_BODY
    assert saber_model.body(SABER_BODY).mass[0] == pytest.approx(NOMINAL_SABER.total_mass)


def test_saber_geoms_do_not_collide_by_default(saber_model):
    saber_id = saber_model.body(SABER_BODY).id
    geoms = [g for g in range(saber_model.ngeom) if saber_model.geom_bodyid[g] == saber_id]
    assert len(geoms) == 3
    assert all(saber_model.geom_contype[g] == 0 and saber_model.geom_conaffinity[g] == 0
               for g in geoms)


def test_collision_option_adds_one_blade_capsule():
    model = spec.g1_saber_model(replace(NOMINAL_SABER, collision=True))
    geom = model.geom("saber_blade_collision")
    assert model.geom_contype[geom.id] == 1 and model.geom_conaffinity[geom.id] == 1


def test_attaching_twice_fails_loudly():
    twice = spec.g1_saber_spec()
    with pytest.raises(ValueError, match="already has a body"):
        spec.attach_saber(twice)


def test_tip_site_sits_at_the_designed_offset_from_the_wrist(saber_model):
    data = mujoco.MjData(saber_model)
    rng = np.random.default_rng(1)
    for _ in range(5):  # any arm pose: the tip is rigid in the wrist frame
        data.qpos[7:] = rng.uniform(saber_model.jnt_range[1:, 0], saber_model.jnt_range[1:, 1])
        mujoco.mj_kinematics(saber_model, data)
        wrist = data.body(spec.PARENT_BODY)
        rotation = wrist.xmat.reshape(3, 3)
        tip_local = rotation.T @ (data.site(TIP_SITE).xpos - wrist.xpos)
        np.testing.assert_allclose(tip_local, spec.tip_in_hand(NOMINAL_SABER), atol=1e-9)
    expected_distance = np.linalg.norm(spec.tip_in_hand(NOMINAL_SABER))
    assert 0.85 < expected_distance < 1.0  # mount offset + grip half + guard + 0.75 m blade


def test_checked_in_mjcf_loads_in_plain_mujoco_and_matches_generator(saber_model):
    assert spec.DEFAULT_MJCF_OUT.exists(), "run `fencing saber build-mjcf`"
    from_file = mujoco.MjModel.from_xml_path(str(spec.DEFAULT_MJCF_OUT))
    assert from_file.nbody == saber_model.nbody
    data_file, data_gen = mujoco.MjData(from_file), mujoco.MjData(saber_model)
    mujoco.mj_kinematics(from_file, data_file)
    mujoco.mj_kinematics(saber_model, data_gen)
    np.testing.assert_allclose(data_file.site(TIP_SITE).xpos, data_gen.site(TIP_SITE).xpos,
                               atol=1e-6, err_msg="checked-in MJCF is stale: rerun build-mjcf")
    assert from_file.body(SABER_BODY).mass[0] == pytest.approx(saber_model.body(SABER_BODY).mass[0])


def test_urdf_snippet_is_well_formed_and_carries_the_mass():
    root = ET.fromstring(f"<robot name='g1'>{spec.urdf_snippet()}</robot>")
    mass = root.find("./link[@name='saber']/inertial/mass")
    assert float(mass.get("value")) == pytest.approx(NOMINAL_SABER.total_mass, abs=1e-4)
    assert root.find("./link[@name='blade_tip']") is not None


# --- blade-tip helper -------------------------------------------------------------------


def test_blade_tip_positions_returns_one_point_per_frame():
    tips = blade.blade_tip_positions(_standing_motion(7))
    assert tips.shape == (7, 3)
    assert np.all(np.isfinite(tips))
    assert np.linalg.norm(tips[-1] - tips[0]) > 0.1  # elbow moved, so the tip moved


def test_blade_tip_follows_root_translation_and_yaw():
    motion = _standing_motion(3)
    base = blade.blade_tip_positions(motion)

    shifted = motion.copy()
    shifted[:, 0:3] += [1.0, -2.0, 0.0]
    np.testing.assert_allclose(blade.blade_tip_positions(shifted), base + [1.0, -2.0, 0.0],
                               atol=1e-9)

    yawed = motion.copy()  # 90 deg about z, xyzw
    yawed[:, 3:7] = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    root = motion[0, 0:3]
    relative = base - root
    rotated = np.stack([-relative[:, 1], relative[:, 0], relative[:, 2]], axis=1) + root
    np.testing.assert_allclose(blade.blade_tip_positions(yawed), rotated, atol=1e-9)


def test_unnormalised_root_quaternion_gives_the_same_tip():
    motion = _standing_motion(2)
    scaled = motion.copy()
    scaled[:, 3:7] *= 3.0
    np.testing.assert_allclose(blade.blade_tip_positions(scaled),
                               blade.blade_tip_positions(motion), atol=1e-12)


def test_blade_trajectory_is_self_consistent():
    trajectory = blade.blade_trajectory(_standing_motion(4))
    np.testing.assert_allclose(np.linalg.norm(trajectory.direction, axis=1), 1.0)
    np.testing.assert_allclose(trajectory.guard + NOMINAL_SABER.blade_length * trajectory.direction,
                               trajectory.tip, atol=1e-9)


def test_blade_tip_error_is_zero_against_itself_and_rejects_length_mismatch():
    motion = _standing_motion(4)
    np.testing.assert_allclose(blade.blade_tip_error(motion, motion), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="frame count mismatch"):
        blade.blade_tip_error(motion, motion[:2])


@pytest.mark.parametrize("bad", [np.zeros((4, 35)), np.zeros(36)])
def test_wrong_csv_shape_is_rejected_with_the_shape(bad):
    with pytest.raises(ValueError, match="expected shape"):
        blade.blade_tip_positions(bad)


def test_load_motion_csv_round_trip(tmp_path):
    path = tmp_path / "m.csv"
    np.savetxt(path, _standing_motion(3), delimiter=",")
    np.testing.assert_allclose(blade.load_motion_csv(path), _standing_motion(3))


# --- IK ---------------------------------------------------------------------------------


def test_ik_recovers_a_pose_it_can_reach():
    """Target = forward kinematics of a random in-range arm pose; IK must find it again."""
    arm = reach.build_arm_model()
    base = reach.stance_qpos(arm.model)
    rng = np.random.default_rng(3)
    q_true = rng.uniform(arm.lower * 0.6, arm.upper * 0.6)
    data = mujoco.MjData(arm.model)
    data.qpos[:] = base
    data.qpos[arm.qpos_adr] = q_true
    mujoco.mj_kinematics(arm.model, data)
    tip = data.site_xpos[arm.tip_site].copy()
    direction = data.site_xmat[arm.tip_site].reshape(3, 3)[:, 0].copy()

    best = min((reach.solve_ik(arm, base, tip, direction, rng.uniform(arm.lower, arm.upper))
                for _ in range(8)), key=lambda r: r.position_error_m)
    assert best.position_error_m < 1e-3
    assert best.angle_error_deg < 0.5
    assert np.all(best.q >= arm.lower - 1e-12) and np.all(best.q <= arm.upper + 1e-12)


def test_saber_action_scale_widens_only_the_right_wrist_pitch_and_yaw():
    import re

    from fencing.motion.io import JOINT_NAMES
    from fencing.saber.mjlab_task import SABER_WRIST_ACTION_SCALE, saber_action_scale

    base = {".*_wrist_roll_joint": 0.44, ".*_wrist_pitch_joint": 0.075, ".*_wrist_yaw_joint": 0.075,
            ".*_elbow_joint": 0.44}
    scale = saber_action_scale(base)

    def resolved(joint):
        hits = [v for k, v in scale.items() if re.fullmatch(k, joint)]
        assert len(hits) == 1, (joint, hits)  # mjlab rejects a joint matched by two keys
        return hits[0]

    assert "right_wrist_pitch_joint" in JOINT_NAMES
    assert resolved("right_wrist_pitch_joint") == resolved("right_wrist_yaw_joint") == SABER_WRIST_ACTION_SCALE
    assert resolved("left_wrist_pitch_joint") == resolved("left_wrist_yaw_joint") == 0.075
    assert resolved("right_wrist_roll_joint") == 0.44 and resolved("left_elbow_joint") == 0.44
    assert base[".*_wrist_pitch_joint"] == 0.075  # the upstream dict is not mutated
