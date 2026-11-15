"""Bending-blade model (fencing.saber.flex), the SaberSpec switch that builds it, and its policy
evaluation (fencing.saber.flex_eval).

Pure-MuJoCo tests run anywhere. The robot tests need mjlab's G1 model and meshes (third_party/mjlab),
and the scene tests also a committed saber export (results/policies/...cut_head_v3_tip/model_2999);
they skip without them.
"""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fencing.paths import G1_MJCF
from fencing.saber import flex_eval as fe
from fencing.saber.flex import (
    MJLAB_G1_ASSETS,
    FlexBlade,
    chain_modes,
    check_stable,
    flex_joint_ids,
    flexible_scene,
    rigid_scene_for,
    saber_mass_check,
    stable_timestep,
)
from fencing.saber.spec import (
    NOMINAL_SABER,
    SaberSpec,
    attach_saber,
    g1_saber_model,
    g1_spec,
    urdf_snippet,
)

EXPORT = fe.export_dir("cut_head_v3")
needs_export = pytest.mark.skipif(
    not (EXPORT / "scene" / "scene.xml").is_file() or not (MJLAB_G1_ASSETS / "pelvis.STL").is_file(),
    reason="needs the committed cut_head_v3 export and third_party/mjlab (scripts/setup_third_party.sh)")
needs_g1 = pytest.mark.skipif(
    not G1_MJCF.is_file() or not (MJLAB_G1_ASSETS / "pelvis.STL").is_file(),
    reason="needs mjlab's G1 model and meshes (scripts/setup_third_party.sh)")
BENDING = SaberSpec(flex_segments=2, flex_f1_flatwise_hz=5.0, flex_f1_edgewise_hz=12.0)
# The committed rigid G1 + saber (commit 95238d9): body/joint/actuator/geom/site counts and the
# physical-parameter hash of tests/test_saber_mjlab.py::_robot_model_digest, computed on the plain
# G1 MJCF so it runs without mjlab.
COMMITTED_RIGID = (32, 30, 0, 71, 8, "1aa7df540000beec271dffbe80a8f27edff0d946c49d05e170250c8363c59312")
BETA1 = 1.8751  # first cantilever mode
SECOND_TO_FIRST = 6.267  # cantilever beam: f2 / f1


def beam_ei(f1: float, saber=NOMINAL_SABER) -> float:
    mu = saber.blade_mass / saber.blade_length
    return (2 * math.pi * f1 / BETA1**2) ** 2 * mu * saber.blade_length**4


@pytest.mark.parametrize("segments", [3, 5])
@pytest.mark.parametrize("f1", [5.0, 10.0])
def test_chain_hits_its_target_frequency_and_matches_beam_theory(segments, f1):
    modes = chain_modes(FlexBlade(segments, f1))
    assert modes.frequencies_hz[0] == pytest.approx(f1, rel=1e-9)
    # A coarse chain is a little stiff in its second mode: 3 segments 6.5x (4% high), 5 segments 6.4x.
    assert modes.frequencies_hz[1] / f1 == pytest.approx(SECOND_TO_FIRST, rel=0.05)
    saber = NOMINAL_SABER
    droop = saber.blade_mass / saber.blade_length * 9.81 * saber.blade_length**4 / (8 * beam_ei(f1))
    assert modes.static_tip_droop_m == pytest.approx(droop, rel=0.02)


@pytest.mark.parametrize("blade", [FlexBlade(3, 3.0), FlexBlade(3, 15.0), FlexBlade(10, 7.0)])
def test_stable_timestep_divides_the_policy_step_and_resolves_the_fastest_mode(blade):
    modes = chain_modes(blade)
    dt = stable_timestep(modes)
    assert round(0.02 / dt) * dt == pytest.approx(0.02, abs=1e-12)
    assert 2 * math.pi * modes.frequencies_hz.max() * dt <= 0.5


def test_measurement_helpers():
    assert fe.wobble_from_video(frames=180, cycles=5, fps=240) == pytest.approx(6.6667, rel=1e-4)
    delta = math.log(10 / 5) / 3
    assert fe.damping_from_decay(10, 5, 3) == pytest.approx(delta / math.sqrt(4 * math.pi**2 + delta**2))
    with pytest.raises(ValueError):
        fe.damping_from_decay(5, 10, 3)
    # A uniform blade's bend test and its frequency must agree with the same beam formulas.
    saber, f1 = NOMINAL_SABER, 6.0
    load, length = 0.2, saber.blade_length
    deflection = load * 9.81 * length**3 / (3 * beam_ei(f1))
    assert fe.uniform_beam_f1(load, deflection, length, saber.blade_mass, length) == pytest.approx(f1, rel=1e-9)


@needs_export
def test_flexible_scene_keeps_mass_and_tip_and_adds_only_unactuated_hinges():
    rigid_xml = fe.rigid_scene(EXPORT)
    rigid = mujoco.MjModel.from_xml_string(rigid_xml)
    xml, _ = flexible_scene(rigid_xml, FlexBlade(3, 7.0))
    model = mujoco.MjModel.from_xml_string(xml)
    hinges = flex_joint_ids(model)
    assert len(hinges) == 6
    assert model.nu == rigid.nu  # no new actuator
    driven = {int(j) for j in model.actuator_trnid[:, 0]}
    assert driven.isdisjoint(hinges)
    assert saber_mass_check(model) == pytest.approx(NOMINAL_SABER.total_mass, abs=1e-9)
    tips = []
    for m in (rigid, model):
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        tips.append(d.site("robot/blade_tip").xpos.copy())
    np.testing.assert_allclose(tips[0], tips[1], atol=1e-6)  # straight blade: same tip as the rigid saber


@needs_export
def test_rigid_scene_for_the_nominal_saber_keeps_the_committed_saber():
    xml = fe.rigid_scene(EXPORT)
    a = mujoco.MjModel.from_xml_string(xml)
    b = mujoco.MjModel.from_xml_string(rigid_scene_for(xml, NOMINAL_SABER))
    saber, tip = a.body("robot/saber").id, a.site("robot/blade_tip").id
    assert b.body_mass[saber] == pytest.approx(a.body_mass[saber])
    np.testing.assert_allclose(b.body_ipos[saber], a.body_ipos[saber], atol=1e-9)
    np.testing.assert_allclose(b.body_inertia[saber], a.body_inertia[saber], atol=1e-6)
    np.testing.assert_allclose(b.site_pos[tip], a.site_pos[tip], atol=1e-9)


@needs_export
def test_the_harness_reproduces_the_committed_sim2sim_on_the_rigid_saber():
    keys = fe.committed_key_steps(EXPORT)
    result = fe.run(EXPORT, fe.rigid_scene(EXPORT), 0.005, keys)
    committed = json.loads((EXPORT / "sim2sim.json").read_text())["nominal"]
    for name in ("joint_angle_error_rad", "body_pos_error_m", "blade_tip_error_m", "blade_tip_error_key_m"):
        assert result["repo_metrics"][name] == pytest.approx(committed[name], abs=1e-9), name
    # With a rigid blade the real tip is the wrist-offset tip the repo scores.
    assert result["real_tip_error_key_m"] == pytest.approx(committed["blade_tip_error_key_m"], abs=1e-9)
    assert result["max_bend_m"] == pytest.approx(0.0, abs=1e-9)


@needs_export
def test_the_harness_matches_the_repo_rollout_run_side_by_side(tmp_path):
    # flex_eval.run copies the default path of fencing.sim2sim.rollout.rollout instead of changing it;
    # running both on the rigid saber catches any change to rollout.py that the copy misses.
    from fencing.sim2sim import rollout as ro

    keys = fe.committed_key_steps(EXPORT)
    scene = tmp_path / "scene.xml"
    scene.write_text(fe.rigid_scene(EXPORT))
    export = replace(ro.load_export(EXPORT), scene_xml=scene)
    repo = ro.rollout(export, config=ro.RolloutConfig(key_frames=keys))["nominal"]
    ours = fe.run(EXPORT, fe.rigid_scene(EXPORT), 0.005, keys)["repo_metrics"]
    for name in ("success_rate", "joint_angle_error_rad", "body_pos_error_m", "blade_tip_error_m",
                 "blade_tip_error_key_m"):
        assert ours[name] == pytest.approx(repo[name], abs=1e-12), name


@needs_export
def test_a_very_stiff_blade_behaves_like_the_rigid_saber():
    keys = fe.committed_key_steps(EXPORT)
    base = fe.rigid_scene(EXPORT)
    xml, modes = flexible_scene(base, FlexBlade(3, 30.0))
    dt = stable_timestep(modes)
    rigid = fe.run(EXPORT, base, dt, keys)
    stiff = fe.run(EXPORT, xml, dt, keys)
    assert stiff["terminated_at_s"] is None
    assert stiff["real_tip_error_key_m"] == pytest.approx(rigid["real_tip_error_key_m"], abs=0.005)


def _digest(model: mujoco.MjModel) -> tuple:
    fields = ("body_mass", "body_inertia", "body_ipos", "body_pos", "jnt_range", "dof_damping", "dof_armature",
              "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
              "geom_size", "geom_friction", "geom_pos", "site_pos")
    digest = hashlib.sha256()
    for name in fields:
        digest.update(np.round(np.asarray(getattr(model, name), dtype=np.float64), 9).tobytes())
    return model.nbody, model.njnt, model.nu, model.ngeom, model.nsite, digest.hexdigest()


@needs_g1
def test_switch_off_builds_exactly_the_committed_rigid_saber():
    assert _digest(g1_saber_model(NOMINAL_SABER)) == COMMITTED_RIGID
    # The other bending settings do nothing while flex_segments is 0.
    assert _digest(g1_saber_model(SaberSpec(flex_f1_flatwise_hz=5.0, flex_f1_edgewise_hz=12.0))) == COMMITTED_RIGID


def test_switch_settings_stay_out_of_the_task_snapshots():
    # tests/test_saber_mjlab.py compares the saber tasks against these files, which store repr(SaberSpec).
    assert repr(BENDING) == repr(NOMINAL_SABER)
    for name in ("saber_task_cfgs.json", "saber_entry_task_cfgs.json"):
        assert repr(NOMINAL_SABER) in (Path(__file__).parent / "data" / name).read_text()


@pytest.mark.parametrize("settings", [
    {"flex_segments": -1},
    {"flex_segments": 1.5, "flex_f1_flatwise_hz": 5.0},
    {"flex_segments": 2},  # no frequency
    {"flex_segments": 2, "flex_f1_flatwise_hz": 5.0, "flex_f1_edgewise_hz": -1.0},
    {"flex_segments": 2, "flex_f1_flatwise_hz": 5.0, "flex_damping_ratio": 1.0},
])
def test_bad_switch_settings_are_refused(settings):
    with pytest.raises(ValueError):
        SaberSpec(**settings)


def test_each_bending_direction_hits_its_own_frequency():
    modes = chain_modes(FlexBlade(2, 5.0, 0.05, 12.5))
    assert modes.frequencies_hz[0] == pytest.approx(5.0, rel=1e-9)
    assert modes.frequencies_edge_hz[0] == pytest.approx(12.5, rel=1e-9)
    assert modes.stiffness_edge / modes.stiffness == pytest.approx((12.5 / 5.0) ** 2, rel=1e-9)
    same = chain_modes(FlexBlade(2, 5.0))
    assert same.stiffness_edge == same.stiffness
    assert FlexBlade.from_saber(SaberSpec(flex_segments=2, flex_f1_flatwise_hz=5.0)).edge_hz == 5.0


@needs_g1
def test_switch_on_hangs_the_blade_pieces_and_keeps_the_saber_mass_and_tip():
    rigid, bent = g1_saber_model(NOMINAL_SABER), g1_saber_model(BENDING)
    hinges = flex_joint_ids(bent)
    assert len(hinges) == 4 and bent.nq == rigid.nq + 4 and bent.nu == rigid.nu
    modes = chain_modes(FlexBlade.from_saber(BENDING))
    for i in range(2):
        assert bent.joint(f"saber_flex_{i}_y").stiffness[0] == pytest.approx(modes.stiffness_edge)
        assert bent.joint(f"saber_flex_{i}_z").stiffness[0] == pytest.approx(modes.stiffness)
    saber_bodies = [b for b in range(bent.nbody) if bent.body(b).name == "saber" or bent.body(b).name.startswith("saber_blade_")]
    assert bent.body_mass[saber_bodies].sum() == pytest.approx(NOMINAL_SABER.total_mass, abs=1e-9)
    tips = []
    for model in (rigid, bent):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        tips.append(data.site("blade_tip").xpos.copy())
    np.testing.assert_allclose(tips[0], tips[1], atol=1e-6)  # straight blade: same tip as the rigid saber


@needs_g1
@needs_export
def test_switch_on_builds_the_same_blade_the_harness_tested():
    xml, _ = flexible_scene(fe.rigid_scene(EXPORT), FlexBlade.from_saber(BENDING))
    harness, built = mujoco.MjModel.from_xml_string(xml), g1_saber_model(BENDING)
    for i in range(2):
        a, b = harness.body(f"robot/saber_blade_{i}"), built.body(f"saber_blade_{i}")
        assert b.mass[0] == pytest.approx(a.mass[0])
        np.testing.assert_allclose(b.inertia, a.inertia, rtol=1e-9)
        np.testing.assert_allclose(b.pos, a.pos, atol=1e-9)
        for axis in ("y", "z"):
            ja, jb = harness.joint(f"robot/saber_flex_{i}_{axis}"), built.joint(f"saber_flex_{i}_{axis}")
            assert jb.stiffness[0] == pytest.approx(ja.stiffness[0], rel=1e-8)
            assert built.dof_damping[jb.dofadr[0]] == pytest.approx(harness.dof_damping[ja.dofadr[0]], rel=1e-8)
    handle = NOMINAL_SABER.grip_mass + NOMINAL_SABER.guard_mass  # grip + guard stay one welded body
    assert harness.body("robot/saber").mass[0] == pytest.approx(handle)
    assert built.body("saber").mass[0] == pytest.approx(handle)


def test_stability_check_refuses_a_blade_that_blows_up_at_the_timestep():
    check_stable(FlexBlade(2, 5.0, 0.05, 12.0), 0.005)  # ran at 5 ms under the saber policies
    with pytest.raises(ValueError, match="blows up"):
        check_stable(FlexBlade(3, 5.0, 0.05, 12.5), 0.005)  # blew up at 5 ms under cut_head_v4


@needs_g1
def test_attach_saber_refuses_an_unstable_blade_and_blade_collision():
    spec = g1_spec()
    spec.option.timestep = 0.005
    with pytest.raises(ValueError, match="blows up"):
        attach_saber(spec, replace(BENDING, flex_segments=3, flex_f1_edgewise_hz=12.5))
    with pytest.raises(NotImplementedError):
        attach_saber(g1_spec(), replace(BENDING, collision=True))


def test_training_and_the_urdf_refuse_a_bending_saber():
    from fencing.saber.mjlab_task import saber_g1_robot_cfg

    with pytest.raises(NotImplementedError, match="bending blade"):
        saber_g1_robot_cfg(BENDING)
    with pytest.raises(NotImplementedError):
        urdf_snippet(BENDING)


@needs_g1
def test_the_keyframer_and_blade_path_tools_refuse_a_bending_saber():
    # They plan with the straight blade; a model with blade hinges has a different qpos layout.
    from fencing.saber import blade, keyframes

    with pytest.raises(NotImplementedError, match="straight blade"):
        keyframes._context(BENDING, 0.9)
    motion = np.zeros((1, 36))
    motion[:, 6] = 1.0  # identity root quaternion (xyzw)
    with pytest.raises(RuntimeError, match="nq"):
        blade.blade_trajectory(motion, BENDING)


def test_hcf_saber_flex_forwards_to_the_bending_blade_tool(capsys):
    from fencing.saber.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["flex", "measured", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "fencing saber flex measured" in out and "--f1-edge" in out


@needs_g1
def test_build_mjcf_turns_the_bending_blade_on_without_touching_the_committed_asset(tmp_path, capsys):
    from fencing.saber import cli
    from fencing.saber.spec import DEFAULT_MJCF_OUT

    before = DEFAULT_MJCF_OUT.read_bytes() if DEFAULT_MJCF_OUT.exists() else None
    out = tmp_path / "bending.xml"
    assert cli.main(["build-mjcf", "--flex-segments", "1", "--flex-f1", "5", "--flex-f1-edge", "12.5",
                     "--out", str(out)]) == 0
    model = mujoco.MjModel.from_xml_path(str(out))
    assert len(flex_joint_ids(model)) == 2
    assert "bending blade" in capsys.readouterr().out
    assert (DEFAULT_MJCF_OUT.read_bytes() if DEFAULT_MJCF_OUT.exists() else None) == before
    assert cli.main(["build-mjcf", "--flex-segments", "1", "--out", str(out)]) == 2  # no frequency given
