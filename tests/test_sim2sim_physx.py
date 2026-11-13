"""`fencing sim2sim --engine physx`: the PhysX (ovphysx, CPU) cross-check.

The pure conversions (joint frames, PhysX COM velocity -> link-origin velocity, the IMU)
are checked against MuJoCo here and run everywhere. The plant itself needs the separate
ovphysx venv (docs/evaluation.md, "Sim-to-sim in PhysX"), so that test skips without it.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
import pytest

from fencing.sim2sim import cli, physx_plant, physx_rollout
from fencing.sim2sim.quat import quat_apply

FREE_BODY_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0.1 -0.2 0.9" quat="0.9 0.1 -0.3 0.2">
      <freejoint/>
      <inertial pos="0.03 -0.02 -0.07" mass="3.8" diaginertia="0.01 0.009 0.008"/>
      <geom type="sphere" size="0.05"/>
      <site name="imu_in_pelvis" pos="0.045 0.01 -0.083" quat="0.96 0.0 0.28 0.0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro site="imu_in_pelvis" name="imu_ang_vel"/>
    <velocimeter site="imu_in_pelvis" name="imu_lin_vel"/>
  </sensor>
</mujoco>
"""


@pytest.mark.parametrize("axis", [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1), (0.3, -0.5, 0.8),
                                  (-1, 1e-9, 0)])
def test_joint_frame_rotates_x_onto_the_joint_axis(axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    q = physx_plant.quat_from_x_to(axis)
    assert np.isclose(np.linalg.norm(q), 1.0)
    np.testing.assert_allclose(quat_apply(q, np.array([1.0, 0.0, 0.0])), axis, atol=1e-9)


def test_quaternion_layout_round_trips_between_mujoco_and_physx():
    wxyz = np.array([0.9, 0.1, -0.3, 0.2])
    xyzw = physx_plant.wxyz_to_xyzw(wxyz)
    np.testing.assert_array_equal(xyzw, [0.1, -0.3, 0.2, 0.9])
    np.testing.assert_array_equal(physx_plant.xyzw_to_wxyz(xyzw), wxyz)


def test_com_velocity_is_origin_velocity_plus_spin_about_the_com_offset():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    origin = np.array([0.2, 0.0, 0.0])
    spin = np.array([0.0, 0.0, 1.0])
    com_local = np.array([1.0, 0.0, 0.0])
    com = physx_plant.origin_to_com_velocity(origin, spin, identity, com_local)
    np.testing.assert_allclose(com, [0.2, 1.0, 0.0])
    np.testing.assert_allclose(physx_plant.com_to_origin_velocity(com, spin, identity, com_local), origin)


def test_imu_from_physx_link_state_matches_mujocos_gyro_and_velocimeter():
    model = mujoco.MjModel.from_xml_string(FREE_BODY_XML)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    data.qvel[:] = rng.normal(size=6) * 2.0
    mujoco.mj_forward(model, data)
    body, site = 1, 0
    quat = data.xquat[body].copy()
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)  # COM, world frame
    angular, com_linear = velocity[:3], velocity[3:]
    # What PhysX hands us: link-frame pose, COM linear velocity, world angular velocity.
    origin_linear = physx_plant.com_to_origin_velocity(com_linear, angular, quat, model.body_ipos[body])
    linear_imu, angular_imu = physx_rollout.imu_readings(quat, origin_linear, angular, model.site_pos[site],
                                                         model.site_quat[site])
    np.testing.assert_allclose(angular_imu, data.sensordata[0:3], atol=1e-10)
    np.testing.assert_allclose(linear_imu, data.sensordata[3:6], atol=1e-10)


def test_engine_defaults_to_mujoco_and_keeps_writing_sim2sim_json(tmp_path):
    args = cli.build_parser().parse_args(["some_run"])
    assert args.engine == "mujoco"
    assert cli.default_output(tmp_path, "mujoco") == tmp_path / "sim2sim.json"


def test_physx_engine_writes_its_own_json_next_to_the_mujoco_one(tmp_path):
    assert cli.default_output(tmp_path, "physx") == tmp_path / "sim2sim_physx.json"


@pytest.mark.parametrize("extra", [["--video"], ["--pd", "explicit"]])
def test_physx_engine_refuses_options_it_does_not_implement(extra, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main([str(tmp_path / "missing_export"), "--engine", "physx", *extra])
    assert raised.value.code == 2
    assert "--engine physx" in capsys.readouterr().err


def test_physx_engine_without_ovphysx_says_where_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "ovphysx", None)
    with pytest.raises(RuntimeError, match="ovphysx"):
        physx_plant.require_ovphysx()


def test_step_dt_check_accepts_a_matching_timestep_and_decimation():
    physx_rollout.check_step_dt(0.005, 4, 0.02)


def test_step_dt_check_names_the_mismatched_values():
    with pytest.raises(ValueError, match=r"timestep 0.004 x decimation 4 != step_dt 0.02"):
        physx_rollout.check_step_dt(0.004, 4, 0.02)


def test_imu_check_accepts_a_site_on_the_root_link():
    model = mujoco.MjModel.from_xml_string(FREE_BODY_XML)
    physx_rollout.check_imu_on_root(model, 0, "pelvis")


def test_imu_check_names_the_body_when_the_site_is_off_the_root_link():
    model = mujoco.MjModel.from_xml_string(FREE_BODY_XML)
    with pytest.raises(ValueError, match=r"'imu_in_pelvis' is on body 'pelvis', not .* 'torso'"):
        physx_rollout.check_imu_on_root(model, 0, "torso")


def _plane_model(attributes: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom name="terrain" type="plane" size="0 0 0.01" {attributes}/></worldbody></mujoco>')


def test_ground_check_accepts_an_identity_plane():
    physx_plant.check_ground_identity(_plane_model(""), 0)


def test_ground_check_accepts_an_identity_plane_in_a_static_body():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="terrain"><geom type="plane" size="0 0 0.01"/></body></worldbody></mujoco>')
    physx_plant.check_ground_identity(model, 0)


@pytest.mark.parametrize("attributes", ['pos="0 0 0.1"', 'pos="1 0 0"', 'euler="10 0 0"'])
def test_ground_check_names_the_pose_of_a_moved_or_tilted_plane(attributes):
    with pytest.raises(ValueError, match=r"ground plane 'terrain' has world pose"):
        physx_plant.check_ground_identity(_plane_model(attributes), 0)


TWO_LINK_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="terrain" type="plane" size="0 0 0.01" friction="1 0.005 0.0001"/>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="floating_base_joint"/>
      <inertial pos="0 0 -0.02" mass="4" diaginertia="0.02 0.02 0.01"/>
      <geom name="pelvis_collision" type="sphere" size="0.07"/>
      <body name="thigh" pos="0 0.1 -0.05" quat="0.98 0.2 0 0">
        <joint name="hip_joint" axis="0 1 0" range="-1.5 2.0" armature="0.01"/>
        <inertial pos="0 0 -0.15" quat="0.9 0 0.43 0" mass="1.5" diaginertia="0.01 0.008 0.002"/>
        <geom name="thigh_collision" type="capsule" size="0.04 0.1" pos="0 0 -0.15"/>
        <body name="shin" pos="0 0 -0.3">
          <joint name="knee_joint" axis="0.2 0.9 0.1" range="0 2.5" armature="0.02"/>
          <inertial pos="0 0 -0.12" mass="1.2" diaginertia="0.008 0.007 0.001"/>
          <geom name="shin_collision" type="capsule" size="0.03 0.1" pos="0 0 -0.12" friction="0.6 0.005 0.0001"
                priority="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip" joint="hip_joint" kp="40" kv="2.5" forcerange="-88 88"/>
    <position name="knee" joint="knee_joint" kp="99" kv="6.3" forcerange="-139 139"/>
  </actuator>
</mujoco>
"""


def test_physx_plant_reproduces_the_mujoco_model(tmp_path):
    pytest.importorskip("ovphysx")
    pytest.importorskip("pxr")
    model = mujoco.MjModel.from_xml_string(TWO_LINK_XML)
    joints = ("knee_joint", "hip_joint")  # policy order differs from the tree order on purpose
    kp, kd = np.array([99.0, 40.0]), np.array([6.3, 2.5])
    usd = tmp_path / "plant.usda"
    authored = physx_plant.build_usd(model, joints, kp, kd, usd)
    assert authored["bodies"] == ["pelvis", "thigh", "shin"]
    plant = physx_plant.PhysxPlant(usd, joints)
    try:
        report = physx_plant.property_report(plant, model, joints, kp, kd)
        assert report["mass_max_abs_diff_kg"] < 1e-5
        assert report["com_max_abs_diff_m"] < 1e-6
        assert report["inertia_max_abs_diff"] < 1e-6
        assert report["kp_max_rel_diff"] < 1e-5 and report["kd_max_rel_diff"] < 1e-5
        assert report["armature_max_abs_diff"] < 1e-6
        assert report["max_force_max_abs_diff"] < 1e-3
        assert report["limit_max_abs_diff_rad"] < 1e-5

        data = mujoco.MjData(model)
        data.qpos[:7] = [0.3, -0.1, 1.2, 0.95, 0.05, -0.1, 0.28]
        data.qpos[:7][3:] /= np.linalg.norm(data.qpos[3:7])
        q = np.array([1.1, -0.4])  # policy order: knee, hip
        for name, value in zip(joints, q):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
        mujoco.mj_kinematics(model, data)
        plant.reset(data.qpos[:3], data.qpos[3:7], np.zeros(3), np.zeros(3), q, np.zeros(2))
        snapshot = plant.snapshot()
        np.testing.assert_allclose(snapshot.q, q, atol=1e-6)
        for link, name in enumerate(plant.link_names):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            np.testing.assert_allclose(snapshot.link_pos[link], data.xpos[body], atol=1e-5)
            assert abs(abs(np.dot(snapshot.link_quat[link], data.xquat[body])) - 1) < 1e-5
    finally:
        plant.close()
