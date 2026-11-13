"""Start from standby (docs/mtc.md "Start from standby"): the logic behind `mtc_sim2sim.sh --standby`.

The robot stands under upstream's standby_controller, then the operator's switch starts the
policy. These tests cover what the host decides from a recording: where the simulator holds the
robot before standby takes it, whether the robot was standing still at the switch, when the
policy converged onto the reference, and how hard the transition drove the joints.
"""

from __future__ import annotations

import numpy as np
import pytest

from fencing.sim2sim import mtc, mtc_standby

FOOT_MJCF = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
      <body name="shin" pos="0 0 -0.4">
        <joint name="knee" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
        <body name="foot" pos="0 0 -0.4">
          <geom type="capsule" fromto="-0.05 0 0 0.1 0 0" size="0.01"/>
          <geom type="box" pos="0 0 0.02" size="0.04 0.03 0.005"/>
          <geom type="sphere" pos="0 0 -0.2" size="0.01" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_standing_qpos_puts_the_lowest_colliding_point_on_the_floor_and_keeps_the_default_pose():
    # The hold keeps the robot in its model default (unitree_description's pose) until standby's
    # first command: standing on the floor, not floating or sunk in it. The visual-only sphere
    # 0.2 m below the foot must not count.
    import mujoco

    model = mujoco.MjModel.from_xml_string(FOOT_MJCF)
    qpos = mtc_standby.standing_qpos(model)
    np.testing.assert_allclose(qpos[3:], model.qpos0[3:])
    np.testing.assert_allclose(qpos[:2], model.qpos0[:2])
    # Foot capsule axis at 0.9 - 0.8 = 0.1 m with qpos0; radius 0.01 -> lowest point 0.09 m.
    assert qpos[2] == pytest.approx(0.9 - 0.09)
    data = mujoco.MjData(model)
    for offset, expected_contacts in ((-1e-4, True), (1e-4, False)):
        data.qpos[:] = qpos
        data.qpos[2] += offset
        mujoco.mj_forward(model, data)
        assert (data.ncon > 0) == expected_contacts
    data.qpos[2] = qpos[2] - 1e-4
    mujoco.mj_forward(model, data)
    assert min(c.dist for c in data.contact[:data.ncon]) == pytest.approx(-1e-4, abs=1e-9)


def test_standing_qpos_refuses_a_colliding_geom_it_cannot_bound_exactly():
    import mujoco

    xml = FOOT_MJCF.replace('<geom type="box" pos="0 0 0.02" size="0.04 0.03 0.005"/>',
                            '<geom type="ellipsoid" size="0.04 0.03 0.02"/>')
    with pytest.raises(ValueError, match="ellipsoid"):
        mtc_standby.standing_qpos(mujoco.MjModel.from_xml_string(xml))


def _standing_recording(switch_t=10.0, standby_on=0.2, speed=0.0, height_drop=0.0, joint_speed=0.0,
                        walking_before=False):
    t = np.arange(0.0, switch_t + 2.0, 0.002)
    pos = np.zeros((len(t), 3))
    pos[:, 0] = speed * t
    pos[:, 2] = 0.75 - height_drop * t
    quat = np.tile([1.0, 0, 0, 0], (len(t), 1))
    lin = np.gradient(pos, t, axis=0)
    activity = [[0.0, 0, 0], [standby_on, 1, 0], [switch_t, 0, 1]]
    if walking_before:
        activity = [[0.0, 0, 0], [standby_on, 0, 1]]
    joint_vel = np.full((len(t), 3), joint_speed)
    return dict(activity=np.array(activity, float), base_t=t, base_pos=pos, base_quat=quat, base_lin_vel=lin,
                joint_t=t, joint_vel=joint_vel, switch_t=switch_t)


def test_a_robot_standing_still_under_standby_at_the_switch_counts():
    report = mtc_standby.standby_state_report(**_standing_recording())
    assert report["still"], report["reasons"]
    assert report["standby_active_s"] == pytest.approx(9.8)
    assert report["pelvis_height_m"] == pytest.approx(0.75)


def test_motion_after_the_switch_does_not_leak_into_the_pre_switch_check():
    # Velocities are central differences: the last sample before the switch also sees the first
    # one after it, when the policy already moves the robot.
    recording = _standing_recording()
    after = recording["base_t"] >= recording["switch_t"]
    recording["base_pos"][after, 1] += 0.01 * (recording["base_t"][after] - recording["switch_t"] + 0.002) / 0.002
    recording["base_lin_vel"] = np.gradient(recording["base_pos"], recording["base_t"], axis=0)
    report = mtc_standby.standby_state_report(**recording)
    assert report["still"], report["reasons"]


@pytest.mark.parametrize("change, reason", [
    (dict(speed=0.05), "pelvis speed"),
    (dict(height_drop=0.01), "pelvis height"),
    (dict(joint_speed=0.2), "joint speed"),
    (dict(standby_on=8.5), "standby_controller active"),
    (dict(walking_before=True), "standby_controller active"),
])
def test_a_robot_not_standing_still_under_standby_at_the_switch_does_not_count(change, reason):
    # docs/mtc.md: unheld starts once handed the policy a falling robot. A run only counts when
    # the recording shows the robot standing still, under standby long enough to finish its ramp.
    report = mtc_standby.standby_state_report(**_standing_recording(**change))
    assert not report["still"]
    assert any(reason in r for r in report["reasons"]), report["reasons"]


def test_step_joint_error_pairs_the_post_step_state_with_the_frame_of_the_step():
    # Same pairing as TrackingScorer through score_recording: step k is scored on the state at
    # the tick after the one carrying frame k (times[k + 1]) against frame k, held at the last.
    times = np.array([0.02, 0.04, 0.06, 0.08])
    joint_t = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
    joint_pos = np.array([[9.0, 9.0], [9.0, 9.0], [0.1, -0.3], [0.2, 0.0], [0.4, 0.4]])
    reference = np.array([[0.0, 0.0], [0.2, 0.2]])
    errors = mtc_standby.step_joint_errors(joint_t, joint_pos, reference, times, num_steps=3)
    np.testing.assert_allclose(errors, [0.2, 0.1, 0.2])


def test_convergence_is_the_first_step_from_which_the_error_stays_under_the_line_for_the_window():
    errors = np.array([0.4, 0.3, 0.14, 0.2, 0.14, 0.13, 0.12, 0.11, 0.3])
    assert mtc_standby.convergence_step(errors, threshold=0.15, window_steps=3) == 4
    assert mtc_standby.convergence_step(errors, threshold=0.15, window_steps=5) is None
    assert mtc_standby.convergence_step(np.array([0.1, 0.1]), threshold=0.15, window_steps=3) is None


def test_default_convergence_line_and_window_are_0_15_rad_for_half_a_second():
    assert mtc_standby.CONVERGED_JOINT_ERROR_RAD == 0.15
    assert mtc_standby.CONVERGED_WINDOW_S == 0.5


def test_peak_fraction_of_limit_per_joint_inside_the_window_only():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    values = np.array([[100.0, 0.0], [-10.0, 3.0], [5.0, -5.0], [1.0, 1.0], [50.0, 50.0]])
    limits = np.array([20.0, 5.0])
    report = mtc_standby.peak_fraction(t, values, limits, ["a", "b"], start=0.5, end=1.5)
    assert report["per_joint"] == {"a": 0.5, "b": 1.0}
    assert report["max"] == 1.0 and report["joint"] == "b"
    assert report["joints_at_limit"] == ["b"]


def test_velocity_limits_are_read_from_the_urdf_by_joint_name():
    urdf = """<robot name="r">
      <joint name="knee" type="revolute"><limit effort="139" lower="-0.1" upper="2.9" velocity="20"/></joint>
      <joint name="wrist" type="revolute"><limit effort="5" lower="-1" upper="1" velocity="22"/></joint>
      <joint name="saber_weld" type="fixed"/>
    </robot>"""
    np.testing.assert_allclose(mtc_standby.urdf_limits(urdf, ["wrist", "knee"], "velocity"), [22.0, 20.0])
    np.testing.assert_allclose(mtc_standby.urdf_limits(urdf, ["knee"], "effort"), [139.0])
    with pytest.raises(KeyError, match="elbow"):
        mtc_standby.urdf_limits(urdf, ["elbow"], "velocity")


def test_standby_runs_write_their_own_json_and_recordings_never_the_held_ones():
    # The held results (mtc.json, $HCF_DATA/mtc/runs/) are the documented Goal 1 numbers; a
    # standby run must not delete or overwrite them, whatever else it does.
    assert mtc.run_outputs(standby=False) == ("mtc.json", "runs")
    assert mtc.run_outputs(standby=True) == ("mtc_standby.json", "runs_standby")


def test_plan_prints_the_output_json_and_runs_directory_for_the_runner(tmp_path, capsys, monkeypatch):
    export = tmp_path / "model_2999"
    export.mkdir()
    (export / "export.json").write_text('{"motion_frames": 150}')
    monkeypatch.setattr("fencing.sim2sim.cli.resolve_export_dir", lambda target, checkpoint: export)
    assert mtc.main([str(export), "--plan", "--standby"]) == 0
    assert capsys.readouterr().out.split() == [str(export), "6", "mtc_standby.json", "runs_standby"]
    assert mtc.main([str(export), "--plan"]) == 0
    assert capsys.readouterr().out.split() == [str(export), "6", "mtc.json", "runs"]
