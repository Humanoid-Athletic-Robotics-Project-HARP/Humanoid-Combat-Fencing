"""Pure logic behind `scripts/mtc_sim2sim.sh`: turning an MTC recording into scored policy steps.

No ROS here: the recording is the flat arrays scripts/mtc/bag_to_npz.py writes, built
synthetically below.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fencing.sim2sim import mtc
from fencing.sim2sim.quat import quat_apply, quat_mul, yaw_quat


def _yaw(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def test_tick_phase_finds_the_physics_sample_where_pd_effort_jumps():
    # 500 Hz samples, policy ticks every 10 samples starting 6 ms after activation: effort
    # drifts smoothly and jumps when a new target arrives.
    dt, activation = 0.002, 0.26
    t = activation - 0.1 + dt * np.arange(600)
    effort = np.zeros((t.size, 3)) + 0.01 * t[:, None]
    ticks = activation + 0.006 + 0.02 * np.arange(30)
    for tick in ticks:
        effort[t >= tick - 1e-9] += np.array([1.0, -2.0, 0.5])
    assert mtc.tick_phase(t, effort, activation, period=0.02) == pytest.approx(0.006, abs=1e-9)


def test_tick_phase_rejects_a_recording_without_samples_after_activation():
    t = np.arange(10) * 0.002
    with pytest.raises(ValueError, match="after activation"):
        mtc.tick_phase(t, np.zeros((10, 2)), activation_t=1.0, period=0.02)


def test_policy_step_times_start_at_activation_plus_phase():
    times = mtc.policy_step_times(activation_t=0.26, phase=0.006, num_steps=3, period=0.02)
    np.testing.assert_allclose(times, [0.266, 0.286, 0.306])


def _frames_the_controller_observes(num_ticks: int) -> list[int]:
    """Reference frame in MTC's observation at each policy tick (third_party/motion_tracking_controller).

    MotionOnnxPolicy::reset runs forward once and rewinds timeStep_, so frame 0 is stored. Each
    tick's observation reads the stored reference (MotionCommandTerm::getValue) before
    forward(observation) runs the ONNX with time_step = timeStep_++ and stores that frame.
    """
    stored, time_step, observed = 0, 0, []
    for _ in range(num_ticks):
        observed.append(stored)
        stored, time_step = time_step, time_step + 1
    return observed


def test_step_k_is_scored_from_the_tick_whose_observation_carries_reference_frame_k():
    activation, phase, steps = 0.26, 0.006, 8
    ticks = mtc.policy_step_times(activation, phase, steps + 3)
    observed = _frames_the_controller_observes(len(ticks))
    scored = mtc.scored_step_times(activation, phase, steps)
    assert len(scored) == steps
    for step, t in enumerate(scored):
        tick = int(np.argmin(np.abs(ticks - t)))
        assert ticks[tick] == pytest.approx(t)
        assert observed[tick] == step


def test_reference_is_placed_from_the_first_tick_whose_observation_carries_the_start_frame():
    # Frame 0 is observed at ticks 0 and 1, and MotionCommandTerm::reset places the motion at
    # activation (tick 0). A later start frame s is first observed at tick s + 1.
    activation, phase = 0.26, 0.006
    ticks = mtc.policy_step_times(activation, phase, 12)
    observed = _frames_the_controller_observes(len(ticks))
    for settle in (0, 1, 3, 7):
        t = mtc.reference_placement_time(activation, phase, settle)
        assert t == pytest.approx(ticks[observed.index(settle)])
    assert mtc.reference_placement_time(activation, phase, 0) == ticks[0]  # settle 0 unchanged, bit for bit


def test_interpolated_samples_reproduce_a_linear_signal_exactly():
    # Irregular stamps with a dropped sample (0.004) and a longer gap: a linear signal comes
    # back exactly at any time between samples, not just at the samples.
    t_src = np.array([0.0, 0.002, 0.006, 0.008, 0.014])
    rows = np.stack([1.0 + 3.0 * t_src, -2.0 * t_src], axis=1)
    t_query = np.array([0.001, 0.004, 0.0071, 0.011, 0.014])
    values, worst = mtc.interpolate_samples(t_src, rows, t_query)
    np.testing.assert_allclose(values, np.stack([1.0 + 3.0 * t_query, -2.0 * t_query], axis=1), atol=1e-12)
    assert worst == pytest.approx(0.003)  # 0.011 is 3 ms from 0.008 and 0.014, as nearest_samples reports


def test_a_dropped_sample_gives_the_same_state_whatever_the_rounding_of_the_tick_time():
    # 0.004 was dropped: 0.002 and 0.006 are equally near, and with the nearest sample a 1e-12 s
    # rounding difference picked one or the other (values 1.0 or 5.0). Interpolation is continuous.
    t_src = np.array([0.0, 0.002, 0.006, 0.008])
    rows = np.array([[0.0], [1.0], [5.0], [6.0]])
    results = [mtc.interpolate_samples(t_src, rows, np.array([0.004 + eps]))[0][0, 0] for eps in (-1e-12, 0.0, 1e-12)]
    np.testing.assert_allclose(results, 3.0, atol=1e-8)


def test_interpolation_does_not_hide_a_run_of_dropped_messages():
    # 30 ms without a sample: the scored state at 0.017 is 15 ms from any real sample, beyond
    # half a policy step, so the run is still INVALID however smooth the interpolation is.
    t_src = np.array([0.0, 0.002, 0.032, 0.034])
    _, worst = mtc.interpolate_samples(t_src, np.zeros((4, 2)), np.array([0.002, 0.017]))
    assert worst == pytest.approx(0.015)
    assert mtc.timing_problems({"max_state_time_mismatch_s": worst}) != []


def test_quaternion_interpolation_follows_a_rotation_and_ignores_the_sign_of_a_sample():
    t_src = np.arange(5) * 0.002
    quats = np.stack([_yaw(0.3 + 5.0 * t) for t in t_src])
    flipped = quats.copy()
    flipped[2] *= -1  # q and -q are the same rotation
    t_query = np.array([0.001, 0.004, 0.0047])
    expected = np.stack([_yaw(0.3 + 5.0 * t) for t in t_query])
    for samples in (quats, flipped):
        q, _ = mtc.interpolate_quats(t_src, samples, t_query)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
        # nlerp over a 10 mrad step: within 1e-8 of the constant-rate rotation; exact at a sample.
        np.testing.assert_allclose(q * np.sign(q[:, :1]), expected, atol=1e-8)


def test_nearest_samples_reports_the_largest_time_gap():
    t_src = np.array([0.0, 0.002, 0.004, 0.010])
    rows = np.arange(8.0).reshape(4, 2)
    picked, worst = mtc.nearest_samples(t_src, rows, np.array([0.0021, 0.007]))
    np.testing.assert_array_equal(picked, rows[[1, 2]])  # 0.007 is 3 ms from 0.004 and 0.010
    assert worst == pytest.approx(0.003)


def test_pelvis_pose_inverts_the_site_offset():
    pelvis_pos, pelvis_quat = np.array([1.0, -2.0, 0.8]), quat_mul(_yaw(0.7), np.array([0.99, 0.1, 0.0, 0.0]))
    pelvis_quat /= np.linalg.norm(pelvis_quat)
    rel_pos, rel_quat = np.array([0.05, 0.0, 0.4]), _yaw(-0.3)
    site_pos = pelvis_pos + quat_apply(pelvis_quat, rel_pos)
    site_quat = quat_mul(pelvis_quat, rel_quat)
    pos, quat = mtc.pelvis_pose_from_site(site_pos, site_quat, rel_pos, rel_quat)
    np.testing.assert_allclose(pos, pelvis_pos, atol=1e-12)
    np.testing.assert_allclose(quat * np.sign(quat[0]), pelvis_quat * np.sign(pelvis_quat[0]), atol=1e-12)


def test_reference_alignment_matches_motion_command_reset_in_the_ground_plane():
    # MotionCommand.cpp:37-43: the motion is moved so its start-frame anchor sits on the robot
    # anchor, rotating by yaw only. Heights are kept: both worlds share the floor at z = 0, and
    # MTC's z shift (from its estimator's anchor height) only biases what the policy observes.
    body_pos = np.array([[[0.0, 0.0, 0.8], [0.3, 0.0, 1.0]], [[0.1, 0.0, 0.8], [0.4, 0.0, 1.0]]])
    body_quat = np.tile(_yaw(0.2), (2, 2, 1))
    lin_vel = np.tile([1.0, 0.0, 0.0], (2, 2, 1))
    robot_anchor_pos, robot_anchor_quat = np.array([2.0, 1.0, 0.75]), _yaw(1.2)
    pos, quat, vel, _ = mtc.align_reference(body_pos, body_quat, lin_vel, lin_vel, robot_anchor_pos,
                                            robot_anchor_quat, anchor_index=0, start_frame=0)
    np.testing.assert_allclose(pos[0, 0, :2], robot_anchor_pos[:2], atol=1e-12)
    np.testing.assert_allclose(pos[:, :, 2], body_pos[:, :, 2], atol=1e-12)
    np.testing.assert_allclose(yaw_quat(quat[0, 0]), yaw_quat(robot_anchor_quat), atol=1e-12)
    # Rigid: distances between bodies and the z offset to the anchor are kept.
    np.testing.assert_allclose(np.linalg.norm(pos[1, 1] - pos[0, 0]), np.linalg.norm(body_pos[1, 1] - body_pos[0, 0]))
    np.testing.assert_allclose(pos[0, 1, 2] - pos[0, 0, 2], 0.2, atol=1e-12)
    np.testing.assert_allclose(vel[0, 0], quat_apply(_yaw(1.0), np.array([1.0, 0.0, 0.0])), atol=1e-12)


def test_base_velocity_from_poses_is_world_linear_and_body_angular():
    # Constant world velocity and a constant yaw rate: MuJoCo's free joint takes linear
    # velocity in the world frame and angular velocity in the body frame.
    t = np.arange(50) * 0.002
    pos = np.stack([0.5 * t, 0.0 * t, 0.8 + 0.0 * t], axis=1)
    quat = np.stack([_yaw(0.3 + 2.0 * ti) for ti in t])
    lin, ang = mtc.base_velocity(t, pos, quat)
    np.testing.assert_allclose(lin[5:-5], np.tile([0.5, 0.0, 0.0], (40, 1)), atol=1e-9)
    # Central differences at 500 Hz: O(dt^2) chord error, about 1e-6 here.
    np.testing.assert_allclose(ang[5:-5], np.tile([0.0, 0.0, 2.0], (40, 1)), atol=1e-5)


def test_repeated_sensor_stamps_collapse_to_the_last_sample():
    rows = np.array([[0.0, 1.0], [0.002, 2.0], [0.002, 3.0], [0.004, 4.0]])
    np.testing.assert_array_equal(mtc.unique_by_time(rows), [[0.0, 1.0], [0.002, 3.0], [0.004, 4.0]])


def test_real_time_factor_is_sim_time_over_wall_time():
    sim = np.array([0.0, 1.0, 2.0])
    wall = np.array([100.0, 102.0, 104.0])
    assert mtc.real_time_factor(sim, wall) == pytest.approx(0.5)


def test_a_recording_that_starts_after_the_first_policy_tick_is_refused():
    # The first steps (and the robot's state when the controller took over) would be missing.
    with pytest.raises(ValueError, match="first policy tick"):
        mtc.require_coverage(first_sample_t=0.396, first_tick_t=0.256)
    mtc.require_coverage(first_sample_t=0.05, first_tick_t=0.256)


def test_gap_to_sim2sim_is_mtc_minus_python_for_shared_metrics():
    mtc_nominal = {"joint_angle_error_rad": 0.0915, "body_pos_error_m": 0.034, "foot_slip_m_s": None}
    sim2sim = {"nominal": {"joint_angle_error_rad": 0.0852, "body_pos_error_m": 0.029, "foot_slip_m_s": 0.006},
               "motion_steps": 6574, "truncated": False}
    gap = mtc.sim2sim_gap(mtc_nominal, sim2sim)
    assert gap["metrics"]["joint_angle_error_rad"]["diff"] == pytest.approx(0.0063)
    assert "foot_slip_m_s" not in gap["metrics"]  # not measured on the MTC side
    assert gap["sim2sim_motion_steps"] == 6574


def test_record_seconds_cover_scored_span_hold_and_margin():
    # 10 s of a 131 s motion, 2 s hold: the controller must run 12 s past activation, plus margin.
    assert mtc.record_seconds(motion_frames=6574, seconds=10.0, hold_seconds=2.0) == 13
    assert mtc.record_seconds(motion_frames=150, seconds=None, hold_seconds=2.0) == 6


TWO_LINK_MJCF = """<mujoco>
  <compiler angle="radian" meshdir="../meshes"/>
  <default><default class="arm"><joint axis="0 1 0" range="-2 2"/></default></default>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom size="0.05"/>
      <body name="upper" pos="0.1 0 0" quat="0.9659 0 0.2588 0">
        <joint name="shoulder" class="arm"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
        <body name="lower" pos="0 0 -0.3">
          <joint name="elbow" axis="1 0 1" pos="0 0 0.02"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_start_qpos_puts_the_pelvis_pose_and_every_joint_angle_at_its_qpos_address():
    # The hold plugin writes this vector into MTC's simulator until walking_controller is active,
    # so the policy takes over a still robot in the reference's first frame, as mjlab resets it.
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_MJCF)
    pelvis_pos, pelvis_quat = np.array([0.2, -0.1, 0.75]), _yaw(0.3)
    qpos = mtc.start_qpos(model, {"elbow": -0.4, "shoulder": 0.7}, pelvis_pos, pelvis_quat)
    assert qpos.shape == (model.nq,)
    np.testing.assert_allclose(qpos[:7], np.concatenate([pelvis_pos, pelvis_quat]))
    assert qpos[model.joint("shoulder").qposadr[0]] == 0.7
    assert qpos[model.joint("elbow").qposadr[0]] == -0.4


def test_start_qpos_normalises_the_pelvis_quaternion():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_MJCF)
    qpos = mtc.start_qpos(model, {"elbow": 0.0, "shoulder": 0.0}, np.zeros(3), 2 * _yaw(0.3))
    np.testing.assert_allclose(qpos[3:7], _yaw(0.3))


def test_start_qpos_refuses_a_model_joint_without_a_start_angle():
    # A joint left at qpos0 would silently start straight: a pose the policy never saw.
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_MJCF)
    with pytest.raises(ValueError, match="elbow"):
        mtc.start_qpos(model, {"shoulder": 0.7}, np.zeros(3), _yaw(0.0))


def test_start_qpos_refuses_an_angle_for_a_joint_the_model_lacks():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_MJCF)
    with pytest.raises(KeyError, match="wrist"):
        mtc.start_qpos(model, {"shoulder": 0.7, "elbow": 0.0, "wrist": 0.1}, np.zeros(3), _yaw(0.0))


def _timing(mismatch):
    return {"max_state_time_mismatch_s": mismatch, "joint_states_max_gap_s": 0.05}


def test_a_scored_state_within_half_a_policy_step_of_its_tick_is_valid():
    # Raw /joint_states gaps do not matter by themselves (the controller runs in the physics
    # loop, not off the recording); only states sampled for a scored tick do.
    assert mtc.timing_problems(_timing(0.010)) == []


def test_a_scored_state_further_than_half_a_policy_step_from_its_tick_invalidates_the_run():
    # Beyond half a period the sample is nearer the neighbouring tick's state than its own.
    problems = mtc.timing_problems(_timing(0.022))
    assert len(problems) == 1 and "0.022" in problems[0]


def _joints(move_after_activation: float):
    t = np.arange(0.0, 2.0, 0.002)
    q = np.zeros((len(t), 3))
    q[t > 1.0, 1] = move_after_activation * np.minimum(t[t > 1.0] - 1.0, 0.1) * 10
    return t, q


def test_joints_that_move_after_activation_show_the_hold_released():
    t, q = _joints(0.05)
    mtc.require_release(t, q, activation_t=1.0)


def test_joints_frozen_after_activation_mean_the_hold_never_released():
    t, q = _joints(0.0)
    with pytest.raises(ValueError, match="released"):
        mtc.require_release(t, q, activation_t=1.0)


def test_a_bag_that_starts_late_is_a_retryable_recording_problem_not_a_policy_failure():
    # Under CPU load rosbag can subscribe after activation; mtc_sim2sim.sh records again on INVALID.
    with pytest.raises(mtc.RecordingIncomplete):
        mtc.require_coverage(first_sample_t=2.34, first_tick_t=0.79)


TORSO_MJCF = """<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom size="0.05"/>
      <body name="torso_link" pos="0 0 0.1">
        <joint name="waist" axis="0 1 0"/>
        <geom size="0.05"/>
        <site name="mid360" pos="0.05 0 0.4" quat="0.9659 0 0.2588 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _torso_recording(model, dropped_t: float) -> dict:
    """A pelvis moving at 0.5 m/s and a waist bending at 2 rad/s, sampled every 2 ms; the
    simulator's mid360 site pose is exact, and the /joint_states sample at dropped_t is missing."""
    import mujoco

    t = np.arange(20) * 0.002
    waist = 0.1 + 2.0 * t
    data = mujoco.MjData(model)
    mid360 = []
    for ti, q in zip(t, waist):
        data.qpos[:7] = [0.5 * ti, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
        data.qpos[7] = q
        mujoco.mj_kinematics(model, data)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[0])
        mid360.append([ti, *data.site_xpos[0], *quat])
    keep = ~np.isclose(t, dropped_t)
    return {"joint_names": np.array(["waist"]), "joint_t": t[keep], "joint_pos": waist[keep, None],
            "joint_vel": np.full((int(keep.sum()), 1), 2.0), "mid360": np.array(mid360)}


def test_replayed_state_at_a_dropped_sample_is_interpolated_and_the_pelvis_pose_stays_exact():
    # With the nearest sample, the waist angle came from 2 ms before or after (float rounding
    # decided), and so did the forward kinematics that turns the exact mid360 pose into the
    # pelvis pose. Interpolated, the state is the same for any rounding and matches the truth.
    import mujoco

    model = mujoco.MjModel.from_xml_string(TORSO_MJCF)
    dropped = 0.020
    state = SimpleNamespace(contract=SimpleNamespace(joint_names=("waist",), body_names=("pelvis",)),
                            body_ids=np.array([1]), qpos_adr=np.array([7]), qvel_adr=np.array([6]))
    replay = mtc.Replayer(model, state, _torso_recording(model, dropped), model.site_pos[0].copy(),
                          model.site_quat[0].copy())
    loaded = []
    for eps in (-1e-12, 0.0, 1e-12):
        assert replay.set_state(dropped + eps) == pytest.approx(0.002)  # still 2 ms from a real sample
        loaded.append((replay.data.qpos.copy(), replay.data.qvel.copy()))
    for qpos, qvel in loaded[1:]:
        np.testing.assert_allclose(qpos, loaded[0][0], atol=1e-9)
        np.testing.assert_allclose(qvel, loaded[0][1], atol=1e-9)
    qpos, qvel = loaded[0]
    assert qpos[7] == pytest.approx(0.1 + 2.0 * dropped, abs=1e-9)
    np.testing.assert_allclose(qpos[:3], [0.5 * dropped, 0.0, 0.8], atol=1e-9)
    np.testing.assert_allclose(qpos[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(qvel[:3], [0.5, 0.0, 0.0], atol=1e-6)


class _MarkerState:
    """MujocoState stand-in: the anchor sits at x = the time the replayed state was taken at."""

    def __init__(self, model, contract):
        self.contract = contract

    def anchor_pose(self, data):
        return np.array([data.t, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])


class _MarkerReplayer:
    def __init__(self, model, state, recording, *site_in_torso):
        self.qadr = 0
        self.data = SimpleNamespace(t=None, qpos=np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]))

    def set_state(self, t: float) -> float:
        self.data.t = float(t)
        return 0.0


class _RecordingScorer:
    """TrackingScorer stand-in that records which replayed state each step began and ended on."""

    instances: list = []

    def __init__(self, model, state, reference, motion_steps, hold_steps, step_dt, settle_steps, key_frames=None):
        self.reference, self.motion_steps, self.hold_steps = reference, motion_steps, hold_steps
        self.total_steps = motion_steps + hold_steps
        self.begun, self.ended = [], []
        _RecordingScorer.instances.append(self)

    def begin_step(self, step, robot_anchor_pos_w, robot_anchor_quat_w):
        self.begun.append(float(robot_anchor_pos_w[0]))

    def end_step(self, step, data):
        self.ended.append(data.t)
        return True

    def result(self):
        return {"motion_frames": self.motion_steps, "motion_steps": self.motion_steps, "hold_steps": self.hold_steps,
                "steps_run": len(self.ended), "terminated_at_step": None, "terminated_at_s": None,
                "termination_terms": [], "has_blade": False, "nominal": {}}


MARKER_ACTIVATION, MARKER_PHASE, MARKER_FRAMES, MARKER_HOLD_S = 0.26, 0.006, 6, 0.04


@pytest.fixture
def marker_scoring(monkeypatch, tmp_path):
    """score_recording on a synthetic recording, with the export, policy, MuJoCo state, replayer
    and TrackingScorer replaced by markers: what is left is the scheduling under test."""
    from fencing.sim2sim import observation, rollout

    scene = tmp_path / "scene.xml"
    scene.write_text(TWO_LINK_MJCF)
    n = MARKER_FRAMES
    reference = rollout.Reference(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1, 3)),
                                  np.tile([1.0, 0.0, 0.0, 0.0], (n, 1, 1)), np.zeros((n, 1, 3)), np.zeros((n, 1, 3)),
                                  source="test")
    contract = SimpleNamespace(joint_names=("shoulder",), body_names=("pelvis",), anchor_body_name="pelvis")
    policy = SimpleNamespace(contract=contract, reference=lambda frames: reference)
    monkeypatch.setattr(rollout, "load_export",
                        lambda d: rollout.Export(d, d / "policy.onnx", scene, {"motion_frames": n}))
    monkeypatch.setattr(rollout, "OnnxPolicy", lambda path: policy)
    monkeypatch.setattr(rollout, "export_key_frames", lambda export: None)
    monkeypatch.setattr(rollout, "TrackingScorer", _RecordingScorer)
    monkeypatch.setattr(observation, "MujocoState", _MarkerState)
    monkeypatch.setattr(mtc, "Replayer", _MarkerReplayer)
    monkeypatch.setattr(mtc, "site_in_body", lambda mjcf: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    monkeypatch.setattr(mtc, "tick_phase", lambda *args, **kwargs: MARKER_PHASE)
    _RecordingScorer.instances = []
    t = np.arange(0.0, MARKER_ACTIVATION + 1.0, 0.002)
    q = np.where(t > MARKER_ACTIVATION, t - MARKER_ACTIVATION, 0.0)[:, None]
    recording = {"activation_t": np.array([MARKER_ACTIVATION]), "joint_names": np.array(["shoulder"]),
                 "joint_t": t, "joint_pos": q, "joint_vel": np.zeros_like(q), "joint_effort": np.zeros_like(q),
                 "mid360": np.column_stack([t, np.zeros((len(t), 3)), np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))])}

    def score(settle_seconds: float = 0.0):
        result = mtc.score_recording(tmp_path, recording, tmp_path / "mtc.xml", None, MARKER_HOLD_S, settle_seconds)
        return result, _RecordingScorer.instances[-1]
    return score


def test_score_recording_scores_step_k_on_the_states_at_the_tick_carrying_frame_k_and_the_next(marker_scoring):
    # The lag test above checks scored_step_times; this checks score_recording uses them: step k
    # begins on the state at the tick whose observation carries frame k and ends on the next tick's.
    result, scorer = marker_scoring()
    ticks = mtc.policy_step_times(MARKER_ACTIVATION, MARKER_PHASE, 20)
    observed = _frames_the_controller_observes(len(ticks))
    assert result["steps_run"] == len(scorer.begun) == scorer.total_steps == MARKER_FRAMES + 2
    for step, (begin, end) in enumerate(zip(scorer.begun, scorer.ended)):
        tick = int(np.argmin(np.abs(ticks - begin)))
        assert begin == pytest.approx(ticks[tick], abs=1e-12)
        assert observed[tick] == step
        assert end == pytest.approx(ticks[tick + 1], abs=1e-12)


@pytest.mark.parametrize("settle", [0, 3])
def test_score_recording_places_the_reference_from_the_first_tick_carrying_the_settle_frame(marker_scoring, settle):
    # The reference's start frame `settle` must sit on the robot as it was when the controller
    # first observed that frame: activation (tick 0) for frame 0, tick settle + 1 after that.
    result, scorer = marker_scoring(settle * 0.02)
    assert result["settle_seconds"] == pytest.approx(settle * 0.02)
    ticks = mtc.policy_step_times(MARKER_ACTIVATION, MARKER_PHASE, 20)
    placement_tick = _frames_the_controller_observes(len(ticks)).index(settle)
    assert scorer.reference.body_pos_w[settle, 0, 0] == pytest.approx(ticks[placement_tick], abs=1e-12)
    if settle > 0:  # the frame the first scored step begins on (frame 0 is also observed at tick 1)
        assert scorer.begun[settle] == pytest.approx(ticks[placement_tick], abs=1e-12)
