"""The sim2sim observation builder reproduces mjlab's actor observation.

Fixture: tests/data/mjlab_obs_g1_tracking.npz, written by `fencing train record-obs` from
a real mjlab rollout (observation noise and encoder bias off). For each recorded step we
load qpos/qvel into the plain upstream G1 MJCF, run mj_forward, rebuild the observation
from that state + the recorded reference frame + last action, and compare per term.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fencing import paths
from fencing.sim2sim.contract import FULL_PRECISION_KEY, full_precision_metadata, parse_metadata
from fencing.sim2sim.observation import (
    MujocoState, build_observation, relative_body_poses, tracking_terminated,
)
from fencing.sim2sim.quat import quat_apply, quat_error_magnitude, quat_mul, yaw_quat

FIXTURE = Path(__file__).parent / "data" / "mjlab_obs_g1_tracking.npz"
TOLERANCE = 1e-4


def _csv(values, decimals=3):
    return ",".join(f"{v:.{decimals}f}" if not isinstance(v, str) else v for v in values)


def _fixture_metadata(fixture) -> dict[str, str]:
    """The metadata mjlab's exporter would attach, from what the fixture recorded."""
    names = [str(n) for n in fixture["term_names"]]
    meta = {
        "joint_names": ",".join(str(n) for n in fixture["policy_joint_names"]),
        "default_joint_pos": _csv(fixture["default_joint_pos"]),
        "action_scale": _csv([0.5] * len(fixture["policy_joint_names"])),
        "observation_names": ",".join(names),
        "observation_terms_scale": _csv([1.0] * len(names)),
        "observation_terms_clip": ",".join("-inf;inf" for _ in names),
        "observation_terms_history_length": ",".join("0" for _ in names),
        "anchor_body_name": str(fixture["anchor_body_name"]),
        "body_names": ",".join(str(n) for n in fixture["body_names"]),
    }
    meta[FULL_PRECISION_KEY] = full_precision_metadata(
        {"default_joint_pos": fixture["default_joint_pos"].tolist()})
    return meta


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.is_file():
        pytest.skip(f"fixture missing: {FIXTURE} (run `fencing train record-obs`)")
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(paths.G1_MJCF))


def test_fixture_matches_upstream_g1_joint_layout(fixture, model):
    mj_names = [str(n).removeprefix("robot/") for n in fixture["mj_joint_names"]]
    assert mj_names == [model.joint(i).name for i in range(model.njnt)]
    assert fixture["qpos"].shape[1] == model.nq and fixture["qvel"].shape[1] == model.nv


def test_rebuilt_observation_matches_mjlab(fixture, model):
    contract = parse_metadata(_fixture_metadata(fixture))
    assert contract.full_precision
    state = MujocoState(model, contract)
    data = mujoco.MjData(model)
    dims = [int(d) for d in fixture["term_dims"]]
    bounds = np.cumsum([0] + dims)
    worst = 0.0
    for i in range(len(fixture["step"])):
        data.qpos[:] = fixture["qpos"][i]
        data.qvel[:] = fixture["qvel"][i]
        mujoco.mj_forward(model, data)
        inputs = state.inputs(
            data,
            fixture["ref_joint_pos"][i], fixture["ref_joint_vel"][i],
            fixture["ref_anchor_pos_w"][i], fixture["ref_anchor_quat_w"][i],
            fixture["last_action"][i],
        )
        obs = build_observation(inputs, contract)
        expected = fixture["obs"][i]
        assert obs.shape == expected.shape
        for name, start, stop in zip(contract.observation_names, bounds[:-1], bounds[1:]):
            error = float(np.abs(obs[start:stop] - expected[start:stop]).max())
            assert error < TOLERANCE, f"step {fixture['step'][i]} term {name}: max |diff| {error:.2e}"
        worst = max(worst, float(np.abs(obs - expected).max()))
    print(f"max abs obs error vs mjlab fixture: {worst:.3e}")


def test_imu_lin_vel_matches_recorded_sensor(fixture, model):
    contract = parse_metadata(_fixture_metadata(fixture))
    state = MujocoState(model, contract)
    data = mujoco.MjData(model)
    data.qpos[:] = fixture["qpos"][0]
    data.qvel[:] = fixture["qvel"][0]
    mujoco.mj_forward(model, data)
    inputs = state.inputs(data, fixture["ref_joint_pos"][0], fixture["ref_joint_vel"][0],
                          fixture["ref_anchor_pos_w"][0], fixture["ref_anchor_quat_w"][0],
                          fixture["last_action"][0])
    np.testing.assert_allclose(inputs.imu_lin_vel, fixture["imu_lin_vel"][0], atol=TOLERANCE)


def _random_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_relative_body_poses_keep_anchor_on_robot_xy_and_reference_height():
    rng = np.random.default_rng(0)
    ref_anchor_pos = np.array([1.0, 2.0, 0.8])
    ref_anchor_quat = _random_quats(rng, 1)[0]
    robot_anchor_pos = np.array([1.3, 1.7, 0.6])
    robot_anchor_quat = _random_quats(rng, 1)[0]
    body_pos = np.stack([ref_anchor_pos, ref_anchor_pos + [0.1, 0.0, -0.4]])
    body_quat = np.stack([ref_anchor_quat, ref_anchor_quat])
    pos, quat = relative_body_poses(ref_anchor_pos, ref_anchor_quat, robot_anchor_pos,
                                    robot_anchor_quat, body_pos, body_quat)
    np.testing.assert_allclose(pos[0], [1.3, 1.7, 0.8], atol=1e-12)
    # Offsets keep their length and their height difference (the shift is a pure yaw).
    np.testing.assert_allclose(np.linalg.norm(pos[1] - pos[0]), np.linalg.norm([0.1, 0.0, -0.4]))
    np.testing.assert_allclose(pos[1, 2] - pos[0, 2], -0.4, atol=1e-12)
    delta = yaw_quat(quat_mul(robot_anchor_quat, [ref_anchor_quat[0], *-ref_anchor_quat[1:]]))
    np.testing.assert_allclose(quat[0], quat_mul(delta, ref_anchor_quat), atol=1e-12)


def test_quat_error_magnitude_is_the_rotation_angle():
    q = np.array([math.cos(0.3), 0.0, 0.0, math.sin(0.3)])
    assert quat_error_magnitude(q, [1.0, 0, 0, 0]) == pytest.approx(0.6)
    assert quat_error_magnitude(q, -q) == pytest.approx(0.0, abs=1e-12)  # double cover
    np.testing.assert_allclose(quat_apply(q, [1.0, 0, 0]), [math.cos(0.6), math.sin(0.6), 0], atol=1e-12)


def test_tracking_terminations_fire_on_height_and_tilt():
    upright = np.array([1.0, 0, 0, 0])
    feet = np.zeros((4, 3))
    ok = tracking_terminated([0, 0, 0.8], upright, [0, 0, 0.75], upright, feet, feet)
    assert ok == {"anchor_pos": False, "anchor_ori": False, "ee_body_pos": False}
    fallen = tracking_terminated([0, 0, 0.8], upright, [0, 0, 0.4], upright, feet, feet)
    assert fallen["anchor_pos"]
    tilted = np.array([math.cos(0.6), math.sin(0.6), 0, 0])  # 1.2 rad roll: 1 - cos = 0.64
    assert not tracking_terminated([0, 0, 0.8], upright, [0, 0, 0.8], tilted, feet, feet)["anchor_ori"]
    tilted = np.array([math.cos(0.8), math.sin(0.8), 0, 0])  # 1.6 rad: 1 - cos = 1.03
    assert tracking_terminated([0, 0, 0.8], upright, [0, 0, 0.8], tilted, feet, feet)["anchor_ori"]
    lifted = feet.copy()
    lifted[0, 2] = 0.3
    assert tracking_terminated([0, 0, 0.8], upright, [0, 0, 0.8], upright, feet, lifted)["ee_body_pos"]


def test_parse_metadata_reads_mjlab_csv_strings():
    meta = {
        "joint_names": "a,b",
        "default_joint_pos": "0.100,-0.200",
        "action_scale": "0.439,0.548",
        "joint_stiffness": "14.251,40.179",
        "joint_damping": "0.907,2.558",
        "observation_names": "command,actions",
        "observation_terms_scale": "1.000,0.500;0.250",
        "observation_terms_clip": "-inf;inf,-1.000;1.000",
        "observation_terms_history_length": "0,0",
        "anchor_body_name": "torso_link",
        "body_names": "pelvis,torso_link",
    }
    contract = parse_metadata(meta)
    assert contract.joint_names == ("a", "b")
    np.testing.assert_allclose(contract.action_scale, [0.439, 0.548])
    assert contract.observation_scales == (1.0, (0.5, 0.25))
    assert contract.observation_clips == ((-math.inf, math.inf), (-1.0, 1.0))
    assert not contract.full_precision
    meta[FULL_PRECISION_KEY] = full_precision_metadata({"action_scale": [0.43859, 0.54795]})
    np.testing.assert_allclose(parse_metadata(meta).action_scale, [0.43859, 0.54795])


def test_parse_metadata_rejects_missing_keys():
    with pytest.raises(ValueError, match="lacks"):
        parse_metadata({"joint_names": "a"})


def test_body_velocity_is_the_body_origin_not_the_centre_of_mass():
    """mjlab's body_link_lin_vel_w is the body frame origin's velocity. A link spinning about
    its own origin with an offset centre of mass must read zero, not omega x com_offset."""
    from fencing.sim2sim.observation import body_origin_lin_vel_w

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="link" pos="0 0 1">
      <joint name="spin" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05" pos="0.3 0 0" mass="1"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 2.0
    mujoco.mj_forward(model, data)
    body = model.body("link").id
    np.testing.assert_allclose(body_origin_lin_vel_w(model, data, [body]), np.zeros((1, 3)), atol=1e-12)
