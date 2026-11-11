"""Our ONNX export against what `motion_tracking_controller` (C++) actually reads.

Sources of each requirement (cited per check below):
- MTC = third_party/motion_tracking_controller @ 742924f (src/, include/, config/).
- LRC = legged_rl_controllers, the base classes MTC derives from (OnnxPolicy, RlController,
  ObservationManager). Its source is not public; we read the headers shipped in
  ros-jazzy-legged-rl-controllers 3.7.0 (qiayuanl/legged_buildfarm, noble-jazzy-amd64), and
  for OnnxPolicy::parseMetadata, whose body is only in liblegged_rl_controllers.so, the key
  string literals referenced from that function's disassembly.

No ONNX fixture is checked in (an export bundles the whole reference motion and needs mjlab
plus a checkpoint), so these tests read the newest exports under $HCF_DATA/eval and skip
when there are none: newest `*dance1*` (plain G1) and newest `*_tip` (saber).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fencing import paths
from fencing.sim2sim.contract import FULL_PRECISION_KEY

ort = pytest.importorskip("onnxruntime")

EXPORT_GLOBS = {"plain_g1": "*dance1*", "saber": "*_tip"}

# MTC MotionOnnxPolicy.cpp:21 and :75 (time_step); LRC OnnxPolicy.h:59 (obs).
REQUIRED_INPUTS = ("obs", "time_step")
# LRC OnnxPolicy.h:60,63,64 (actions); MTC MotionOnnxPolicy.cpp:24-30 and :76-79 (the rest).
REQUIRED_OUTPUTS = ("actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")

# LRC OnnxPolicy::parseMetadata reads these with getMetadataStr (OnnxPolicy.h:68-74), which
# throws when a key is missing. Float lists go through getMetadataVector -> parseCsv<double>
# (OnnxPolicy.h:24-40, 75-80): split on ',' and std::stod each item.
LRC_FLOAT_KEYS = ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale")
LRC_STRING_LIST_KEYS = ("joint_names", "command_names", "observation_names")
# MTC MotionOnnxPolicy.cpp:46 (anchor_body_name, plain string) and :48 (body_names, CSV).
MTC_KEYS = ("anchor_body_name", "body_names")
REQUIRED_KEYS = LRC_STRING_LIST_KEYS + LRC_FLOAT_KEYS + MTC_KEYS
# Read inside a try/catch ("observation_history_lengths not usable (...), set to default"),
# so absence is fine: every term keeps historyLength_ = 1 (ObservationManager.h:59).
LRC_OPTIONAL_HISTORY_KEY = "observation_history_lengths"

# The G1 joint list the controller's robot config names (MTC config/g1/controllers.yaml:24-33).
# Joints are matched by name, not position (ObservationManager.h:146-150 getJointIndex).
G1_CONTROLLER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def observation_term_size(name: str, num_joints: int, num_actions: int, num_bodies: int) -> int:
    """Size of each observation term the controller can build, or KeyError if it cannot."""
    sizes = {
        # MTC MotionTrackingController.cpp:69-76 -> MotionObservation.h:25,34,43,52.
        "motion_anchor_pos_b": 3, "motion_ref_pos_b": 3,
        "motion_anchor_ori_b": 6, "motion_ref_ori_b": 6,
        "robot_body_pos": 3 * num_bodies, "robot_body_ori": 6 * num_bodies,
        # LRC RlController::parserObservation (names from its literals) -> ObservationManager.h.
        "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,  # :108, :117, :126
        "joint_pos": num_joints, "joint_vel": num_joints,  # :143 (minus default pos, :164)
        "actions": num_actions,  # :179 LastActionObservationTerm
        "phase": 4,  # :190
        # :219 CommandObservationTerm = the command manager; the only command here is MTC's
        # "motion" term, 2 * joints (MotionCommand.h:33, values MotionCommand.cpp:10).
        "command": 2 * num_joints,
    }
    return sizes[name]


def _newest_export(pattern: str) -> Path | None:
    run_dirs = sorted(d for d in paths.EVAL.glob(pattern) if d.is_dir())
    for run_dir in reversed(run_dirs):
        models = sorted(run_dir.glob("model_*/policy.onnx"),
                        key=lambda p: int(p.parent.name.removeprefix("model_")))
        if models:
            return models[-1]
    return None


class Export:
    def __init__(self, onnx_path: Path):
        self.path = onnx_path
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.inputs = {i.name: list(i.shape) for i in self.session.get_inputs()}
        self.outputs = {o.name: list(o.shape) for o in self.session.get_outputs()}
        export_json = onnx_path.parent / "export.json"
        self.summary = json.loads(export_json.read_text()) if export_json.is_file() else {}

    def csv(self, key: str) -> list[str]:
        # parseCsv: std::getline on ','. No trimming, empty items kept (stod("") would throw).
        return self.meta[key].split(",")

    def stod_list(self, key: str) -> np.ndarray:
        return np.array([_stod(item) for item in self.csv(key)])


def _stod(text: str) -> float:
    """std::stod: leading whitespace, decimal/exponent, inf/nan. Raises like it on garbage."""
    if text.strip() == "":
        raise ValueError("std::stod on an empty item throws std::invalid_argument")
    return float(text)


@pytest.fixture(scope="module", params=sorted(EXPORT_GLOBS))
def export(request) -> Export:
    onnx_path = _newest_export(EXPORT_GLOBS[request.param])
    if onnx_path is None:
        pytest.skip(f"no {request.param} export under {paths.EVAL} (run `hcf train export`)")
    return Export(onnx_path)


def test_export_has_every_input_and_output_the_controller_looks_up_by_name(export):
    assert set(REQUIRED_INPUTS) <= set(export.inputs), export.inputs
    assert set(REQUIRED_OUTPUTS) <= set(export.outputs), export.outputs


def test_every_metadata_key_the_controller_reads_is_present(export):
    missing = [k for k in REQUIRED_KEYS if k not in export.meta]
    assert not missing, f"{export.path}: controller would throw on {missing}"


def test_float_lists_parse_with_stod_and_have_one_value_per_joint(export):
    joints = export.csv("joint_names")
    for key in LRC_FLOAT_KEYS:
        values = export.stod_list(key)
        # Applied elementwise against the joint/action vectors in RlController; a scalar
        # action_scale (which mjlab can write) would not broadcast in Eigen.
        assert values.shape == (len(joints),), f"{key}: {values.size} values, {len(joints)} joints"
        assert np.isfinite(values).all(), key


def test_joint_names_are_the_g1_joints_the_controller_drives(export):
    joints = export.csv("joint_names")
    assert all(name == name.strip() and name for name in joints), "parseCsv does not trim"
    assert len(set(joints)) == len(joints)
    assert set(joints) == set(G1_CONTROLLER_JOINTS)


def test_tensor_shapes_pass_motion_policy_check_input_output(export):
    joints, bodies = export.csv("joint_names"), export.csv("body_names")

    def rows_cols(shape):
        # MotionOnnxPolicy.cpp:63-72: rank 2 is (s0, s1), rank 3 is (s1, s2), else throw.
        assert len(shape) in (2, 3), shape
        return tuple(shape) if len(shape) == 2 else tuple(shape[1:])

    assert rows_cols(export.outputs["body_pos_w"]) == (len(bodies), 3)  # :81-88
    assert rows_cols(export.outputs["body_quat_w"]) == (len(bodies), 4)  # :85
    assert rows_cols(export.outputs["joint_pos"])[1] == len(joints)  # :89
    assert rows_cols(export.outputs["joint_vel"])[1] == len(joints)  # :89
    # LRC OnnxPolicy.h:59-60 read shape[1] of obs and actions; MotionOnnxPolicy.cpp:19-21
    # feeds time_step as a (1, 1) float tensor.
    assert export.inputs["obs"][0] == 1 and len(export.inputs["obs"]) == 2
    assert export.outputs["actions"] == [1, len(joints)]
    assert export.inputs["time_step"] == [1, 1]


def test_anchor_body_is_one_of_the_tracked_bodies(export):
    # MotionCommand.cpp:26-35 throws "Anchor body ... not found in body names" otherwise.
    assert export.meta["anchor_body_name"] in export.csv("body_names")


def test_tracking_policy_uses_only_the_motion_command(export):
    # The "command" observation is the whole command manager (ObservationManager.h:222); for
    # it to equal mjlab's reference joint pos/vel, "motion" (MotionTrackingController.cpp:57)
    # must be the only command term.
    assert export.csv("command_names") == ["motion"]


def test_every_observation_term_is_one_the_controller_can_build(export):
    # Unknown names make parserObservation return false ("Unknown observation name: %s").
    joints, bodies = export.csv("joint_names"), export.csv("body_names")
    unknown = []
    for name in export.csv("observation_names"):
        try:
            observation_term_size(name, len(joints), len(joints), len(bodies))
        except KeyError:
            unknown.append(name)
    assert not unknown


def test_observation_the_controller_assembles_has_the_obs_input_width(export):
    # OnnxPolicy::run checks "OnnxPolicy: input obs has N elements, expected M". Each term
    # contributes size * historyLength (ObservationManager.h:46, :96-101), history 1 here.
    joints, bodies = export.csv("joint_names"), export.csv("body_names")
    total = sum(observation_term_size(n, len(joints), len(joints), len(bodies))
                for n in export.csv("observation_names"))
    assert total == export.inputs["obs"][1]


def test_mjlab_history_scale_and_clip_are_identity_because_the_controller_ignores_them(export):
    # The controller reads none of mjlab's observation_terms_{scale,clip,history_length}; it
    # only knows observation_history_lengths (absent -> 1 per term). Our observations match
    # only while mjlab's are no-ops: history 0 (= one frame), scale 1, clip +-inf.
    assert LRC_OPTIONAL_HISTORY_KEY not in export.meta, (
        "if present, setHistoryLengths rejects 0 (ObservationManager.h:74-77)")
    num_terms = len(export.csv("observation_names"))
    history = export.meta.get("observation_terms_history_length")
    if history is not None:
        assert [float(h) for h in history.split(",")] == [0.0] * num_terms
    scale = export.meta.get("observation_terms_scale")
    if scale is not None:
        assert [float(s) for s in scale.split(",")] == [1.0] * num_terms
    clip = export.meta.get("observation_terms_clip")
    if clip is not None:
        assert clip.split(",") == ["-inf;inf"] * num_terms


def _rounded_and_exact(export: Export):
    # The C++ parses the 3-decimal CSV strings and never reads FULL_PRECISION_KEY, so it runs
    # with the rounded action scale and gains, not the ones the policy trained with.
    if FULL_PRECISION_KEY not in export.meta:
        pytest.skip(f"{FULL_PRECISION_KEY} absent: nothing to compare the rounded values to")
    precise = json.loads(export.meta[FULL_PRECISION_KEY])
    for key in LRC_FLOAT_KEYS:
        if key in precise:
            yield key, export.stod_list(key), np.asarray(precise[key], dtype=np.float64)


def test_controller_reads_gains_and_action_scale_rounded_to_3_decimals(export):
    for key, rounded, exact in _rounded_and_exact(export):
        assert np.abs(rounded - exact).max() <= 5e-4 + 1e-9, key


@pytest.mark.xfail(strict=True, reason=(
    "real mismatch: mjlab rounds to 3 decimals and the C++ ignores hcf_full_precision, so the "
    "G1 wrist pitch/yaw action scale 0.0745 runs as 0.075 (+0.67%; the 0.4386 joints +0.10%)"))
def test_controller_action_scale_and_gains_are_within_0_2_percent_of_the_trained_values(export):
    for key, rounded, exact in _rounded_and_exact(export):
        nonzero = np.abs(exact) > 1e-6
        assert (np.abs(rounded - exact)[nonzero] / np.abs(exact[nonzero])).max() < 2e-3, key


def test_time_step_past_the_motion_end_returns_the_last_frame(export):
    # MotionOnnxPolicy.cpp:20 increments time_step every policy step with no upper bound.
    run = lambda t: export.session.run(None, {  # noqa: E731
        "obs": np.zeros((1, export.inputs["obs"][1]), np.float32),
        "time_step": np.array([[t]], np.float32),
    })
    names = [o.name for o in export.session.get_outputs()]
    late, later = run(1e6), run(1e6 + 1)
    for name, a, b in zip(names, late, later):
        if name != "actions":
            np.testing.assert_array_equal(a, b, err_msg=name)
    frames = export.summary.get("motion_frames")
    if frames:
        last = run(frames - 1)
        for name, a, b in zip(names, last, late):
            if name != "actions":
                np.testing.assert_array_equal(a, b, err_msg=name)


def test_body_quaternions_are_unit_length_in_wxyz_rows(export):
    # MotionOnnxPolicy.cpp:36-37 reads w from column 0 and xyz from columns 1-3; a unit norm
    # does not prove the order, but sim2sim (same wxyz reading) tracking does.
    out = export.session.run(["body_quat_w"], {
        "obs": np.zeros((1, export.inputs["obs"][1]), np.float32),
        "time_step": np.zeros((1, 1), np.float32),
    })[0].reshape(-1, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_tracked_bodies_are_frames_of_the_controllers_g1_urdf(export):
    # MotionCommand.cpp:15-25 looks every body up as a pinocchio frame of the robot_description
    # (unitree_description g1 robot.xacro, mujoco.launch.py:67-77). Only checkable where that
    # package is unpacked; the rootless ROS install in $HCF_DATA/mtc provides it.
    urdf = paths.DATA / "mtc" / "g1_mujoco.urdf"
    if not urdf.is_file():
        pytest.skip(f"{urdf} absent (xacro output of unitree_description g1 robot.xacro)")
    import xml.etree.ElementTree as ET

    links = {link.get("name") for link in ET.parse(urdf).getroot().iter("link")}
    missing = [b for b in export.csv("body_names") if b not in links]
    assert not missing
    assert export.meta["anchor_body_name"] in links


def test_rounded_numbers_survive_the_csv_round_trip():
    # Guard for _stod: the formats mjlab writes ("0.439", "-inf") parse as the C++ would.
    assert _stod("0.439") == 0.439 and _stod("-inf") == -math.inf
    with pytest.raises(ValueError):
        _stod("")
