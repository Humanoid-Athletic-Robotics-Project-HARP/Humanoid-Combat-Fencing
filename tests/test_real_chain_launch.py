"""scripts/mtc/real_chain.launch.py without ROS (docs/mtc.md, "Entry from standby").

The launch file needs ROS 2 packages this venv lacks, so they are stubbed: the tests check what
the file decides (the argument it declares, the controller config and spawner lists it builds),
not ROS itself. The stubs mirror the Jazzy behaviour relied on: a DeclareLaunchArgument without a
default raises when executed and the argument was not given, and LaunchDescription.entities is the
live list of actions, executed in order.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "scripts" / "mtc" / "real_chain.launch.py"

UPSTREAM = '''
class LaunchDescription:
    def __init__(self, entities):
        self.entities = list(entities)
    def add_action(self, action):
        self.entities.append(action)

def setup_controllers(context):
    return ["upstream setup"]

def generate_launch_description():
    # upstream builds OpaqueFunction(function=setup_controllers) here, looked up at build time
    return LaunchDescription(["upstream arguments", ("opaque", setup_controllers)])
'''


class _Config:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


class _Declare:
    def __init__(self, name, default_value=None, description=""):
        self.name, self.default_value, self.description = name, default_value, description


@pytest.fixture
def launch(tmp_path, monkeypatch):
    share = tmp_path / "share"
    (share / "launch").mkdir(parents=True)
    (share / "launch" / "real.launch.py").write_text(UPSTREAM)
    written = {}

    def generate_temp_config(path, package, kv_pairs):
        written["kv"] = list(kv_pairs)
        return "/tmp/controllers.yaml"

    stubs = {
        "ament_index_python": types.ModuleType("ament_index_python"),
        "ament_index_python.packages": types.SimpleNamespace(get_package_share_directory=lambda package: str(share)),
        "launch": types.ModuleType("launch"),
        "launch.actions": types.SimpleNamespace(
            DeclareLaunchArgument=_Declare,
            SetLaunchConfiguration=lambda name, value: ("set", name, value)),
        "launch.substitutions": types.SimpleNamespace(LaunchConfiguration=_Config),
        "legged_bringup": types.ModuleType("legged_bringup"),
        "legged_bringup.launch_utils": types.SimpleNamespace(
            control_spawner=lambda names, inactive=False, param_file=None: ("spawn", tuple(names), inactive),
            download_wandb_onnx=lambda path: pytest.fail("no wandb download expected"),
            generate_temp_config=generate_temp_config,
            get_controller_names=lambda path, package: ["state_estimator", "standby_controller", "walking_controller"],
            resolve_policy_paths=lambda path, package: []),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("real_chain_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.written = written
    return module


def _context(**overrides):
    context = {"robot_type": "g1", "policy_path": "/p/entry.onnx", "wandb_path": "", "start_step": "",
               "ext_pos_corr": "false", "chain_policy_path": "/p/motion.onnx"}
    context.update(overrides)
    return context


def test_chain_policy_path_is_required_and_declared_before_upstream_builds_its_controllers(launch):
    entities = launch.generate_launch_description().entities
    declared = entities[0]
    assert isinstance(declared, _Declare) and declared.name == "chain_policy_path"
    # No default: Jazzy's DeclareLaunchArgument.execute raises 'Required launch argument ... was
    # not provided', and it runs before upstream's OpaqueFunction reads the value.
    assert declared.default_value is None
    assert entities[-1] == ("opaque", launch.setup_controllers)


def test_an_empty_chain_policy_path_fails_loudly(launch):
    with pytest.raises(RuntimeError, match="chain_policy_path"):
        launch.setup_controllers(_context(chain_policy_path=""))


def test_walking_controller1_runs_the_motion_policy_and_is_spawned_inactive(launch):
    actions = launch.setup_controllers(_context())
    kv = dict(launch.written["kv"])
    assert kv["walking_controller.policy.path"] == "/p/entry.onnx"
    assert kv["controller_manager.ros__parameters.walking_controller1.type"] == \
        "motion_tracking_controller/MotionTrackingController"
    assert kv["walking_controller1.policy.path"] == "/p/motion.onnx"
    assert kv["walking_controller1.update_rate"] == "50"
    assert kv["walking_controller1.motion.start_step"] == "0"
    assert ("spawn", ("state_estimator", "standby_controller"), False) in actions
    assert ("spawn", ("walking_controller", "walking_controller1"), True) in actions
