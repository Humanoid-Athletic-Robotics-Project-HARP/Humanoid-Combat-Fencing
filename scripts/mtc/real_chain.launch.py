"""motion_tracking_controller's real.launch.py plus a second policy (docs/mtc.md, "Entry from standby").

    ros2 launch real_chain.launch.py network_interface:=<if> policy_path:=<entry.onnx> \
        chain_policy_path:=<motion.onnx>

Loads third_party/motion_tracking_controller/launch/real.launch.py (left unmodified) from the
package share and keeps every node it starts (ros2_control_node, robot_state_publisher, rosbag2,
unitree_bringup's teleop). Only its controller setup is replaced, by a copy of upstream's
setup_controllers that also adds walking_controller1: walking_controller's type and parameters
with policy.path = chain_policy_path, spawned inactive. chain_policy_path is required: omitted, the
launch stops with 'Required launch argument "chain_policy_path" was not provided'; given empty, it
stops with a RuntimeError. Use upstream real.launch.py for a single policy. unitree_bringup's
config/g1/joy.yaml already switches to it:

    L1 + A   standby_controller      (deactivates walking_controller, walking_controller1)
    R1 + A   walking_controller      = policy_path, the entry  (from standby)
    R1 + X   walking_controller1     = chain_policy_path, the motion (from the entry's end hold)
    B / select / left stick           stop all three

R1 + X is not interlocked: pressed straight from standby it starts the motion policy without the
entry, the start docs/hardware_tests.md says not to use for the policies that fall from standby.
Not run on the robot or in this repo's sim (it needs the Unitree network interface); the sim
equivalent is scripts/mtc/mujoco_custom.launch.py chain_policy_path:=. tests/test_real_chain_launch.py
checks the generated parameters and spawner lists with ROS stubbed; a launch on the robot's PC is
still the real check.
"""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration
from legged_bringup.launch_utils import (
    control_spawner, download_wandb_onnx, generate_temp_config, get_controller_names, resolve_policy_paths,
)

PACKAGE = 'motion_tracking_controller'
CHAIN_CONTROLLER = 'walking_controller1'


def _upstream():
    path = os.path.join(get_package_share_directory(PACKAGE), 'launch', 'real.launch.py')
    spec = importlib.util.spec_from_file_location('mtc_real_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_controllers(context):
    """upstream real.launch.py setup_controllers, plus walking_controller1 running chain_policy_path."""
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    policy_path_value = LaunchConfiguration('policy_path').perform(context)
    wandb_path_value = LaunchConfiguration('wandb_path').perform(context)
    start_step_value = LaunchConfiguration('start_step').perform(context)
    ext_pos_corr = LaunchConfiguration('ext_pos_corr').perform(context)
    chain_policy_path = LaunchConfiguration('chain_policy_path').perform(context)
    if not chain_policy_path:
        raise RuntimeError('chain_policy_path is empty: real_chain.launch.py starts two policies; '
                           'use real.launch.py for one')

    if not policy_path_value and wandb_path_value:
        policy_path_value = download_wandb_onnx(wandb_path_value)

    controllers_config_path = f'config/{robot_type_value}/controllers.yaml'
    kv_pairs = resolve_policy_paths(controllers_config_path, PACKAGE)
    if policy_path_value:
        kv_pairs.append(('walking_controller.policy.path',
                         os.path.abspath(os.path.expanduser(os.path.expandvars(policy_path_value)))))
    if start_step_value:
        kv_pairs.append(('walking_controller.motion.start_step', start_step_value))
    if ext_pos_corr.lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/glim/odom"))
    kv_pairs += [
        (f'controller_manager.ros__parameters.{CHAIN_CONTROLLER}.type',
         'motion_tracking_controller/MotionTrackingController'),
        (f'{CHAIN_CONTROLLER}.update_rate', '50'),  # as walking_controller in config/g1/controllers.yaml
        (f'{CHAIN_CONTROLLER}.policy.path', os.path.abspath(os.path.expanduser(os.path.expandvars(chain_policy_path)))),
        (f'{CHAIN_CONTROLLER}.motion.start_step', '0'),
    ]
    temp_controllers_config_path = generate_temp_config(controllers_config_path, PACKAGE, kv_pairs)
    set_controllers_yaml = SetLaunchConfiguration(name='controllers_yaml', value=temp_controllers_config_path)

    # get_controller_names reads the package's own file, which has no walking_controller1.
    all_controllers = get_controller_names(controllers_config_path, PACKAGE) + [CHAIN_CONTROLLER]
    active_list = ["state_estimator", "standby_controller"]
    inactive_list = [c for c in all_controllers if c not in active_list]
    param_file = LaunchConfiguration('controllers_yaml')
    return [set_controllers_yaml, control_spawner(active_list, param_file=param_file),
            control_spawner(inactive_list, inactive=True, param_file=param_file)]


def generate_launch_description():
    upstream = _upstream()
    upstream.setup_controllers = setup_controllers  # looked up when its OpaqueFunction is built
    description = upstream.generate_launch_description()
    # First, so its missing-argument error comes before upstream's OpaqueFunction reads the value.
    # No default: a launch without it would otherwise start one policy under a two-policy name.
    description.entities.insert(0, DeclareLaunchArgument(
        'chain_policy_path', description='policy for walking_controller1 (R1 + X), the motion after the entry'))
    return description
