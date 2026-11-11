"""motion_tracking_controller's mujoco.launch.py with a deterministic start.

Runs INSIDE the rootfs (started by scripts/mtc/record.sh). Same controller config, controller
lists, spawners, robot model and simulator node as third_party/motion_tracking_controller/launch/
mujoco.launch.py (left unmodified). Differences:

- hold_qpos_file:= one line of numbers, a qpos of the simulator's MJCF. Adds the hcf_mtc_hold
                 physics plugin (scripts/mtc/hcf_mtc_hold), which keeps the robot frozen in that
                 pose until walking_controller first commands torque. Upstream nothing holds the
                 robot between simulator start and activation, which takes 0.2-3.5 s of sim time
                 depending on CPU load, and the unpowered G1 topples at ~0.7-1.0 s.
- urdf:= mjcf:=  optional model overrides (default unitree_description's, as upstream), for a
                 modified plant such as the G1 holding a saber.
- spawn_delay:=   seconds to wait before starting the spawners (default 0). A test hook: it
                 makes activation late on purpose, to check that the start does not depend on it.
- ext_pos_corr:= as upstream.
- no teleop      joy_linux/joy_teleop only serve the gamepad; nothing to read headless.

mujoco_sim_ros2 opens <share of model_package>/<model_file>; an override MJCF is reached through
the `hcf_mtc_models` package whose share directory is `/` (see record.sh).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from legged_bringup.launch_utils import generate_temp_config, get_controller_names, resolve_policy_paths

PACKAGE = 'motion_tracking_controller'


def spawner(names, param_file, inactive=False):
    # legged_bringup's control_spawner with a longer wait for the manager: under CPU load the
    # simulator (which hosts the controller manager) can take longer than the 10 s default.
    args = list(names) + ['--param-file', param_file, '--controller-manager-timeout', '120']
    if inactive:
        args.append('--inactive')
    return Node(package='controller_manager', executable='spawner', arguments=args, output='screen')


def setup(context):
    policy_path = os.path.abspath(LaunchConfiguration('policy_path').perform(context))
    urdf = LaunchConfiguration('urdf').perform(context)
    mjcf = LaunchConfiguration('mjcf').perform(context)
    hold_qpos_file = LaunchConfiguration('hold_qpos_file').perform(context)
    spawn_delay = float(LaunchConfiguration('spawn_delay').perform(context))

    controllers_config_path = 'config/g1/controllers.yaml'
    kv_pairs = resolve_policy_paths(controllers_config_path, PACKAGE)
    kv_pairs.append(('walking_controller.policy.path', policy_path))
    kv_pairs.append(('walking_controller.motion.start_step', LaunchConfiguration('start_step').perform(context)))
    if LaunchConfiguration('ext_pos_corr').perform(context).lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/mid360"))
    temp_config = generate_temp_config(controllers_config_path, PACKAGE, kv_pairs)
    all_controllers = get_controller_names(controllers_config_path, PACKAGE)
    active = ["state_estimator", "walking_controller"]
    inactive = [c for c in all_controllers if c not in active]

    if urdf:
        with open(urdf) as f:
            robot_description = {"robot_description": f.read()}
    else:
        xacro_file = os.path.join(get_package_share_directory('unitree_description'), 'urdf', 'g1', 'robot.xacro')
        robot_description = {"robot_description": Command(['xacro ', xacro_file, ' robot_type:=g1 simulation:=mujoco'])}
    model = ({"model_package": "hcf_mtc_models", "model_file": os.path.abspath(mjcf)} if mjcf
             else {"model_package": "unitree_description", "model_file": "/mjcf/g1.xml"})

    plugins = ["mujoco_ros2_control::MujocoRos2ControlPlugin"]
    hold = {}
    if hold_qpos_file:
        with open(hold_qpos_file) as f:
            hold = {"hold_qpos": [float(v) for v in f.read().split()]}
        plugins.append("hcf_mtc_hold::HoldPlugin")  # after ros2_control: sees this step's commands
    simulator = Node(
        package='mujoco_sim_ros2', executable='mujoco_sim',
        parameters=[{**model, **hold, "physics_plugins": plugins, "use_sim_time": True},
                    robot_description, temp_config],
        output='screen')
    state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[robot_description, {'publish_frequency': 500.0, 'use_sim_time': True}])
    spawners = [spawner(active, temp_config), spawner(inactive, temp_config, inactive=True)]
    if spawn_delay > 0:
        spawners = [TimerAction(period=spawn_delay, actions=spawners)]
    return [*spawners, simulator, state_publisher]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy_path'),
        DeclareLaunchArgument('urdf', default_value=''),
        DeclareLaunchArgument('mjcf', default_value=''),
        DeclareLaunchArgument('start_step', default_value='0'),
        DeclareLaunchArgument('ext_pos_corr', default_value='false'),
        DeclareLaunchArgument('hold_qpos_file', default_value=''),
        DeclareLaunchArgument('spawn_delay', default_value='0'),
        OpaqueFunction(function=setup),
    ])
