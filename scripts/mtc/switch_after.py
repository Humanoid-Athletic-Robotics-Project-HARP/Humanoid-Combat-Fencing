"""Runs INSIDE the rootfs. The operator's R1 + A, sent <seconds> of sim time after standby is active.

    switch_after.py <seconds after standby_controller activation> <wall-clock cap seconds> [<chain seconds>]

On the real robot, unitree_bringup's teleop (config/g1/joy.yaml, `walking_controller`, buttons
[5, 0] = R1 + A) calls /controller_manager/switch_controller with exactly the request below.
motion_tracking_controller's real.launch.py includes that teleop unchanged. Here no gamepad is
attached, so this node sends the same request itself once standby_controller has been active
for <seconds> of sim time. Exit 0 when the controller manager accepted it, 1 when it refused,
2 when the wall-clock cap passed first (standby never activated, or the sim is too slow).

<chain seconds> (the chained start, docs/mtc.md "Entry from standby"): then also the operator's
R1 + X (joy.yaml `walking_controller1`, buttons [5, 2]), <chain seconds> of sim time after
walking_controller (the entry policy) became active, switching to walking_controller1 (the motion
policy, added by mujoco_custom.launch.py chain_policy_path:=). Same exit codes, for both switches.
"""

import sys
import time

import rclpy
from controller_manager_msgs.msg import ControllerManagerActivity
from controller_manager_msgs.srv import SwitchController
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

ACTIVE = 3  # lifecycle_msgs/State.PRIMARY_STATE_ACTIVE
# unitree_bringup config/g1/joy.yaml, joy_teleop.walking_controller.service_request (commit 8a0f308).
# walking_controller1 does not exist in motion_tracking_controller's config; strictness 1
# (BEST_EFFORT) skips it, on the robot as here.
REQUEST = {"activate_controllers": ["walking_controller"],
           "deactivate_controllers": ["standby_controller", "walking_controller1"],
           "strictness": 1, "activate_asap": True}
# joy_teleop.walking_controller1.service_request, buttons [5, 2] (R1 + X), same file and commit.
CHAIN_REQUEST = {"activate_controllers": ["walking_controller1"],
                 "deactivate_controllers": ["standby_controller", "walking_controller"],
                 "strictness": 1, "activate_asap": True}


def main() -> int:
    seconds, wall_cap = float(sys.argv[1]), float(sys.argv[2])
    chain = float(sys.argv[3]) if len(sys.argv) > 3 else None
    rclpy.init()
    node = rclpy.create_node("hcf_standby_switch")
    state = {"now": None, "standby_controller": None, "walking_controller": None}

    def on_clock(msg):
        state["now"] = msg.clock.sec + 1e-9 * msg.clock.nanosec

    def on_activity(msg):
        for controller in msg.controllers:
            stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            if controller.name == "standby_controller" and controller.state.id == ACTIVE and state["standby_controller"] is None:
                state["standby_controller"] = stamp
            if controller.name == "walking_controller" and controller.state.id == ACTIVE and state["walking_controller"] is None:
                state["walking_controller"] = stamp

    node.create_subscription(Clock, "/clock", on_clock, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", on_activity,
                             QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    client = node.create_client(SwitchController, "/controller_manager/switch_controller")
    start = time.monotonic()
    switches = [("standby_controller", seconds, REQUEST)]
    if chain is not None:
        switches.append(("walking_controller", chain, CHAIN_REQUEST))
    for after, delay, request in switches:
        status = switch_when(node, client, state, after, delay, request, start, wall_cap)
        if status:
            return status
    return 0


def switch_when(node, client, state, after, delay, request, start, wall_cap) -> int:
    """Send `request` once controller `after` has been active for `delay` s of sim time."""
    last_print = 0.0
    while time.monotonic() - start < wall_cap:
        rclpy.spin_once(node, timeout_sec=0.05)
        now, since = state["now"], state[after]
        if time.monotonic() - last_print > 5:
            print(f"wall {time.monotonic() - start:.0f}s sim {now} {after} active since {since}", flush=True)
            last_print = time.monotonic()
        if now is None or since is None or now < since + delay:
            continue
        if not client.wait_for_service(timeout_sec=1.0):
            continue
        print(f"switch requested at sim {now:.3f} ({after} active since {since:.3f}): {request}", flush=True)
        future = client.call_async(SwitchController.Request(**request))
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(1.0, wall_cap - (time.monotonic() - start)))
        response = future.result()
        print(f"switch response: {response} (sim {state['now']})", flush=True)
        return 0 if response is not None and response.ok else 1
    print(f"wall-clock cap {wall_cap}s reached: sim {state['now']} {after} active since {state[after]}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
