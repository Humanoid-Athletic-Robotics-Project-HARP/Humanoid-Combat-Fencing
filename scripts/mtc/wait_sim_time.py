"""Runs INSIDE the rootfs. Exit 0 once sim time is <seconds> past walking_controller activation.

    wait_sim_time.py <seconds after activation> <wall-clock cap seconds>

Exit 2 when the wall-clock cap passes first (sim too slow, or the controller never activated).
"""

import sys
import time

import rclpy
from controller_manager_msgs.msg import ControllerManagerActivity
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock

ACTIVE = 3  # lifecycle_msgs/State.PRIMARY_STATE_ACTIVE


def main() -> int:
    seconds, wall_cap = float(sys.argv[1]), float(sys.argv[2])
    rclpy.init()
    node = rclpy.create_node("hcf_wait_sim_time")
    state = {"now": None, "activated": None}

    def on_clock(msg):
        state["now"] = msg.clock.sec + 1e-9 * msg.clock.nanosec

    def on_activity(msg):
        for controller in msg.controllers:
            if controller.name == "walking_controller" and controller.state.id == ACTIVE and state["activated"] is None:
                state["activated"] = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    node.create_subscription(Clock, "/clock", on_clock, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    # The activity topic is latched (transient local): a late subscriber still gets the last state.
    node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", on_activity,
                             QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    start = time.monotonic()
    last_print = 0.0
    while time.monotonic() - start < wall_cap:
        rclpy.spin_once(node, timeout_sec=0.2)
        now, activated = state["now"], state["activated"]
        if time.monotonic() - last_print > 5:
            print(f"wall {time.monotonic() - start:.0f}s sim {now} activated {activated}", flush=True)
            last_print = time.monotonic()
        if now is not None and activated is not None and now >= activated + seconds:
            print(f"done: sim {now:.3f} >= activation {activated:.3f} + {seconds}", flush=True)
            return 0
    print(f"wall-clock cap {wall_cap}s reached: sim {state['now']} activated {state['activated']}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
