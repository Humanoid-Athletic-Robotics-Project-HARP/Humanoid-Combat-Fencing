"""Runs INSIDE the rootfs (needs rosbag2_py). Flatten an MTC recording into plain arrays.

    bag_to_npz.py <bag dir> <out.npz>

Only copies fields; all interpretation (time alignment, base pose, scoring) happens in
fencing.sim2sim.mtc on the host, where it is unit-tested. Times are message header stamps
(simulation time; the launch runs with use_sim_time).
"""

import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ACTIVE = 3


def stamp(header) -> float:
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


def odometry_row(msg) -> list[float]:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    v, w = msg.twist.twist.linear, msg.twist.twist.angular
    return [stamp(msg.header), p.x, p.y, p.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z, w.x, w.y, w.z]


def main() -> int:
    bag, out = sys.argv[1], sys.argv[2]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    arrays = {"joint_t": [], "joint_pos": [], "joint_vel": [], "joint_effort": [],
              "mid360": [], "imu_in_pelvis": [], "odom": [], "clock": [], "activation_t": [],
              "clock_wall": [], "stat_t": [], "stat_period": [], "stat_samples": [],
              # One row per /controller_manager/activity message: stamp, standby active, walking active.
              "activity": [],
              # The chained start (docs/mtc.md "Entry from standby"): walking_controller1's
              # activation stamps and stats, one row per activity message of stamp, walking
              # active, walking1 active. Empty without walking_controller1.
              "activation1_t": [], "activity_chain": [], "stat1_t": [], "stat1_period": [], "stat1_samples": []}
    joint_names = None
    stat_names = {}  # names_version -> names (pal_statistics splits names and values)
    prefix = "walking_controller.stats/periodicity/"
    prefix1 = "walking_controller1.stats/periodicity/"
    while reader.has_next():
        topic, data, received_ns = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/joint_states":
            joint_names = list(msg.name)
            arrays["joint_t"].append(stamp(msg.header))
            arrays["joint_pos"].append(list(msg.position))
            arrays["joint_vel"].append(list(msg.velocity))
            arrays["joint_effort"].append(list(msg.effort) or [np.nan] * len(msg.name))
        elif topic in ("/mid360", "/imu_in_pelvis", "/odom"):
            arrays[topic[1:]].append(odometry_row(msg))
        elif topic == "/clock":
            arrays["clock"].append(msg.clock.sec + 1e-9 * msg.clock.nanosec)
            arrays["clock_wall"].append(1e-9 * received_ns)  # wall clock: real-time factor
        elif topic == "/controller_manager/activity":
            active = {c.name for c in msg.controllers if c.state.id == ACTIVE}
            if "walking_controller" in active:
                arrays["activation_t"].append(stamp(msg.header))
            arrays["activity"].append([stamp(msg.header), "standby_controller" in active,
                                       "walking_controller" in active])
            if "walking_controller1" in active:
                arrays["activation1_t"].append(stamp(msg.header))
            if any(c.name == "walking_controller1" for c in msg.controllers):
                arrays["activity_chain"].append([stamp(msg.header), "walking_controller" in active,
                                                 "walking_controller1" in active])
        elif topic == "/controller_manager/statistics/names":
            stat_names[msg.names_version] = list(msg.names)
        elif topic == "/controller_manager/statistics/values":
            names = stat_names.get(msg.names_version)
            if names and prefix + "average" in names:
                arrays["stat_t"].append(stamp(msg.header))
                arrays["stat_period"].append([msg.values[names.index(prefix + k)]
                                              for k in ("average", "min", "max", "current_value")])
                arrays["stat_samples"].append(msg.values[names.index(prefix + "sample_count")])
            if names and prefix1 + "average" in names:
                arrays["stat1_t"].append(stamp(msg.header))
                arrays["stat1_period"].append([msg.values[names.index(prefix1 + k)]
                                               for k in ("average", "min", "max", "current_value")])
                arrays["stat1_samples"].append(msg.values[names.index(prefix1 + "sample_count")])
    if joint_names is None:
        raise SystemExit("no /joint_states in the bag: the simulator never ran")
    result = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
    result["joint_names"] = np.array(joint_names)
    result["stat_names"] = np.array(sorted({n for names in stat_names.values() for n in names}))
    np.savez(out, **result)
    print({k: v.shape for k, v in result.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
