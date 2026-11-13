#!/bin/bash
# Runs INSIDE the rootless Ubuntu 24.04 rootfs (see docs/mtc.md); started by scripts/mtc_sim2sim.sh.
#
#   record.sh <policy.onnx> <out dir> <sim seconds after activation> <start qpos file | -> [<urdf> <mjcf>]
#
# Runs motion_tracking_controller's controllers and simulator headless (Xvfb) through
# mujoco_custom.launch.py: upstream's mujoco.launch.py plus the hcf_mtc_hold physics plugin,
# which keeps the robot frozen in the start qpos (the reference's first frame, written by
# `python -m fencing.sim2sim.mtc --write-start-qpos`) until walking_controller first commands
# torque, so the policy takes over the same still robot whatever the CPU load. "-" runs without
# the hold, as upstream. rosbag2 records from before the simulator starts until sim time is
# <seconds> past the activation (or a wall-clock cap); then every process this script started
# is stopped and the bag is flattened to out/bag.npz. <urdf> <mjcf> override the robot models.
# $MTC_LAUNCH_ARGS is passed to the launch (e.g. "ext_pos_corr:=true").
# $MTC_STANDBY_SWITCH_S (start from standby, docs/mtc.md "Start from standby"): spawn
# standby_controller active and walking_controller inactive, as upstream's real.launch.py, and
# send the operator's R1 + A switch (switch_after.py) that many sim seconds after standby is
# active. The hold then releases at standby's first command, not the policy's.
# $MTC_CHAIN_POLICY + $MTC_CHAIN_SWITCH_S (the chained start, docs/mtc.md "Entry from standby";
# needs $MTC_STANDBY_SWITCH_S): <policy.onnx> is the entry policy (walking_controller), and
# $MTC_CHAIN_POLICY runs as walking_controller1, switched to (R1 + X) $MTC_CHAIN_SWITCH_S sim
# seconds after the entry became active. <seconds> then count from walking_controller1's activation.
set -o pipefail
POLICY="${1:?policy.onnx}"; OUT="${2:?out dir}"; SECONDS_AFTER="${3:?seconds}"; HOLD_QPOS="${4:?start qpos file or -}"
URDF="${5:-}"; MJCF="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/jazzy/setup.bash
source /root/colcon_ws/install/setup.bash
# First, before anything can fail: a previous run's bag.npz left in OUT would be scored as this run's.
rm -rf "$OUT"; mkdir -p "$OUT"

HOLD_ARGS=()
if [ "$HOLD_QPOS" != "-" ]; then
  # Build the hold plugin when its sources changed (a few seconds; the sources live in this repo).
  HOLD_WS=/root/hcf_ws
  SRC_SHA=$(cat "$HERE"/hcf_mtc_hold/CMakeLists.txt "$HERE"/hcf_mtc_hold/package.xml "$HERE"/hcf_mtc_hold/plugins.xml \
            "$HERE"/hcf_mtc_hold/src/*.cpp | sha256sum | cut -d' ' -f1)
  if [ "$(cat "$HOLD_WS/src.sha" 2>/dev/null)" != "$SRC_SHA" ]; then
    rm -rf "$HOLD_WS"; mkdir -p "$HOLD_WS"
    colcon --log-base "$HOLD_WS/log" build --base-paths "$HERE/hcf_mtc_hold" --build-base "$HOLD_WS/build" \
      --install-base "$HOLD_WS/install" --cmake-args -DCMAKE_BUILD_TYPE=Release >"$HOLD_WS/build.log" 2>&1 || {
      echo "hcf_mtc_hold build failed, see $HOLD_WS/build.log"; exit 4; }
    echo "$SRC_SHA" >"$HOLD_WS/src.sha"
  fi
  source "$HOLD_WS/install/setup.bash"
  # Activate the controller 3 s late (wall clock): rosbag only subscribes once the simulator's
  # topics exist, and under CPU load that took up to ~2.3 s of sim time, after an early
  # activation (the bag then misses the start and the run is INVALID). Harmless with the hold,
  # which makes the start independent of activation time (docs/mtc.md, delay3 runs).
  # $MTC_LAUNCH_ARGS comes later on the command line and overrides it.
  HOLD_ARGS=(hold_qpos_file:="$HOLD_QPOS" spawn_delay:=3)
fi
STANDBY_S="${MTC_STANDBY_SWITCH_S:-}"
[ -n "$STANDBY_S" ] && HOLD_ARGS+=(start_controller:=standby_controller)
CHAIN_POLICY="${MTC_CHAIN_POLICY:-}"; CHAIN_S="${MTC_CHAIN_SWITCH_S:-}"; SCORED=walking_controller
if [ -n "$CHAIN_POLICY" ]; then
  [ -n "$STANDBY_S" ] && [ -n "$CHAIN_S" ] || { echo "MTC_CHAIN_POLICY needs MTC_STANDBY_SWITCH_S and MTC_CHAIN_SWITCH_S"; exit 2; }
  HOLD_ARGS+=(chain_policy_path:="$CHAIN_POLICY"); SCORED=walking_controller1
fi
export ROS_DOMAIN_ID="${MTC_ROS_DOMAIN_ID:-77}" ROS_LOCALHOST_ONLY=1 RCUTILS_COLORIZED_OUTPUT=0
TOPICS=(/clock /joint_states /mid360 /imu_in_pelvis /odom /controller_manager/activity
        /controller_manager/statistics/names /controller_manager/statistics/values)
WORK=/tmp/mtc_record; rm -rf "$WORK"; mkdir -p "$WORK"

cleanup() {
  # Everything below was started by this script inside its own PID namespace.
  # Let rosbag2 finish the mcap footer and metadata.yaml before anything is killed. SIGTERM,
  # not SIGINT: background jobs of a non-interactive shell ignore SIGINT.
  if [ -n "${BAG_PID:-}" ]; then
    kill -TERM "$BAG_PID" 2>/dev/null
    for _ in $(seq 150); do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 0.2; done
  fi
  [ -n "${LAUNCH_PID:-}" ] && kill -TERM "$LAUNCH_PID" 2>/dev/null
  sleep 3
  pkill -9 -f mujoco_sim; pkill -9 -f 'ros2 '; pkill -9 -f spawner; pkill -9 -f joy_; pkill -9 -f robot_state_publisher
  pkill -9 Xvfb
  true
}
trap cleanup EXIT

# A display for GLFW: mujoco_sim has no headless mode. Remove a stale lock from a killed run.
rm -f /tmp/.X77-lock /tmp/.X11-unix/X77
Xvfb :77 -screen 0 640x480x24 >"$WORK/xvfb.log" 2>&1 &
export DISPLAY=:77
for _ in $(seq 100); do [ -e /tmp/.X11-unix/X77 ] && break; sleep 0.1; done

# The recorder starts BEFORE the simulator so the whole run, from the first sim step, is in the
# bag (one started once /clock existed missed the activation and the first policy steps).
# -p 10: look for the (not yet existing) topics every 10 ms.
ros2 bag record -s mcap -p 10 -o "$WORK/bag" --topics "${TOPICS[@]}" >"$WORK/bag.log" 2>&1 &
BAG_PID=$!
for _ in $(seq 100); do grep -q -i -E 'listening|recording' "$WORK/bag.log" 2>/dev/null && break; sleep 0.1; done

MODEL_ARGS=()
if [ -n "$MJCF" ]; then
  # mujoco_sim_ros2 opens <share of model_package> + model_file: a package whose share is "/"
  # lets model_file be an absolute path.
  PREFIX=/root/hcf_prefix
  mkdir -p "$PREFIX/share/ament_index/resource_index/packages"
  touch "$PREFIX/share/ament_index/resource_index/packages/hcf_mtc_models"
  ln -sfn / "$PREFIX/share/hcf_mtc_models"
  export AMENT_PREFIX_PATH="$PREFIX:$AMENT_PREFIX_PATH"
  MODEL_ARGS=(mjcf:="$MJCF")
  [ -n "$URDF" ] && MODEL_ARGS+=(urdf:="$URDF")
fi
# shellcheck disable=SC2086  # MTC_LAUNCH_ARGS is a list of key:=value words
ros2 launch "$HERE/mujoco_custom.launch.py" policy_path:="$POLICY" "${HOLD_ARGS[@]}" "${MODEL_ARGS[@]}" ${MTC_LAUNCH_ARGS:-} \
  >"$WORK/launch.log" 2>&1 &
LAUNCH_PID=$!
WALL_CAP="${MTC_WALL_CAP_S:-$((SECONDS_AFTER * 4 + 120))}"

WAIT_STATUS=0
if [ -n "$STANDBY_S" ]; then
  # Standby first; the policy is activated only by the switch, like the operator's button.
  python3 "$HERE/switch_after.py" "$STANDBY_S" "$WALL_CAP" ${CHAIN_S:+"$CHAIN_S"} >"$WORK/switch.log" 2>&1 || {
    SWITCH_STATUS=$?; echo "standby switch failed (status $SWITCH_STATUS), see $OUT/switch.log"; WAIT_STATUS=5; }
fi
# Wait until the walking controller is active, then until sim time passes activation + seconds.
if [ "$WAIT_STATUS" -eq 0 ]; then
  python3 "$HERE/wait_sim_time.py" "$SECONDS_AFTER" "$WALL_CAP" "$SCORED" \
    >"$WORK/wait.log" 2>&1
  WAIT_STATUS=$?
fi
ros2 control list_controllers >"$WORK/controllers.txt" 2>&1
cleanup; trap - EXIT
cp -r "$WORK/bag" "$WORK"/*.log "$WORK/controllers.txt" "$OUT/" 2>/dev/null
python3 "$HERE/bag_to_npz.py" "$OUT/bag" "$OUT/bag.npz" >"$OUT/convert.log" 2>&1 || {
  echo "bag conversion failed, see $OUT/convert.log"; exit 3; }
exit $WAIT_STATUS
