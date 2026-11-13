# motion_tracking_controller sim-to-sim

`scripts/mtc_sim2sim.sh <eval dir>` runs an exported policy through BeyondMimic's deployment
controller, [`motion_tracking_controller`](https://github.com/HybridRobotics/motion_tracking_controller)
(MTC, vendored read-only at `third_party/motion_tracking_controller`), in its own MuJoCo
simulator, and scores the run with the metrics, thresholds and verdict of `hcf sim2sim`. It
writes `<eval dir>/mtc.json` next to `sim2sim.json`.

## Why it matters for the Goal 1 gate

ADR 0001's sim-to-sim gate is "the ONNX runs in `motion_tracking_controller`'s MuJoCo launch".
`hcf sim2sim` (docs/evaluation.md) is a Python stand-in that follows the same metadata
contract on mjlab's scene. MTC is the real thing: the C++ controller that would run on the
robot, its ONNX parsing (3-decimal metadata, no `hcf_full_precision`), its observation
assembly, its state estimator (which the Python path does not have: it reads ground truth),
ros2_control timing, and a different MuJoCo plant (`unitree_description`'s `g1.xml`, MuJoCo
3.2.7). A policy that passes both has passed everything we can check before hardware.

## What is covered and what is not

- **Covered:** plain-G1 tracking policies (29 DoF) and saber policies (the same G1 with the
  training saber welded to the right wrist, picked automatically, see "Saber plant"), the full
  controller and estimator, nominal tracking metrics including the blade tip (no pushes or
  robustness sweeps).
- **Covered separately (`--standby`):** the start from upstream's standby controller and the
  operator's `R1 + A` switch, issued as the same `switch_controller` request the joystick
  binding sends, not through the joystick nodes. See "Start from standby". All other results
  in this file start held in the first frame.
- **Not covered:** hardware I/O (`real.launch.py`, unitree_systems) and the joystick nodes
  themselves.

## Build the rootless environment from scratch

No sudo and no Docker on this machine, so ROS 2 Jazzy lives in an Ubuntu 24.04 rootfs entered
with a user namespace (`unshare -r`) and `chroot`. About 3.7 GB under `$HCF_DATA/mtc`.

*Provenance:* reconstructed from the existing rootfs (its apt `history.log`, `dpkg.log`, dpkg
diversions, `apt.conf.d`, colcon logs) and the prior engineer's scripts. The installed
versions below were checked against `dpkg-query` in the rootfs. The sequence was **not re-run
end to end**. The one deliberate change from how it was first done: the chown shim (step 3)
goes in before the big install. The first time, it was added after the install had failed in
dpkg.

```bash
# 1. Base system (Ubuntu Base 24.04.5, ~30 MB), unpacked as fake root.
M=$HOME/hcf-data/mtc; mkdir -p $M/noble
curl -L -o $M/ubuntu-base-24.04.5.tar.gz \
  https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/ubuntu-base-24.04.5-base-amd64.tar.gz
unshare -r tar -xzf $M/ubuntu-base-24.04.5.tar.gz -C $M/noble
```

2. Create `$M/enter.sh`. It runs a command as fake root inside the rootfs, with `/dev`,
   `/proc` and `/sys` mounted and `$HOME/hcf-data` bind-mounted at `/hostdata`. That is the
   only host path the rootfs sees, which is why eval dirs must be under `$HCF_DATA`. Its
   environment is empty (`env -i`).

```bash
#!/bin/bash
R=$HOME/hcf-data/mtc/noble
exec unshare -r --mount --pid --fork --kill-child bash -c '
R="$0"
mkdir -p "$R/dev" "$R/proc" "$R/sys" "$R/hostdata"
mount --rbind /dev "$R/dev"
mount -t proc proc "$R/proc"
mount --rbind /sys "$R/sys"
mount --bind "$HOME/hcf-data" "$R/hostdata"
cp /etc/resolv.conf "$R/etc/resolv.conf"
exec chroot "$R" /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root TERM=dumb DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 /bin/bash -c "$*"
' "$R" "$@"
```

```bash
chmod +x $M/enter.sh
# 3. Make apt/dpkg work in a single-uid user namespace:
#    - apt drops to the _apt user for downloads; that uid is not mapped -> sandbox as root.
#    - dpkg chowns files to groups that are not mapped (e.g. root:adm) and aborts -> divert
#      chown to a wrapper that ignores failures. Everything stays owned by you on the host.
$M/enter.sh 'printf "APT::Sandbox::User \"root\";\nAcquire::Retries \"3\";\nAPT::Install-Recommends \"false\";\n" > /etc/apt/apt.conf.d/99rootless'
$M/enter.sh 'dpkg-divert --local --rename --add /usr/bin/chown &&
  printf "#!/bin/sh\n/usr/bin/chown.distrib \"\$@\" 2>/dev/null || true\n" > /usr/bin/chown && chmod 755 /usr/bin/chown'
$M/enter.sh 'apt-get update -qq && apt-get install -y -qq curl ca-certificates gnupg'

# 4. ROS 2 Jazzy + legged_control2 + the author's MuJoCo stack (qiayuanl buildfarms).
cat > $M/noble/root/install_ros.sh <<'EOF'
set -e
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list
for f in legged simulation unitree; do echo "deb [trusted=yes] https://github.com/qiayuanl/${f}_buildfarm/raw/noble-jazzy-amd64/ ./" > /etc/apt/sources.list.d/qiayuanl_${f}.list; done
nice -n 15 apt-get update -qq
nice -n 15 apt-get install -y -q ros-jazzy-ros-base ros-jazzy-legged-control-base ros-jazzy-mujoco-ros2-control ros-jazzy-unitree-description ros-jazzy-unitree-systems ros-jazzy-rosbag2-storage-mcap ros-dev-tools build-essential cmake git python3-rosdep
EOF
$M/enter.sh 'bash /root/install_ros.sh'

# 5. The official ROS repo's ros-jazzy-mujoco-ros2-control 0.1.2 (ros-controls' project) has a
#    higher version than the author's 0-2026.10.06.16.24 and gets installed instead. MTC's
#    launch needs the author's (mujoco_sim_ros2 + MujocoRos2ControlPlugin): downgrade
#    explicitly. An `apt-get upgrade` would undo this. `apt-mark hold ros-jazzy-mujoco-ros2-control`
#    should prevent that (suggested, not tried).
$M/enter.sh 'apt-get install -y -q --allow-downgrades ros-jazzy-mujoco-sim-ros2 ros-jazzy-mujoco-ros2-control=0-2026.10.06.16.24'
$M/enter.sh 'apt-get install -y -q xvfb libgl1-mesa-dri'   # mujoco_sim needs a GLFW display

# 6. Workspace: unitree_bringup (git) + MTC (copied from the vendored submodule).
$M/enter.sh 'mkdir -p /root/colcon_ws/src && git clone https://github.com/qiayuanl/unitree_bringup.git /root/colcon_ws/src/unitree_bringup'
cp -r "<repo>/third_party/motion_tracking_controller" $M/noble/root/colcon_ws/src/
rm -rf $M/noble/root/colcon_ws/src/motion_tracking_controller/.git
$M/enter.sh 'source /opt/ros/jazzy/setup.bash && cd /root/colcon_ws && nice -n 15 colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo --packages-up-to motion_tracking_controller unitree_bringup'
```

Versions in the working rootfs: legged_* 3.7.0-2026.11.03, mujoco-sim-ros2 and
mujoco-ros2-control 0-2026.10.06, unitree-description 1.3.0-2026.10.06, unitree_bringup at
8a0f308, MTC at 742924f (the submodule commit). `scripts/mtc/record.sh` builds the start-hold
plugin (`scripts/mtc/hcf_mtc_hold`, below) into `/root/hcf_ws` on first use and whenever its
sources change (a few seconds).

## Run

```bash
scripts/mtc_sim2sim.sh ~/hcf-data/eval/<run>/model_<N>          # writes model_<N>/mtc.json
#   [--seconds S] [--hold-seconds H] [--settle-seconds T] [--checkpoint N] [--keep] [--no-hold]
#   [--standby [--switch-after S]]   # start from standby: mtc_standby.json + runs_standby/, see "Start from standby"
```

The plant follows the policy: saber policies run on the saber plant, plain ones on
`unitree_description`'s G1 (see "Saber plant"). There is no flag to override that.

CPU only, `nice -n 15`, about real time: roughly 20 s for a 4 s motion, plus scoring. The ROS
recording and logs go to `$HCF_DATA/mtc/runs/<name>/`. `launch.log` there shows the hold and
release lines. One run at a time: the recorder uses a fixed ROS domain (77) and X display
(:77). `MTC_LAUNCH_ARGS` is forwarded to the launch, e.g. `ext_pos_corr:=true` (upstream's
external position correction from `/mid360`) or `spawn_delay:=3` (a test hook that activates
the controller late on purpose).

**What makes a run fail before scoring (exit 1, no `mtc.json`).** The old `mtc.json` and the
run dir are deleted before recording, so a failed run never leaves a stale score behind, and
neither does an old bag. The runner scores only when the recorder exits 0 (whole span recorded)
or 2 (wall-clock cap: scored and reported incomplete, so the verdict is PARTIAL). Any other
status stops it: hold plugin build, bag conversion, `enter.sh`. With the hold on, it also stops
if `launch.log` has no release line. The scorer stops with an error if the joints do not move
in the 0.5 s after activation (`require_release`).

**INVALID and retries.** When a scored state was recorded more than 10 ms (half a policy
step) from its tick, the run is INVALID: `verdict: "INVALID"`, `invalid_reasons`, exit 3. Such a
state is nearer the neighbouring tick's state than its own. The cause is rosbag dropping
`/joint_states` or `/mid360` messages under CPU load. The controller runs inside the physics
loop, so the run itself is unaffected; only the recording is too coarse to judge it.
`mtc_sim2sim.sh` records again, up to 3 attempts. `mtc.json` records `attempt` and the gap
statistics under `timing` (`joint_states_max_gap_s`, `mid360_max_gap_s`,
`max_state_time_mismatch_s` and its limit).

Pipeline: `mtc_sim2sim.sh` → (in the rootfs) `scripts/mtc/record.sh` →
`scripts/mtc/mujoco_custom.launch.py`. That launch is upstream's `mujoco.launch.py` with the
same controller config, spawners, robot model and simulator, plus the start hold and minus
the gamepad nodes. rosbag2 records it. `scripts/mtc/bag_to_npz.py` flattens the bag, and
`python -m fencing.sim2sim.mtc` scores it (`src/fencing/sim2sim/mtc.py`).

## Deterministic start

**The problem.** Upstream nothing holds the robot between simulator start and controller
activation. mujoco_sim_ros2 steps in real time from its first moment, and walking_controller
becomes active only after the spawner, the ONNX load and the switch. That took 0.19 to 4.5 s
of sim time here, depending on CPU load (a GPU training job and other users share the CPU).
The unpowered G1 starts straight-legged in `unitree_description`'s pose and topples at about
0.7-1.0 s. So some runs handed the policy a falling or fallen robot: lunge_v3,
advance_lunge_v3, and retreat_v1 (joint error 0.156), and advance_v1 passed once and failed on
other runs.

**The mechanism: a start hold in the simulator.** `scripts/mtc/hcf_mtc_hold` is a ~90-line
mujoco_sim_ros2 physics plugin, loaded after `MujocoRos2ControlPlugin`. It freezes the whole
robot (free joint and all hinges, zero velocity) in a given qpos. It releases at the first
physics step in which any actuator command is non-zero, which is the first step the activated
controller drives. While it holds, it applies as `qfrc_applied` the inverse-dynamics force
that makes qacc = 0 in the held state (below, "What the estimator sees"). The qpos is the reference motion's first frame (pelvis pose and joint
angles from the ONNX), written by `python -m fencing.sim2sim.mtc <dir> --write-start-qpos`.
That is the state mjlab resets to in training and `hcf sim2sim` starts from. Sim time, the
controller manager, the state estimator and the activation all run as upstream. Only the
robot's state is pinned. The release is decided inside the physics loop rather than by a ROS
message, so executor latency does not move it.

**Start state, as recorded.** In every run the robot was exactly the start qpos until release
(joint deviation 0 and constant pelvis-site height over the hold). It was released within one
physics step (2 ms) of the activation stamp: thrust_v3_tip, for example, released at 0.384 s
against activation at 0.386 s. `mtc.json` has both (`hold_release_sim_s`,
`timing.activation_sim_s`). The fencing motions start with the pelvis at 0.711-0.713 m,
upright, in their first-frame stance. `mtc.json` reports `start_hold`, the pelvis height and
tilt at activation, and the activation time.

**Options weighed.**

| Option | Verdict |
|---|---|
| Upstream standby controller, then switch (the hardware procedure) | Rejected. The standby controller must itself activate before the robot falls, so the race stays. It also hands the policy the standby stance, not the reference's first frame. Upstream's MuJoCo launch does not use it either: walking_controller is spawned active. |
| Start the sim paused until the controller is active | Not possible. mujoco_sim_ros2 has no pause parameter (its only parameters are `model_package`, `model_file`, `physics_plugins`). The controller manager is updated from the physics loop, so a paused sim never finishes the activation. |
| Weld or elastic band on the base, released at activation | Rejected. The joints are still unpowered under the band, so the limbs sag and the pose at release depends on how long the hold lasted. |
| Start MuJoCo in the first-frame pose (posed MJCF, `ref` per joint) | Rejected and deleted. Joints read ~0 at the start (cause unconfirmed), and it does not fix the race: the posed robot still falls if activation is late. |
| **Freeze the full state in the first frame until the first command** | **Chosen.** Deterministic by construction, starts where training and `hcf sim2sim` start, no `third_party/` changes. |

**The tradeoff.** This tests the policy from its training start state. It does not test the
hardware procedure: standby stance, operator switch, robot already loaded on its feet. It also
needs a small C++ plugin built in the rootfs. `--no-hold` reproduces the upstream start.

**What the estimator sees during the hold** (fixed 2026-11-11). The state estimator activates
with the walking controller, a few tens of ms before the release, and integrates the IMU from
then on. The first version of the hold only reset qpos/qvel around each step. The sensors were
computed inside the step from the collapsing, unpowered robot: an accelerometer near 0, i.e.
free fall. So at release `/odom` said the pelvis was sinking at 0.6 m/s (parry_quinte), 1.0
(en_garde) or 1.5 m/s (lunge_v3). The earlier text here said the opposite (that the IMU read a
still robot), and that was wrong. Now each held step applies the inverse-dynamics force for
qacc = 0, so the step's own forward dynamics, and with it every sensor, sees a robot at rest:
accelerometer +g, gyro 0. The plugin removes its own force before checking for a command. A
first version that cleared it in `PostUpdate` released at 0.004 s on its own free-joint force,
because `PostUpdate` does not run before the next `PreUpdate` there. Measured from the bags:
`/odom` speed at release is now 0.000 m/s (it was 0.6-1.5). `mtc.json` reports it as
`estimator.at_first_tick`.

*Still harsh, and why.* The estimated pelvis height at release is **+8 cm** off (0.794 m
against 0.712 m on parry_quinte; see `at_first_tick.height_error_m` per run). It is constant
during the hold and corrects within ~0.1 s of release (0.710 m at +0.1 s). Our reading: the
estimator corrects height from leg kinematics only for feet it judges in contact, and it
judges contact from joint torques. During the hold those torques are the inverse-dynamics hold
force, not a loaded stance: up to ~4,000 N m on one ankle. So it keeps its initial height
(0.794 m, about `g1.xml`'s default pelvis height 0.793 m) until real torques arrive. That is
inferred, not traced in the estimator's source (closed, `legged_controllers`). Giving it a
consistent height would need contact forces that carry the robot's weight in the held pose,
i.e. feet at their contact equilibrium depth, which the reference's first frame does not
guarantee. The policy's first anchor observation therefore carries an ~8 cm height error that
the estimator removes in ~0.1 s. MotionCommandTerm::reset places the reference at the
estimated anchor height, so the offset is in the reference too.

**Determinism evidence, first hold version** (2026-11-11, before the sensor fix above;
`nice -n 15`, load average 8.6-10 on 24 threads with a GPU training job running; the `delay3`
runs add `spawn_delay:=3`; scored with the 1.0 s settle that was the default then). The final
code's evidence is under "Determinism, final code" below:

| motion | run | activation (sim s) | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | verdict |
|---|---|---|---|---|---|---|---|
| en_garde_v3 | 1 | 3.366 | 0.07037 | 3.81 | 1.36 | 0.0022 | PASS |
| en_garde_v3 | 2 | 4.058 | 0.07044 | 3.81 | 1.31 | 0.0035 | PASS |
| en_garde_v3 | 3 | 0.224 | 0.07043 | 3.81 | 1.30 | 0.0010 | PASS |
| en_garde_v3 | 4 | 0.226 | 0.07036 | 3.81 | 1.37 | 0.0010 | PASS |
| en_garde_v3 | delay3 | 3.008 | 0.07049 | 3.82 | 1.30 | 0.0010 | PASS |
| en_garde_v3 | 5 | 0.602 | 0.07020 | 3.82 | 1.40 | 0.0040 | PASS |
| advance_v1 | 1 | 0.190 | 0.07381 | 3.55 | 2.41 | 0.0027 | PASS |
| advance_v1 | 2 | 0.236 | 0.07381 | 3.55 | 2.41 | 0.0027 | PASS |
| advance_v1 | 3 | 0.716 | 0.07371 | 3.55 | 2.42 | 0.0027 | PASS |
| advance_v1 | 4 | 0.420 | 0.07369 | 3.51 | 2.48 | 0.0030 | PASS |
| advance_v1 | delay3 | 3.398 | 0.07371 | 3.55 | 2.42 | 0.0027 | PASS |

Activation varied from 0.19 to 4.06 s, and the joint-error spread is 2.9e-4 rad (en_garde_v3,
5 runs plus delay3) and 1.2e-4 rad (advance_v1). Every verdict is the same. The remaining spread is the policy tick's phase inside the 2 ms
physics step relative to activation (runs with the same phase, advance_v1 1 and 2, are
identical to every digit).

## Scoring conventions

- **Ground truth, not the estimator.** The pelvis pose comes from the simulator's `mid360`
  site pose (exact in simulation) plus forward kinematics. `/odom` (the estimator) is only
  reported, in `estimator`.
- **Policy ticks.** Every 20 ms of sim time from activation. The phase inside the period is
  read from the PD torque jumps.
- **Reference frame per tick: the controller runs one tick behind** (since 2026-11-11, see
  "One-tick reference lag" below). At tick k MTC's observation carries reference frame
  max(k - 1, 0), and sim2sim (like mjlab) observes frame k at step k. So the scorer scores step k
  (frame k) from tick k + 1, with tick k + 2 as its post-step state. The first tick (frame 0 a
  second time) is not scored. With settle 0 the reference is still placed at the activation tick
  (frame 0 is observed at ticks 0 and 1, and `MotionCommandTerm::reset` places it at tick 0),
  and the estimator report still uses ticks 0, 1, .... `mtc.json` records
  `timing.controller_reference_lag_steps` (1).
- **State at a tick: interpolated** (since 2026-11-11, see "Sample interpolation" below).
  `/joint_states` positions and velocities, and the pelvis track (the `mid360` pose plus forward
  kinematics), are linearly interpolated to the tick time. Orientation uses nlerp with the
  quaternion sign aligned. The INVALID rule still uses the distance from the tick to the nearest
  *real* sample, so a run of dropped messages is still caught.
- **Settle window: none by default** (`--settle-seconds T`, default 0, scores from step 0
  like `hcf sim2sim`). With T > 0 the first T seconds after activation are neither scored nor
  checked for terminations. The reference is placed like
  `MotionCommandTerm::reset` places it (yaw and x/y onto the robot's anchor, heights kept),
  but at the end of the settle window. The start frame is frame `settle`, and the pose is the
  robot's at the first tick whose observation carries that frame: tick settle + 1, where its
  step starts being scored. Before 2026-11-11 the scorer used tick `settle`, one tick early
  (settle 0 is unaffected). *Why:* it was introduced when the start was unheld. The
  robot then entered the motion from `unitree_description`'s pose, and its first steps moved it
  (dance1: ~0.4 m in 1 s), an offset the global metrics would otherwise carry for the whole
  run. *With the start hold this reason is gone.* The robot starts in the first frame, and the
  settle now only drops the start transient from the scores. That makes MTC more lenient than
  `hcf sim2sim`, which scores from step 0: foot slip on en_garde_v3 is 0.001 with the settle
  and 0.041 without, against 0.049 in Python. So the default is 0 (since 2026-11-11, it was
  1.0 before); the option stays for unheld (`--no-hold`) runs.
- **Thresholds and verdict.** Those of `fencing.train.thresholds`, same as `sim2sim.json`,
  plus INVALID (see Run) when the recording is too coarse to score.

## Results, plain G1 (2026-11-11, start hold on)

*Scored before the one-tick reference lag fix.* The effect of that fix on the current
recordings is in "One-tick reference lag" below. In short, the plain-G1 numbers barely move,
and lunge_v3 crosses its joint line to agree with Python.

MTC rows are scored from step 0 (settle 0, the default): a rescore of the recording with
`--settle-seconds 0`. For dance1 that is the first of its two runs. The last column is the same
recording scored with the old 1.0 s settle, kept as a note. Python rows come from each dir's
`sim2sim.json`. Success and recovery were 1 in every row. Thresholds: joint ≤ 0.10 rad,
body ≤ 5 cm, anchor ≤ 25 cm, slip ≤ 0.10 m/s.

| motion (eval dir) | source | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | verdict | note: MTC with 1.0 s settle (joint, body, anchor, slip, verdict) |
|---|---|---|---|---|---|---|---|
| dance1 (`2026-11-10_13-16-56_dance1_subject2_p12v2r/model_9999`) | MTC | 0.110 | 4.10 | **33.8** | 0.052 | FAIL | 0.110 / 0.110, 4.11 / 4.11, **30.5 / 28.6**, 0.053 / 0.054, FAIL (2 runs) |
| dance1 (`2026-11-11_13-16-07_dance1_subject2_p12v2r30k/model_29998`) | MTC | 0.0944 | 3.33 | 24.5 | 0.037 | PASS | one run |
| | Python | 0.108 | 3.92 | 9.8 | 0.058 | FAIL | |
| en_garde_v3 (`2026-11-10_04-44-31_en_garde_v3_pipe/model_2999`) | MTC | 0.076 | 3.74 | 3.1 | 0.041 | PASS | 0.070, 3.81, 1.4, 0.001, PASS |
| | Python | 0.077 | 3.79 | 2.3 | 0.049 | PASS | |
| advance_v1 (`2026-11-09_19-10-15_advance_v1_pipe/model_2999`) | MTC | 0.079 | 3.19 | 3.5 | 0.093 | PASS | 0.074, 3.51, 2.5, 0.003, PASS |
| | Python | 0.078 | 3.10 | 3.3 | 0.082 | PASS | |
| retreat_v1 (`2026-11-09_20-40-56_retreat_v1_pipe/model_2999`) | MTC | 0.086 | 2.86 | 2.3 | 0.063 | PASS | 0.088, 3.35, 2.4, 0.012, PASS |
| | Python | 0.087 | 2.94 | 1.9 | 0.045 | PASS | |
| lunge_v3 (`2026-11-10_04-02-40_lunge_v3_pipe/model_2999`) | MTC | **0.101** | 3.89 | 13.5 | 0.081 | FAIL | 0.097, 4.01, 19.1, 0.060, PASS |
| | Python | 0.099 | 3.75 | 11.8 | 0.073 | PASS | |
| advance_lunge_v3 (`2026-11-10_02-34-53_advance_lunge_v3_pipe/model_2999`) | MTC | 0.098 | 3.65 | 8.7 | 0.020 | PASS | 0.098, 3.90, 6.5, 0.020, PASS |
| | Python | 0.098 | 3.72 | 5.8 | 0.019 | PASS | |

No `lunge_v4_pipe` or `advance_lunge_v4_pipe` eval dirs existed at the time of these runs.

**Agreement.** MTC's joint error is within 0.002 rad of Python's on all six motions, and body
error within 0.25 cm. The controller and our ONNX export agree. Five of six verdicts agree;
lunge_v3 is a threshold-edge case (below). With the old 1.0 s settle all six agreed.

**Disagreements and why.**

- **dance1, anchor position (30.5 cm MTC vs 9.8 cm Python, threshold 25 cm).** Both fail on
  joint error (0.110 / 0.108), and MTC fails anchor as well. Evidence that it is the state
  estimator: the estimator's mean pelvis error on this run is 31.3 cm, about equal to the
  anchor error. It is not the old start step. The estimator is ~2 cm off for the first 5 s and
  then drifts with distance travelled: ~7 cm at 10 s, ~19 cm at 20 s, ~50 cm from 90 s on (a
  131 s motion that covers a lot of floor). The policy observes the reference anchor relative
  to the *estimated* pose, so the robot ends up tracking a reference that drifts with the
  estimate. The five short motions have estimator errors of 0.6-2.4 cm. (The drift-over-time
  figures compare `/odom` with the torso-mounted `mid360` site, so they are approximate to a
  few cm.) This is correlation, not proof. One rerun with upstream's `ext_pos_corr:=true` did
  not help: anchor 33.6 cm, estimator 55.8 cm. I did not confirm that the correction was
  actually active in that run, so that experiment is inconclusive. At 30,000 iterations the
  anchor error is 24.5 cm against 5.8 cm in Python, just inside the 25 cm gate, the same drift.
- **lunge_v3, joint error (0.101 vs 0.099, threshold 0.100).** A threshold-edge
  case. The two values differ by 0.002 rad, the same size as the gap on the other motions, and
  sit either side of 0.100. With the 1.0 s settle MTC passes (0.097). The policy is marginal on
  joint error in both simulators.

## Known gaps

- **State estimator world position.** With the unheld start, the estimator picked up a ~29 cm
  position step in the first second and kept it. With the start hold it is 0.6-2.4 cm on short
  motions, but legged odometry still drifts on long, travelling motions (dance1: ~50 cm after
  90 s). That inflates the anchor error, because the policy's anchor observation uses the
  estimate. Upstream's `ext_pos_corr:=true` (lidar position on `/mid360`) is meant to bound
  the drift, but one dance1 run with it was no better (see Results). Open: check that the
  estimator actually consumes `/mid360` in simulation.
- **Metadata precision.** The C++ reads mjlab's 3-decimal metadata: the wrist pitch/yaw action
  scale 0.0745 runs as 0.075 (+0.67%). This is a strict xfail in `tests/test_mtc_contract.py`.

## Saber plant

Saber policies (mjlab task `Fencing-Tracking-Flat-G1-Saber`) were trained on the G1 with a rigid
saber welded to `right_wrist_yaw_link` (`fencing.saber.spec.attach_saber`, ADR 0002).
`unitree_description`'s G1 has no saber, so `mtc_sim2sim.sh` gives MTC a plant that does. The
controller itself needs no change: on all eleven v3 `_tip` exports, `observation_names`,
`body_names`, `anchor_body_name`, `joint_names` and `command_names` are identical to the plain
`en_garde_v3_pipe` export, and the only metadata difference is the right wrist pitch/yaw action
scale (0.250 instead of 0.075), which the controller reads from the ONNX.

**Detection** (`fencing.sim2sim.mtc_plant.policy_plant`). Two independent records:
`export.json`'s `task` (saber if one of its dash-separated words is `Saber`) and the export's
`scene/scene.xml` (the exact mjlab scene; saber if it has a `saber` body). They must agree. One
alone decides when the other is missing. If they disagree, or neither exists, the runner stops
with `[mtc] FAIL: ...` before starting ROS. The scorer checks again: it refuses to score a
saber policy against an MJCF without the saber body, and a plain one against an MJCF with it.
There is no override flag.

**How the models are built** (`python -m fencing.sim2sim.mtc_plant <export dir>`, run by
`mtc_sim2sim.sh` for every saber run, so they always follow the current `saber/spec.py`):

- **MJCF** `$HCF_DATA/mtc/models/g1_saber.xml`. The rootfs's own
  `unitree_description/mjcf/g1.xml` goes through `MjSpec` + `attach_saber` (venv MuJoCo 3.11).
  Only the resulting `<body name="saber">` element (explicit inertial, three visual geoms with
  contype/conaffinity 0, `guard` and `blade_tip` sites) is copied into the original XML under
  `right_wrist_yaw_link`. Everything else stays what the rootfs's MuJoCo 3.2.7 already loads.
  `meshdir` is rebased to a relative path (`../noble/opt/ros/jazzy/share/unitree_description/meshes/g1`),
  so the same file loads on the host (start qpos, scoring) and in the rootfs (`/hostdata/...`).
  The weld adds no qpos, so the hold plugin's start vector is unchanged.
- **URDF** `$HCF_DATA/mtc/models/g1_saber.urdf`: `$HCF_DATA/mtc/g1_mujoco.urdf` (the xacro
  output of `robot.xacro robot_type:=g1 simulation:=mujoco`; byte-identical to a fresh xacro run
  on 2026-11-11) plus `saber.spec.urdf_snippet()` before `</robot>`. It is used by
  robot_state_publisher and the controller's pinocchio model. The saber is not a tracked body,
  so this only keeps their kinematics and mass consistent with the plant.
- Both go to `record.sh` as its `<urdf> <mjcf>` override (the `hcf_mtc_models` share trick), and
  the MJCF goes to `--write-start-qpos` and scoring as `--mtc-mjcf`. `mtc.json` records `plant`,
  `physics.simulator` (the MJCF path) and `plant_urdf`.

**Numerical equivalence with mjlab** (`tests/test_mtc_plant.py`, run in the gate; it builds the
plant fresh and compares it with the newest `*_tip` export's `scene.xml`. On 2026-11-11 that was
parry_tierce_high_v3_tip):

| quantity | MTC plant | mjlab scene.xml | difference |
|---|---|---|---|
| robot mass (pelvis subtree) | 33.541142 kg | 33.541142 kg | 0 (plain g1.xml: 33.341142) |
| saber mass, COM in saber frame | 0.2 kg, (0.23225, 0, 0) m | same | 0 |
| saber inertia (diagonal, about COM) | 5.2325e-05, 1.39455e-02, 1.39328e-02 kg m² | same | 0 |
| mount in wrist frame | pos (0.1, 0, 0), quat (0.965926, 0, -0.258819, 0) | same | 0 |
| blade tip in wrist frame | (0.810141, 0, 0.410000) m | same | 0 (1.3e-7 m from `spec.tip_in_hand`: 6-digit quaternion in both XMLs) |
| saber collision geoms | none | none | equal |
| URDF blade tip in wrist frame | (0.810141, 0, 0.410000) m | | 2.9e-7 m (printed precision) |

The test asserts 1e-6 for the MJCF quantities and the URDF's printed precision (1e-4 m for the
tip, 1e-4 kg, 1e-5 relative inertia) for the URDF. The G1 itself was compared once by hand: every
body's mass, inertia, COM and pose agree exactly between `g1.xml` and the mjlab scene.

**What differs from mjlab.** Nothing on the saber. The plant differences are the ones that
already apply to plain-G1 runs: MTC steps at 2 ms with MuJoCo's Euler integrator, mjlab at 5 ms with
implicitfast; `g1.xml` has 0.1 N m joint frictionloss where the mjlab scene has 0; armature
agrees to 2.5e-8. Training randomized the saber mass (0.12-0.35 kg, `MASS_RANDOMIZATION_KG`). MTC,
like `hcf sim2sim`, uses the nominal 0.20 kg in the scene.

**Blade tip.** Scoring writes the recorded states into the export's mjlab scene, so the
blade-tip error comes from the same `rollout.TrackingScorer` code as `blade_tip_error_m` in
`sim2sim.json`, with the same 0.10 m threshold from `fencing.train.thresholds`.

### Results, saber policies (2026-11-11, start hold on, settle 0)

*Scored before the one-tick reference lag fix* (next section). Most of the blade-tip gap
described under "Agreement" was that lag.

One run each, model_2999, newest matching eval dir. Python rows come from each dir's
`sim2sim.json`. Thresholds: joint ≤ 0.10 rad, body ≤ 5 cm, blade tip ≤ 10 cm, anchor ≤ 25 cm,
slip ≤ 0.10 m/s, success ≥ 0.95, recovery ≥ 0.90. No `advance_lunge_cut_v4_tip`, `lunge_v4_pipe` or
`advance_lunge_v4_pipe` eval dirs existed when these ran.

| motion (eval dir) | source | joint (rad) | body (cm) | blade tip (cm) | anchor (cm) | slip (m/s) | success | recovery | verdict |
|---|---|---|---|---|---|---|---|---|---|
| extension (`2026-11-10_06-14-28_extension_v3_tip/model_2999`) | MTC | 0.075 | 4.08 | 1.77 | 2.36 | 0.037 | 1 | 1 | PASS |
| | Python | 0.074 | 4.17 | 1.32 | 2.19 | 0.041 | 1 | 1 | PASS |
| parry_prime (`2026-11-10_10-09-18_parry_prime_v3_tip/model_2999`) | MTC | 0.075 | 4.00 | 2.55 | 2.07 | 0.035 | 1 | 1 | PASS |
| | Python | 0.075 | 4.20 | 1.51 | 1.89 | 0.047 | 1 | 1 | PASS |
| parry_seconde (`2026-11-10_10-54-06_parry_seconde_v3_tip/model_2999`) | MTC | 0.091 | 4.26 | 3.26 | 1.96 | 0.042 | 1 | 1 | PASS |
| | Python | 0.091 | 4.44 | 2.53 | 2.59 | 0.045 | 1 | 1 | PASS |
| parry_tierce_high (`2026-11-10_12-26-40_parry_tierce_high_v3_tip/model_2999`) | MTC | 0.089 | 3.76 | 1.48 | 2.70 | 0.042 | 1 | 1 | PASS |
| | Python | 0.090 | 3.97 | 1.11 | 2.56 | 0.038 | 1 | 1 | PASS |
| parry_quinte (`2026-11-10_07-03-50_parry_quinte_v3_tip/model_2999`) | MTC run 1 | 0.073 | 4.66 | 2.00 | 2.58 | 0.029 | 1 | 1 | PASS |
| | MTC run 2 | 0.073 | 4.64 | 2.00 | 2.59 | 0.030 | 1 | 1 | PASS |
| | MTC run 3 | 0.073 | 4.65 | 2.00 | 2.59 | 0.031 | 1 | 1 | PASS |
| | Python | 0.072 | 4.57 | 1.58 | 2.82 | 0.032 | 1 | 1 | PASS |
| parry_sixte (`2026-11-10_11-42-25_parry_sixte_v3_tip/model_2999`) | MTC | 0.074 | 4.15 | 1.09 | 3.25 | 0.036 | 1 | 1 | PASS |
| | Python | 0.073 | 4.23 | 1.11 | 3.30 | 0.035 | 1 | 1 | PASS |
| cut_head (`2026-11-10_05-31-03_cut_head_v3_tip/model_2999`) | MTC | 0.085 | 3.82 | 3.48 | 2.02 | 0.045 | 1 | 1 | PASS |
| | Python | 0.084 | 3.92 | 2.63 | 2.26 | 0.046 | 1 | 1 | PASS |
| cut_flank (`2026-11-10_07-47-56_cut_flank_v3_tip/model_2999`) | MTC | 0.069 | 3.98 | 2.42 | 1.82 | 0.045 | 1 | 1 | PASS |
| | Python | 0.068 | 4.24 | 1.75 | 2.38 | 0.048 | 1 | 1 | PASS |
| cut_chest (`2026-11-10_08-36-30_cut_chest_v3_tip/model_2999`) | MTC | 0.077 | 3.79 | 2.37 | 2.08 | 0.043 | 1 | 1 | PASS |
| | Python | 0.078 | 3.99 | 1.58 | 2.17 | 0.045 | 1 | 1 | PASS |
| thrust (`2026-11-10_09-21-15_thrust_v3_tip/model_2999`) | MTC | 0.079 | 4.24 | 1.83 | 1.62 | 0.041 | 1 | 1 | PASS |
| | Python | 0.079 | 4.59 | 1.69 | 2.32 | 0.047 | 1 | 1 | PASS |
| advance_lunge_cut (`2026-11-10_03-14-21_advance_lunge_cut_v3_tip/model_2999`) | MTC run 1 | 0.114 | 3.74 | 2.30 | 5.89 | 0.015 | 1 | 1 | FAIL |
| | MTC run 2 | 0.113 | 3.64 | 2.21 | 6.04 | 0.015 | 1 | 1 | FAIL |
| | MTC run 3 | 0.114 | 3.67 | 2.25 | 6.15 | 0.016 | 1 | 1 | FAIL |
| | Python | 0.109 | 3.57 | 1.81 | 5.40 | 0.015 | 1 | 1 | FAIL |

**Agreement.** All eleven verdicts agree (ten PASS; advance_lunge_cut FAILs joint error in
both, 0.114 vs 0.109). Joint error is within 0.002 rad of Python on ten motions and 0.005 on
advance_lunge_cut. Body error is 0.08-0.35 cm lower in MTC on nine motions and higher on
parry_quinte (+0.09) and advance_lunge_cut (+0.17). The blade tip moves most: it is 0.1-1.0 cm
*higher* in MTC on ten motions (parry_sixte equal), still well under the 10 cm threshold. That pattern is measured; its cause is not established. Candidates are the
plant differences above (a 0.82 m lever arm magnifies small wrist-orientation differences) and
the estimator-based anchor observation. None was isolated.

**Determinism with the saber plant.** Three runs each, settle 0:

| motion | run | activation (sim s) | joint (rad) | body (cm) | blade tip (cm) | anchor (cm) | verdict |
|---|---|---|---|---|---|---|---|
| parry_quinte_v3_tip | 1 | 1.266 | 0.07259 | 4.657 | 2.000 | 2.579 | PASS |
| parry_quinte_v3_tip | 2 | 0.228 | 0.07264 | 4.643 | 1.998 | 2.593 | PASS |
| parry_quinte_v3_tip | 3 | 0.276 | 0.07270 | 4.645 | 1.997 | 2.586 | PASS |
| advance_lunge_cut_v3_tip | 1 | 0.228 | 0.11411 | 3.738 | 2.303 | 5.892 | FAIL |
| advance_lunge_cut_v3_tip | 2 | 0.356 | 0.11265 | 3.640 | 2.213 | 6.037 | FAIL |
| advance_lunge_cut_v3_tip | 3 | 0.302 | 0.11389 | 3.669 | 2.247 | 6.155 | FAIL |

parry_quinte spreads by 1.1e-4 rad joint and 0.03 mm tip, the same size as the plain-G1 runs.
advance_lunge_cut (7 s, travelling, lunge) spreads by 1.5e-3 rad joint, 0.9 mm tip and 2.6 mm
anchor. That is ten times more, but still far from changing a verdict. These are not like for like with the plain-G1
determinism table, which used the 1.0 s settle. The likely source is the same tick-phase
effect, amplified over a longer, dynamic motion (not verified).

## One-tick reference lag (scorer fixed 2026-11-11)

**Symptom.** At the key frame, fast cuts scored much worse in MTC than in Python:
cut_head_v3_tip 11.2 cm against 4.2, cut_chest_v3_tip 6.1 against 2.6. That failed the 5 cm key
gate. Slow and held targets agreed.

**Cause: the controller observes the previous frame.** `MotionOnnxPolicy::forward`
(third_party/motion_tracking_controller, src/MotionOnnxPolicy.cpp) runs the ONNX with
`time_step = timeStep_++` and stores the reference it returns. The observation for a tick is
assembled before that tick's `forward(observation)`, from the stored reference
(`MotionCommandTerm::getValue`, `getAnchorPositionLocal`, ...). `reset()` runs one forward and
rewinds, which leaves frame 0 stored. So tick k observes frame max(k - 1, 0). The controller
holds frame 0 for two ticks and then runs the whole motion exactly one tick (20 ms) behind
activation. mjlab and `hcf sim2sim` observe frame k at step k. The scorer used to pair tick k
with frame k, which scored MTC 20 ms late. (The order inside `RlController::step` is compiled,
not read. But `forward` takes the observation as its argument, so the observation exists
before the new frame is stored.)

**Evidence: shift against error, existing recordings.** The recorded state was compared with
the reference shifted by s steps (s > 0: robot late). Whole steps use the reference index.
Fractional steps use the 2 ms MTC samples and the 5 ms physics substeps of the Python rollout.
Same `TrackingScorer` throughout. Values: blade tip at the key / whole-motion blade tip (cm).

| s (steps late) | -0.25 | 0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| cut_head, Python | 5.48 / **2.48** | 4.15 / 2.63 | 3.15 / 3.42 | 3.68 / 3.89 | 4.43 / 4.39 | 5.17 / 4.91 | 5.82 / 5.42 | |
| cut_head, MTC | 13.04 / 3.96 | 11.23 / 3.48 | 7.28 / 2.63 | 5.65 / **2.47** | 4.20 / 2.65 | 3.56 / 3.03 | 3.22 / 3.43 | 4.94 / 4.39 |
| cut_chest, Python | 3.18 / **1.44** | 2.59 / 1.58 | 3.78 / 2.30 | 4.51 / 2.69 | 5.14 / 3.09 | 5.67 / 3.50 | 6.09 / 3.90 | |
| cut_chest, MTC | 8.07 / 2.81 | 6.10 / 2.34 | 3.55 / 1.60 | **2.74 / 1.44** | 1.81 / 1.61 | 1.91 / 1.90 | 2.80 / 2.32 | 4.11 / 3.10 |

On this 0.25-step grid the MTC curve is the Python curve moved by one step. The whole-motion
minima are equal (2.48 / 2.47, 1.44 / 1.44) and sit at the grid points -0.25 (Python) and +0.75
(MTC), a 1.0-step difference at 0.25-step resolution. Joint error, a mean over 29 joints and the
whole motion, changes by less than 0.001 rad over this range. Sword-arm joints (right shoulder,
elbow, wrist) around the cut_head key: MTC at tick k+1 differs from Python's step k by
0.006-0.010 rad mean, and MTC at tick k by 0.02-0.11 rad. After the one-tick shift, any further
offset is smaller than this grid resolves. Whatever the estimator, overshoot and plant
differences add to the cut gap is below what these two motions at 0.25-step resolution can show.

The MTC rows used the nearest recorded sample. Where a sample was dropped at a tick, two
neighbours were equally near (see "Sample interpolation"). cut_head's 4.20 at s = 1.0 matches
the nearest-sample rule with ties going to the later neighbour. Recomputed that way it is 4.21;
the scorer broke ties the other way and got 4.67. I did not have the shift study's code, so its
tie rule is inferred from this match.

**What it means for hardware.** Tracking is unaffected: against the frame it was given, the
deployed controller tracks the cuts as well as Python does. What changes is the timeline. Every
frame, the hit included, arrives one policy tick (20 ms) later than activation + frame × 20 ms.
The motion starts one tick late; it does not run slow. That matters only if something is
synchronised to the controller's activation time. These simulators do not model the real
robot's sensing and communication latency on top of this.

**Fix.** `fencing.sim2sim.mtc` scores step k from the tick whose observation carries frame k
(`CONTROLLER_REFERENCE_LAG_STEPS = 1`, `scored_step_times`; test
`test_step_k_is_scored_from_the_tick_whose_observation_carries_reference_frame_k` models the C++
mechanism). The controller is unchanged (read-only). All 19 `mtc.json` files under
`$HCF_DATA/eval` were backed up to `$HCF_DATA/mtc/backup_mtc_json_pre_lag_fix/`. 18 of them
were re-scored from their own `bag.npz` with their recorded settings. dance1 has no bag (below),
so its backup is identical to its current file.

cut_head's key error after the fix (4.7 cm at the time, 4.4 cm now) differs from the shift
table's s = 1.0 value (4.20). That is the same recording. The cause is the tie-break at a
dropped sample in the key window, now fixed ("Sample interpolation"). The Results tables further
up differ from the "before" columns below by more than a tie-break can explain. For example,
extension's body error is 4.08 there and 4.18 here, both at settle 0. They look like earlier
recordings or an earlier scorer version; which of the two was not checked.

Bladework, before → after (Python `sim2sim.json`). "After" is the current scorer (lag fix plus
sample interpolation, re-scored 2026-11-11); the interpolation's own share is in "Sample
interpolation". Key and tip in cm, re-anchored unless marked:

| motion | key | key global | whole tip | joint (rad) | verdict |
|---|---|---|---|---|---|
| cut_head_v3 | 11.2 → 4.4 (4.2) | 13.7 → 6.8 (5.0) | 3.48 → 2.66 (2.63) | 0.0839 → 0.0837 (0.0841) | FAIL → PASS |
| cut_chest_v3 | 6.1 → 1.9 (2.6) | 13.8 → 8.0 (7.7) | 2.34 → 1.61 (1.58) | 0.0775 → 0.0773 (0.0778) | FAIL → PASS |
| cut_flank_v3 | 3.5 → 2.3 (2.7) | 4.6 → 3.1 (3.4) | 2.51 → 2.04 (1.75) | 0.0681 → 0.0681 (0.0684) | PASS |
| parry_seconde_v3 | 4.7 → 2.8 (2.5) | 5.6 → 3.5 (3.9) | 3.21 → 2.51 (2.53) | 0.0900 → 0.0895 (0.0905) | PASS |
| parry_prime_v3 | 3.0 → 2.0 (1.6) | 3.4 → 2.6 (1.3) | 2.59 → 1.76 (1.51) | 0.0743 → 0.0741 (0.0750) | PASS |
| thrust_v3 | 2.4 → 1.9 (3.3) | 4.7 → 3.6 (4.8) | 1.78 → 1.54 (1.69) | 0.0784 → 0.0781 (0.0791) | PASS |
| extension_v3 | 1.4 → 1.5 (1.1) | 0.9 → 0.8 (2.0) | 1.74 → 1.53 (1.32) | 0.0746 → 0.0743 (0.0745) | PASS |
| parry_sixte_v3 | 1.5 → 1.6 (1.6) | 1.3 → 1.4 (2.4) | 1.11 → 1.18 (1.11) | 0.0723 → 0.0722 (0.0729) | PASS |
| parry_quinte_v3 | 1.5 → 1.5 (2.0) | 4.5 → 4.2 (4.0) | 2.01 → 1.70 (1.58) | 0.0725 → 0.0727 (0.0724) | PASS |
| parry_tierce_high_v3 | 2.0 → 1.7 (1.6) | 2.0 → 1.8 (2.1) | 1.50 → 1.12 (1.11) | 0.0896 → 0.0892 (0.0898) | PASS |
| advance_lunge_cut_v4 | 2.7 → 1.8 (1.8) | 16.4 → 17.3 (14.2) | 2.65 → 2.41 (2.05) | 0.1023 → 0.1023 (0.0964) | FAIL (joint) |

After the fix every key error is within 1.4 cm of Python, with differences in both directions;
before, the fast cuts were up to 7 cm worse. The whole-motion tip is now -0.15 to +0.36 cm
from Python (higher on 9 of 11), against 0.0 to +1.1 cm before. That remainder is the unexplained part of the old
"blade tip higher in MTC" pattern. advance_lunge_cut_v3_tip (no key frames) moves from 2.29 to
2.06 cm tip and stays FAIL on joint error (0.1134).

Plain G1, before → after (current scorer, as above) (Python):

| motion | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | verdict |
|---|---|---|---|---|---|
| en_garde_v3 | 0.0756 → 0.0758 (0.0772) | 3.73 → 3.75 (3.79) | 2.8 → 2.8 (2.3) | 0.048 → 0.045 (0.049) | PASS |
| advance_v1 | 0.0785 → 0.0790 (0.0776) | 3.19 → 3.18 (3.10) | 3.5 → 3.3 (3.3) | 0.096 → 0.080 (0.082) | PASS |
| retreat_v1 | 0.0856 → 0.0862 (0.0866) | 2.85 → 2.89 (2.94) | 2.0 → 2.0 (1.9) | 0.045 → 0.036 (0.044) | PASS |
| lunge_v3 | 0.1004 → 0.0996 (0.0992) | 3.79 → 3.74 (3.75) | 13.5 → 13.5 (11.8) | 0.066 → 0.066 (0.073) | FAIL → PASS |
| advance_lunge_v3 | 0.0972 → 0.0965 (0.0979) | 3.58 → 3.58 (3.72) | 9.5 → 9.4 (5.8) | 0.026 → 0.018 (0.019) | PASS |
| advance_lunge_v4 | 0.0804 → 0.0797 (0.0805) | 3.09 → 3.06 (3.08) | 7.0 → 6.8 (4.8) | 0.017 → 0.014 (0.017) | PASS |

Joint error moves by at most 0.0008 rad and body error by at most 0.05 cm. The only verdict
change is lunge_v3, the threshold-edge case under "Disagreements". It moves across 0.100 and
now agrees with Python. dance1 could not be re-scored: its run dir is empty and its `mtc.json`
is an INVALID later attempt. On an older unheld dance1 bag (`$HCF_DATA/mtc/rec_dance1`, 1.0 s
settle, too coarse to be valid), the fix moves joint error from 0.1006 to 0.0995, body error
from 3.70 to 3.61 cm, and leaves anchor at 11.9 cm. That figure is from the nearest-sample
scorer; the bag is not under an eval dir and was not re-scored with interpolation.

## Sample interpolation (scorer fixed 2026-11-11)

**Symptom.** The same bag could give different key-frame errors. cut_head_v3_tip's key error
was 4.67 cm in `mtc.json` and 4.20 cm in the shift study at s = 1.0, against a 5 cm gate.

**Cause: ties at dropped samples.** The scorer took the recorded sample nearest each tick.
rosbag drops `/joint_states` messages under CPU load. Where the sample at a tick is missing, the
samples 2 ms before and after are equally near, and float rounding in the tick time picked one.
On cut_head that happened at 18 of the 281 scored ticks, one of them (step 39) in the ±2-step
key window. The `mid360` pelvis track had the same flaw one level down. Its forward kinematics
used the nearest joint sample, so at a dropped joint sample it combined the exact site pose
with joint angles 2 ms off. Rescoring the cut_head bag with the nearest-sample rule gives a key
error of 4.67 cm when ties go to the earlier sample and 4.21 cm when they go to the later one.

**Fix.** `fencing.sim2sim.mtc` interpolates instead (`interpolate_samples`,
`interpolate_quats`, used by `Replayer.set_state` and `Replayer.base_track`). Joint positions
and velocities and the pelvis position and velocities are linear in time. Orientation uses nlerp
with the sign aligned; over 2 ms that is within ~1e-9 of slerp. At a tick with a real sample the
state is that sample, as before. At a dropped one it is the midpoint, the same whatever the
rounding. The INVALID limit (half a policy step) still applies to the distance from the tick to
the nearest real sample, so interpolation cannot bridge a run of dropped messages. Tests are in
`tests/test_mtc_replay.py` (tie, linear signal, dropped run, quaternion sign, a replayed
dropped sample with an exact pelvis pose).

**Re-score.** The 19 `mtc.json` were backed up to
`$HCF_DATA/mtc/backup_mtc_json_pre_interp/`. The 18 with a bag were re-scored from it with
their recorded settings (all settle 0, hold 2 s, untruncated), using the normal scoring CLI
(`python -m fencing.sim2sim.mtc <export dir> --recording <bag.npz> --extra <run>/extra.json`,
plus `--mtc-mjcf` for saber runs). dance1 has no bag and was not re-scored. Nearest →
interpolated (Python `sim2sim.json`); tip and key in cm, re-anchored:

| motion | joint (rad) | body (cm) | whole tip | key | slip (m/s) | verdict |
|---|---|---|---|---|---|---|
| cut_head_v3 | 0.0837 → 0.0837 (0.0841) | 3.81 → 3.81 (3.92) | 2.67 → 2.66 (2.63) | 4.67 → 4.43 (4.15) | 0.043 → 0.042 (0.046) | PASS |
| cut_chest_v3 | 0.0773 → 0.0773 (0.0778) | 3.91 → 3.91 (3.99) | 1.61 → 1.61 (1.58) | 1.81 → 1.85 (2.59) | 0.043 → 0.042 (0.045) | PASS |
| cut_flank_v3 | 0.0681 → 0.0681 (0.0684) | 4.07 → 4.07 (4.24) | 2.04 → 2.04 (1.75) | 2.28 → 2.28 (2.69) | 0.044 → 0.042 (0.048) | PASS |
| parry_seconde_v3 | 0.0895 → 0.0895 (0.0905) | 4.38 → 4.38 (4.44) | 2.50 → 2.51 (2.53) | 2.76 → 2.76 (2.47) | 0.041 → 0.040 (0.045) | PASS |
| parry_prime_v3 | 0.0741 → 0.0741 (0.0750) | 4.11 → 4.11 (4.20) | 1.76 → 1.76 (1.51) | 1.98 → 1.98 (1.62) | 0.038 → 0.038 (0.047) | PASS |
| thrust_v3 | 0.0781 → 0.0781 (0.0791) | 4.43 → 4.43 (4.59) | 1.54 → 1.54 (1.69) | 1.87 → 1.87 (3.27) | 0.057 → 0.043 (0.047) | PASS |
| extension_v3 | 0.0743 → 0.0743 (0.0745) | 4.18 → 4.18 (4.17) | 1.54 → 1.53 (1.32) | 1.50 → 1.50 (1.10) | 0.043 → 0.039 (0.041) | PASS |
| parry_sixte_v3 | 0.0722 → 0.0722 (0.0729) | 4.06 → 4.06 (4.23) | 1.18 → 1.18 (1.11) | 1.57 → 1.57 (1.64) | 0.037 → 0.034 (0.035) | PASS |
| parry_quinte_v3 | 0.0727 → 0.0727 (0.0724) | 4.64 → 4.64 (4.57) | 1.70 → 1.70 (1.58) | 1.46 → 1.46 (1.99) | 0.032 → 0.032 (0.032) | PASS |
| parry_tierce_high_v3 | 0.0893 → 0.0892 (0.0898) | 3.82 → 3.82 (3.97) | 1.12 → 1.12 (1.11) | 1.70 → 1.69 (1.64) | 0.039 → 0.036 (0.038) | PASS |
| advance_lunge_cut_v4 | 0.1023 → 0.1023 (0.0964) | 3.26 → 3.26 (3.04) | 2.41 → 2.41 (2.05) | 1.80 → 1.80 (1.84) | 0.030 → 0.019 (0.023) | FAIL (joint) |
| advance_lunge_cut_v3 | 0.1134 → 0.1134 (0.1089) | 3.68 → 3.68 (3.57) | 2.06 → 2.06 (1.81) | - | 0.014 → 0.014 (0.015) | FAIL (joint) |
| en_garde_v3 | 0.0758 → 0.0758 (0.0772) | 3.75 → 3.75 (3.79) | - | - | 0.048 → 0.045 (0.049) | PASS |
| advance_v1 | 0.0790 → 0.0790 (0.0776) | 3.18 → 3.18 (3.10) | - | - | 0.086 → 0.080 (0.082) | PASS |
| retreat_v1 | 0.0863 → 0.0862 (0.0866) | 2.89 → 2.89 (2.94) | - | - | 0.037 → 0.036 (0.044) | PASS |
| lunge_v3 | 0.0996 → 0.0996 (0.0992) | 3.74 → 3.74 (3.75) | - | - | 0.070 → 0.066 (0.073) | PASS |
| advance_lunge_v3 | 0.0966 → 0.0965 (0.0979) | 3.58 → 3.58 (3.72) | - | - | 0.024 → 0.018 (0.019) | PASS |
| advance_lunge_v4 | 0.0797 → 0.0797 (0.0805) | 3.06 → 3.06 (3.08) | - | - | 0.016 → 0.014 (0.017) | PASS |

No verdict changes. Joint error moves by at most 2.3e-5 rad (the 0.0001 steps in the table
are rounding), body error by at most 0.002 cm, and whole-motion tip by at most 0.011 cm. The
cut_head key error lands between the two tie-break
values (4.43), 0.28 cm from Python. Anchor error and the timing check
(`max_state_time_mismatch_s`) are unchanged on every run.

Foot slip moves most, always down, by up to 0.014 m/s (thrust 0.057 → 0.043, now closer to
Python's 0.047). Slip is computed from body velocities, and the pelvis velocity is a finite
difference of the pelvis track. A pelvis pose built with joint angles 2 ms off was displaced by
up to 0.3 mm (thrust), and differencing that over 4 ms gave pelvis velocity spikes of up to
0.12 m/s, which the slip metric picked up. Measured on thrust and cut_head (0.04 m/s there); the
other motions are expected to behave the same way but were not checked one by one.

## Goal 1 final set (2026-11-11)

The newest passing policy of every Goal 1 motion, in all three simulators. Generated from each eval dir's `metrics.json` (mjlab, 1024 episodes), `sim2sim.json` (CPU MuJoCo) and `mtc.json` (this controller; one-tick lag and interpolated sampling applied, start hold on, settle 0). Each cell is mjlab / Python sim2sim / MTC. The copies are in [`results/`](../results/SUMMARY.md).

| motion (run) | verdicts | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | recovery | blade tip (cm) | tip at key (cm) |
|---|---|---|---|---|---|---|---|---|
| en_garde_v3 (`2026-11-10_04-44-31_en_garde_v3_pipe`) | PASS / PASS / PASS | 0.077 / 0.077 / 0.076 | 3.8 / 3.8 / 3.8 | 2.9 / 2.3 / 2.8 | 0.044 / 0.049 / 0.045 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| advance_v1 (`2026-11-09_19-10-15_advance_v1_pipe`) | PASS / PASS / PASS | 0.079 / 0.078 / 0.079 | 3.2 / 3.1 / 3.2 | 3.3 / 3.3 / 3.3 | 0.096 / 0.082 / 0.080 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| retreat_v1 (`2026-11-09_20-40-56_retreat_v1_pipe`) | PASS / PASS / PASS | 0.088 / 0.087 / 0.086 | 3.0 / 2.9 / 2.9 | 2.4 / 1.9 / 2.0 | 0.051 / 0.044 / 0.036 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| extension_v3 (`2026-11-10_06-14-28_extension_v3_tip`) | PASS / PASS / PASS | 0.075 / 0.074 / 0.074 | 4.2 / 4.2 / 4.2 | 2.5 / 2.2 / 2.4 | 0.036 / 0.041 / 0.039 | 1.00 / 1.00 / 1.00 | 1.4 / 1.3 / 1.5 | 1.1 / 1.1 / 1.5 |
| lunge_v5 (`2026-11-11_11-15-56_lunge_v5_pipe`) | PASS / PASS / PASS | 0.087 / 0.087 / 0.088 | 4.0 / 4.0 / 3.9 | 6.3 / 5.2 / 5.8 | 0.054 / 0.051 / 0.046 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| parry_prime_v3 (`2026-11-10_10-09-18_parry_prime_v3_tip`) | PASS / PASS / PASS | 0.075 / 0.075 / 0.074 | 4.2 / 4.2 / 4.1 | 2.4 / 1.9 / 2.1 | 0.039 / 0.047 / 0.038 | 1.00 / 1.00 / 1.00 | 1.7 / 1.5 / 1.8 | 1.5 / 1.6 / 2.0 |
| parry_seconde_v3 (`2026-11-10_10-54-06_parry_seconde_v3_tip`) | PASS / PASS / PASS | 0.091 / 0.091 / 0.089 | 4.5 / 4.4 / 4.4 | 2.6 / 2.6 / 2.0 | 0.042 / 0.045 / 0.040 | 1.00 / 1.00 / 1.00 | 2.6 / 2.5 / 2.5 | 2.2 / 2.5 / 2.8 |
| parry_tierce_high_v3 (`2026-11-10_12-26-40_parry_tierce_high_v3_tip`) | PASS / PASS / PASS | 0.090 / 0.090 / 0.089 | 4.0 / 4.0 / 3.8 | 2.8 / 2.6 / 2.8 | 0.039 / 0.038 / 0.036 | 1.00 / 1.00 / 1.00 | 1.2 / 1.1 / 1.1 | 1.4 / 1.6 / 1.7 |
| parry_quinte_v3 (`2026-11-10_07-03-50_parry_quinte_v3_tip`) | PASS / PASS / PASS | 0.072 / 0.072 / 0.073 | 4.5 / 4.6 / 4.6 | 3.2 / 2.8 / 2.7 | 0.033 / 0.032 / 0.032 | 1.00 / 1.00 / 1.00 | 1.7 / 1.6 / 1.7 | 2.4 / 2.0 / 1.5 |
| parry_sixte_v3 (`2026-11-10_11-42-25_parry_sixte_v3_tip`) | PASS / PASS / PASS | 0.073 / 0.073 / 0.072 | 4.2 / 4.2 / 4.1 | 3.7 / 3.3 / 3.5 | 0.037 / 0.035 / 0.034 | 1.00 / 1.00 / 1.00 | 1.2 / 1.1 / 1.2 | 1.6 / 1.6 / 1.6 |
| cut_head_v4 (`2026-11-11_13-49-44_cut_head_v4_tip`) | PASS / PASS / PASS | 0.089 / 0.088 / 0.088 | 4.5 / 4.5 / 4.5 | 2.2 / 1.8 / 1.6 | 0.047 / 0.049 / 0.042 | 1.00 / 1.00 / 1.00 | 2.5 / 2.4 / 2.4 | 2.1 / 2.0 / 2.9 |
| cut_flank_v3 (`2026-11-10_07-47-56_cut_flank_v3_tip`) | PASS / PASS / PASS | 0.068 / 0.068 / 0.068 | 4.2 / 4.2 / 4.1 | 2.4 / 2.4 / 1.9 | 0.046 / 0.048 / 0.042 | 1.00 / 1.00 / 1.00 | 1.9 / 1.8 / 2.0 | 2.5 / 2.7 / 2.3 |
| cut_chest_v3 (`2026-11-10_08-36-30_cut_chest_v3_tip`) | PASS / PASS / PASS | 0.078 / 0.078 / 0.077 | 4.0 / 4.0 / 3.9 | 2.3 / 2.2 / 1.9 | 0.044 / 0.045 / 0.042 | 1.00 / 1.00 / 1.00 | 1.7 / 1.6 / 1.6 | 3.2 / 2.6 / 1.9 |
| thrust_v3 (`2026-11-10_09-21-15_thrust_v3_tip`) | PASS / PASS / PASS | 0.080 / 0.079 / 0.078 | 4.6 / 4.6 / 4.4 | 2.7 / 2.3 / 2.0 | 0.044 / 0.047 / 0.043 | 1.00 / 1.00 / 1.00 | 1.8 / 1.7 / 1.5 | 2.8 / 3.3 / 1.9 |
| advance_lunge_v4 (`2026-11-11_09-00-26_advance_lunge_v4_pipe`) | PASS / PASS / PASS | 0.081 / 0.081 / 0.080 | 3.2 / 3.1 / 3.1 | 5.6 / 4.8 / 6.8 | 0.021 / 0.017 / 0.014 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| advance_lunge_cut_v5 (`2026-11-11_11-15-05_advance_lunge_cut_v5_tip`) | PASS / PASS / PASS | 0.076 / 0.074 / 0.076 | 3.1 / 3.0 / 3.1 | 7.3 / 6.5 / 8.0 | 0.025 / 0.023 / 0.020 | 1.00 / 1.00 / 1.00 | 1.9 / 1.8 / 2.0 | 1.9 / 2.1 / 1.5 |

cut_head_v4 replaced cut_head_v3 (`2026-11-10_05-31-03_cut_head_v3_tip`: joint 0.085 / 0.084 / 0.084, body 4.0 / 3.9 / 3.8 cm, tip at key 4.1 / 4.2 / 4.4 cm) on 2026-11-11. v3 passed here but failed the key-frame gate in PhysX (6.1 cm); v4's attack move is 0.3 s instead of 0.2 s. The sections above that analyse cut_head (lag, interpolation, standby) were measured on v3.

Footwork motions have no blade, so their blade columns are `–`. Each policy also has a rendered CPU-MuJoCo rollout, `sim2sim.mp4`, next to its scores (robot in white, reference as a green ghost).

## Start from standby (2026-11-11)

Every result above starts held in the motion's first frame. The real robot does not start
there. `real.launch.py` brings up `standby_controller` active and the policy inactive, and the
operator switches in the policy with `R1 + A`. This section runs that start in this simulator
for the 16 final policies.

### Method

```bash
scripts/mtc_sim2sim.sh ~/hcf-data/eval/<run>/model_2999 --standby [--switch-after S]   # default S = 10
#   writes model_2999/mtc_standby.json, records under $HCF_DATA/mtc/runs_standby/<name>/
```

- **Controllers as on the robot.** The launch arg `start_controller:=standby_controller`
  (`scripts/mtc/mujoco_custom.launch.py`) spawns `state_estimator` and `standby_controller`
  active and `walking_controller` inactive, as `real.launch.py` lines 54-59 do. The
  controller config is the same file the real launch uses (MTC's `config/g1/controllers.yaml`:
  standby `default_position`, kp, kd) and the controller type is the same
  (`legged_controllers/StandbyController`).
- **Getting the robot onto its feet.** The robot is held still in `unitree_description`'s
  default pose (all joints 0), lowered onto the floor (`standing_qpos`,
  `src/fencing/sim2sim/mtc_standby.py`). The existing start hold (`hcf_mtc_hold`) releases it
  at standby's first torque command. Standby then ramps to its pose. Its header has
  `totalTime_{2.}`, so the ramp presumably takes 2 s (the controller is closed source). Why
  not hold it already in the standby pose: standby's torque would be exactly 0 there, and the
  hold releases only on a non-zero command.
- **The switch is the operator's button.** `scripts/mtc/switch_after.py` waits until standby
  has been active for S seconds of sim time. Then it calls
  `/controller_manager/switch_controller` with the request that `unitree_bringup`'s
  `config/g1/joy.yaml` binds to `R1 + A`:
  `activate [walking_controller]`, `deactivate [standby_controller, walking_controller1]`,
  `strictness 1`, `activate_asap true`. Every run's controller manager answered "Deactivated
  [standby_controller state_estimator] / Activated [walking_controller state_estimator]". It
  warned twice that there is no `walking_controller1`, a name in upstream's binding that the
  MTC launch does not create. **The switch restarts `state_estimator`.** At the first policy
  tick the estimate was 0.008 m/s off in speed and 0.2 cm off in height in every run.
- **No start hold on the policy.** The hold acts only before standby. The policy starts on
  whatever standby left: the reference is placed on the torso at activation, as on the robot.
- **S = 10 s.** A pilot with S = 5 showed the torso still swaying fore and aft after the
  ramp. The sway decayed from ±21 to ±9 to ±2 mm per 0.5 s, so it took about 5 s to settle.
- **Proof the robot was standing still, or the run does not count.** The scorer checks the
  last 1.0 s before the switch. The window ends 10 ms early because central differences would
  otherwise pick up motion from after the switch. Limits: standby active for at least that
  window plus the 2 s ramp; pelvis speed under 0.01 m/s; pelvis height range under 2 mm; tilt
  under 0.2 rad; every joint slower than 0.05 rad/s. Standby activity comes from
  `/controller_manager/activity` (the `activity` array in `bag.npz`). A run that fails this
  is INVALID (exit 3), the same as a dropped-message run, and `mtc_sim2sim.sh` records it
  again. In all 16 counted runs: standby active 10.00 s; pelvis 0.749 m (plain G1) or 0.750 m
  (saber plant); height range ≤ 1.5 µm; pelvis speed ≤ 0.1 mm/s; joint speed ≤ 6·10⁻⁵ rad/s;
  tilt 0.086-0.091 rad. thrust_v3 needed a second attempt (attempt 1 INVALID: dropped
  messages). Every other policy counted on attempt 1.
- **What is measured** (`mtc_standby.json`, `standby`):
  - **Convergence:** the first policy step from which the joint error, measured as everywhere
    in this file, stays under 0.15 rad for 0.5 s (25 steps).
  - **Peaks in the first second after the switch,** as a fraction of each joint's limit:
    joint speed against the URDF `velocity` limit, and the applied torque (`/joint_states`
    effort, which the simulator clamps at the limit) against the URDF `effort` limit. Both
    limits come from the controller's own URDF. The table also gives the time each joint
    spent at the clamp.
  - **Transition:** how far the torso (anchor) moved in xy from the switch to convergence;
    the pelvis dip; how far each foot moved and lifted.
- **Scores, whole run and after convergence.** The whole run is scored exactly as `mtc.json`
  is, from the first step, with unchanged thresholds. "After convergence" rescores the same
  recording with the existing settle window (`--settle-seconds`, "Scoring conventions") set to
  the convergence step. Steps before it are not scored, and the reference is placed on the
  robot's torso there.

**Why the after-convergence score is fair.**

- It answers a different question from the whole run: does the policy track the motion once
  the switch transient is over? The transient itself is not hidden. It is reported in the
  transition table and still counts in the whole-run row.
- To show what the window alone does, the held recording is scored over the same window.
  Against its own whole-run `mtc.json`, joint error moves by at most 0.005 rad and body error
  by at most 0.12 cm. The exceptions are the two windows that skip more than 2 s: on
  parry_tierce_high and cut_head the held joint error drops by 0.013 and 0.008 rad, so the
  window does help those two. Slip and anchor are lower in every held window, because the
  window skips the start. So compare each standby after-convergence row with the held row
  over the same window, not with `mtc.json`.
- The held-recording rescores are scratch only. They were never written over `mtc.json`, and
  the held files' checksums were unchanged after every run.
- The one leniency is the re-placement. Torso displacement from the transition is forgiven in
  the anchor column. That is why the anchor displacement is reported separately.
- When the convergence step falls after a blade key frame, tip-at-key cannot be scored, and
  the verdict is PARTIAL. That applies to parry_tierce_high and cut_head, and also to their
  held windows.

### Transition

Joint error at the first policy step was 0.26-0.38 rad from standby, against 0.03-0.09 rad
from the held start. Held starts converge at step 1 (0.04 s) in all 16 runs.

| motion | stayed up | joint error at first step (rad) | converged after switch (s) | peak joint speed / limit, 1st s (held start) | peak torque / limit, 1st s | time at the torque clamp (s) | torso moved (cm) | pelvis dip (cm) | feet moved L / R (cm) | max foot lift (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| en_garde_v3 | yes | 0.27 | 0.42 | 0.15 left hip roll (0.39) | 0.55 left ankle pitch | none | 15.0 | 1.8 | 8 / 22 | 2.6 |
| advance_v1 | **fell** (terminated 0.52 s, hand/ankle height) | 0.35 | never | 0.43 left wrist yaw (0.47) | 1.00 left ankle pitch | left ankle pitch 0.04, wrists ≤ 0.05 | 94 | 62 | 42 / 128 | 48 |
| retreat_v1 | yes | 0.26 | 0.40 | 0.20 left hip yaw (0.44) | 0.92 right wrist yaw | none | 5.5 | 3.7 | 3 / 2 | 1.7 |
| lunge_v5 | yes | 0.30 | 1.48 | 0.39 right hip yaw (0.20) | 0.56 right hip pitch | none | 19.4 | 4.0 | 9 / 30 | 6.2 |
| advance_lunge_v4 | yes | 0.34 | 0.24 | 0.26 left hip roll (0.24) | 0.59 left ankle pitch | none | 2.9 | 4.3 | 6 / 26 | 7.0 |
| extension_v3 | yes | 0.30 | 0.14 | 0.54 right wrist yaw (0.33) | 1.00 right wrist pitch | right wrist pitch 0.10, yaw 0.05 | 3.7 | 0.9 | 2 / 4 | 0.6 |
| parry_sixte_v3 | yes | 0.36 | 0.80 | 0.65 right wrist yaw (0.40) | 1.00 left ankle pitch | left ankle pitch 0.10, right hip pitch 0.07, right ankle pitch 0.12, right wrist ≤ 0.08 | 29.9 | 9.3 | 15 / 38 | 13.1 |
| parry_tierce_high_v3 | yes | 0.33 | 2.20 | 0.59 right wrist pitch (0.41) | 1.00 right wrist pitch | right wrist pitch 0.17, yaw 0.06 | 11.9 | 3.7 | 2 / 20 | 2.9 |
| parry_quinte_v3 | **fell** (terminated 0.40 s, hand/ankle height) | 0.38 | never | 0.85 left knee (0.35) | 1.00 left knee | 8 joints incl. right hip pitch 0.49, right ankle pitch 0.35, waist pitch 0.30 | 88 | 68 | 60 / 55 | 48 |
| parry_prime_v3 | yes | 0.33 | 0.42 | 0.44 right wrist pitch (0.39) | 1.00 right wrist pitch | right wrist pitch 0.09, yaw 0.04 | 11.3 | 2.3 | 3 / 11 | 1.7 |
| parry_seconde_v3 | yes | 0.31 | 0.38 | 0.49 right wrist pitch (0.32) | 1.00 right wrist pitch | right wrist pitch 0.17, yaw 0.01 | 15.6 | 3.0 | 1 / 21 | 6.8 |
| thrust_v3 | yes | 0.32 | 0.18 | 0.49 right wrist yaw (0.37) | 1.00 right wrist pitch | right wrist pitch 0.09, yaw 0.03 | 2.3 | 2.1 | 3 / 2 | 0.3 |
| cut_chest_v3 | yes | 0.32 | 0.20 | 0.56 right wrist yaw (0.40) | 1.00 right wrist pitch | right wrist pitch 0.10, yaw 0.09 | 6.0 | 2.7 | 1 / 29 | 5.9 |
| cut_flank_v3 | yes | 0.35 | 0.72 | 0.51 right wrist pitch (0.43) | 1.00 right wrist pitch | right wrist pitch 0.09, yaw 0.04 | 16.9 | 2.6 | 2 / 18 | 6.4 |
| cut_head_v3 | yes | 0.34 | 2.90 | 0.55 right wrist pitch (0.43) | 1.00 right wrist pitch | right wrist pitch 0.17, yaw 0.05 | 13.1 | 2.9 | 8 / 26 | 11.0 |
| advance_lunge_cut_v5 | yes | 0.34 | 0.26 | 0.69 right wrist yaw (0.23) | 1.00 right wrist pitch | right wrist pitch 0.04, yaw 0.09 | 2.0 | 7.2 | 2 / 6 | 1.0 |

"Fell": the episode terminated (a hand or ankle more than 0.25 m off its reference height),
the pelvis dropped 62-68 cm, and the joints never converged.

- **Torque.** In the held-start recordings the clamp shows up only in the first 0.01 s, as
  the hold lets go (parry_seconde 0.11 s and cut_head 0.05 s on the right wrist pitch). Clamp
  time later than that is caused by the standby start.
- **Wrists.** Every saber policy drives the right wrist pitch and yaw into their clamp for
  0.04-0.17 s. The guard's right wrist yaw is 1.17 rad from the standby pose.
- **Legs.** The legs reach the clamp only in the two falls and in parry_sixte.
- "Torso moved", "feet moved" and the dip are measured from the switch to convergence. For
  the two falls they run to the end of the recorded motion.

### Scores

Joint, body, anchor and slip as everywhere in this file. "Held, from T s" is the held
recording scored over the same window (scratch only, see above). "Held, whole run" is
`mtc.json`. Footwork has no blade (`–`).

| motion | run, scored span | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | blade tip (cm) | tip at key (cm) | verdict |
|---|---|---|---|---|---|---|---|---|
| en_garde_v3 | standby, whole run | 0.096 | 5.06 | 12.5 | 0.099 | – | – | FAIL |
| | standby, from 0.38 s | 0.082 | 4.55 | 11.3 | 0.055 | – | – | PASS |
| | held, from 0.38 s | 0.072 | 3.87 | 1.3 | 0.002 | – | – | PASS |
| | held, whole run (`mtc.json`) | 0.076 | 3.75 | 2.8 | 0.045 | – | – | PASS |
| advance_v1 | standby, whole run | 0.317 | 18.88 | 9.7 | 0.403 | – | – | FAIL |
| | standby, after convergence | never converged | | | | | | – |
| | held, whole run (`mtc.json`) | 0.079 | 3.18 | 3.3 | 0.080 | – | – | PASS |
| retreat_v1 | standby, whole run | 0.121 | 4.54 | 4.6 | 0.037 | – | – | FAIL |
| | standby, from 0.36 s | 0.097 | 3.70 | 5.9 | 0.014 | – | – | PASS |
| | held, from 0.36 s | 0.085 | 2.95 | 2.3 | 0.007 | – | – | PASS |
| | held, whole run (`mtc.json`) | 0.086 | 2.89 | 2.0 | 0.036 | – | – | PASS |
| lunge_v5 | standby, whole run | 0.110 | 5.00 | 16.7 | 0.082 | – | – | FAIL |
| | standby, from 1.44 s | 0.090 | 4.15 | 15.4 | 0.055 | – | – | PASS |
| | held, from 1.44 s | 0.089 | 3.90 | 11.4 | 0.042 | – | – | PASS |
| | held, whole run (`mtc.json`) | 0.088 | 3.88 | 5.8 | 0.046 | – | – | PASS |
| advance_lunge_v4 | standby, whole run | 0.088 | 3.47 | 7.0 | 0.029 | – | – | PASS |
| | standby, from 0.20 s | 0.083 | 3.23 | 7.8 | 0.018 | – | – | PASS |
| | held, from 0.20 s | 0.081 | 3.11 | 6.6 | 0.014 | – | – | PASS |
| | held, whole run (`mtc.json`) | 0.080 | 3.06 | 6.8 | 0.014 | – | – | PASS |
| extension_v3 | standby, whole run | 0.100 | 6.26 | 11.1 | 0.028 | 4.17 | 4.5 | FAIL |
| | standby, from 0.10 s | 0.097 | 6.18 | 12.4 | 0.023 | 3.03 | 4.6 | FAIL |
| | held, from 0.10 s | 0.074 | 4.24 | 2.1 | 0.021 | 1.50 | 1.5 | PASS |
| | held, whole run (`mtc.json`) | 0.074 | 4.18 | 2.4 | 0.039 | 1.53 | 1.5 | PASS |
| parry_sixte_v3 | standby, whole run | 0.096 | 5.41 | 31.0 | 0.112 | 3.87 | 2.9 | FAIL |
| | standby, from 0.76 s | 0.072 | 4.40 | 7.5 | 0.026 | 1.08 | 2.9 | FAIL |
| | held, from 0.76 s | 0.068 | 4.15 | 1.6 | 0.001 | 0.99 | 1.6 | PASS |
| | held, whole run (`mtc.json`) | 0.072 | 4.06 | 3.5 | 0.034 | 1.18 | 1.6 | PASS |
| parry_tierce_high_v3 | standby, whole run | 0.128 | 6.20 | 11.8 | 0.072 | 5.55 | 8.3 | FAIL |
| | standby, from 2.16 s | 0.097 | 4.83 | 6.0 | 0.069 | 1.35 | – | PARTIAL |
| | held, from 2.16 s | 0.076 | 3.71 | 1.6 | 0.001 | 0.24 | – | PARTIAL |
| | held, whole run (`mtc.json`) | 0.089 | 3.82 | 2.8 | 0.036 | 1.12 | 1.7 | PASS |
| parry_quinte_v3 | standby, whole run | 0.376 | 13.68 | 11.9 | 0.845 | 37.00 | – | FAIL |
| | standby, after convergence | never converged | | | | | | – |
| | held, whole run (`mtc.json`) | 0.073 | 4.64 | 2.7 | 0.032 | 1.70 | 1.5 | PASS |
| parry_prime_v3 | standby, whole run | 0.097 | 5.48 | 15.6 | 0.065 | 4.17 | 1.0 | FAIL |
| | standby, from 0.38 s | 0.084 | 5.05 | 11.7 | 0.041 | 2.07 | 1.0 | FAIL |
| | held, from 0.38 s | 0.072 | 4.19 | 1.7 | 0.002 | 1.64 | 2.0 | PASS |
| | held, whole run (`mtc.json`) | 0.074 | 4.11 | 2.1 | 0.038 | 1.76 | 2.0 | PASS |
| parry_seconde_v3 | standby, whole run | 0.126 | 6.97 | 11.8 | 0.052 | 6.04 | 5.2 | FAIL |
| | standby, from 0.34 s | 0.119 | 6.82 | 9.7 | 0.046 | 4.06 | 5.3 | FAIL |
| | held, from 0.34 s | 0.087 | 4.47 | 2.3 | 0.003 | 2.59 | 2.8 | PASS |
| | held, whole run (`mtc.json`) | 0.089 | 4.38 | 2.0 | 0.040 | 2.51 | 2.8 | PASS |
| thrust_v3 | standby, whole run | 0.118 | 6.49 | 7.2 | 0.013 | 5.38 | 1.3 | FAIL |
| | standby, from 0.14 s | 0.114 | 6.36 | 8.3 | 0.008 | 4.00 | 1.3 | FAIL |
| | held, from 0.14 s | 0.077 | 4.51 | 3.4 | 0.019 | 1.48 | 1.9 | PASS |
| | held, whole run (`mtc.json`) | 0.078 | 4.43 | 2.0 | 0.043 | 1.54 | 1.9 | PASS |
| cut_chest_v3 | standby, whole run | 0.136 | 6.98 | 20.2 | 0.075 | 6.17 | 4.6 | FAIL |
| | standby, from 0.16 s | 0.132 | 6.85 | 19.2 | 0.066 | 4.62 | 4.5 | FAIL |
| | held, from 0.16 s | 0.075 | 3.95 | 3.8 | 0.016 | 1.53 | 1.9 | PASS |
| | held, whole run (`mtc.json`) | 0.077 | 3.91 | 1.9 | 0.042 | 1.61 | 1.9 | PASS |
| cut_flank_v3 | standby, whole run | 0.109 | 5.83 | 21.3 | 0.104 | 5.42 | 5.0 | FAIL |
| | standby, from 0.68 s | 0.077 | 4.77 | 17.7 | 0.072 | 2.71 | 5.0 | FAIL |
| | held, from 0.68 s | 0.063 | 4.13 | 1.8 | 0.002 | 1.88 | 2.3 | PASS |
| | held, whole run (`mtc.json`) | 0.068 | 4.07 | 1.9 | 0.042 | 2.04 | 2.3 | PASS |
| cut_head_v3 | standby, whole run | 0.161 | 7.51 | 12.0 | 0.078 | 8.87 | 10.2 | FAIL |
| | standby, from 2.86 s | 0.102 | 5.47 | 2.5 | 0.092 | 2.04 | – | FAIL |
| | held, from 2.86 s | 0.076 | 3.58 | 0.4 | 0.001 | 0.99 | – | PARTIAL |
| | held, whole run (`mtc.json`) | 0.084 | 3.81 | 2.2 | 0.042 | 2.66 | 4.4 | PASS |
| cut_head_v4 | standby, whole run | 0.127 | 6.67 | 6.9 | 0.023 | 5.44 | 9.1 | FAIL |
| | standby, from 0.16 s | 0.123 | 6.58 | 7.9 | 0.006 | 4.10 | 9.1 | FAIL |
| | held, whole run (`mtc.json`) | 0.088 | 4.48 | 1.6 | 0.042 | 2.36 | 2.9 | PASS |
| advance_lunge_cut_v5 | standby, whole run | 0.087 | 3.73 | 8.0 | 0.052 | 3.21 | 1.5 | PASS |
| | standby, from 0.22 s | 0.083 | 3.54 | 8.7 | 0.046 | 1.85 | 1.5 | PASS |
| | held, from 0.22 s | 0.076 | 3.12 | 9.2 | 0.020 | 1.97 | 1.5 | PASS |
| | held, whole run (`mtc.json`) | 0.076 | 3.08 | 8.0 | 0.020 | 2.03 | 1.5 | PASS |

"From T s" is the convergence step × 20 ms, the settle window's own unit. The transition
table's convergence time is that step's tick measured from the switch, which adds the tick
phase and the one-tick lag, about 0.04 s.

**Failing checks after convergence:**

| motion | failing check | value | threshold |
|---|---|---|---|
| extension | body | 6.2 cm | 5 cm |
| parry_prime | body | 5.05 cm | 5 cm |
| parry_seconde | joint | 0.119 rad | 0.10 rad |
| parry_seconde | body | 6.8 cm | 5 cm |
| parry_seconde | tip at key | 5.3 cm | 5 cm |
| parry_sixte | torso orientation | 0.246 rad | 0.2 rad |
| cut_head | joint | 0.102 rad | 0.10 rad |
| cut_head | body | 5.5 cm | 5 cm |
| cut_flank | tip at key | 5.0 cm | 5 cm |
| cut_chest | joint | 0.132 rad | 0.10 rad |
| cut_chest | body | 6.8 cm | 5 cm |
| thrust | joint | 0.114 rad | 0.10 rad |
| thrust | body | 6.4 cm | 5 cm |

On parry_sixte the robot turned during the transition, so it does the motion at a different
heading.

### What it shows

- **Switches cleanly from standby** (stayed up, converged within 0.5 s, no leg at the torque
  clamp, passes after convergence):
  - **Without a fail anywhere:** advance_lunge_v4 and advance_lunge_cut_v5 pass the whole run
    as well.
  - **The whole run fails only on the start transient:** en_garde_v3, retreat_v1.
  - **Converges more slowly (1.5 s), the torso moves 19 cm, still passes after
    convergence:** lunge_v5.
- **Stays up, but the motion is not tracked to the gate afterwards:** extension, parry prime,
  parry seconde, parry sixte, parry tierce high (PARTIAL), and all four hits. The joint error
  drops under 0.15 rad, but body error is 4.4-6.9 cm after convergence, against 3.6-4.5 cm
  for the held start over the same window. The checks that fail are listed above: mostly body
  and joint error; on parry_sixte the torso heading; on cut_flank tip at key, at the
  threshold. The likely reason is that the feet do not get back to the reference stance: the right foot
  moved 11-38 cm during the transition (except thrust and extension), and the left ankle's own
  error stays high. This was checked per body on en_garde only (left ankle 19 cm). On the
  others it is an inference.
- **Falls in simulation:** advance_v1 and parry_quinte. Both have the same standby start as
  motions that survive (parry_quinte shares its first frame with all the `*_v3` saber
  motions), so the size of the offset alone does not predict a fall. These policies are the
  least robust to this start.
- **`motion.start_step` does not help.** Every reference frame (ONNX reference, by joint name)
  was compared with standby's `default_position`, as the mean |frame − default| over the 29
  joints. No motion has a frame much closer than frame 0. The smallest mean offset anywhere
  in a motion is 0.24-0.35 rad, against 0.24-0.42 at frame 0. Within the first 2 s, the best
  frame's leg offsets are no smaller (largest leg offset 0.44-1.08 rad). The gap is the stance
  itself: standby stands feet together, symmetric, and every motion starts in a fencing
  stance with the feet 0.33-0.48 m apart. A non-zero `start_step` would also skip part of a
  motion that was evaluated only from frame 0.
- **What would help, per this simulator:** a start pose closer to the motion's first frame.
  That means either a lead-in that walks from the standby pose into the guard, trained or
  added to the reference, or a standby `default_position` set to the guard's joint angles.
  The standby default only changes the joints, not where the feet are. Neither option was
  tested here.

### Limits

- **Sim standby equals real standby only as far as the config is shared.** Verified: the
  same `controllers.yaml` (default pose, kp, kd) and the same controller type as
  `real.launch.py`. Not verified:
  - that the closed-source `legged_controllers` binary in the rootfs is the build on the
    robot;
  - the ramp time (2 s is read from the header, not measured on hardware);
  - the motor dynamics: the plant has no torque-speed curve, and the effort limits are the
    URDF's.
- **`unitree_bringup`'s own `controllers.yaml` differs** (standby ankle pitch −0.363). Only
  MTC's file was used, because `real.launch.py` loads MTC's.
- **The robot reached standby from a lowered, straight-legged pose.** How the real robot gets
  into standby (gantry, standing first) is unknown. A robot that is still swaying at the
  switch was not tested. The pilot showed about 5 s of sway after the ramp.
- **One run per policy.** The determinism sections above show run-to-run spread of a few
  0.001 rad on held starts. The standby transient may spread more. The two falls and the
  marginal fails (parry_prime body 5.05, cut_flank tip at key 5.0) should be run again before
  anything is decided on them.
- **Wrist torque.** The wrist clamp times are measured on a rigid, welded saber.
- **The estimator restart at the switch is simulated as upstream does it.** Real estimator
  behaviour on the robot is a separate unknown ("Known gaps").

## Entry from standby (2026-11-11)

The fix for the start problem above. A short **entry** policy takes the robot from standby's
pose into the guard. The motion policy is then switched in from the entry's end hold, on its own
first frame. Both switches are the operator's buttons, which `unitree_bringup` already has:

```
L1 + A  standby_controller        robot stands in default_position (as before)
R1 + A  walking_controller        = the entry policy (7.5 s, ends holding the guard)
R1 + X  walking_controller1       = the motion policy, started from the entry's hold
```

Five entry policies were trained on the same reference. **Recommended: `stand_to_en_garde_v1_feetjoint05`**
(2026-11-12). It is the only one that passes its own gates in all five simulators (mjlab, CPU
MuJoCo, MTC held, MTC standby, PhysX) and lets all ten chained saber motions pass, one run each.
`stand_to_en_garde_v1_feetjoint` also chains all ten but fails PhysX (it drifts in the hold). See
"Paid for joint angles" below. The older `stand_to_en_garde_v1_tip` and `stand_to_en_garde_v1_feet`
are kept for comparison.

### Why the standby start misses, and why two policies fall (evidence)

The standby-start recordings ("Start from standby") were compared with the held-start
recordings of the same policies. The window was the same: from the convergence step to the end of the motion.

- **The error that lasts is the stance, not the sword arm.** Right-arm joint error after
  convergence is about the held run's (thrust 0.095 vs 0.066 rad, cut_head 0.068 vs 0.073,
  seconde 0.078 vs 0.090). Legs and waist are not: thrust legs 0.145 vs 0.103, waist 0.116 vs
  0.039; seconde legs 0.158 vs 0.096; cut_chest legs 0.187 vs 0.110. The difference is where
  the right foot stands relative to the left, in the anchor's heading frame, at the end of the
  motion, against the reference's (0.298, −0.152) m. Standby vs held: thrust 30.5 vs 2.1 cm,
  seconde 27.6 vs 3.3, cut_head 24.2 vs 5.7, en_garde 18.4 vs 5.9, extension 17.4 vs 5.2,
  tierce high 12.8 vs 1.9, cut_chest 11.9 vs 1.5. sixte (2.6 vs 6.9) and cut_flank (3.8 vs
  3.9) end in the reference stance and are the two closest to the gate. On thrust, cut_head and
  tierce high the error does not decay over the motion (thrust 0.115 rad in the first 50 steps
  after convergence, 0.118 in the last 50, 0.122 in the hold). The policies never re-step into
  the trained stance. The wrist torque clamp at the switch lasts about 0.2 s and does not knock
  the arm off for good.
- **The falls are the policies' own response, not a larger offset.** All 11 static saber
  references share frame 0 exactly (pelvis 0.711 m, same joints), so parry_quinte, parry_prime and
  the others got the same start state. In the first 0.6 s after the switch:
  - en_garde: two orderly steps (right foot lifts 2.5 cm and moves 21 cm, then the left
    18 cm; pelvis dips at most 1.7 cm).
  - parry_prime (survives): shuffles of ≤ 1.6 cm lift.
  - parry_quinte: hops. Both feet are off the floor by 0.15 s (right 17 cm, left 12 cm up)
    and the pelvis rises 7 cm. It lands with the left foot 65 cm away and the pelvis drops
    16 cm, with right hip pitch, right ankle pitch and waist pitch at the torque limit.
  - advance_v1: swings the left foot 40 cm with 10 cm of lift, then the pelvis collapses
    (−21 cm by 0.6 s).
- **The footwork motions do not start in the guard.** The first frames of advance_v1 (stance
  29 cm from en_garde's), retreat_v1 (27 cm), lunge_v5 (12 cm) and the advance-lunges (52 cm)
  are as far from the guard as from standby. An entry into the guard therefore helps only the
  saber motions. Of the footwork motions only advance_v1 fails from standby.

### What upstream does about the start pose

Nothing at deployment. Training starts each episode at an adaptively sampled frame, with
random offsets: mjlab `tasks/tracking/mdp/commands.py` `_resample_command`, `sampling_mode`
adaptive; `tracking_env_cfg.py` sets `pose_range` x/y ±0.05 m, z ±0.01 m, small angles, root
velocity kicks and `joint_position_range` ±0.1 rad. BeyondMimic's `whole_body_tracking`
is the same. Play mode turns the randomisation off and starts at frame 0 (`config/g1/env_cfgs.py`).
`motion_tracking_controller`'s README says only that the robot should start in the standby
controller, then `R1 + A`. `MotionCommandTerm::reset` re-anchors the reference's first frame
on the robot's torso position and heading at activation. Joints, foot placement and stance are
left to the policy. Nothing in upstream bridges the standby pose to a motion's first frame.

### Two policies in one controller manager

No upstream file is changed. `generate_temp_config` (legged_bringup) accepts new keys, so our
launch files add a second `MotionTrackingController`, `walking_controller1`, with its own
`policy.path`, spawned inactive next to `walking_controller`:

- sim: `scripts/mtc/mujoco_custom.launch.py chain_policy_path:=`, driven by
  `scripts/mtc_sim2sim.sh <motion> --standby --entry <entry>`. `scripts/mtc/switch_after.py`
  sends joy.yaml's `walking_controller` request (R1 + A) and then its `walking_controller1`
  request (R1 + X), byte for byte.
- robot: `scripts/mtc/real_chain.launch.py` loads upstream's `real.launch.py` from the
  package share and swaps in a copy of its `setup_controllers` that adds `walking_controller1`.
  All other nodes are upstream's, the teleop included. Checked in the rootfs only, not on the robot:
  the generated `temp_controllers.yaml` has `walking_controller1` of type
  `motion_tracking_controller/MotionTrackingController` with the given `policy.path`, and the
  inactive spawner lists `walking_controller walking_controller1`.
- `unitree_bringup`'s `config/g1/joy.yaml` (commit 8a0f308) already binds R1 + X ([5, 2]) to
  activate `walking_controller1` and deactivate standby and `walking_controller`. R1 + A and
  L1 + A deactivate `walking_controller1`. R1 + X is not interlocked: pressed straight from
  standby it starts the motion without the entry.

**Pilot, before any new training** (one run): standby → extension_v3_tip (as the entry) →
parry_quinte_v3_tip. The policy that falls 0.40 s after a standby switch was switched in from
extension's end hold (extension ends in the guard). It stayed up, converged at step 0 (0.04 s),
and passed every gate on the whole run: joint 0.072 rad, body 4.7 cm, blade 1.4 cm, blade at
key 1.2 cm. The joint offset at the switch was 0.097 rad mean, against 0.364 from standby. The
robot stood still under the entry before the switch. Peak torque in the first second: 1.00 on
the left wrist pitch, legs ≤ 0.70 (left ankle pitch). The JSON is
`$HCF_DATA/eval/2026-11-10_07-03-50_parry_quinte_v3_tip/model_2999/mtc_entry_pilot_extension.json`.
extension itself still fails its gates from standby. It is a launch pad here, not a
deliverable.

### The entry motion

`stand_to_en_garde_v1` (`fencing motion entry motions/parry_quinte_v3.csv --publish`,
`src/fencing/motion/entry.py`), 7.5 s at 30 fps:

```
0.0-1.0  hold standby's default_position (joint for joint, from MTC's controllers.yaml), feet flat
1.0-1.8  weight onto the left foot          (CoM to 1.5 cm inside the left sole)
1.8-2.6  right foot steps 11.5 cm into place (lift straight up 6 cm, move, set down)
2.6-3.4  weight onto the right foot
3.4-4.2  left foot steps 11.5 cm back into place
4.2-5.0  settle into the guard (pelvis, waist and arms finish their min-jerk blend)
5.0-7.5  hold the guard = frame 0 of every static saber motion, exactly
```

The soles are placed by leg IK and the pelvis is moved so that the CoM stays over the stance
sole while the other foot is in the air. One exception: the guard stance is wider than the
G1's hips. Its soles are 26 cm apart sideways and the standby soles 23.6 cm, and the ankle roll
limit is ±0.26 rad. So over the last quarter of each step, as the foot is lowered, the CoM
starts toward its next target. That makes the end of each step dynamic, not static: on the
back-foot step the CoM is outside the right sole for the last 4 frames (0.13 s), up to 5 cm
outside, while the left foot is still 0.6-2.8 cm up. On the front-foot step it is at most
0.8 cm outside, with the foot under 0.3 cm (`tests/test_motion_entry.py`, run on both
en_garde_v3 and the published configuration). Releasing only at touchdown would push the ankle
roll further past its limit. No joint goes beyond a limit, but both ankle rolls sit exactly on
their ±0.2618 rad limit for part of the motion (frames at 30 fps, within 1e-4 rad):
`left_ankle_roll_joint` at -0.2618 for frames 97-126 (1 s: the weight transfer, its own swing
and the landing), `right_ankle_roll_joint` for frames 69-71 (its swing) and 114-120 (while it is
the stance ankle). `tests/test_motion_entry.py` pins these frames. Quality check: pass (slip
0.4 cm, penetration 0.1 cm, max 3.1 rad/s). It reports 0 limit violations because it counts
values beyond a limit (by more than 1e-3 rad), not values on it. Arm and blade never pass through the body
(the blade keyframer's self-collision check, every frame). Trained like the saber motions:
`Fencing-Tracking-Flat-G1-Saber`, 3000 iterations, 4096 envs, tag `tip`.

The same generator cannot build an entry for **advance_v1**. Its first stance is 40 cm wide
across the heading, and the leg IK cannot set the second foot down within 1 cm under the
quasi-static rule. advance_v1 has no entry. See "Limits" below.

### Results (2026-11-11, one MTC run each, saber plant for saber policies)

**The entry policy**, `2026-11-11_15-31-43_stand_to_en_garde_v1_tip` (model_2999):

| Where | Verdict | Joint (rad) | Body (cm) | Blade (cm) | Notes |
|---|---|---|---|---|---|
| mjlab, 1024 episodes | PASS | 0.058 | 3.3 | 1.3 | push, noise, friction, saber mass robustness 1.00 |
| CPU MuJoCo sim2sim | PASS | 0.055 | 3.1 | 1.2 | |
| MTC, held start (`mtc.json`) | PASS | 0.056 | 3.2 | 1.3 | |
| MTC, from standby (`mtc_standby.json`) | PASS | 0.056 | 3.2 | 1.3 | converged at step 0. Peak torque in the first second 0.40 (right wrist pitch), legs ≤ 0.20 |

At its end hold the entry stands with the pelvis 1.9 cm above the guard's. Its back leg is
straighter (left hip pitch −0.29 rad, left ankle pitch +0.25 rad off the guard), so the motion
policies start with a 0.065 rad mean joint offset, against 0.364 from standby.

**Standby → entry → motion, first entry (`stand_to_en_garde_v1_tip`).** These reports were
`mtc_entry.json` until the retrain below. They are now `mtc_entry_stand_to_en_garde_v1_tip.json`
in each motion's eval dir, and their recordings are under `$HCF_DATA/mtc/runs_entry_v1/`
(cut_head_v3, superseded, was not re-run and keeps `mtc_entry.json`). R1 + A 10 s after standby is active; R1 + X
10 s after R1 + A. The entry reference is 7.5 s and ends with 2.5 s of the guard; it had ended
2.5 s before R1 + X, and the policy holds its last frame past the end.
The motion is scored from the second switch with the usual 2 s end hold. Each chained report
records which entry ran: `entry.run` and `entry.onnx_sha256` (here
`2026-11-11_15-31-43_stand_to_en_garde_v1_tip`, sha256 `1b08dc23a521d524…`; added to the
reports after scoring, from the export, whose ONNX predates them; `entry.provenance_note`
says so). R1 + X also restarts `state_estimator`: the controller manager deactivates and
reactivates it with the policy controller (`switch.log`, "Deactivated controllers:
[ walking_controller state_estimator ]"). R1 + A from standby does not. "Torque" is the peak
fraction of the joint's limit in the first second after each switch:

Both verdict columns are whole-run verdicts (the top-level `verdict` of the JSON).

| # | Motion (policy) | From standby directly, whole run (`mtc_standby.json`) | Standby → v1 entry → motion, whole run | Stays up | Converged | Failing check | Torque at R1 + X: any joint / legs |
|---|---|---|---|---|---|---|---|
| 4 | extension_v3_tip | FAIL | FAIL | yes | 0.04 s | body 5.18 cm | 1.00 left wrist pitch / 0.58 |
| 6 | parry_prime_v3_tip | FAIL | FAIL | yes | 0.04 s | body 5.15 cm | 1.00 / 0.52 |
| 7 | parry_seconde_v3_tip | FAIL | FAIL | yes | 0.04 s | body 5.23 cm | 1.00 / 0.58 |
| 8 | parry_tierce_high_v3_tip | FAIL (after convergence: PARTIAL) | **PASS** | yes | 0.04 s | | 1.00 / 0.46 |
| 9 | parry_quinte_v3_tip | FAIL (**fell** at 0.40 s) | **PASS** | yes | 0.04 s | | 1.00 / 0.59 |
| 10 | parry_sixte_v3_tip | FAIL | **PASS** | yes | 0.04 s | | 1.00 / 0.50 |
| 11 | thrust_v3_tip | FAIL | FAIL | yes | 0.04 s | body 5.57 cm | 1.00 / 0.54 |
| 12 | cut_chest_v3_tip | FAIL | **PASS** | yes | 0.04 s | | 1.00 / 0.50 |
| 13 | cut_flank_v3_tip | FAIL | FAIL | yes | 0.04 s | body 5.09 cm | 1.00 / 0.55 |
| 15 | cut_head_v4_tip (final) | FAIL | FAIL | yes | 0.04 s | body 5.40 cm | 1.00 / 0.59 |
| - | cut_head_v3_tip (superseded) | FAIL | FAIL | yes | 0.04 s | body 5.15 cm, tip at key 5.6 cm | 1.00 / 0.50 |

The "any joint" 1.00 at R1 + X is the left wrist pitch in every run, a joint with a small
limit. The R1 + A switch into the entry peaks at 0.40 (right wrist pitch) and legs 0.20.

**advance_v1: standby → retreat_v1 → advance_v1.** No entry could be generated for it (above).
But retreat_v1's last frame is 1.3 cm and 0.027 rad mean from advance_v1's first (stance
(0.141, −0.419) vs (0.153, −0.415) m). R1 + X was sent 5 s after R1 + A: retreat_v1 is 1.9 s,
so 3.1 s after its reference ended. retreat_v1 is not clean from standby: its whole run fails the joint
gate (0.121 rad, the same in the chain's `entry` block), because of the 0.4 s transition after
R1 + A. From convergence on it passes. It is still usable as the first policy: it stays up,
converges in 0.40 s, and stands still before R1 + X (pelvis speed 0.0006 m/s; the stillness
check says `false` only for the guard's pelvis tilt, see Limits). What R1 + X needs from it is
its end pose, and that matches advance_v1's start. In that chain advance_v1 stayed up,
converged at step 0 and passed every gate on the whole run: joint 0.081 rad, body 3.6 cm.
Torque at R1 + X: 1.00 (right wrist pitch), legs 0.42. From standby directly it fell at 0.52 s.
No training was needed. The robot walks backward through the retreat before the advance.

**Motions switched in directly from standby** (unchanged, "Start from standby"): en_garde_v3,
retreat_v1 and lunge_v5 pass after convergence (0.42, 0.40 and 1.48 s) and fail their whole
runs. advance_lunge_v4 and advance_lunge_cut_v5 pass the whole run.

**Comparison: extension_v3_tip as the entry** (no new training; `mtc_entry_via_extension.json`,
quinte in `mtc_entry_pilot_extension.json`). Run for the eight final saber motions other than
extension and cut_head_v4, plus the superseded cut_head_v3. All stayed up and converged at step
0, with a 0.097 rad offset at the switch. PASS: parry_quinte, parry_sixte, cut_chest.
parry_tierce_high failed the joint gate (0.102 rad; body 4.94 cm). prime, seconde, thrust and
cut_flank failed the body gate at 5.05-5.45 cm (cut_head_v3: body 5.31 cm and tip at key
6.0 cm). The trained entry is a little better: offset 0.065 rad, and it adds tierce high as a
PASS. It also passes its own gates from standby, which extension does not.

### Retrained on the entry task: `stand_to_en_garde_v1_feet` (2026-11-11)

One retrain, aimed at the cause found below ("What it shows", first entry). Same reference
(`motions/stand_to_en_garde_v1.csv`), new task `Fencing-Tracking-Flat-G1-Saber-Entry`: the
saber task plus a `feet_pos` reward on the two ankle roll links, std 0.1 m, weight 1.0
([saber.md, mjlab tasks](saber.md#mjlab-tasks)). Observations, actions and the ONNX contract
are unchanged: inputs, outputs and metadata keys match v1's, and only `run_path` differs. 3000
iterations, 4096 envs. Run `2026-11-11_18-02-31_stand_to_en_garde_v1_feet`, policy sha256
`40e9c72c42f27d38462ab382a4ae85e7f3304dd4c4aa61aa579d1f2f26afcce5`.

**Why the reference was kept.** The v1 policy did not try the steps and fall short. Its feet
lifted 0.4 cm against 6 cm, which is no step at all. A step that was too fast would show a
partial lift. The step is also slow for a G1: 11.5 cm in 0.32 s, lift 6 cm in 0.24 s, swing
speeds 0.25-0.36 m/s. So the cause is what the policy is paid for, not what it is shown. The
result bears this out: on the same reference the new policy steps.

**The entry policy:**

| Where | Verdict | Joint (rad) | Body (cm) | Notes |
|---|---|---|---|---|
| mjlab, 1024 episodes | **FAIL** | **0.1007** (gate 0.1) | 3.4 | the only failing check; success 1.00, blade 1.5 cm, slip 0.008 m/s |
| CPU MuJoCo sim2sim | PASS | 0.0986 | | |
| MTC, held start (`mtc.json`) | PASS | 0.0992 | 3.33 | |
| MTC, from standby (`mtc_standby.json`) | PASS | 0.0985 | 3.35 | converged at step 0. R1 + A torque 0.34 (right wrist pitch), legs 0.23 |

It fails the mjlab joint gate by 0.0007 rad. In the other three simulators it passes that
gate by 0.0008 rad (MTC held), 0.0014 (sim2sim) and 0.0015 (MTC standby). Paying for foot
position moved error into the joints: v1 had 0.058 rad. The gates were not changed.

Robustness (mjlab, 1024 episodes per pass, `metrics.json`): `robustness.<pass>.success_rate`,
the gated field, is 1.00 in all four passes for both entries. What dropped is
`robustness.<pass>.recovery_rate`, which is reported but not gated: push 0.888 → 0.777, obs_noise
0.998 → 0.901, friction 1.000 → 0.999, saber_mass 1.000 → 1.000 (v1_tip → v1_feet). The
anchor (torso) orientation error rose from 0.04 to 0.13 rad in every pass.

**Longer training does not clear the gate (`stand_to_en_garde_v1_feet5k`, 2026-11-12).** The
coordinator resumed v1_feet from checkpoint 2999 for 2000 more iterations, on the same task
and reference. The last checkpoint is `model_4998` (the resume counts from 2999). Run
`2026-11-11_19-48-51_stand_to_en_garde_v1_feet5k`, policy sha256
`f32014601368a70afb9bc4bc7a8e4623adc623b8d2ac00c3fabc1cb691550a08`. It places its feet better
still, but the joint error did not fall. It fails the joint gate in three of the four
simulators, so it was not chained. Three entries, one reference:

| | v1_tip (saber task, 3000 it) | v1_feet (entry task, 3000 it) | v1_feet5k (entry task, 5000 it) |
|---|---|---|---|
| mjlab verdict, joint (rad) | PASS, 0.058 | **FAIL**, 0.1007 | **FAIL**, 0.1013 |
| sim2sim verdict, joint | PASS, 0.055 | PASS, 0.0986 | **FAIL**, 0.10002 |
| MTC held verdict, joint | PASS, 0.056 | PASS, 0.0992 | **FAIL**, 0.1003 |
| MTC standby verdict, joint | PASS, 0.056 | PASS, 0.0985 | PASS, 0.0999 |
| Body error, mjlab / MTC held (cm) | 3.3 / 3.2 | 3.4 / 3.3 | 3.2 / 3.1 |
| Robustness success rate (gated), all four passes | 1.00 | 1.00 | 1.00 |
| `recovery_rate` push / obs_noise (mjlab) | 0.888 / 0.998 | 0.777 / 0.901 | 0.767 / 0.911 |
| Anchor orientation error, mjlab (rad) | 0.04 | 0.13 | 0.135 |
| Pelvis height error at the end, mjlab (cm) | 1.9 | 1.0 | 0.4 |
| Foot xy error at the end, L / R, MTC held (cm) | 9.9 / 4.8 | 2.1 / 2.7 | 0.1 / 1.8 |
| Foot lift L / R, MTC held (cm; reference 6.9 / 6.1) | 0.4 / 0.6 | 5.0 / 1.6 | 6.1 / 1.7 |
| End stance, MTC held (m; guard 0.295, −0.145) | (0.180, −0.212) | (0.318, −0.161) | (0.313, −0.145) |
| Saber chains run through it | 10: 4 PASS, 6 FAIL (body) | 10: 10 PASS | not run (fails its own gates) |
| sha256 of `policy.onnx` | `1b08dc23…` | `40e9c72c…` | `f3201460…` |

The feet entries' joint error sits at the 0.1 rad line whatever the training length. Which
joints carry it was not measured, since `metrics.json` has no per-joint error. Over 2000 more iterations, foot placement and pelvis height improved and
the joint error did not. More of the same training is not the lever. A further attempt would
change what the joints are paid for, or the reference's leg IK, and that is outside this
brief. (Paying for the joints was done on 2026-11-12: see "Paid for joint angles" below.)

**Foot placement** (`scripts/entry_foot_placement.py <entry export dir> <bag.npz>`. MTC, from the first policy tick to 1 s into the end hold; xy displacement
error against the reference's own displacement; stance = front foot minus back foot in the
heading frame; guard (0.295, −0.145) m):

| Entry | Run | Left foot xy error | Right foot xy error | Lift left / right (reference 6.9 / 6.1 cm) | End stance (m) | Ankle body error, mjlab L / R |
|---|---|---|---|---|---|---|
| v1_tip | held | 9.9 cm | 4.8 cm | 0.4 / 0.6 cm | (0.180, −0.212) | 10.8 / 7.4 cm |
| v1_tip | standby | 9.8 cm | 6.1 cm | 0.4 / 0.4 cm | (0.169, −0.220) | |
| v1_feet | held | 2.1 cm | 2.7 cm | 5.0 / 1.6 cm | (0.318, −0.161) | 4.3 / 5.0 cm |
| v1_feet | standby | 2.3 cm | 2.8 cm | 4.9 / 1.5 cm | (0.321, −0.160) | |

The back (left) foot now steps. The front foot lifts 1.6 cm and slides the rest of the way.
At the end its stance is 2.3 cm long and 1.6 cm wide of the guard, against 12 cm short and 7.6 cm
wide for v1. The thrust recording at R1 + X: stance (0.321, −0.159), pelvis 0.721 m (guard
0.711). Its pelvis height error at the end of the reference is 1.0 cm, against 1.9 cm.

**Standby → v1_feet → motion** (`mtc_entry.json`, the same switch times as v1: R1 + A after
10 s, R1 + X 10 s later). One run each. Every recording was complete, with no INVALID reason,
real-time factor 1.0 and the first policy tick 2 ms after each switch:

| # | Motion | Verdict, whole run | Joint (rad) | Body (cm) | v1 entry body (cm) | Entry part | Torque at R1 + X: any joint / legs |
|---|---|---|---|---|---|---|---|
| 4 | extension_v3_tip | **PASS** | 0.076 | 4.04 | 5.18 | PASS | 1.00 right wrist pitch / 0.77 left ankle pitch |
| 6 | parry_prime_v3_tip | **PASS** | 0.069 | 4.25 | 5.15 | PASS | 1.00 left wrist pitch / 0.46 right ankle pitch |
| 7 | parry_seconde_v3_tip | **PASS** | 0.089 | 4.42 | 5.23 | PASS | 1.00 right wrist pitch / 0.42 |
| 8 | parry_tierce_high_v3_tip | **PASS** | 0.089 | 3.99 | 4.79 | PASS | 1.00 right wrist yaw / 0.46 |
| 9 | parry_quinte_v3_tip | **PASS** | 0.070 | 4.03 | 4.84 | PASS | 1.00 right wrist yaw / 0.67 |
| 10 | parry_sixte_v3_tip | **PASS** | 0.068 | 3.59 | 4.93 | PASS | 0.92 right wrist yaw / 0.53 |
| 11 | thrust_v3_tip | **PASS** | 0.080 | 4.54 | 5.57 | PASS | 1.00 right wrist yaw / 0.44 |
| 12 | cut_chest_v3_tip | **PASS** | 0.073 | 3.40 | 4.87 | PASS | 1.00 right wrist pitch / 0.72 |
| 13 | cut_flank_v3_tip | **PASS** | 0.063 | 4.10 | 5.09 | PASS | 1.00 right wrist yaw / 0.48 |
| 15 | cut_head_v4_tip | **PASS** | 0.087 | 4.31 | 5.40 | PASS | 1.00 right wrist pitch / 0.53 |

All ten stayed up, converged at step 0 (0.04 s) and passed after convergence too. Body error
is 3.40-4.54 cm. From their own held starts the same ten scored 3.82-4.64 cm in MTC (`mtc.json`)
and 4.01-4.55 cm in mjlab (`metrics.json`), so they no longer pay for the start. "Entry part" is the entry scored from R1 + A to R1 + X in the same recording. The
stillness check before R1 + X says `still: false` in all ten, for one reason only: pelvis tilt
0.397 rad against standby's 0.2 limit. The guard leans by design, as in the retreat → advance
chain, and v1 never reached the guard's lean. Leg torque at R1 + X is higher than with v1:
up to 0.77 (left ankle pitch, extension), against 0.59.

### Paid for joint angles: `stand_to_en_garde_v1_feetjoint05` (2026-11-12, recommended)

**Where v1_feet's extra joint error sits.** The exported ONNX policies were rolled out in CPU
sim2sim, with the per-joint mean |error| taken over the motion part (a scratch script around
`fencing.sim2sim.rollout`; the totals equal the reported `joint_angle_error_rad`). v1_feet is at
0.0986 and v1_tip at 0.0555. Phase means, v1_feet against v1_tip (rad):

| Phase (s) | hold standby 0-1 | weight left 1-1.8 | right step 1.8-2.6 | weight right 2.6-3.4 | left step 3.4-4.2 | settle 4.2-5.0 | guard hold 5.0-7.5 |
|---|---|---|---|---|---|---|---|
| v1_tip | 0.035 | 0.041 | 0.058 | 0.047 | 0.069 | 0.064 | 0.064 |
| v1_feet | 0.041 | 0.053 | 0.074 | 0.076 | 0.097 | 0.128 | **0.143** |
| v1_feetjoint | 0.018 | 0.023 | 0.039 | 0.019 | 0.030 | 0.041 | 0.039 |
| v1_feetjoint05 | 0.018 | 0.027 | 0.048 | 0.030 | 0.045 | 0.054 | 0.058 |

- **It is the guard hold and the settle into it, not the steps.** The hold is 125 of the 374
  scored steps. It accounts for about 0.026 rad of v1_feet's extra 0.043, and the settle for
  0.007 more.
- **It is spread over the whole body.** The largest errors in the hold: waist roll 0.30 rad,
  left hip pitch 0.30, left shoulder roll 0.29, right ankle pitch 0.27, waist pitch 0.25, right
  knee 0.21, right hip pitch 0.21, left wrist pitch 0.22. Contribution to the mean over the whole
  motion, v1_feet against v1_tip: legs 0.042 vs 0.030, waist 0.013 vs 0.003, left arm 0.028 vs
  0.011, right arm 0.016 vs 0.012. In the hold the pelvis leans 22.9° from vertical against the
  reference's 13.1° (v1_tip 9.9°). The soles are flat (≤ 0.3°).
- **It is not the ankle-roll limit frames.** The ankle rolls' mean error is about the same
  (v1_feet L 0.055 / R 0.062 rad, v1_tip 0.054 / 0.074). On the 50 left and 20 right steps where
  the reference sits on the limit, v1_feet's error is lower than v1_tip's (0.134 vs 0.142, 0.088
  vs 0.111). Together the two ankle rolls are about 0.004 rad of the mean in both policies.
- **Cause: the guard cannot be held as given with both feet in place.** The entry ends on
  `parry_quinte_v3` frame 0, the shared first frame of the saber motions. In that pose the
  whole-body CoM lies **5.8 cm outside the convex hull of both soles' capsule axes**, measured
  on the generator's own G1 model. The real sole edge is about 1 cm further out than the axes.
  In the saber scene the result is 5.9 cm: the saber's 0.20 kg moves it by 0.1 cm. The back sole
  is also 0.65 cm above the floor. `en_garde_v3` frame 0 gives 5.4 cm. A policy that puts both
  feet where the reference does has to leave the posture to stay up. v1_tip left the foot
  placement instead (its stance was 12 cm short and 7.6 cm wide) and kept the posture. v1_feet
  kept the feet and left the posture. The upstream reward pays for body positions, orientations
  and velocities but not for joint angles, so nothing steered which posture the policy picked.
  Yet the joint angles are exactly what the 0.1 rad gate measures.
- **Why not a v2 motion.** The end frame must stay `parry_quinte_v3` frame 0, so a v2 cannot
  rebalance the guard. A shorter end hold would cut the hold's share of the score, but end holds
  under the eval's 2 s leave policies falling in the post-motion hold (pipeline.md, known errors;
  evaluation.md, Protocol). Longer early phases would only dilute the score. All three were
  rejected.

**The change.** Two new tasks, additive: `Fencing-Tracking-Flat-G1-Saber-Entry-Joint` and
`-Joint05`. Each is the entry task plus one term, `joint_pos` = exp(−mean_j e_j² / 0.15²), over
the same joints and reference as the joint gate, at weight 1.0 and 0.5 (`fencing.saber.mjlab_task`,
`ENTRY_JOINT_WEIGHTS`). Observations, actions, events and every other term match the entry task.
`tests/test_saber_mjlab.py` checks this. A new snapshot (`tests/data/saber_entry_task_cfgs.json`)
pins the entry task as it was before. One round of two 3000-iteration trainings, 4096 envs, at the
same time (91 min). Runs `2026-11-12_07-52-56_stand_to_en_garde_v1_feetjoint` (policy sha256
`5560a095b29b2747e21ef15d6b7df36e419da8ec05a6eadc037466442e62fc4a`) and
`2026-11-12_07-52-58_stand_to_en_garde_v1_feetjoint05` (sha256
`b5be9916da54ac33729c0f12bf213c49b0b76750d404b71451140533b481515d`). The second round of the
budget was not needed.

**All entry candidates.** One run per simulator. MTC runs went one at a time on a quiet machine,
with no INVALID result:

| | v1_tip | v1_feet | v1_feet5k | v1_feetjoint | **v1_feetjoint05** |
|---|---|---|---|---|---|
| Task (3000 it unless noted) | saber | entry | entry, 5000 it | entry + joint_pos 1.0 | entry + joint_pos 0.5 |
| mjlab verdict, joint (rad) | PASS, 0.0580 | FAIL, 0.1007 | FAIL, 0.1013 | PASS, 0.0335 | **PASS, 0.0459** |
| sim2sim verdict, joint | PASS, 0.0555 | PASS, 0.0986 | FAIL, 0.1000 (0.10002) | PASS, 0.0314 | **PASS, 0.0435** |
| MTC held verdict, joint | PASS, 0.0560 | PASS, 0.0992 | FAIL, 0.1003 | PASS, 0.0317 | **PASS, 0.0441** |
| MTC standby verdict, joint | PASS, 0.0561 | PASS, 0.0985 | PASS, 0.0999 | PASS, 0.0318 | **PASS, 0.0438** |
| PhysX CPU verdict, joint (`sim2sim_physx.json`) | PASS, 0.0572 | PASS, 0.0987 | FAIL, 0.1012 | **FAIL**, 0.0325: `recovery_rate` 0 | **PASS, 0.0436** |
| PhysX final pelvis speed (m/s; recovery needs < 0.2) | 0.003 | 0.014 | 0.004 | **0.277** | 0.004 |
| Body, mjlab / sim2sim / MTC held / standby (cm) | 3.26 / 3.10 / 3.17 / 3.17 | 3.44 / 3.30 / 3.33 / 3.35 | 3.23 / 3.09 / 3.11 / 3.11 | 2.74 / 2.62 / 2.62 / 2.65 | 2.95 / 2.82 / 2.84 / 2.87 |
| Body, PhysX (cm) | 2.95 | 3.25 | 3.17 | 2.39 | 2.68 |
| Robustness `success_rate` (gated), four passes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `recovery_rate` push / obs_noise / friction (not gated) | 0.888 / 0.998 / 1.000 | 0.777 / 0.901 / 0.999 | 0.767 / 0.911 / 1.000 | 0.703 / 0.755 / 0.985 | 0.745 / 0.819 / 1.000 |
| Anchor orientation error, mjlab (rad) | 0.037 | 0.133 | 0.135 | 0.106 | 0.121 |
| Pelvis height error at the end, mjlab (cm) | 1.9 | 1.0 | 0.4 | 0.7 | 1.3 |
| Foot xy error at the end L / R, MTC held (cm) | 9.9 / 4.8 | 2.1 / 2.7 | 0.1 / 1.8 | 2.8 / 3.4 | 4.5 / 0.9 |
| Same, MTC standby (cm) | 9.8 / 6.1 | 2.3 / 2.8 | | 2.7 / 3.1 | 4.2 / 1.0 |
| Foot lift L / R, MTC held (cm; reference 6.9 / 6.1) | 0.4 / 0.6 | 5.0 / 1.6 | 6.1 / 1.7 | 7.3 / 1.7 | 5.8 / 0.9 |
| End stance, MTC held (m; guard 0.295, −0.145) | (0.180, −0.212) | (0.318, −0.161) | (0.313, −0.145) | (0.317, −0.161) | (0.309, −0.171) |
| Ankle body error, mjlab L / R (cm) | 10.8 / 7.4 | 4.3 / 5.0 | | 5.5 / 6.2 | 6.0 / 5.9 |
| Pelvis lean in the guard hold, CPU MuJoCo (reference 13.1°) | 9.9° | 22.9° | | 7.9° | 8.7° |
| Saber chains run through it | 10: 4 PASS | 10: 10 PASS | not run | 10: 10 PASS | **10: 10 PASS** |

`joint_pos` removed the posture error without giving up the stance, at both weights. The joint
error fell to a third (weight 1.0) and under half (weight 0.5) of v1_feet's (0.031 and 0.044 vs
0.099 rad in sim2sim), and it is lower in every phase. In the guard hold the pelvis leans 7.9°
and 8.7° (reference 13.1°, v1_feet 22.9°), with the soles flat. Weight 1.0 ends within 2 mm of
v1_feet's stance; weight 0.5 lands the feet 1 cm wider.

**PhysX separates them.** Weight 1.0 does not settle in PhysX: its pelvis is still moving at
0.289 m/s 1 s into the post-motion hold, 0.468 m/s at 4 s and 0.277 m/s at the end, so
`recovery_rate` is 0 and the verdict FAIL. Run with `--hold-seconds 8`, it terminates at 13.72 s
on `ee_body_pos`. The same policy holds 8 s in CPU MuJoCo. Its foot slip in PhysX is 0.056 m/s
against weight 0.5's 0.015. Weight 0.5 settles (0.004 m/s) and also holds 8 s. Why weight 1.0
drifts in PhysX alone was not investigated. Both were chained.

**Standby → entry → motion, both joint entries** (`scripts/mtc_sim2sim.sh <motion> --standby
--entry <entry>`, the same commands and switch times as before: R1 + A after 10 s, R1 + X 10 s
later). Each entry's reports are kept as `mtc_entry_stand_to_en_garde_v1_<tag>.json` (and `.log`)
in every eval dir, with their recordings in `$HCF_DATA/mtc/runs_entry_v1_<tag>/` (tags `feet`,
`feetjoint`, `feetjoint05`). The plain `mtc_entry.json` is the last chain run, v1_feetjoint05. One
run each, one at a time. Every recording was complete, with no INVALID reason and attempt 1. The
real-time factor was 1.00, except 0.99 for prime through v1_feetjoint05:

| # | Motion | Verdict, whole run: v1_feetjoint05 / v1_feetjoint | Joint (rad), j05 / fj | Body (cm), j05 / fj | v1_feet body | v1_tip body | After convergence, entry part | Leg torque at R1 + X, j05 / fj |
|---|---|---|---|---|---|---|---|---|
| 4 | extension_v3_tip | **PASS** / PASS | 0.074 / 0.073 | 4.38 / 4.30 | 4.04 | 5.18 | PASS, PASS (both) | 0.49 right ankle pitch / 0.62 right ankle pitch |
| 6 | parry_prime_v3_tip | **PASS** / PASS | 0.069 / 0.070 | 4.42 / 4.25 | 4.25 | 5.15 | PASS, PASS | 0.51 / 0.66 |
| 7 | parry_seconde_v3_tip | **PASS** / PASS | 0.089 / 0.088 | 4.60 / 4.50 | 4.42 | 5.23 | PASS, PASS | 0.61 / 0.68 |
| 8 | parry_tierce_high_v3_tip | **PASS** / PASS | 0.088 / 0.088 | 4.15 / 4.07 | 3.99 | 4.79 | PASS, PASS | 0.53 / 0.63 |
| 9 | parry_quinte_v3_tip | **PASS** / PASS | 0.068 / 0.069 | 4.12 / 4.24 | 4.03 | 4.84 | PASS, PASS | 0.65 left ankle pitch / 0.68 left ankle pitch |
| 10 | parry_sixte_v3_tip | **PASS** / PASS | 0.068 / 0.067 | 3.98 / 3.76 | 3.59 | 4.93 | PASS, PASS | 0.51 / 0.60 |
| 11 | thrust_v3_tip | **PASS** / PASS | 0.079 / 0.079 | 4.72 / 4.62 | 4.54 | 5.57 | PASS, PASS | 0.48 / 0.61 |
| 12 | cut_chest_v3_tip | **PASS** / PASS | 0.074 / 0.073 | 4.13 / 4.00 | 3.40 | 4.87 | PASS, PASS | 0.54 / 0.65 left ankle pitch |
| 13 | cut_flank_v3_tip | **PASS** / PASS | 0.062 / 0.062 | 4.31 / 4.21 | 4.10 | 5.09 | PASS, PASS | 0.48 / 0.61 |
| 15 | cut_head_v4_tip | **PASS** / PASS | 0.086 / 0.086 | 4.63 / 4.47 | 4.31 | 5.40 | PASS, PASS | 0.47 left ankle pitch / 0.60 left ankle pitch |

All twenty stayed up and converged at step 0 (0.04 s). The "any joint" torque peak at R1 + X is
1.00 on the left wrist pitch in all twenty. Unnamed leg joints are the right ankle pitch.
Through v1_feetjoint05 the body error is 3.98-4.72 cm, 0.1-0.2 cm above v1_feetjoint's (quinte
0.1 below) and
0.1-0.7 cm above v1_feet's, and still under the gate. Thrust is the closest, at 0.28 cm under
(v1_feetjoint 0.38). The joint offset from the motion's first frame at R1 + X is 0.058 rad mean
(v1_feetjoint 0.039, v1_feet 0.143, v1_tip 0.065). The stillness check before R1 + X says
`still: true` in all twenty: pelvis tilt 0.152 rad (v1_feetjoint 0.138) against standby's 0.2
limit, where v1_feet gave 0.397 and `false`. The R1 + A switch into v1_feetjoint05 peaked at
0.35 (right wrist pitch), legs 0.21 (right knee); v1_feetjoint 0.44 / 0.24, v1_feet 0.34 / 0.23.
The entry part scored 0.0438 rad in every v1_feetjoint05 chain. A low-priority test run overlapped
the extension, prime and seconde chains through v1_feetjoint05. None of the three reported an
INVALID reason or a timing mismatch.

**Decision: load `stand_to_en_garde_v1_feetjoint05`** as `walking_controller` for the ten saber
motions. Policy `$HCF_DATA/eval/2026-11-12_07-52-58_stand_to_en_garde_v1_feetjoint05/model_2999/policy.onnx`,
sha256 `b5be9916da54ac33729c0f12bf213c49b0b76750d404b71451140533b481515d`. It is the only entry
that passes in all five simulators and chains all ten motions. v1_feetjoint is rejected for its
PhysX FAIL, although it is ahead on joint error and chained body error. A pelvis that will not
settle in one of the two physics engines is the risk that matters most on the robot. What the
choice costs:

- **Chained body error up to 0.2 cm higher than through v1_feetjoint** (nine of ten; quinte is
  0.1 cm lower). Thrust's margin is 0.28 cm,
  the thinnest of any chain.
- **Disturbance recovery below v1_feet's.** `recovery_rate` under push is 0.745 and under
  obs_noise 0.819, against v1_feet's 0.777 / 0.901 and v1_tip's 0.888 / 0.998 (v1_feetjoint
  0.703 / 0.755). The field is not gated, and every robustness `success_rate` is 1.00: no episode
  terminated. The other episodes ended the hold with the pelvis more than 5 cm off height or
  moving faster than 0.2 m/s. Why it is lower was not investigated.
- **The left foot lands 4.5 cm off.** The right foot is within 0.9 cm, and the stance is 1 cm
  wider than the guard's. The front foot still slides most of its step (lift 0.9 cm against 6.1).
- One run per simulator and per chain. Margins: joint 0.054-0.056 rad under the gate in all five
  simulators. Chained body error is 0.28-1.02 cm under the gate.

### What it shows

- **Nothing falls any more.** Every Goal 1 policy stays up from standby in this simulator:
  directly for five footwork motions, through the entry for the ten saber motions, and
  through retreat_v1 for advance_v1.
- **With `stand_to_en_garde_v1_feetjoint05` as the entry (as with v1_feetjoint and v1_feet), 13 of 16
  motion policies pass every gate on the whole run from standby, and the other 3 only after
  settling.** Whole run: advance_lunge_v4
  and advance_lunge_cut_v5 switched in directly, advance_v1 through retreat_v1, and all ten
  saber motions through the entry. Only after convergence: en_garde_v3, retreat_v1 and
  lunge_v5. Switched in directly, their whole runs fail on the transition (converged 0.42, 0.40
  and 1.48 s after the switch), and they pass from convergence on. v1_feetjoint05 passes its own
  gates in all five simulators, which v1_feet (mjlab joint 0.1007) and v1_feetjoint (PhysX
  recovery) did not. With the v1_tip entry the whole-run count was 7 of 16.
- **The motion policies start on their reference.** Each one converged at its first scored step.
- **Cause of the six v1 failures: the v1 entry did not take its steps, so the motion started
  in the wrong stance.** Checked per body and per foot on the v1 runs:
  - The rise in error, chained against held, is in the back leg. Left ankle +7.7 to +8.9 cm and
    left knee +6.1 to +6.5 cm for thrust, extension and cut_head_v4 (prime, seconde and cut_flank
    were not broken down). For quinte, which passed, the rise is +3.4 and +2.4 cm. No other
    body rises by more than 2.0 cm, and the right ankle's error falls.
  - At R1 + X the robot's stance was (0.171, −0.221) m against the guard's (0.295, −0.145): the
    front foot 12 cm short, the feet 7.6 cm further apart sideways, the pelvis 0.73 m against
    0.711 (thrust recording; unchanged 1 s and 3 s later, so waiting longer would not help).
  - The v1 entry's feet moved 3.3 and 6.9 cm against 10.9 and 11.5 cm, and lifted 0.4 cm
    against 6 cm. It was the same in mjlab, so the cause is training, not the simulator.
  - Neither our gate nor the reward saw it. Two feet 11 cm off among 14 tracked bodies still
    gave a 3.3 cm mean body error. `motion_relative_body_position_error_exp` (std 0.3 m, mean
    over 14 bodies) charged about 2% of one term for it. Every footwork policy here has 7-18 cm
    foot errors (`metrics.json`).
  - The fix confirms the cause. The same reference, with the feet paid for, gives a 2-3 cm
    foot placement, and all ten chains pass.
- **Design choice, by this evidence.** A (entry + chain) for the saber motions, a chain through
  an existing policy for advance_v1, C (direct switch) for the five footwork motions that pass
  from standby. B (a lead-in baked into each motion) would mean 16 retrains, and every
  validated policy would change. A changes none of them.

### Limits

- One run per configuration. The recommended entry (v1_feetjoint05) clears the joint gate by
  0.054-0.056 rad in all five simulators. Its ten chains clear the body gate by 0.28-1.02 cm,
  thrust being the thinnest. v1_feet's margins were 0.0007 rad over the mjlab joint gate and
  0.0008-0.0015 rad under it elsewhere. No gate was changed for any entry.
- The joint-reward entries recover from disturbances less often than v1_feet: `recovery_rate`
  under push 0.745 (v1_feetjoint05) and 0.703 (v1_feetjoint), obs noise 0.819 and 0.755. The
  field is reported, not gated: every robustness `success_rate` is 1.00. The cause was not
  investigated. On the robot a shove during the entry's hold is the case it points to.
- v1_feetjoint's PhysX drift (pelvis 0.28-0.47 m/s in the post-motion hold, terminated at
  13.72 s with an 8 s hold) was not explained. Only two joint_pos weights were tried, so
  whether a weight between 0.5 and 1.0 would keep weight 1.0's margins and settle in PhysX is
  unknown.
- The guard (`parry_quinte_v3` frame 0) has its CoM about 5 cm outside the G1's double support
  (above). The entry reference is therefore not statically holdable as written, and nor is the
  first frame of any saber motion. This was measured kinematically, from the reference poses. How
  much of the saber motions' own 4-4.6 cm body error comes from it was not measured.
- The stillness check before R1 + X uses standby's limits. With v1_feetjoint05 it reports
  `still: true` (pelvis tilt 0.152 rad against 0.2; v1_feetjoint 0.138). With v1_feet it reported `false` on pelvis
  tilt alone (0.397-0.41 rad). That check is reported for the second switch, never INVALID. The
  pelvis tilt it reports does not match the plain pelvis tilt of the scored reference (0.229 rad
  in the guard), and that check's definition was not looked into.
- The v1 entry's own MTC held and standby runs were started while another agent's MTC run may
  have been active. Both recorded without timing problems (no INVALID), but CPU contention was
  not ruled out. The feet entry's runs and its ten chains ran one at a time.
- Foot placement is still not a gate. It is measured by `scripts/entry_foot_placement.py`,
  not by `fencing sim2sim mtc`.
- On the robot: `real_chain.launch.py` was checked by building its controller config in the
  rootfs, and its decisions with ROS stubbed (`tests/test_real_chain_launch.py`).
  `chain_policy_path` is required: a launch without it stops with "Required launch argument",
  and an empty value raises. The joystick mapping is read from joy.yaml, not tested. R1 + X is not interlocked
  against being pressed from standby. Everything in "Start from standby / Limits" applies (real
  standby binary, ramp, motors, estimator).
