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
- **Not covered:** hardware I/O (`real.launch.py`, unitree_systems), the standby controller and
  the joystick switch.

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
  actually active in that run, so that experiment is inconclusive.
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
| cut_head_v3 (`2026-11-10_05-31-03_cut_head_v3_tip`) | PASS / PASS / PASS | 0.085 / 0.084 / 0.084 | 4.0 / 3.9 / 3.8 | 2.5 / 2.3 / 2.2 | 0.043 / 0.046 / 0.042 | 1.00 / 1.00 / 1.00 | 2.7 / 2.6 / 2.7 | 4.1 / 4.2 / 4.4 |
| cut_flank_v3 (`2026-11-10_07-47-56_cut_flank_v3_tip`) | PASS / PASS / PASS | 0.068 / 0.068 / 0.068 | 4.2 / 4.2 / 4.1 | 2.4 / 2.4 / 1.9 | 0.046 / 0.048 / 0.042 | 1.00 / 1.00 / 1.00 | 1.9 / 1.8 / 2.0 | 2.5 / 2.7 / 2.3 |
| cut_chest_v3 (`2026-11-10_08-36-30_cut_chest_v3_tip`) | PASS / PASS / PASS | 0.078 / 0.078 / 0.077 | 4.0 / 4.0 / 3.9 | 2.3 / 2.2 / 1.9 | 0.044 / 0.045 / 0.042 | 1.00 / 1.00 / 1.00 | 1.7 / 1.6 / 1.6 | 3.2 / 2.6 / 1.9 |
| thrust_v3 (`2026-11-10_09-21-15_thrust_v3_tip`) | PASS / PASS / PASS | 0.080 / 0.079 / 0.078 | 4.6 / 4.6 / 4.4 | 2.7 / 2.3 / 2.0 | 0.044 / 0.047 / 0.043 | 1.00 / 1.00 / 1.00 | 1.8 / 1.7 / 1.5 | 2.8 / 3.3 / 1.9 |
| advance_lunge_v4 (`2026-11-11_09-00-26_advance_lunge_v4_pipe`) | PASS / PASS / PASS | 0.081 / 0.081 / 0.080 | 3.2 / 3.1 / 3.1 | 5.6 / 4.8 / 6.8 | 0.021 / 0.017 / 0.014 | 1.00 / 1.00 / 1.00 | – / – / – | – / – / – |
| advance_lunge_cut_v5 (`2026-11-11_11-15-05_advance_lunge_cut_v5_tip`) | PASS / PASS / PASS | 0.076 / 0.074 / 0.076 | 3.1 / 3.0 / 3.1 | 7.3 / 6.5 / 8.0 | 0.025 / 0.023 / 0.020 | 1.00 / 1.00 / 1.00 | 1.9 / 1.8 / 2.0 | 1.9 / 2.1 / 1.5 |

Footwork motions have no blade, so their blade columns are `–`. Each policy also has a rendered CPU-MuJoCo rollout, `sim2sim.mp4`, next to its scores (robot in white, reference as a green ghost).
