# Hardware test plan (Phase 1.6)

The plan a person follows at the real G1 for roadmap Phase 1.6. **Nothing in this repository sends commands to the robot.** People run every test, under [`safety/checklist.md`](../safety/checklist.md), which this plan does not replace. Nothing here has been run on hardware yet.

Every statement about how the real robot's software is operated cites its upstream source file. Where upstream does not say how to do something, this page says **not documented upstream: confirm with the lab's G1 operator**. All numbers about the policies come from `results/` (copied from the eval dirs) or were computed from the policy files themselves (see [Space](#space-each-motion-needs)).

## Decide before the first session

These are open risks for a person to decide on. None of them is settled by simulation.

1. **No Isaac Sim cross-check.** The Phase 1.6 entry rule is "only for motions that passed Phase 1.5 in both simulators", and "both" was written as Isaac Sim and MuJoCo ([roadmap.md](roadmap.md), Phase 1.5 "Done when"). Isaac Sim has not run on this machine. The 16 policies pass in mjlab, plain CPU MuJoCo and `motion_tracking_controller`'s own MuJoCo sim ([mtc.md](mtc.md#goal-1-final-set-2026-11-11)), all three MuJoCo-based. Decide whether that is enough, or wait for Isaac Sim.
2. **The start state was never tested the way the robot will start.** Training reset each episode to a random frame of the reference, perturbed by up to ±5 cm and ±0.1 rad on the root, ±0.1 rad on each joint, plus root velocity kicks (`results/policies/<run>/training/params/env.yaml`, `commands.motion`: `sampling_mode: adaptive`, `pose_range`, `joint_position_range`). Every evaluation, and every simulator test, started exactly on the motion's first frame, held there until the controller activated ([evaluation.md](evaluation.md#protocol), [mtc.md](mtc.md#deterministic-start)). On the real robot the policy is switched in from the standby controller (`third_party/motion_tracking_controller/README.md`, "Real Experiments"). The standby pose (`config/g1/controllers.yaml`, `standby_controller.default_position`) is a symmetric, feet-together stance. It differs from each motion's first frame by 0.24-0.42 rad per joint on average and up to 1.20 rad on single joints, and the feet stand in a different place ([table](#entry-conditions)). **Option (a) has now been run in simulation** ([mtc.md, Start from standby](mtc.md#start-from-standby-2026-11-11)). In `motion_tracking_controller`'s MuJoCo sim the robot stood still in `standby_controller` for 10 s, and then the `R1 + A` switch request started the policy, with no start hold on the policy. Results for the 16 policies:

- **Fell: advance_v1 and parry_quinte.** Both terminated within 0.52 s, and legs reached the torque clamp.
- **Stayed up, converged and passed the gate after the transient:** en_garde, retreat, lunge, advance-lunge, advance-lunge-cut. The last two also pass the whole run.
- **Stayed up but missed the gate even after converging:** extension, the other four parries and all four hits. Body error was 4.4-6.9 cm, and the feet did not return to the stance.
- **Saber motions:** every one drives the right wrist pitch and yaw into their torque limit for 0.04-0.17 s after the switch.
- **During the transient:** the torso moved 2-30 cm and the right foot up to 38 cm.

Changing `start_step` does not help: no motion has a frame much closer to standby than frame 0.

**This is one simulator run per policy.** It used the same controller config file and controller type as `real.launch.py`. The robot's actual standby binary, ramp time and motors are not verified to match.

Remaining options:
- (b) Have the lab set the standby pose to the guard's joint angles for the session. That changes the robot's controller config, is outside this repo, and was not simulated. It would not move the feet apart.
- (b') Add a lead-in from the standby pose to the guard. **Built and simulated as an entry policy** ([mtc.md, Entry from standby](mtc.md#entry-from-standby-2026-11-11)). The entry steps from the standby pose into the saber guard and holds it. The motion policy is a second controller, `walking_controller1`, switched in from that hold with `R1 + X`. `unitree_bringup`'s `config/g1/joy.yaml` already binds that button; `scripts/mtc/real_chain.launch.py` adds the controller without changing upstream files. Five entry policies were trained on the same reference:
  - `stand_to_en_garde_v1_tip` (saber task) passes its own gates. It shuffles its feet instead of stepping, and the motion then starts with the front foot about 12 cm short of the guard. Chained after it, four saber motions passed every gate. Six (extension, prime, seconde, thrust, flank cut, head cut v4) failed only the body gate, at 5.1-5.6 cm against 5 cm.
  - `stand_to_en_garde_v1_feet` (entry task, which also pays for foot position) steps to within 2-3 cm of the guard stance. Chained after it, **all ten saber motions passed every gate** (body 3.4-4.5 cm). But the entry itself fails the mjlab joint gate by 0.0007 rad (0.1007 against 0.1). It passes sim2sim, MTC held and MTC standby, by 0.0014, 0.0008 and 0.0015 rad. Its push and noise recovery rates in mjlab dropped (push 0.888 → 0.777, obs_noise 0.998 → 0.901; reported, not gated), and every robustness success rate is 1.00.
  - `stand_to_en_garde_v1_feet5k` (v1_feet trained 2000 iterations longer) places its feet within 2 cm, but fails the joint gate in mjlab, sim2sim and MTC held (0.1003-0.1013 rad), and PhysX (0.1012). It was not chained and is not a candidate.
  - **`stand_to_en_garde_v1_feetjoint05`** (2026-11-12; the entry task plus a reward on joint angles, weight 0.5) passes its own gates in all five simulators: joint 0.0459 rad in mjlab, 0.0435 in sim2sim, 0.0441 in MTC held, 0.0438 in MTC standby and 0.0436 in PhysX, against the 0.1 gate. Its feet land within 4.5 / 0.9 cm, 1 cm wider than the guard, and **all ten chained saber motions passed every gate** (body 3.98-4.72 cm; thrust is 0.28 cm under the gate). Why v1_feet missed the joint gate: the guard's CoM lies about 5 cm outside the G1's double support, so a policy that places both feet has to depart from the posture to stand. Nothing paid for joint angles, so v1_feet bent its waist and arms, with 0.14 rad of mean joint error in the guard hold ([mtc.md, Paid for joint angles](mtc.md#paid-for-joint-angles-stand_to_en_garde_v1_feetjoint05-2026-11-12-recommended)). Its cost: disturbance recovery below v1_feet's. In mjlab, `recovery_rate` is push 0.745 and obs_noise 0.819 (v1_feet 0.777 / 0.901, v1_tip 0.888 / 0.998). That field is reported, not gated, and every robustness success rate is 1.00.
  - `stand_to_en_garde_v1_feetjoint` (same, weight 1.0) has the lowest joint error (0.031-0.034 rad in mjlab and both MuJoCo setups, against the 0.1 gate) and also chains all ten saber motions (body 3.76-4.62 cm). **It fails PhysX**: its pelvis never settles in the post-motion hold (0.28 m/s at the end, against 0.2; `recovery_rate` 0), and with an 8 s hold it terminates at 13.72 s. The cause was not found. Rejected for that.
  - advance_v1 passed every gate when chained after retreat_v1, whose last frame is within 1.3 cm of advance_v1's first.
- (c) Accept the risk on the gantry, for the motions that switched cleanly in simulation only.

Recommendation: (c) for the five clean footwork motions (en garde, retreat, lunge, both advance-lunges). For the ten saber motions, (b'): standby → `R1 + A` (entry, 7.5 s, wait for its hold) → `R1 + X` (motion). **Entry to load: `stand_to_en_garde_v1_feetjoint05`**. It is the only entry that passes its own gates in all five simulators (mjlab, CPU MuJoCo, MTC held, MTC standby, PhysX) and lets all ten chained saber motions pass. The evidence is one run each, in simulation:
   - `stand_to_en_garde_v1_feetjoint05`: `$HCF_DATA/eval/2026-11-12_07-52-58_stand_to_en_garde_v1_feetjoint05/model_2999/policy.onnx`, sha256 `b5be9916da54ac33729c0f12bf213c49b0b76750d404b71451140533b481515d`.

   Kept for comparison, not recommended: `stand_to_en_garde_v1_feetjoint` (sha256 `5560a095b29b2747e21ef15d6b7df36e419da8ec05a6eadc037466442e62fc4a`; PhysX FAIL, does not settle), `stand_to_en_garde_v1_feet` (sha256 `40e9c72c42f27d38462ab382a4ae85e7f3304dd4c4aa61aa579d1f2f26afcce5`; mjlab joint FAIL 0.1007) and `stand_to_en_garde_v1_tip` (sha256 `1b08dc23a521d524ba81de5dd466db8d496bf1dd66a2fdab7646e3342f262372`; six chains miss the body gate). Robustness, mjlab `recovery_rate` under push / obs_noise (not gated; every `success_rate` is 1.00): v1_feetjoint05 0.745 / 0.819, v1_feetjoint 0.703 / 0.755, v1_feet 0.777 / 0.901, v1_tip 0.888 / 0.998. The recommended entry recovers less often than v1_feet from a shove or noisy observations in its hold. That is the first thing to watch on the gantry.

   Check the sha256 of the file on the robot's PC before loading it. For advance_v1: standby → `R1 + A` with retreat_v1 loaded as `walking_controller` → `R1 + X` (advance_v1). Never press `R1 + X` straight from standby: nothing interlocks it, and that is the start under which parry_quinte and advance_v1 fell in simulation. All of this is simulation only, one run each. How the real robot behaves at either switch is not known (see [entry conditions](#entry-conditions)).
3. **There is no low-gain or slow-playback mode.** The roadmap and the checklist ask for "low gains and slower playback first". The controller declares two parameters, `policy.path` and `motion.start_step` (`src/MotionTrackingController.cpp`, lines 13, 23-24). The PD gains are read from the ONNX metadata (`joint_stiffness`, `joint_damping`; [evaluation.md](evaluation.md#sim-to-sim)), and the reference is bundled in the ONNX at 50 Hz (`third_party/mjlab/src/mjlab/tasks/tracking/rl/runner.py`, `_OnnxMotionModel`). Lower gains or a slower reference would be a different policy from the one that passed simulation; a slower motion would need a new reference and a new training run. Decide whether to drop that item for these policies or to make slowed variants first (Phases 1.4-1.5 work).
4. **Where the margins are smallest.**
   - Blade tip at the moment of the hit: the head cut (`cut_head_v4`) has the largest tip error at the key frame in the C++ controller, 2.9 cm against the provisional 5 cm gate. Every other key frame is at 2.8 cm or less (parry seconde is next). The head cut's body error, 4.5 cm against the 5 cm gate, is now its thinnest margin ([mtc.md](mtc.md#goal-1-final-set-2026-11-11)).
   - Joint error: parry seconde 0.089-0.091 rad, parry tierce high 0.089-0.090 and the lunge 0.087-0.088, against 0.10. The lunge family passed only after five versions, and `lunge_v3`'s pass was luck in the end hold ([roadmap.md](roadmap.md), 1.5.4).
   - Wrist torque: the wrist pitch and yaw motors are limited to 5 Nm ([ADR 0002](adr/0002-saber-grip-and-mount.md)). [saber.md](saber.md#payload-at-en-garde) estimates 3.04 Nm at wrist yaw for the nominal saber and 5.19 Nm for a 0.35 kg one, at an assumed 100 rad/s². The references' peak wrist yaw accelerations (second difference of the ONNX reference at 50 Hz) are 123 rad/s² on advance-lunge-cut, 110 on tierce high and 105 on the flank cut (the v4 head cut is lower, at 69). Scaled the same way, that is about 3.7 Nm with the nominal saber and 5.4-6.4 Nm with a 0.35 kg one. This is inertia about one axis only, no gravity and no coupling, so treat it as an order of magnitude. Simulation carried the same 5 Nm limit and passed the 0.12-0.35 kg saber-mass sweep at 1.00 on every saber policy (`results/summary.csv`, `robust_saber_mass`).
   - Blade speed: the reference tip reaches 17.4 m/s (parry seconde), 16.2 (parry prime) and 12.7 (head cut). That is a foam blade moving at about 60 km/h, under 1 m from the robot.
   - Legs: no simulator run recorded joint torques against their limits, so there is no torque margin for the lunge's legs at all.
5. **Stretch targets.** Phase 1.6 is done when en garde, advance/retreat, the lunge, at least one parry and at least one hit run on the real G1 ([roadmap.md](roadmap.md), Phase 1.6). Advance-lunge with hit is a stretch target. The extension, the other parries and hits, and the footwork-only advance-lunge are optional. Decide which optional tests to run at all.
6. **The saber hardware does not exist yet.** The printed socket that replaces the rubber hand is a hardware to-do: measure the wrist flange, then design the socket so it reproduces the 0.10 m grip offset and the 30° pitch ([ADR 0002](adr/0002-saber-grip-and-mount.md), Consequences). The weapon must weigh 0.12-0.35 kg in total with the socket, nominal 0.20 kg. The footwork policies were trained on the G1 with its rubber hands and no saber (task `Mjlab-Tracking-Flat-Unitree-G1`), so they should run in that configuration.
7. **Gantry and blade.** The travelling motions move the pelvis up to 1.10 m and the blade tip up to 2.9 m from the start ([Space](#space-each-motion-needs)); the gantry has to follow without pulling. Quinte, tierce high and prime raise the tip to 1.41-1.64 m, near the harness lines. The saber has no collision in simulation ([ADR 0002](adr/0002-saber-grip-and-mount.md), decision 5), so a blade hitting a rope, the robot's own legs or the floor (parry seconde takes the tip to 0.10 m) was never modelled.
8. **Identifying the run.** The checklist asks for the commit and W&B link in `log.md`. Every Goal 1 run was logged at commit `2f8d260+dirty` with no W&B ([log.md](log.md)), so the commit does not pin the policy. This plan identifies each policy by the SHA-256 of its `policy.onnx` instead. Decide whether that satisfies the checklist item.

## How a policy runs on the real robot (upstream)

Everything in this table is from `third_party/motion_tracking_controller` (pinned at 742924f, [mtc.md](mtc.md)). Paths below are relative to that folder.

| Step | What upstream says | Source |
|---|---|---|
| Disclaimer | Running these models on real robots is dangerous, for research only, at your own risk | `README.md`, "Real Experiments" |
| Software | ROS 2 Jazzy, `legged_control2`, `ros-jazzy-unitree-description`, `ros-jazzy-unitree-systems`, and the `unitree_bringup` repo, built with colcon | `README.md`, "Installation" |
| Connect | Ethernet cable to the robot; set the adapter to static IP `192.168.123.11`; find the interface name with `ifconfig` | `README.md`, "Real Experiments" |
| Launch | `ros2 launch motion_tracking_controller real.launch.py network_interface:=<if> policy_path:=<absolute path or ~/...>.onnx` | `README.md`; `launch/real.launch.py` lines 33-35 expand the path |
| Start step | `start_step` launch argument, default `0`. Keep 0: every policy was evaluated from frame 0 only | `launch/real.launch.py` lines 36-37, 146-150 |
| External position correction | `ext_pos_corr` launch argument, default `false`. When true, the estimator takes position from `/glim/odom` and stops trusting contact height. Keep it off: our simulator tests ran without it, and the one sim run with it was inconclusive ([mtc.md](mtc.md#known-gaps)) | `launch/real.launch.py` lines 38-40, 151-155 |
| Controllers at launch | `state_estimator` and `standby_controller` are spawned active; `walking_controller` (the policy) is spawned inactive | `launch/real.launch.py` lines 53-59 |
| Load check | The console prints `Load Onnx model from <path> successfully !` when the policy is configured. Check that the path is the file in the [test table](#test-order-and-policy-files) | `src/MotionTrackingController.cpp` line 32 |
| Operator controls | The robot should start in the standby controller. Unitree remote: `L1 + A` standby (joint position control), `R1 + A` the policy, `B` e-stop (damping) | `README.md`, "Real Experiments" |
| Standby pose and gains | Hip pitch −0.312, knee 0.669, ankle pitch −0.33 rad on both legs, waist 0, arms (0.2, ±0.2, 0, 0.6, 0, 0, 0); leg kp 150-350, arm kp 40 | `config/g1/controllers.yaml`, `standby_controller` |
| Rates | Controller manager 500 Hz, policy 50 Hz | `config/g1/controllers.yaml` |
| What the policy does on activation | Places the reference so that its first frame's torso (`torso_link`) matches the robot's current torso position and yaw, then steps through the reference one frame per 20 ms tick | `src/MotionCommand.cpp` lines 37-41; `src/MotionOnnxPolicy.cpp` |
| Timeline | The controller observes the previous reference frame, so every frame, the hit included, arrives one tick (20 ms) after activation + frame × 20 ms | [mtc.md](mtc.md#one-tick-reference-lag-scorer-fixed-2026-11-11) |
| After the last frame | The ONNX clamps the time step at the last frame, so the policy keeps holding the final pose until the operator switches away. Simulation tested 2 s of that hold | `third_party/mjlab/src/mjlab/tasks/tracking/rl/runner.py` line 33; [evaluation.md](evaluation.md#protocol) |
| Recording | The launch runs `ros2 bag record -s mcap -a` with a regex that excludes most Unitree topics. No output folder is set, so the bag lands where `ros2 bag` puts it by default (a `rosbag2_<date>` folder in the launch's working directory; confirm) | `launch/real.launch.py` lines 104-130 |
| Joystick node | Included from `unitree_bringup`'s `teleop.launch.py`, which is not vendored here | `launch/real.launch.py` lines 132-136, 165-168 |

`third_party/whole_body_tracking/README.md` points to `motion_tracking_controller` for sim-to-real and adds nothing about the procedure. `third_party/mjlab`'s README and docs do not describe G1 deployment.

**Not documented upstream: confirm with the lab's G1 operator.**

- How to put the robot into the state where `real.launch.py` can take low-level control (any Unitree mode change on the remote before launching).
- How the robot is brought onto its feet before standby: lowered from the gantry, or standing first.
- What the standby controller does when the operator switches in: how fast it moves to its pose, and whether that is safe from a fencing stance.
- What `R1 + A` does to the transition from the standby pose to the motion's first frame. Nothing in the vendored code smooths it; the policy simply starts.
- Whether the policy can be started a second time without relaunching, and whether a second activation restarts the motion from frame 0 (`RlController` is in the closed `legged_controllers` package).
- What the `B` damping e-stop does physically (gains, whether it latches), and how to recover from it.
- How to lower gains or slow playback (see [risk 3](#decide-before-the-first-session): the controller has no such parameter).
- Which topic carries joint states in the real bag, and whether `/odom` is recorded. Run `ros2 bag info` on the first session's bag.
- Joint, velocity and torque safety limits in the Unitree hardware interface, and motor temperature limits.
- Whether the lab runs GLIM (`/glim/odom`), needed only for `ext_pos_corr:=true`.
- The network latency between the control PC and the robot.
- How much slack the harness should have, and whether the gantry can follow a 1.1 m travel.

## Test order and policy files

Same order as Phase 1.5. Footwork first, with no saber and the rubber hands on. Then the saber policies with the foam saber on its socket. Within the parries and the hits the order is slowest blade first (peak reference tip speed, [Blade sweep](#blade-sweep)), so the head cut, the largest key-frame tip error and the thinnest body margin, comes last among the hits. That ordering inside a group is this plan's choice, not the roadmap's.

Each file is `results/policies/<run>/model_2999/policy.onnx`. Check its SHA-256 (`sha256sum`) against the table before loading it. The scores next to it are `metrics.json`, `sim2sim.json` and `mtc.json`, and the simulator video is `results/videos/<motion>_<suffix>.mp4` (also `model_2999/sim2sim.mp4`).

| # | Block | Motion | Run (`results/policies/...`) | Task | SHA-256 of `policy.onnx` | Needed for Phase 1.6 |
|---|---|---|---|---|---|---|
| 1 | A, no saber | en garde | `2026-11-10_04-44-31_en_garde_v3_pipe` | plain G1 | `a98ebddb1feb27efc244171fa9e96407449e932086dd9c7d9cc76e8a115c9471` | yes |
| 2 | A | advance | `2026-11-09_19-10-15_advance_v1_pipe` | plain G1 | `d1571c7c649abd2607ea2f59fe626bd079160267ba7b53b610311fcea11f950a` | yes |
| 3 | A | retreat | `2026-11-09_20-40-56_retreat_v1_pipe` | plain G1 | `fbf6c23e354010520d5a641da6ee156b3a98d48cc015ab6bfd6a479b5dba9d31` | yes |
| 4 | A | lunge | `2026-11-11_11-15-56_lunge_v5_pipe` | plain G1 | `2e8df191e84cd07b9c0e8a50e497abf5100c3cbd8f159462ca1e7b9ac35c53a2` | yes |
| 5 | A | advance-lunge (footwork only) | `2026-11-11_09-00-26_advance_lunge_v4_pipe` | plain G1 | `d41373f9c5898c392496a6a66034c7677a17f903a535ae65226ed0cf3a8f33b1` | optional; the footwork of the stretch target |
| 6 | B, foam saber | extension | `2026-11-10_06-14-28_extension_v3_tip` | saber | `9f8088b78e08e43a7ede0c05004be10d8758a6a6572241077aa0258cd189af7b` | optional |
| 7 | B | parry sixte | `2026-11-10_11-42-25_parry_sixte_v3_tip` | saber | `813092f18f2c5f2985f42678c61afd5a7b3b4e6f9ce1158dd9ceef3ea8581570` | one parry needed |
| 8 | B | parry tierce high | `2026-11-10_12-26-40_parry_tierce_high_v3_tip` | saber | `e94a5f88bc2a383b79b7a8277ac7a0270042be92c6e1ebfa550671bb48321aab` | one parry needed |
| 9 | B | parry quinte | `2026-11-10_07-03-50_parry_quinte_v3_tip` | saber | `2dc354d2920b738f883e7ee944aa8e0e54097ad6dec6c1fd28e0beb0057a494c` | one parry needed |
| 10 | B | parry prime | `2026-11-10_10-09-18_parry_prime_v3_tip` | saber | `85e94ae1f101f481dadb82c19a47c8b6b0cb78d4d8648e06f675abba47e99b46` | one parry needed |
| 11 | B | parry seconde | `2026-11-10_10-54-06_parry_seconde_v3_tip` | saber | `c77ce49184b43541a0ff266725cb9bba20c98b6a5edca470b307a722d6ee3e03` | one parry needed |
| 12 | B | point thrust | `2026-11-10_09-21-15_thrust_v3_tip` | saber | `3255fc4f43bbb3fbb7fe9428ad0f2c88b4efe8d25aba3284c1a5035782342362` | one hit needed |
| 13 | B | chest cut | `2026-11-10_08-36-30_cut_chest_v3_tip` | saber | `10683765d30edf58e3c495b2372ee14eee98a867f2b3a9345a29f0dcfff85df1` | one hit needed |
| 14 | B | flank cut | `2026-11-10_07-47-56_cut_flank_v3_tip` | saber | `6ea4b83e3d0e425c1a0166ae13c7ff0533594162d0b0bd9d9abb2cfefebf5a06` | one hit needed |
| 15 | B | head cut | `2026-11-11_13-49-44_cut_head_v4_tip` | saber | `b8af2ecefb6a691d98e10aefdd460d3c05ffc1ff2865eb483a2cdf1a730a7a60` | one hit needed |
| 16 | C, stretch | advance-lunge with head cut | `2026-11-11_11-15-05_advance_lunge_cut_v5_tip` | saber | `6134a6e7271ff142ac8eb5cfa06c5f9b14444f652b8ca7f3286add7e439a06dc` | stretch target |

"Plain G1" is task `Mjlab-Tracking-Flat-Unitree-G1`; "saber" is `Fencing-Tracking-Flat-G1-Saber` (`export.json`, `task`). A saber policy on a robot without the saber, or a plain policy on a robot with it, is a robot it was not trained on.

**Moving on.** Phase 1.5 required each motion to pass before the next. This plan proposes the same at the robot: go to the next row only after the current one has run cleanly (no abort, still for 2 s after the end) on three attempts in a row, and do every first attempt on the gantry or harness (checklist). The count of three is a proposal for the person running the session to confirm.

## Entry conditions

**For every test.**

- Every item of [`safety/checklist.md`](../safety/checklist.md) ticked.
- The right file: SHA-256 matches the table, and the launch console names it (`Load Onnx model from ...`).
- The right robot configuration: rubber hands and no socket for block A; socket and foam saber at 0.12-0.35 kg for blocks B and C ([risk 6](#decide-before-the-first-session)).
- `start_step` 0 and `ext_pos_corr` false (the launch defaults).
- The robot stable in standby (`L1 + A`) before the switch (checklist). Give it at least 5 s after standby's pose ramp: in simulation the torso swayed fore and aft for about 5 s after the 2 s ramp, and every simulated switch was made from a robot standing still ([mtc.md](mtc.md#start-from-standby-2026-11-11)).
- The robot's torso facing the direction you want "forward" to be. The policy places the whole motion relative to the torso's position and heading at the moment of `R1 + A` (`src/MotionCommand.cpp` lines 37-41), so the space below is drawn from there.

**Start pose of each motion against standby.** On activation the robot is in the standby pose, not the motion's first frame ([risk 2](#decide-before-the-first-session)). The table shows how far apart they are: first-frame joint angle minus the standby `default_position`, from the ONNX reference at time step 0. Feet are the ankle positions at frame 0 in the start heading frame (x where the torso faces, y to its left, from the pelvis).

| Motions | First-frame pelvis height (m) | Feet at frame 0 (left; right) (m) | Mean joint offset from standby (rad) | Largest offsets (rad) |
|---|---|---|---|---|
| en garde | 0.711 | (−0.26, −0.04); (+0.04, −0.19), 0.33 apart | 0.27 | left shoulder yaw −0.82, right wrist pitch −0.71, left wrist pitch −0.55 |
| advance | 0.713 | (−0.00, +0.35); (+0.15, −0.05), 0.43 apart | 0.26 | right wrist yaw −0.93, right hip pitch −0.65, left wrist roll +0.65 |
| retreat | 0.703 | (−0.01, +0.32); (+0.20, −0.07), 0.44 apart | 0.24 | right wrist yaw −0.89, right hip pitch −0.77, left wrist roll +0.53 |
| lunge | 0.722 | (−0.21, +0.06); (+0.11, −0.15), 0.38 apart | 0.33 | left hip yaw +0.88, left shoulder yaw −0.84, right hip roll −0.80, right hip yaw −0.75 |
| advance-lunge | 0.643 | (−0.00, +0.21); (−0.04, −0.27), 0.48 apart | 0.38 | right wrist roll −0.97, right shoulder yaw −0.91, left hip roll +0.83, left knee +0.74 |
| extension, all parries, all hits (the `*_v3` set, same first frame) | 0.711 | as en garde | 0.37 | right wrist yaw −1.17, right elbow +0.84, right wrist pitch −0.83, left shoulder yaw −0.82 |
| advance-lunge with cut | 0.643 | as advance-lunge | 0.42 | right shoulder yaw −1.20, right wrist yaw −1.12, left hip roll +0.83, left knee +0.74 |

**How each motion switched in from standby in simulation** ([mtc.md](mtc.md#start-from-standby-2026-11-11), one run each, MTC MuJoCo; [risk 2](#decide-before-the-first-session) for what is and is not verified). "Converged" is joint error under 0.15 rad for 0.5 s. Torque is the peak in the first second after the switch as a fraction of the joint's limit.

| # | Motion | From standby in sim | Converged after switch (s) | Torso / right foot moved (cm) | Peak torque (joint) | Entry condition this suggests |
|---|---|---|---|---|---|---|
| 1 | en garde | clean; passes after convergence | 0.42 | 15 / 22 | 0.55 (left ankle pitch) | standby start acceptable on the gantry; expect one recovering step |
| 2 | advance | **fell** at 0.52 s | never | 94 / 128 | 1.00 (left ankle pitch) | **do not start from standby** until a lead-in or a guard-pose standby is simulated |
| 3 | retreat | clean; passes after convergence | 0.40 | 6 / 2 | 0.92 (right wrist yaw) | standby start acceptable on the gantry |
| 4 | lunge | stays up; passes after convergence | 1.48 | 19 / 30 | 0.56 (right hip pitch) | acceptable on the gantry; slower settling, the lunge starts displaced |
| 5 | advance-lunge | clean; passes whole run | 0.24 | 3 / 26 | 0.59 (left ankle pitch) | standby start acceptable on the gantry |
| 6 | extension | stays up; fails body error (6.2 cm) | 0.14 | 4 / 4 | 1.00 (right wrist pitch) | gantry; expect worse tracking than the held-start numbers |
| 7 | parry sixte | stays up; torso turns, fails heading (0.25 rad) | 0.80 | 30 / 38 | 1.00 (left ankle pitch; legs at the limit) | gantry; mark the start heading and expect a turned stance |
| 8 | parry tierce high | stays up; PARTIAL (key frame inside the transient) | 2.20 | 12 / 20 | 1.00 (right wrist pitch) | gantry; the key frame is not tracked from standby |
| 9 | parry quinte | **fell** at 0.40 s | never | 88 / 55 | 1.00 (left knee; 8 joints at the limit) | **do not start from standby** until a lead-in or a guard-pose standby is simulated |
| 10 | parry prime | stays up; fails body error (5.05 cm, at the threshold) | 0.42 | 11 / 11 | 1.00 (right wrist pitch) | gantry |
| 11 | parry seconde | stays up; fails joint, body, tip at key | 0.38 | 16 / 21 | 1.00 (right wrist pitch) | gantry; expect worse tracking |
| 12 | point thrust | stays up; fails joint and body | 0.18 | 2 / 2 | 1.00 (right wrist pitch) | gantry; expect worse tracking |
| 13 | chest cut | stays up; fails joint and body | 0.20 | 6 / 29 | 1.00 (right wrist pitch) | gantry; expect worse tracking |
| 14 | flank cut | stays up; fails tip at key (5.0 cm, at the threshold) | 0.72 | 17 / 18 | 1.00 (right wrist pitch) | gantry |
| 15 | head cut | stays up; fails joint (0.127 rad), body (6.7 cm) and tip at key (9.1 cm). The left foot never reaches its stance (30 cm off after convergence) | 0.16 | 3 / 10 | 1.00 (right wrist pitch) | gantry; the cut is not tracked from standby |
| 16 | advance-lunge with cut | clean; passes whole run | 0.26 | 2 / 6 | 1.00 (right wrist pitch, 0.09 s) | standby start acceptable on the gantry |

`start_step` does not help any of them: the closest frame to standby in any motion is no closer than frame 0 (mean 0.24-0.35 rad). Keep `start_step` 0. A lead-in into the guard was then simulated (next table). A standby `default_position` at the guard's joint angles was not. Whether the real standby behaves like the simulated one is not verified beyond the shared config file and controller type.

**Chained start in simulation** ([mtc.md, Entry from standby](mtc.md#entry-from-standby-2026-11-11), one MTC MuJoCo run each). Launch with `scripts/mtc/real_chain.launch.py policy_path:=<entry or retreat ONNX> chain_policy_path:=<motion ONNX>`. Operator sequence: `L1 + A` standby and wait ≥ 5 s; `R1 + A` (entry); wait until the entry has finished and holds the guard; `R1 + X` (motion). What was simulated: `R1 + A` 10 s after standby was active, and `R1 + X` 10 s after `R1 + A` for `stand_to_en_garde` (a 7.5 s reference, so 2.5 s after it ended) and 5 s after `R1 + A` for retreat_v1 (1.9 s, so 3.1 s after it ended). Shorter waits were not simulated. Torque is the peak in the first second after each switch as a fraction of the joint's limit. The table is for the recommended entry, `stand_to_en_garde_v1_feetjoint05` (2026-11-12). Its `R1 + A` switch peaked at 0.35 (right wrist pitch), legs 0.21 (right knee), in every run (v1_feetjoint: 0.44 / 0.24; v1_feet: 0.34 / 0.23; v1_tip: 0.40 / 0.20). Leg torque at `R1 + X` is 0.47-0.65 (v1_feetjoint: 0.60-0.68; v1_feet: 0.42-0.77; v1_tip: up to 0.59). The joint offset from the motion's first frame at `R1 + X` is 0.058 rad mean (v1_feetjoint 0.039, v1_feet 0.143, v1_tip 0.065).

| # | Motion | First policy (`R1 + A`) | Result after `R1 + X` in sim | Converged after `R1 + X` (s) | Peak torque at `R1 + X`: any joint / legs | Entry condition this suggests |
|---|---|---|---|---|---|---|
| 2 | advance | retreat_v1 | PASS whole run (joint 0.081 rad, body 3.6 cm) | 0.04 | 1.00 right wrist pitch / 0.42 | chain after retreat on the gantry; never from standby directly |
| 4 | extension | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.38 cm); with v1_feetjoint: PASS 4.30 cm; with v1_feet: PASS 4.04 cm; with v1_tip: FAIL body 5.18 cm | 0.04 | 1.00 left wrist pitch / 0.49 right ankle pitch | chained, on the gantry |
| 6 | parry prime | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.42 cm); with v1_feetjoint: PASS 4.25 cm; with v1_feet: PASS 4.25 cm; with v1_tip: FAIL body 5.15 cm | 0.04 | 1.00 / 0.51 | chained, on the gantry |
| 7 | parry seconde | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.60 cm); with v1_feetjoint: PASS 4.50 cm; with v1_feet: PASS 4.42 cm; with v1_tip: FAIL body 5.23 cm | 0.04 | 1.00 / 0.61 | chained, on the gantry |
| 8 | parry tierce high | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.15 cm); with v1_feetjoint: PASS 4.07 cm; with v1_feet: PASS 3.99 cm; with v1_tip: PASS | 0.04 | 1.00 / 0.53 | chained, on the gantry |
| 9 | parry quinte | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.12 cm); with v1_feetjoint: PASS 4.24 cm; with v1_feet: PASS 4.03 cm; with v1_tip: PASS | 0.04 | 1.00 / 0.65 left ankle pitch | chained only; never from standby directly |
| 10 | parry sixte | stand_to_en_garde_v1_feetjoint05 | PASS (body 3.98 cm); with v1_feetjoint: PASS 3.76 cm; with v1_feet: PASS 3.59 cm; with v1_tip: PASS | 0.04 | 1.00 / 0.51 | chained, on the gantry |
| 11 | thrust | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.72 cm); with v1_feetjoint: PASS 4.62 cm; with v1_feet: PASS 4.54 cm; with v1_tip: FAIL body 5.57 cm | 0.04 | 1.00 / 0.48 | chained, on the gantry |
| 12 | chest cut | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.13 cm); with v1_feetjoint: PASS 4.00 cm; with v1_feet: PASS 3.40 cm; with v1_tip: PASS | 0.04 | 1.00 / 0.54 | chained, on the gantry |
| 13 | flank cut | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.31 cm); with v1_feetjoint: PASS 4.21 cm; with v1_feet: PASS 4.10 cm; with v1_tip: FAIL body 5.09 cm | 0.04 | 1.00 / 0.48 | chained, on the gantry |
| 15 | head cut (cut_head_v4) | stand_to_en_garde_v1_feetjoint05 | PASS (body 4.63 cm); with v1_feetjoint: PASS 4.47 cm; with v1_feet: PASS 4.31 cm; with v1_tip: FAIL body 5.40 cm | 0.04 | 1.00 / 0.47 left ankle pitch | chained, on the gantry |

What simulation cannot show here: whether the real G1 steps out of standby into the guard the way the simulated one does (foot friction, motor torque-speed limits, the real estimator restarting at `R1 + X`: in simulation the controller manager restarts `state_estimator` together with the policy controller at that switch, not at `R1 + A`); whether the operator presses `R1 + X` while the entry is still moving; and how two `MotionTrackingController` instances behave in the robot's controller manager. That last point was checked only by generating the controller config in the rootfs and by `tests/test_real_chain_launch.py` with ROS stubbed. `real_chain.launch.py` refuses to start without `chain_policy_path`. The 1.00 wrist-pitch peaks are on a welded, rigid saber.

The right foot is the front foot in every motion (left-handed source footage is mirrored, [roadmap.md](roadmap.md) 1.4d). Measured from the torso's facing direction, the line through the feet runs forward and to the robot's right, which is why the travelling motions go that way ([Space](#space-each-motion-needs)).

## Space each motion needs

**How this was computed.** For each policy, the reference bundled in its own `policy.onnx` was read for every time step (the exact motion the robot will track). The saber G1 model (`src/fencing/saber/assets/g1_29dof_saber.xml`, the saber removed for plain policies) was posed in each frame. Every geom's bounding box was taken in world coordinates, which over-estimates the body by a few centimetres. Positions are in the **start heading frame**: origin at the pelvis at frame 0 (the torso is within 1 cm of it), +x where the torso faces at frame 0, +y to the robot's left. That is the frame the controller anchors the motion to on activation (`src/MotionCommand.cpp` lines 37-41). The blade tip is the `blade_tip` site, 0.82 m along the blade ([saber.md](saber.md)). Forward kinematics reproduce the ONNX body positions to within 4e-7 m. CPU only; the script is in the [appendix](#appendix-envelope-script).

**Margin.** The checklist asks for at least 2 m clear beyond the motion's reach in every direction. "Clear radius" is the largest distance of any part of the robot (blade included) from the start pelvis, rounded up to 0.1 m, plus 2.0 m. A circle needs no heading, so it holds even if the robot is not facing the way you expect. The x/y extents let you check a smaller rectangle once the heading is known. The margin also has to absorb the real robot's deviation from the reference, which the simulators put at 2-8 cm of torso drift ([mtc.md](mtc.md#goal-1-final-set-2026-11-11)) and the real robot may exceed.

| # | Motion | Reference length (s) | Pelvis travel, end (x, y) (m) | Lowest pelvis (m) | Robot x extent (m) | Robot y extent (m) | Reach radius (m) | Clear radius (m) |
|---|---|---|---|---|---|---|---|---|
| 1 | en garde | 3.30 | (0.00, −0.01) | 0.71 | −0.34 to +0.42 | −0.25 to +0.35 | 0.42 | **2.5** |
| 2 | advance | 1.80 | (+0.25, −0.21), 0.33 m | 0.69 | −0.13 to +0.65 | −0.59 to +0.48 | 0.80 | **2.8** |
| 3 | retreat | 1.90 | (−0.29, +0.16), 0.33 m | 0.68 | −0.39 to +0.35 | −0.39 to +0.63 | 0.66 | **2.7** |
| 4 | lunge | 5.78 | (+0.28, −0.50), 0.57 m | 0.56 | −0.29 to +1.14 | −1.29 to +0.37 | 1.64 | **3.7** |
| 5 | advance-lunge | 7.08 | (+0.06, −1.10), 1.10 m | 0.58 | −0.26 to +0.50 | −2.33 to +0.50 | 2.34 | **4.4** |
| 6 | extension | 4.38 | (0.00, −0.01) | 0.71 | −0.34 to +1.20 | −0.25 to +0.46 | 1.25 | **3.3** |
| 7 | parry sixte | 4.00 | (0.00, −0.01) | 0.71 | −0.34 to +0.86 | −0.25 to +0.35 | 0.91 | **3.0** |
| 8 | parry tierce high | 4.18 | (0.00, −0.01) | 0.71 | −0.34 to +0.98 | −0.25 to +0.35 | 1.00 | **3.0** |
| 9 | parry quinte | 4.00 | (0.00, −0.01) | 0.71 | −0.34 to +0.86 | −0.30 to +0.67 | 0.91 | **3.0** |
| 10 | parry prime | 4.00 | (0.00, −0.01) | 0.71 | −0.36 to +0.86 | −0.25 to +0.81 | 0.91 | **3.0** |
| 11 | parry seconde | 4.00 | (0.00, −0.01) | 0.71 | −0.34 to +0.86 | −0.25 to +0.87 | 0.91 | **3.0** |
| 12 | point thrust | 3.78 | (0.00, −0.01) | 0.71 | −0.34 to +1.25 | −0.25 to +0.47 | 1.30 | **3.4** |
| 13 | chest cut | 3.68 | (0.00, −0.01) | 0.71 | −0.34 to +1.20 | −0.48 to +0.35 | 1.22 | **3.3** |
| 14 | flank cut | 3.64 | (0.00, −0.01) | 0.71 | −0.34 to +0.87 | −0.25 to +0.88 | 1.16 | **3.2** |
| 15 | head cut | 3.70 | (0.00, −0.01) | 0.71 | −0.34 to +1.11 | −0.25 to +0.89 | 1.20 | **3.2** |
| 16 | advance-lunge with cut | 7.10 | (+0.06, −1.10), 1.10 m | 0.58 | −0.26 to +0.78 | −2.92 to +0.50 | 2.92 | **5.0** |

Reference lengths include each motion's own end hold. The tallest point of the robot is 1.25-1.32 m except where the blade goes overhead (next table).

**The travelling motions go to the robot's right, not straight ahead.** The fencer stands side-on with the right foot forward, and the motion is anchored to where the torso faces. The lunge ends 0.57 m away at 61° to the right of the torso's facing direction, and the advance-lunge 1.10 m at 87° to the right, almost sideways. The advance goes 40° to the right, the retreat 151° (back and to the left). The torso also turns during the motion: 29° on the lunge, 19° on the advance-lunge, 14° on tierce high, 9° on the retreat and the thrust, 5° or less elsewhere.

### Blade sweep

Blade tip in the start heading frame, from the reference. Heights are above the floor. The key frame is when the reference reaches the parry or the hit ([evaluation.md](evaluation.md#key-frames), the motion's sidecar `key_frames`); on the controller timeline it comes 0.02 s later ([mtc.md](mtc.md#one-tick-reference-lag-scorer-fixed-2026-11-11)).

| # | Motion | Tip x (m) | Tip y (m) | Tip height (m) | Tip reach from start pelvis (m) | Peak tip speed (m/s, at s) | Key frame (s) |
|---|---|---|---|---|---|---|---|
| 6 | extension | +0.85 to +1.19 | +0.30 to +0.45 | 1.03 to 1.21 | 1.25 | 2.0 at 0.82 | 1.07 |
| 7 | parry sixte | +0.74 to +0.85 | +0.02 to +0.30 | 1.21 to 1.51 | 0.90 | 2.6 at 0.76 | 0.90 |
| 8 | parry tierce high | +0.48 to +0.97 | −0.15 to +0.30 | 1.16 to **1.64** | 0.99 | 4.9 at 0.90 | 1.07 |
| 9 | parry quinte | +0.07 to +0.85 | +0.30 to +0.67 | 1.21 to 1.48 | 0.90 | 7.4 at 0.72 | 0.90 |
| 10 | parry prime | −0.35 to +0.85 | +0.30 to +0.80 | 0.30 to 1.41 | 0.90 | **16.2** at 0.72 | 0.90 |
| 11 | parry seconde | −0.02 to +0.85 | +0.02 to +0.87 | **0.10** to 1.26 | 0.90 | **17.4** at 0.72 | 0.90 |
| 12 | point thrust | +0.85 to +1.25 | +0.30 to +0.45 | 0.93 to 1.21 | 1.30 | 3.3 at 0.76 | 0.97 |
| 13 | chest cut | +0.85 to +1.20 | −0.46 to +0.30 | 0.79 to 1.33 | 1.22 | 10.0 at 0.72 | 0.87 |
| 14 | flank cut | +0.59 to +0.86 | +0.30 to +0.88 | 0.59 to 1.21 | 1.16 | 10.1 at 0.70 | 0.83 |
| 15 | head cut | +0.34 to +1.11 | +0.30 to +0.89 | 0.40 to 1.21 | 1.19 | 12.7 at 0.72 | 0.90 |
| 16 | advance-lunge with cut | −0.01 to +0.77 | −2.91 to −0.97 | 0.39 to 1.30 | 2.91 | 8.7 at 4.80 | 3.10 |

In the guard (frame 0 of every `*_v3` motion) the tip is 0.85 m ahead of the pelvis and 0.30 m to the robot's left, at 1.21 m height. On advance-lunge-cut the blade travels with the body: the tip ends almost 3 m to the robot's right.

## Expected behaviour and the sim numbers to compare

The C++ controller (MTC) column is the closest to hardware: same controller, same ONNX parsing, its own state estimator ([mtc.md](mtc.md#why-it-matters-for-the-goal-1-gate)). mjlab and plain-MuJoCo numbers are in [mtc.md](mtc.md#goal-1-final-set-2026-11-11) and [`results/SUMMARY.md`](../results/SUMMARY.md); they agree with MTC to within 0.002 rad on joint error. Every policy had success 1.00 and recovery 1.00 in all three simulators, and 1.00 in every robustness pass (push, observation noise, friction 0.3-1.2, saber mass) in mjlab (`results/summary.csv`). Thresholds are in [evaluation.md](evaluation.md#thresholds-provisional).

| # | Motion | What it should look like | Joint (rad) | Body (cm) | Torso drift (cm) | Foot slip (m/s) | Tip at key (cm) |
|---|---|---|---|---|---|---|---|
| 1 | en garde | Settles into the fencing stance and holds it. No steps | 0.076 | 3.8 | 2.8 | 0.045 | – |
| 2 | advance | One advance, 0.33 m net, then still | 0.079 | 3.2 | 3.3 | 0.080 | – |
| 3 | retreat | One retreat, 0.33 m net, then still | 0.086 | 2.9 | 2.0 | 0.036 | – |
| 4 | lunge | Lunge (pelvis down to 0.56 m), recovery to en garde, 0.57 m net travel, then still | 0.088 | 3.9 | 5.8 | 0.046 | – |
| 5 | advance-lunge | Advance, lunge, recovery; 1.10 m net travel, then still | 0.080 | 3.1 | 6.8 | 0.014 | – |
| 6 | extension | Sword arm extends, blade forward; legs still | 0.074 | 4.2 | 2.4 | 0.039 | 1.5 |
| 7 | parry sixte | Hand at chest height outside, blade up and forward; holds, back to guard; legs still | 0.072 | 4.1 | 3.5 | 0.034 | 1.6 |
| 8 | parry tierce high | Tierce with the hand at chest height, blade near vertical; holds, back to guard; legs still | 0.089 | 3.8 | 2.8 | 0.036 | 1.7 |
| 9 | parry quinte | Blade horizontal over the head, holds, back to guard | 0.073 | 4.6 | 2.7 | 0.032 | 1.5 |
| 10 | parry prime | Blade hanging down across the body, holds, back to guard | 0.074 | 4.1 | 2.1 | 0.038 | 2.0 |
| 11 | parry seconde | Blade down and out, holds, back to guard | 0.089 | 4.4 | 2.0 | 0.040 | 2.8 |
| 12 | point thrust | Arm and blade extend in a line, holds, back to guard | 0.078 | 4.4 | 2.0 | 0.043 | 1.9 |
| 13 | chest cut | Arm extended, blade angled right and down; back to guard | 0.077 | 3.9 | 1.9 | 0.042 | 1.9 |
| 14 | flank cut | Arm extended at chest height, blade sweeping left; back to guard | 0.068 | 4.1 | 1.9 | 0.042 | 2.3 |
| 15 | head cut | Arm extended at shoulder height, blade angled down; back to guard | 0.088 | **4.5** | 1.6 | 0.042 | 2.9 |
| 16 | advance-lunge with cut | Advance, lunge with a head cut at touchdown, recovery; 1.10 m net travel | 0.076 | 3.1 | 8.0 | 0.020 | 1.5 |

"Body" is drift-free tracked-body error, "torso drift" is world-frame torso (anchor) error ([evaluation.md](evaluation.md#metrics)). On hardware, only some of these can be measured; see [Comparing with simulation](#comparing-with-simulation).

## Abort criteria

The operator presses `B` (damping e-stop, `README.md`) at any of these. With the gantry or harness on, that is the safe stop.

**From the checklist, for every motion:** oscillation, a foot slipping, a joint hitting its limit, overheating, unexpected contact.

**From the simulator's own failure checks** ([evaluation.md](evaluation.md#protocol)), which a person can judge by eye: the pelvis sinking much lower than in the sim video (the simulator ends an episode when the torso is 0.25 m off the reference height), the torso tipping, or a hand or ankle far off its reference height (0.25 m).

**Specific to these motions:**

| Motions | Stop when |
|---|---|
| en garde, extension, all parries, all hits | Any step at all. The reference pelvis moves 1 cm and the feet stay planted |
| advance, retreat | More than one step, or the robot ends more than about 0.3 m from the reference end point (0.33 m away) |
| lunge, advance-lunge, advance-lunge with cut | The robot goes the wrong way (the travel is to the robot's right, see [Space](#space-each-motion-needs)), does not stop after the recovery, or the gantry pulls |
| every motion, at the end | Not still within 2 s of the end of the reference. Simulation tested 2 s of the final hold and nothing longer. The policy keeps holding the last frame until someone switches away (`runner.py` line 33) |
| every saber motion | The blade touches the harness lines, the gantry, the robot's own legs or the floor, or anything inside the cleared area. Buzzing or chatter at the wrist, which would be the 5 Nm wrist motors at their limit |
| at the switch (`R1 + A`) | A violent jump from the standby pose to the first frame ([risk 2](#decide-before-the-first-session)) |

How to end a clean run normally (back to standby with `L1 + A`, or straight to damping) is not documented upstream for a robot standing in a fencing stance: confirm with the lab's G1 operator.

## What to record

For every attempt, including aborted ones:

- **Video** (checklist). Two angles if possible: one along the robot's right side (the direction of travel) and one from the front. Start before `R1 + A` and stop after the robot is back in standby or damped. A visible clock, or the operator calling the switch, helps match the video to the bag.
- **The ROS bag** that `real.launch.py` records (`launch/real.launch.py` lines 124-130). Note its folder name.
- **The launch console output**, including the `Load Onnx model from ...` line, saved to a file.
- **The policy:** run name and SHA-256 from the test table.
- **The setup:** saber measured mass (blocks B and C), gantry or harness and how it was rigged, floor surface, battery level at start.
- **Floor marks:** tape at both feet and the pelvis's floor projection before the switch; mark where the feet end up. That gives the travel to compare with the [Space](#space-each-motion-needs) table.
- **Motor temperatures** and any hardware issue (checklist).

## Results template

One row per attempt. Keep the table in the session notes, and add one `hw` row per motion per session to [log.md](log.md).

| Date | # | Motion | Run | SHA-256 (first 12) | Attempt | Saber (kg) | Restraint | Outcome | Still 2 s after end | Travel measured / reference (m) | Abort reason | Video | Bag | What differed from sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | 1 | en garde | `2026-11-10_04-44-31_en_garde_v3_pipe` | `a98ebddb1feb` | 1 | – | gantry | completed / e-stop / caught / fell | yes / no | 0.0 / 0.01 | | link | folder | |

The matching [log.md](log.md) row, in its existing columns:

| Date | Stack | Motion | Data | Commit | W&B | Result | Notes |
|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | hw | en_garde | en_garde_v3 | this repo's short SHA | — | PASS / FAIL: n of N attempts completed, still after the end in n; travel x.xx m against 0.01; aborts and why | run `2026-11-10_04-44-31_en_garde_v3_pipe`, ONNX sha256 `a98ebddb1feb...`, restraint, saber mass, video links, bag folders, what differed from sim |

## Comparing with simulation

**What can be compared directly.** Completed without an abort (sim success 1.00), still after the end (sim recovery 1.00), net travel from the floor marks (the Space table), and foot slip and stepping by eye from the video.

**What needs a tool this repo does not have.** No scorer for a real bag exists. `fencing.sim2sim.mtc` scores simulator bags, and its body, anchor and blade-tip metrics come from the simulator's exact pelvis pose, which a real robot does not have ([mtc.md](mtc.md#scoring-conventions)). Joint error could be computed from the recorded joint states against the ONNX reference, pairing tick k with reference frame k − 1 (the one-tick lag). Body and blade-tip error on hardware would depend on the robot's own state estimate, or on an external measurement (motion capture or video).

**What simulation did not cover.** These are where the real robot can differ, and the first places to look when it does.

| Not covered in simulation | Why it matters | Source |
|---|---|---|
| Isaac Sim cross-check | All three simulators are MuJoCo-based, with related contact models | [evaluation.md](evaluation.md#sim-to-sim), roadmap 1.1 |
| Start from the standby controller and the `R1 + A` switch: **partly covered** | Simulated in MTC MuJoCo, one run per policy, with the same controller config and controller type ([entry conditions](#entry-conditions)). Still not covered: the robot's own standby binary and ramp time, how the robot gets into standby, a robot still swaying at the switch (sim sway took about 5 s to settle), the joystick nodes, and the real state estimator, which the switch restarts | [mtc.md](mtc.md#start-from-standby-2026-11-11) |
| Real hardware I/O (`real.launch.py`, `unitree_systems`) | The simulators ran the MuJoCo launch only | [mtc.md](mtc.md#what-is-covered-and-what-is-not) |
| Actuator model | The plants are mjlab's G1 and `unitree_description`'s `g1.xml`: PD actuators with effort limits, joint armature, 0.1 N m frictionloss in `g1.xml` and none in mjlab. No motor torque-speed curve, backlash, heating, or variation between motors. No run recorded joint torques against their limits | [mtc.md](mtc.md#saber-plant), [saber.md](saber.md) |
| Saber flex and the real socket | The saber is one rigid body welded to the wrist. A foam blade bends, and the socket is not designed yet | [ADR 0002](adr/0002-saber-grip-and-mount.md) |
| Saber contact | The saber has no collision geometry. A blade touching the legs, the floor or the harness was never simulated | [ADR 0002](adr/0002-saber-grip-and-mount.md), decision 5 |
| Saber mass outside 0.12-0.35 kg, or a different mount | Only that mass range was randomized in training; geometry was fixed | [saber.md](saber.md#mjlab-tasks) |
| Floor friction outside 0.3-1.2 | That is the friction range trained and tested; nominal 0.6 | [evaluation.md](evaluation.md#passes) |
| State estimator at activation | In the simulator the estimated pelvis height was 7-15 cm high at activation (0.08 m on most policies, 0.15 m on the advance-lunge pair) and corrected within about 0.1 s. That came from the simulator's start hold, so the real robot, already standing on its feet, may behave differently | [mtc.md](mtc.md#deterministic-start), `mtc.json` `estimator.at_first_tick` |
| Estimator drift on travelling motions | The estimator's largest horizontal position error in the C++ controller sim was 1.1-2.6 cm on the static and short motions, 3.8 cm on the advance-lunge, 5.3 cm on the lunge and 8.2 cm on advance-lunge-cut. Drift grows with distance travelled (about 50 cm after 90 s on dance1). The real robot's IMU and feet are not simulated | [mtc.md](mtc.md#known-gaps), `mtc.json` `estimator.pelvis_pos_error_xy_m` |
| Communication latency | The simulators model the controller's one-tick reference lag, not the sensing and network latency between the robot and the control PC | [mtc.md](mtc.md#one-tick-reference-lag-scorer-fixed-2026-11-11) |
| Metadata precision | The C++ reads 3-decimal metadata; the plain-G1 wrist action scale runs as 0.075 instead of 0.0745. MTC ran with the same rounding, so this is covered, but it is a known difference from training | [mtc.md](mtc.md#known-gaps) |
| Holding the last frame longer than 2 s | The eval hold was 2 s; the robot will hold until the operator switches | [evaluation.md](evaluation.md#protocol) |
| Pushes during the motion on the C++ controller | MTC ran nominal only; pushes and other robustness passes ran in mjlab | [mtc.md](mtc.md#what-is-covered-and-what-is-not) |
| A real environment | Harness forces, gantry lines, an uneven or soft floor, people moving nearby | |

## Appendix: envelope script

Run with the project's Python (`~/hcf-data/venvs/fencing/bin/python hw_envelope.py <repo> <run>`), CPU only. It prints one motion's numbers for the Space and Blade sweep tables. The tables above were produced by an equivalent longer script that also computed the start-pose offsets; this shorter one was checked to reproduce the lunge, head cut (the previous version's row, with the 0.2 s attack) and advance-lunge-cut rows exactly. The v4 head cut rows were computed with it. Frame 0 of the previous and the current head cut motion is identical, so their start-pose offsets are the same.

```python
"""Space envelope of one policy's reference motion. Usage: python hw_envelope.py <repo> <run>."""
import json, sys
from pathlib import Path
import mujoco, numpy as np, onnxruntime as ort

repo, run = Path(sys.argv[1]), sys.argv[2]
d = repo / "results/policies" / run / "model_2999"
exp = json.loads((d / "export.json").read_text())
sess = ort.InferenceSession(str(d / "policy.onnx"), providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
jn, bn = meta["joint_names"].split(","), meta["body_names"].split(",")
pel, anc = bn.index("pelvis"), bn.index(meta["anchor_body_name"])
obs = np.zeros((1, sess.get_inputs()[0].shape[1]), np.float32)
ref = [sess.run(["joint_pos", "body_pos_w", "body_quat_w"], {"obs": obs, "time_step": np.array([[t]], np.float32)})
       for t in range(exp["motion_frames"])]  # the reference bundled in the ONNX, 50 Hz
J, P, Q = (np.array([r[i][0] for r in ref]) for i in range(3))

m = mujoco.MjModel.from_xml_path(str(repo / "src/fencing/saber/assets/g1_29dof_saber.xml"))
x = mujoco.MjData(m)
saber = "Saber" in exp["task"].split("-")
sab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "saber")
tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "blade_tip")
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]
box = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)

w, qx, qy, qz = Q[0, anc]  # start heading frame: torso yaw at frame 0, origin at the pelvis
yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
local = lambda xy: (xy - P[0, pel, :2]) @ R.T

pts, tips = [], []
for t in range(len(J)):
    x.qpos[:3], x.qpos[3:7] = P[t, pel], Q[t, pel]
    x.qpos[adr] = J[t]
    mujoco.mj_kinematics(m, x)
    for g in range(m.ngeom):  # every robot geom's bounding box, world corners
        if m.geom_bodyid[g] == 0 or (m.geom_bodyid[g] == sab and not saber):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        pts.append(x.geom_xpos[g] + (c + box * h) @ x.geom_xmat[g].reshape(3, 3).T)
    tips.append(x.site_xpos[tip_id].copy())
pts, tips = np.concatenate(pts), np.array(tips)
L, pl = local(pts[:, :2]), local(P[:, pel, :2])
print(f"{run}: {len(J) * 0.02:.2f} s, pelvis end {pl[-1].round(2)} m, lowest pelvis {P[:, pel, 2].min():.2f} m")
print(f"  body x {L[:, 0].min():+.2f}..{L[:, 0].max():+.2f}  y {L[:, 1].min():+.2f}..{L[:, 1].max():+.2f}"
      f"  radius {np.linalg.norm(L, axis=1).max():.2f} m  top {pts[:, 2].max():.2f} m")
if saber:
    tl, v = local(tips[:, :2]), np.linalg.norm(np.diff(tips, axis=0), axis=1) / 0.02
    print(f"  tip x {tl[:, 0].min():+.2f}..{tl[:, 0].max():+.2f}  y {tl[:, 1].min():+.2f}..{tl[:, 1].max():+.2f}"
          f"  z {tips[:, 2].min():.2f}..{tips[:, 2].max():.2f}  peak speed {v.max():.1f} m/s at {(v.argmax() + 1) * 0.02:.2f} s")
```

The start-pose offsets use the same ONNX reference at time step 0, minus `standby_controller.default_position` from `third_party/motion_tracking_controller/config/g1/controllers.yaml`, matched by joint name. The wrist accelerations in [risk 4](#decide-before-the-first-session) are the largest second difference of the right wrist yaw and pitch reference angles at 50 Hz.
