# Hardware test checklist

**Nothing in this repository sends commands to a real robot.** Hardware tests are run by people only. Work through this list for **every** session. If any item can't be ticked, don't run the test.

> Draft. Review it with whoever is responsible for the G1 before the first Phase 1.6 session, and have them sign off on it.

## Before the session

- [ ] The policy passed the evaluation gate in simulation, including `motion_tracking_controller` MuJoCo sim-to-sim, with the same ONNX file you will run
- [ ] The run and the ONNX file are recorded in `docs/log.md` (commit, W&B link)
- [ ] At least two people present: an **operator** (remote and e-stop) and a **spotter** (gantry / harness, watching the space)
- [ ] Everyone knows the stop controls: Unitree remote `B` = damping e-stop. Test the e-stop at the start of every session
- [ ] Battery charged. Robot inspected: no loose cables, no damaged joints or feet
- [ ] The `policy.onnx` you will load has the SHA-256 listed for that test in [`docs/hardware_tests.md`](../docs/hardware_tests.md), and the launch console's `Load Onnx model from ...` line names that file
- [ ] The open decisions at the top of `docs/hardware_tests.md` have been made and written down, in particular how the robot gets from the standby pose to the motion's first frame
- [ ] The robot is in the configuration the policy was trained on: rubber hands and no saber socket for footwork (plain G1) policies; socket and foam saber for saber policies

## Space and equipment

- [ ] **Gantry or harness attached** for every first attempt of a motion, and for every motion with a lunge or jump
- [ ] Cleared area: at least 2 m beyond the motion's reach in every direction (lunges travel forward)
- [ ] No people inside the cleared area while the policy runs, other than the spotter at the gantry
- [ ] **Foam or padded saber only.** Never a metal blade. The weapon mass matches the sim model (Phase 1.3)
- [ ] Floor is flat, dry and non-slip. Friction is roughly in the simulated range
- [ ] The cleared radius for this motion in `docs/hardware_tests.md` is marked on the floor around the start position, and the robot's facing direction is marked. The travelling motions go to the robot's right, not straight ahead
- [ ] The gantry or harness can follow the motion's travel (up to 1.1 m) without pulling, and the harness lines are clear of the blade's overhead sweep (up to 1.64 m tip height)
- [ ] Saber socket reproduces the simulated mount: grip centre 0.10 m from the wrist link, blade pitched 30° towards the thumb side ([ADR 0002](../docs/adr/0002-saber-grip-and-mount.md)). Total weapon mass measured, within 0.12-0.35 kg

## Running

- [ ] Start in the standby controller (`L1 + A`), check the robot is stable, then switch to the policy (`R1 + A`)
- [ ] First attempt: lower gains and/or slower playback if available. Go up step by step
- [ ] Record video of every attempt
- [ ] Stop at the first sign of: oscillation, foot slipping, joint limit hits, overheating, unexpected contact
- [ ] Switch away from the policy within 2 s of the motion's end. The policy holds the last frame until someone switches; simulation tested only 2 s of that hold

## After

- [ ] Log every attempt in `docs/log.md` (stack `hw`): pass/fail, video link, what differed from sim
- [ ] Note motor temperatures and any hardware issues
- [ ] Keep the ROS bag that `real.launch.py` records and the launch console output with the video, named by attempt
