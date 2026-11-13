# Roadmap

Teaching a Unitree G1 humanoid (29 DoF) to fence with a saber, from replaying human fencing motions up to fencing on its own.

The roadmap has four **goals**. A goal is a capability the robot shows. Each goal is split into **phases**, which are concrete engineering steps. Goal 1 is planned in detail. Goals 2–4 are outlines that we will expand once Goal 1 is done and we know what actually worked.

| Goal | Capability | Method | Status |
|---|---|---|---|
| **1** | Footwork and saber coordination (replayed motions) | Video → GVHMR → GMR → BeyondMimic (mjlab) | Simulation done in mjlab, CPU MuJoCo and the C++ controller sim (2026-11-11). Open: Isaac Sim cross-check, SMPL-X/GVHMR, W&B, hardware tests |
| **2** | Targeted hits and blade perception | Goal-conditioned policy + head-camera perception | Planned |
| **3** | Footwork and bladework combined, aimed at a target | Goal-conditioned whole-body policy | Planned |
| **4** | Reactive and autonomous fencing | Reactive policy (e.g. guided diffusion) + real-time perception | Stretch |

**Rules that apply to every goal**

- **Simulation first.** Every policy passes sim-to-sim before it goes near hardware: trained in mjlab, then run in `motion_tracking_controller`'s MuJoCo launch, plus Isaac Sim where practical ([ADR 0001](adr/0001-training-stack.md)).
- **Humans run hardware.** Nothing in this repo sends commands to the real robot. Hardware tests follow [`safety/checklist.md`](../safety/checklist.md): gantry or harness, cleared space, foam or padded weapons only.
- **Upstream code stays upstream.** mjlab, BeyondMimic, Isaac Lab, GMR, GVHMR and motion_tracking_controller are pinned submodules in `third_party/`. Setup: [`setup.md`](setup.md).
- **Log every run** in [`log.md`](log.md): date, motion, data version, commit, W&B link, result.

---

## Goal 1: Footwork and saber coordination

**Target:** the G1 performs fencing footwork, saber actions, and combinations of the two, learned from video of human fencers. This is first in simulation, then in human-supervised hardware tests.

**Motions in scope (in order of difficulty)**

| # | Motion | What it tests |
|---|---|---|
| 1 | En garde (static) | Holding balance in a low, wide stance with the saber up |
| 2 | Advance / retreat | Footwork and shifting weight |
| 3 | Extension | Precise arm movement while balanced |
| 4 | Lunge | Fast, explosive whole-body movement; landing and recovery |
| 5 | Parries (e.g. prime, tierce, quinte) | Precise blade positions; wrist range |
| 6 | Hits (head, flank, chest cuts; point thrust) | Fast arm and wrist motion that ends at a precise blade position |
| 7 | Advance-lunge with saber hit | Footwork, lunge and cut combined in one motion |

**Not in scope for Goal 1:** reacting to anything. Every policy in Goal 1 replays a fixed reference motion. Aiming at a target belongs to Goal 2, reacting to an opponent to Goal 4.

### Phase 1.0: Repository setup ✅

- [x] Repo structure, license, contribution docs
- [x] Upstream code pinned as git submodules (`scripts/setup_third_party.sh`)

- [x] Setup scripts: `scripts/setup_mac.sh`, `scripts/setup_linux_gpu.sh`, kinematic replay `scripts/replay_csv.sh`
- [x] Training stack decided: mjlab primary, Isaac Lab kept ([ADR 0001](adr/0001-training-stack.md))

**Done when:** a fresh clone plus `setup_third_party.sh` gives the pinned upstream code.

### Phase 1.1: Environments

- [x] Mac: mjlab + GMR installed, LAFAN1 downloaded, CSV → NPZ and kinematic replay verified on CPU
- [x] Linux machine with an NVIDIA GPU: the PC under WSL2 (RTX 5080), not native Ubuntu. Recorded in `log.md`
- [ ] Install mjlab, Isaac Sim 4.5 + Isaac Lab v2.1.0, `whole_body_tracking`, GVHMR (`scripts/setup_linux_gpu.sh`). *mjlab, GMR, MediaPipe and GVHMR code installed; Isaac Sim is not available on WSL2. Checked 2026-11-11: headless Isaac Sim still needs an NVIDIA Vulkan driver, which WSL2 does not expose, and upstream closed the issue as not planned. The untried route is a native Windows install (20–50 GB on a drive with 70 GB free, EULA to accept), left for you; details in [setup.md](setup.md)*
- [ ] Manual downloads: SMPL / SMPL-X models, GVHMR checkpoints. *GVHMR checkpoints present; SMPL-X needs your registration, so pose uses the MediaPipe fallback* ([setup.md](setup.md#manual-steps-licences-and-logins-cannot-be-scripted))
- [ ] Set up the W&B organization and motion registry. *Skipped for now: runs use `WANDB_MODE=disabled` and local NPZs*
- [ ] Run an mjlab dummy-agent check and Isaac Lab's own sample task to confirm the installs. *mjlab verified (smoke test + training); Isaac Lab not installed*
- [x] ROS 2 Jazzy + `motion_tracking_controller` (Docker on 22.04) for sim-to-sim; can wait until Phase 1.2's last step. *Done without Docker or root (2026-11-11): an Ubuntu 24.04 rootfs under `~/hcf-data/mtc`, entered with `unshare -r` + chroot. Build steps and limits in [mtc.md](mtc.md)*

**Done when:** an mjlab task and an Isaac Lab sample both train without errors and log to W&B.

### Phase 1.2: Prove the pipeline on a known motion

Train one motion that BeyondMimic already supports, before adding any fencing data. If this fails, the problem is the setup, not our data.

- [x] Choose a reference motion from the BeyondMimic examples: LAFAN1 `dance1_subject2`
- [ ] Train with the standard recipe and leave it unchanged, in mjlab and in Isaac Lab. *mjlab run paused at iteration 1867 of 10000 to free the GPU for fencing runs. Resumed 2026-11-10 13:16 from `model_1500` as `dance1_subject2_p12v2r` (`hcf train run ... --resume 2026-11-09_14-57-04_dance1_subject2_p12v2 --checkpoint 1500 --iterations 8500`, unchanged recipe, ends at 10000). The warm start was checked: episode length 357 steps 40 iterations after the resume, against 7.5 for a fresh start. **Result at 10000 it: mjlab FAIL and sim2sim FAIL, on the mean joint gate only:** joint 0.108 (sim2sim 0.108) against 0.10. Everything else passes: success 1.00, body 3.9 cm, anchor 9.8 cm and 0.068 rad, slip 0.059, recovery 0.94 (sim2sim 1.00), push 0.97, noise and friction 1.00. Without the six wrist joints it is 0.096 (wrists alone 0.163), the same pattern as the saber runs. The unchanged upstream recipe on its own reference motion misses our provisional joint gate, so the gate is too tight as written, not the setup broken (see [evaluation.md](evaluation.md)). Whether 10000 it is enough for this 133 s clip, compared with the paper's runs, I have not checked. No Isaac Lab. **Update 2026-11-11: that reading was wrong.** The 10000-it run was a third of the upstream length. `dance1_subject2_p12v2r30k` (resumed from `model_9999`, 30,000 it in total, unchanged recipe, `model_29998`) passes every gate in mjlab: joint 0.093, body 3.3 cm, anchor 5.8 cm, slip 0.046, recovery 1.00. The gate stands. The only thing missing is the Isaac Lab run.*
- [ ] Evaluate in both, then MuJoCo sim-to-sim with `motion_tracking_controller`. *`hcf train eval`, the CPU MuJoCo `hcf sim2sim` and now `scripts/mtc_sim2sim.sh` (the C++ controller's MuJoCo launch) all run. dance1 in the C++ controller: joint 0.110 against 0.108 in Python sim2sim, so the two agree; it fails the same provisional joint gate, and the anchor position gate too (30 cm, probably state-estimator drift over this long travelling motion; not confirmed). Isaac Lab is still missing, so the box stays open. **At 30,000 it (2026-11-11, `p12v2r30k`) all four pass:** mjlab joint 0.093, CPU MuJoCo 0.093, CPU PhysX 0.095, C++ controller 0.094 with anchor 24.5 cm (gate 25, the same odometry drift). Isaac Lab is the only thing missing*
- [x] Check that mjlab's ONNX export loads in `motion_tracking_controller` ([ADR 0001](adr/0001-training-stack.md) open item). *It loads and runs unchanged (2026-11-11). `tests/test_mtc_contract.py` pins the inputs, outputs and metadata keys the C++ reads. One known difference: the C++ uses mjlab's 3-decimal metadata, so the wrist action scale is 0.075 instead of 0.0745 (+0.67%), with no measurable effect on tracking*
- [x] Write down the metrics we will reuse for every fencing motion: [evaluation.md](evaluation.md) (thresholds provisional)

**Done when:** the known motion tracks in both simulators with numbers close to the ones BeyondMimic reports.

### Phase 1.3: Saber and hardware decisions

Runs alongside 1.1–1.2. It has to be settled before training any saber motion.

- [x] **Grip:** the Dex3 hand holding a handle, or a fixed mount on the wrist? A fixed mount is simpler and more repeatable. Decide and record it as an ADR. *Fixed mount, [ADR 0002](adr/0002-saber-grip-and-mount.md)*
- [x] **Wrist range:** check that the G1's wrist joint limits can reach the parry and cut positions. If they can't, list which positions are possible. *10 of 12 at the 30° mount, [saber.md](saber.md); the wrist action scale had to be widened for training*
- [x] **Training weapon:** a foam or padded saber, its length and mass. Check the effect on the arm's payload and the robot's balance. *0.20 kg nominal, randomized 0.12–0.35 kg; at 0.35 kg fast cuts need 5.19 Nm at the 5 Nm wrist*
- [ ] **Sim model:** add the saber to the G1 model (URDF/MJCF) as a rigid body with correct mass and inertia, in both Isaac Sim and MuJoCo. *MuJoCo/mjlab done (`Fencing-Tracking-Flat-G1-Saber`); the URDF snippet for Isaac is untested*
- [x] **Blade tip:** define a blade-tip frame so hit and parry accuracy can be measured.

**Done when:** the G1 model with the saber loads in both simulators and the grip decision is written down.

### Phase 1.4: Fencing motion dataset

Turn fencing video into G1 reference trajectories.

**1.4a Source video**
- [x] Collect clips in [`data/video_log.md`](../data/video_log.md): links only, never the video files. *83 videos from `hcf scrape run`; downloads stay in `$HCF_DATA/videos/raw`*
- [x] Prefer one fencer, side-on camera, steady shot, full body visible, good lighting. *`hcf screen` scores people count, view, camera motion and full-body visibility and ranks usable ranges into `shortlist.json`. Lighting is not scored. 22 videos have no usable range and are marked `rejected`*
- [x] Use saber specifically: footwork and blade actions differ from foil and épée. *The search queries are saber-only. The weapon in each video is not verified automatically: the lunge clip's raised rear arm looked like foil or épée form and was replaced (1.5.4)*
- [ ] Tag each clip with the motion type, start and end time, and the fencer's handedness. *Motion is tagged on all 83 rows and start/end on 61 (`scrape log --annotate`, 2026-11-11). Handedness (2026-11-11): 38 rows are tagged, 10 from the original pose runs and 28 from a 12 s pose window on each shortlisted video, every one checked by eye against its overlay. 23 `new` rows stay empty: the guesser was undecided on 6, and 17 guesses were withheld because the overlay showed a wrong arm, no fencing, or a switched track. The guesser's arm cue adds almost nothing and MediaPipe swaps left and right in back and side views, so do not trust it unchecked; row #3 is mirrored footage. The 22 `rejected` rows have no usable range. Quality (A/B/C) is left for a person*

**1.4b Isolate the fencer**
- [x] For clips with two fencers: detect and track people, and keep one fencer's track through the clip (`hcf screen`; it can still switch fencers in crowded shots, and the plausibility gate rejects those)
- [ ] Check how GVHMR's built-in detection handles two fencers close together. Add a separate tracking step if it doesn't cope.
- [x] Cut clips into single-motion segments (`fencing.motion.segment`, plus manual `motion trim`)

**1.4c Human pose (GVHMR)**
- [ ] Run GVHMR to get SMPL-X body motion in world coordinates. *Blocked on SMPL-X registration; MediaPipe world landmarks used instead*
- [x] Check visually: overlay the pose on the video and reject bad segments (overlay PNGs plus the automatic plausibility gate)
- [x] Note what GVHMR can't see: wrist rotation and blade angle are poorly captured. Parries and cuts may need hand-edited wrist motion or keyframes

**1.4d Retarget to the G1 (GMR)**
- [x] Retarget onto the G1 with GMR (from MediaPipe landmarks for now, not SMPL-X)
- [x] Add the saber: drive the wrist so the blade lands in the intended parry or cut position (`motion blade` keyframes)
- [x] Mirror for handedness so every motion uses the same sword hand (`CleanConfig.mirror_left_handed`, on by default)

**1.4e Clean up the data** (where most of the effort goes)
- [x] Foot contacts: no sliding while a foot is planted, no feet below the floor
- [x] Vertical drift: remove the slow up-and-down drift of the pelvis from pose estimation
- [x] Segment selection: trim to clean start and end poses, ideally starting and ending in en garde
- [x] Joint limits and speeds within the G1's specs
- [x] Save as small CSVs in `motions/`, with a version number and a short note on where each came from. *All footwork motions (en garde, advance, retreat, lunge, advance-lunge) are video-derived; the lunge and advance-lunge come from video_log #35. Bladework is keyframed on those bases. Each CSV has a JSON sidecar with its source and cleanup steps*
- [x] Kinematic replay check: `scripts/replay_csv.sh motions/<motion>_vN.csv`. *2026-11-11: the newest version of all 16 Goal 1 CSVs replays without physics (`fencing motion check` forward kinematics, and mjlab's `csv_to_npz` for every trained motion). 12 pass the quality gates. 4 warn: `advance_v1` planted-foot float 3.5 cm; `lunge_v3` slip 2.8 cm and floor penetration 0.9 cm; `advance_lunge_v3` and `advance_lunge_cut_v3` penetration 1.0 cm. Contact sheets (`fencing motion render`) were inspected for en garde, advance, lunge, advance-lunge-cut, head cut and quinte: feet planted, lunge shape and recovery plausible. The sheets draw the plain G1 without the blade, and nobody has watched the full-speed mp4s.*

**Done when:** each of the seven motions has at least one cleaned, versioned reference CSV that plays back properly in simulation without physics (kinematic replay).

### Phase 1.5: Train and evaluate in simulation

Same BeyondMimic recipe for every motion. If a motion fails, **fix the data first** (Phase 1.4e), and only change the recipe as a last resort, written down as an ADR.

Train in this order. Each motion has to pass before starting the next:

- [x] 1.5.1 En garde. *`en_garde_v1` at 3000 it (2026-11-09): mjlab PASS and CPU-MuJoCo sim2sim PASS, which agree within about 1 mm. Joint error 0.089 vs 0.10, slip 0.094 vs 0.10, recovery 0.98, push/noise/friction 1.00. The margins are thin, the thresholds are provisional, and nothing has run in `motion_tracking_controller` yet, so this stays unticked. With pushes on, about a third of envs fall during the hold after the motion. Success only scores the motion itself. `en_garde_v2` (front foot turned down the line; see 1.5.6) tracked better: joint 0.077, body 3.8 cm, and the right-leg offset gone. But it drifted about 6 cm forward during the motion and fell 1.3 s into the hold (recovery 0.07, FAIL). Cause: mjlab resamples an env when its motion ends, so no policy ever practises more than the reference's 0.5 s end hold. `en_garde_v3` adds a 2 s end hold. **`en_garde_v3_pipe` (2026-11-10, 3000 it): mjlab PASS and sim2sim PASS**, better than v1 on every metric: joint 0.077 (v1 0.089), body 3.8 cm (4.2), slip 0.044 (0.094, so the thin margin is gone), anchor 2.9 cm, recovery 1.00 with no terminations, robustness 1.00. The right hip yaw offset is gone. What's left is right hip pitch 0.23 rad and ankle roll 0.20. `en_garde_v3` is the new en garde base for the `*_v3` bladework set. **C++ controller (2026-11-11): PASS**, joint 0.076.*
- *Ordering deviation: to use the GPU overnight, the later motions were queued in parallel without waiting for each earlier one to pass. Their verdicts are in `log.md`.*
- [x] 1.5.2 Advance / retreat. *`advance_v1` (2026-11-09, 3000 it): mjlab PASS and sim2sim PASS. Joint error 0.079, body 3.2 cm, slip 0.096 vs 0.10 (thin margin), recovery and robustness 1.00. `retreat_v1` (3000 it): mjlab PASS and sim2sim PASS. Joint error 0.088, body 3.0 cm, slip 0.051, recovery 1.00. **C++ controller (2026-11-11): both PASS**, advance joint 0.079 and slip 0.080, retreat joint 0.086.*
- [x] 1.5.3 Extension. *`extension_v2_tip` (2026-11-09, 3000 it): mjlab PASS and sim2sim PASS. Blade tip 1.8 cm, body 4.9 cm vs 5.0, joint 0.080, robustness including saber mass 1.00. It needed two recipe changes: the wider wrist action scale ([saber.md](saber.md#wrist-action-scale)) and a blade-tip reward ([ADR 0003](adr/0003-blade-tip-reward.md)). `motion_tracking_controller` has not been run yet, so this stays unticked. **`extension_v3_tip` (2026-11-10, `en_garde_v3` base, 2 s end hold): mjlab PASS and sim2sim PASS**, better than v2 across the board: blade tip 1.4 cm (v2 1.8), joint 0.075 (0.080), body 4.2 cm (4.9, so the body margin is back), slip 0.036. **C++ controller (2026-11-11): PASS**, joint 0.074, tip 1.5 cm, tip at the key frame 1.5 cm.*
- [x] 1.5.4 Lunge, including landing and recovery to en garde. *`lunge_v2_pipe` (2026-11-09, 3000 it): mjlab FAIL and sim2sim FAIL on joint error, 0.137 vs 0.10. Everything else passes: body 4.5 cm, slip 0.074, recovery and robustness 1.00, though anchor error is 23.6 cm against 25. The error is spread over the whole clip, mostly the rear arm and the wrists. The training joint error was still falling (1.31 at 2000 it, 1.01 at 3000 it, against about 0.66 for runs that passed), so the run looked under-trained. The resume `lunge_v2_more` (6000 it in total) only reached 0.131: training joint error flattened at 0.97 and anchor error dropped to 9.6 cm. So the problem is in the data. In every frame the retarget pinned the rear arm at its joint limits (shoulder roll +2.23 against a +2.25 limit, pitch −2.90, a raised foil/épée rear arm), and the sword arm hangs inside the torso (up to 3.7 cm). Rear elbow error 0.33 rad and shoulder 0.24 are the largest per-joint errors, with low torque, so the policy is choosing not to follow. `lunge_v3` replaces the rear arm with the `en_garde_v2` saber rear-arm pose and clears the sword arm of the torso (`fixes.clear_arm_torso_collision`). **`lunge_v3_pipe` (2026-11-10, 3000 it): mjlab PASS and sim2sim PASS.** Joint 0.09996 (sim2sim 0.0992), against 0.131 for v2 after 6000 it. Body 3.8 cm, anchor 12.6 cm, slip 0.077, recovery and robustness 1.00. The joint margin is a hair's breadth, so treat it as a borderline pass. The rear arm is the saber pose, not the video's. `lunge_v4` (same wrist fix as 1.5.7, both arms; 2026-11-11, 3000 it) tracked better, joint 0.092, but FAILED recovery in mjlab and sim2sim (0.09 and 0.00): it fell in the eval hold. `lunge_v3` and `v4` only have a 0.5 s end hold, so v3's recovery pass was luck (the 1.5.1 lesson again). **`lunge_v5` = `lunge_v4` plus a 2 s end hold, nothing else changed (2026-11-11, 3000 it): mjlab PASS, sim2sim PASS and C++ controller PASS.** Joint 0.087 / 0.087 / 0.088, body 4.0 cm, anchor 6.3 cm (v3 12.6), slip 0.054, recovery 1.00 in all three, push, noise and friction 1.00. The joint margin is no longer a hair's breadth.*
- [x] 1.5.5 Parries. *`parry_quinte_v2_tip` (2026-11-09, 3000 it): mjlab PASS and sim2sim PASS. Blade tip 2.5 cm, body 4.0 cm, joint 0.089, robustness including saber mass 1.00. The other four parries (prime, seconde, sixte, tierce high) have published motions but have not been trained. **`parry_quinte_v3_tip` (2026-11-10, `en_garde_v3` base, 2 s end hold): mjlab PASS and sim2sim PASS.** Joint 0.072 (v2 0.089), blade tip 1.7 cm (2.5), body 4.5 cm (4.0), slip 0.033. **`parry_prime_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.075, tip 1.7 cm, body 4.2 cm, slip 0.039). **`parry_seconde_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.091, tip 2.6 cm, body 4.5 cm, slip 0.042). **`parry_sixte_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.073, tip 1.2 cm, body 4.2 cm, slip 0.037). **`parry_tierce_high_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.090, tip 1.2 cm, body 4.0 cm, slip 0.039). All five Goal 1 parries (prime, seconde, tierce_high, quinte, sixte) now pass on the `en_garde_v3` base. **C++ controller (2026-11-11): all five PASS**; blade tip at the parry key frame 1.5–2.8 cm against the provisional 5 cm gate.*
- [x] 1.5.6 Hits. *`cut_head_v2_tip` (2026-11-10, 3000 it): mjlab FAIL and sim2sim FAIL, but narrowly: joint error 0.106 vs 0.10. Blade tip averages 3.0 cm (worst moment 32 cm, mid-cut), body 4.3 cm, robustness 1.00. The training curve had flattened at about 0.66, the same as the passing runs, so more iterations probably won't help. The largest errors are the right hip yaw (0.375 rad) and right ankle roll (0.27 rad). Every policy on the `en_garde_v1` video base shows these: en_garde 0.28/0.25, extension 0.30/0.26. The reference right leg is strongly twisted (hip yaw −0.77, hip roll −0.37), and the policy settles 0.28 rad further into hip yaw with the ankle rolled to −0.19 (limit −0.26). The feet are flat (about 1° tilt) and there is no self-collision. Checked: the reference front foot points 47° off the fencing line, and a saber fencer points it down the line. Its COM sits 1.8 cm outside the support polygon, so the reference stance can't be held statically. `en_garde_v2` turns that foot down the line by leg IK (hip yaw −0.03, hip roll +0.04, COM 0.9 cm inside) and is training. It cleared the offset, but its policy fell in the hold. `en_garde_v3` (plus a 2 s end hold, see 1.5.1) passed, and the static bladework set was regenerated on it as `*_v3` (`motion blade-all --end-hold 2`). **`cut_head_v3_tip` (2026-11-10, 3000 it): mjlab PASS and sim2sim PASS.** Joint 0.085 (sim2sim 0.084, v2 0.106), blade tip 2.7 cm on average (worst moment 38 cm, mid-cut), body 4.0 cm, recovery 1.00, all robustness including saber mass 1.00. Also on the `en_garde_v3` base, both trained for the first time: **`cut_flank_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.068, tip 1.9 cm, body 4.2 cm, slip 0.046), and **`cut_chest_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.078, tip 1.7 cm, body 4.0 cm, slip 0.044). **`thrust_v3_tip`: mjlab PASS and sim2sim PASS** (joint 0.080, tip 1.8 cm, body 4.6 cm, slip 0.044). **C++ controller (2026-11-11): all four PASS.** Blade tip at the moment of the hit (new key-frame metric, Python sim2sim / C++ controller): head cut 4.2 / 4.4 cm, flank 2.7 / 2.3, chest 2.6 / 1.9, thrust 3.3 / 1.9, against a provisional 5 cm gate. The head cut is the thin one. **`cut_head_v4_tip` (2026-11-11, 3000 it) replaces v3:** the attack move takes 0.3 s instead of 0.2 s, because v3's asked the wrist pitch motor for 5.2 times its torque limit ([saber.md](saber.md)). It passes mjlab, sim2sim, the C++ controller and PhysX, with the tip at the hit at 2.1 / 2.0 / 2.9 / 3.1 cm (v3: 4.1 / 4.2 / 4.4 / 6.1, a PhysX fail). Joint 0.089, body 4.5 cm (v3 4.0), so body error is now its thinnest margin.*
- [x] 1.5.7 Advance-lunge with hit. *Footwork only, `advance_lunge_v2_pipe` (2026-11-10, 3000 it): mjlab FAIL and sim2sim FAIL on recovery alone, 0.00 vs 0.90. The motion itself tracks: success 1.00, joint 0.091, body 3.5 cm, slip 0.032, robustness 1.00. But 1022 of 1024 envs fall about 1.5 s into the eval's post-motion hold (sim2sim at 7.0 s). The clip is cut mid-recovery, with the back foot still sliding, and had only a 0.5 s synthesized end hold to train on. `advance_lunge_v3` is the same data with a 2 s end hold (new `CleanConfig.end_hold_s`). It got **mjlab PASS and sim2sim PASS** (2026-11-10): recovery 1.00, joint 0.099 (thin), body 3.8 cm, slip 0.023, push 0.998. The end-hold length was the whole problem. `advance_lunge_cut_v2_tip` (3000 it): mjlab FAIL on recovery alone, 0.82 vs 0.90, and sim2sim PASS. It had no terminations, but 18 % of envs were still moving faster than 0.2 m/s at the end of the hold. The rest tracks well: joint 0.099, blade tip 2.5 cm, body 3.2 cm. `advance_lunge_cut_v3` (same keys on the `advance_lunge_v3` base, 2 s end hold; 3000 it): recovery fixed (1.00, was 0.82), blade tip 1.9 cm, body 3.7 cm, slip 0.018, all robustness including saber mass 1.00. But it fails the joint gate in both mjlab and sim2sim: 0.110 vs 0.10 (v2: 0.099). The error is in the wrists (right wrist yaw 0.32 rad, pitch 0.23, roll 0.20, plus the rear wrists), which are redundant once the blade tip is placed. With near-identical runs landing 0.01 apart, the provisional 0.10 mean-joint gate is a coin flip for saber motions. Decision for the user: exclude the wrists from the joint gate, or give them their own per-group threshold, since the tip gate already covers the sword wrist. Measured in sim2sim: without the six wrist joints this run is at 0.080 (wrists alone 0.225), advance_lunge_cut_v2 0.082, extension_v2 0.079, parry_quinte_v2 0.082. cut_head_v2 is still over at 0.102 without wrists, because its problem is the en_garde_v1 leg offset. **2026-11-11:** 3000 more iterations changed nothing (`advance_lunge_cut_v3_tip6k`: joint 0.110, sim2sim 0.109), so the cause is the reference, not training time. The rear wrist's pitch and yaw came from the retargeter, not the video, and sat up to 1.2 rad from the default pose, which the 0.075 rad wrist action scale cannot reach (see pipeline.md known errors). `advance_lunge_v4` and `advance_lunge_cut_v4` set those unobserved wrist angles to neutral. **`advance_lunge_cut_v4_tip` (3000 it): mjlab PASS and sim2sim PASS.** Joint 0.098 (thin; v3 0.110), blade tip 2.1 cm, body 3.2 cm, slip 0.026, recovery and all robustness including saber mass 1.00. The rear wrist error is gone; what is left is the sword wrist (yaw 0.31 rad), which the blade-tip reward leaves free, and the first third of the motion (0.132 against 0.08 later). In the C++ controller v4 FAILED the joint gate (0.102). **Cause found (2026-11-11): the reference used two different sword-arm poses for the same en garde guard** (wrist yaw −1.12 rad at the start, +1.45 at the end), because the blade keyframer re-solved the return guard from the cut pose. The policy held a compromise pose for the first 1.4 s; that was the whole first-third error. The keyframer now reuses the first solution for a repeated target (`advance_lunge_cut_v2` and `v3` had the same defect; the `*_v3` static set regenerates byte-identically). **`advance_lunge_cut_v5_tip` (3000 it): mjlab PASS, sim2sim PASS and C++ controller PASS**, joint 0.076 / 0.074 / 0.076 (v4 0.098 / 0.096 / 0.102), blade tip 1.9 cm, tip at the hit 1.9 / 2.1 / 1.5 cm, body 3.1 cm, slip 0.025, recovery and all robustness including saber mass 1.00. Footwork only: **`advance_lunge_v4_pipe` (wrists neutral, 3000 it): PASS in all three**, joint 0.081 / 0.081 / 0.080 (v3 0.099), recovery 1.00.*

For each motion:
- [ ] Train in mjlab; log to W&B. *Trained in mjlab with local tensorboard logs; no W&B account is set up, so this stays unticked*
- [x] Evaluate in mjlab against the reference. *`hcf train eval`, 1024 episodes per pass, for every run in `log.md`*
- [x] Sim-to-sim: same ONNX in `motion_tracking_controller`'s MuJoCo launch; Isaac Lab cross-check where practical. *The same ONNX runs in plain CPU MuJoCo (`hcf sim2sim`) for every run, and since 2026-11-11 in `motion_tracking_controller`'s MuJoCo launch (`scripts/mtc_sim2sim.sh`, [mtc.md](mtc.md)). **All 16 final policies pass there (2026-11-11)**: the table is at the end of [mtc.md](mtc.md) and in [`results/SUMMARY.md`](../results/SUMMARY.md). Two scorer fixes came out of this: the C++ controller runs the reference one tick (20 ms) behind mjlab's convention, which the scorer now accounts for, and recorded states are interpolated to the tick. No Isaac Lab cross-check: not practical on this PC (Phase 1.1). **A second physics engine instead (2026-11-11):** the same ONNX runs in CPU PhysX through `ovphysx` (`fencing sim2sim <export dir> --engine physx`, [evaluation.md](evaluation.md#sim-to-sim-in-physx)). It is the engine Isaac Sim uses, without Isaac Lab's actuator and sensor layers, so it is not a substitute for the Isaac Lab run. 15 of 16 passed at first; the failure was `cut_head_v3`'s blade tip at the hit (6.1 cm vs 5), which led to `cut_head_v4` (1.5.6). All 16 pass now. **Start from the standby controller:** tested in the C++ controller sim; most policies do not survive a direct switch from the standby pose, see [mtc.md](mtc.md#start-from-standby-2026-11-11) and [hardware_tests.md](hardware_tests.md)*
- [x] Robustness: push the robot, add sensor noise, vary friction and saber mass. *All four passes run in every eval; saber mass only on the saber task*
- [x] Record the result in `log.md`. *`fencing pipeline train` appends the row; resumed runs get a manual row*

**Done when:** all seven motions pass the evaluation gate below in both simulators. *Status 2026-11-11: all seven motions (16 policies) pass in mjlab, in CPU MuJoCo and in `motion_tracking_controller`'s C++ controller and MuJoCo sim. "Both simulators" was written as Isaac Sim and MuJoCo; Isaac Sim has not been run, so that half is still open.*

### Phase 1.6: Human-supervised hardware tests

Test plan, policy files, space per motion and the decisions to make first: [hardware_tests.md](hardware_tests.md).
- [ ] Only for motions that passed Phase 1.5 in both simulators
- [ ] Follow `safety/checklist.md` for every session
- [ ] Same order as 1.5. Do footwork without the saber first, then add the foam saber
- [ ] Gantry or harness on for every first attempt; low gains and slower playback first
- [ ] Record video of every test and log the result
- [ ] Compare against simulation: where does the real robot differ, and why?

**Done when:** en garde, advance/retreat, lunge and at least one parry and one hit run on the real G1 under supervision. Advance-lunge with hit is a stretch target for Goal 1.

### Goal 1 evaluation

Use the same metrics for every motion:

| Metric | Meaning |
|---|---|
| Body tracking error | Mean joint-angle and body-position error against the reference |
| Blade-tip error | Distance between the robot's blade tip and the reference blade tip at the key frame (the parry position or the moment of the hit). Scored as `blade_tip_error_key_m` since 2026-11-11, next to the whole-motion mean |
| Success rate | Share of rollouts that finish without falling, out of N per motion |
| Foot slip | Sliding of a planted foot |
| Recovery | Returns to a stable en garde after the motion |
| Sim-to-sim gap | Difference between the Isaac Sim and MuJoCo results |

Pass/fail thresholds will be set after Phase 1.2, based on what BeyondMimic gets on known motions.

### Goal 1 risks

| Risk | Mitigation |
|---|---|
| GVHMR misses wrist and blade angle | Hand-edit wrist keyframes for parries and cuts; treat blade position as its own target |
| Fencing video has occlusion and two fencers | Prefer solo drill and lesson videos; add a tracking step |
| The lunge is too explosive for the G1's motor limits | Slow it down first, then speed it up; clamp joint speeds in the retargeted data |
| The saber's mass upsets balance | Light foam saber; include it in the sim model and vary its mass during training |
| Wrist range can't reach some parries | List the parries it can reach; change the grip or the mount angle |

---

## Goal 2: Targeted hits and blade perception

**Target:** the G1 hits a given target on a stationary dummy, and can track an opponent's blade from its own camera.

Goal 1 policies only replay motions. This goal needs a **goal-conditioned policy** that takes the target as input. That is a change of method from BeyondMimic tracking alone.

### Phase 2.1: Hit a target (goal-conditioned)
- Target position given directly (no perception yet), on a stationary dummy
- Train a policy that takes the target as input, starting from Goal 1's hit motions
- Measure blade-tip error at the moment of the hit across many target positions

### Phase 2.2: Perception, offline
- Check the G1's head camera first (believed to be a depth camera; confirm the model, frame rate and field of view) before adding any hardware
- Detect and track the opponent's body and blade in recorded video
- The blade is thin and fast, so expect this to be the hard part. Measure delay and accuracy

### Phase 2.3: Perception on the robot
- Run detection on the robot or a tethered computer within a fixed delay budget
- Hit targets detected by the camera, not given as coordinates

**Done when:** the G1 hits targets on a stationary dummy that it found with its own camera, to a set accuracy.

---

## Goal 3: Footwork and bladework combined, aimed at a target

**Target:** given a target, the G1 picks the distance and the action, for example an advance-lunge cut to the flank.

### Phase 3.1: Distance and footwork
- Close distance to a target with advances and retreats before attacking

### Phase 3.2: Combinations aimed at a target
- Advance-lunge, lunge with cut, and similar combinations aimed at a target the camera found
- Choose between actions based on distance

### Phase 3.3: Hardware tests against a stationary target

**Done when:** from different starting distances, the G1 closes in and hits the chosen target with a combination.

---

## Goal 4: Reactive and autonomous fencing

**Target:** the G1 reacts to an opponent in real time and, in the end, fences a simple bout on its own.

### Phase 4.1: Parry in real time
- Track the incoming blade and parry in time
- Start with slow, scripted attacks in simulation, then a human at slow speed with foam weapons

### Phase 4.2: Reactive policy
- Policy that reacts to the opponent's movement (e.g. guided diffusion over fencing motions)
- Parry-riposte and keeping distance

### Phase 4.3: Autonomous bout
- Simple bouts against a human under strict safety rules
- Rules and scoring are kept simple at first

**Done when:** the G1 fences a short, supervised bout against a human on its own.

Goal 4 is open research. Expect this section to be rewritten once Goals 1–3 are done.

---

## Open questions

- Saber grip: Dex3 hand or fixed mount? (Phase 1.3)
- Which parries and cuts can the G1's wrist actually reach? (Phase 1.3)
- What is the G1 head camera's exact model, frame rate and field of view? (Phase 2.2)
- Run perception on the robot or on a tethered computer? (Phase 2.3)
- What safety setup is needed once the robot reacts to a human? (Goal 4)
