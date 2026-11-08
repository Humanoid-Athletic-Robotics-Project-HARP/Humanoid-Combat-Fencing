# Roadmap

Teaching a Unitree G1 humanoid (29 DoF) to fence with a saber, from replaying human fencing motions up to fencing on its own.

The roadmap has four **goals**. A goal is a capability the robot shows. Each goal is split into **phases**, which are concrete engineering steps. Goal 1 is planned in detail. Goals 2–4 are outlines that we will expand once Goal 1 is done and we know what actually worked.

| Goal | Capability | Method | Status |
|---|---|---|---|
| **1** | Footwork and saber coordination (replayed motions) | Video → GVHMR → GMR → BeyondMimic | In progress |
| **2** | Targeted hits and blade perception | Goal-conditioned policy + head-camera perception | Planned |
| **3** | Footwork and bladework combined, aimed at a target | Goal-conditioned whole-body policy | Planned |
| **4** | Reactive and autonomous fencing | Reactive policy (e.g. guided diffusion) + real-time perception | Stretch |

**Rules that apply to every goal**

- **Simulation first.** Every policy passes Isaac Sim and MuJoCo sim-to-sim before it goes near hardware.
- **Humans run hardware.** Nothing in this repo sends commands to the real robot. Hardware tests follow [`safety/checklist.md`](../safety/checklist.md): gantry or harness, cleared space, foam or padded weapons only.
- **Upstream code stays upstream.** BeyondMimic, GMR and GVHMR are pinned submodules in `third_party/`.
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

**Done when:** a fresh clone plus `setup_third_party.sh` gives the pinned upstream code.

> Note: as of 2026-11-08 the `main` branch only has `README.md`, `LICENSE` and `.gitignore`. The setup work needs to be pushed before this phase really counts as done.

### Phase 1.1: GPU environment check

- [ ] Linux machine with an NVIDIA GPU; record GPU model, driver and CUDA version in `log.md`
- [ ] Install Isaac Sim 4.5 and Isaac Lab v2.1.0
- [ ] Install `whole_body_tracking` (BeyondMimic) at the pinned commit
- [ ] Set up the W&B organization and motion registry
- [ ] Run Isaac Lab's own sample task to confirm the install

**Done when:** an Isaac Lab sample trains without errors and logs to W&B.

### Phase 1.2: Prove the pipeline on a known motion

Train one motion that BeyondMimic already supports, before adding any fencing data. If this fails, the problem is the setup, not our data.

- [ ] Choose a reference motion from the BeyondMimic examples
- [ ] Train with the standard recipe and leave it unchanged
- [ ] Evaluate in Isaac Sim, then MuJoCo sim-to-sim
- [ ] Write down the metrics we will reuse for every fencing motion (see [Evaluation](#goal-1-evaluation))

**Done when:** the known motion tracks in both simulators with numbers close to the ones BeyondMimic reports.

### Phase 1.3: Saber and hardware decisions

Runs alongside 1.1–1.2. It has to be settled before training any saber motion.

- [ ] **Grip:** the Dex3 hand holding a handle, or a fixed mount on the wrist? A fixed mount is simpler and more repeatable. Decide and record it as an ADR.
- [ ] **Wrist range:** check that the G1's wrist joint limits can reach the parry and cut positions. If they can't, list which positions are possible.
- [ ] **Training weapon:** a foam or padded saber, its length and mass. Check the effect on the arm's payload and the robot's balance.
- [ ] **Sim model:** add the saber to the G1 model (URDF/MJCF) as a rigid body with correct mass and inertia, in both Isaac Sim and MuJoCo.
- [ ] **Blade tip:** define a blade-tip frame so hit and parry accuracy can be measured.

**Done when:** the G1 model with the saber loads in both simulators and the grip decision is written down.

### Phase 1.4: Fencing motion dataset

Turn fencing video into G1 reference trajectories.

**1.4a Source video**
- [ ] Collect clips in [`data/video_log.md`](../data/video_log.md): links only, never the video files
- [ ] Prefer one fencer, side-on camera, steady shot, full body visible, good lighting
- [ ] Use saber specifically: footwork and blade actions differ from foil and épée
- [ ] Tag each clip with the motion type, start and end time, and the fencer's handedness

**1.4b Isolate the fencer**
- [ ] For clips with two fencers: detect and track people, and keep one fencer's track through the clip
- [ ] Check how GVHMR's built-in detection handles two fencers close together. Add a separate tracking step if it doesn't cope.
- [ ] Cut clips into single-motion segments

**1.4c Human pose (GVHMR)**
- [ ] Run GVHMR to get SMPL-X body motion in world coordinates
- [ ] Check visually: overlay the pose on the video and reject bad segments
- [ ] Note what GVHMR can't see: wrist rotation and blade angle are poorly captured. Parries and cuts may need hand-edited wrist motion or keyframes

**1.4d Retarget to the G1 (GMR)**
- [ ] Retarget SMPL-X onto the G1 with GMR
- [ ] Add the saber: drive the wrist so the blade lands in the intended parry or cut position
- [ ] Mirror for handedness so every motion uses the same sword hand

**1.4e Clean up the data** (where most of the effort goes)
- [ ] Foot contacts: no sliding while a foot is planted, no feet below the floor
- [ ] Vertical drift: remove the slow up-and-down drift of the pelvis from pose estimation
- [ ] Segment selection: trim to clean start and end poses, ideally starting and ending in en garde
- [ ] Joint limits and speeds within the G1's specs
- [ ] Save as small CSVs in `motions/`, with a version number and a short note on where each came from

**Done when:** each of the seven motions has at least one cleaned, versioned reference CSV that plays back properly in simulation without physics (kinematic replay).

### Phase 1.5: Train and evaluate in simulation

Same BeyondMimic recipe for every motion. If a motion fails, **fix the data first** (Phase 1.4e), and only change the recipe as a last resort, written down as an ADR.

Train in this order. Each motion has to pass before starting the next:

- [ ] 1.5.1 En garde
- [ ] 1.5.2 Advance / retreat
- [ ] 1.5.3 Extension
- [ ] 1.5.4 Lunge, including landing and recovery to en garde
- [ ] 1.5.5 Parries
- [ ] 1.5.6 Hits
- [ ] 1.5.7 Advance-lunge with hit

For each motion:
- [ ] Train in Isaac Sim; log to W&B
- [ ] Evaluate in Isaac Sim against the reference
- [ ] MuJoCo sim-to-sim: same policy, different physics engine
- [ ] Robustness: push the robot, add sensor noise, vary friction and saber mass
- [ ] Record the result in `log.md`

**Done when:** all seven motions pass the evaluation gate below in both simulators.

### Phase 1.6: Human-supervised hardware tests

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
| Blade-tip error | Distance between the robot's blade tip and the reference blade tip at the key frame (the parry position or the moment of the hit) |
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
