# Saber model, mount and reach

How the training saber is modelled and bolted to the G1, and how well the right arm can put the blade where saber parries and cuts need it. The decision and its reasons are in [ADR 0002](adr/0002-saber-grip-and-mount.md). This page holds the details and the numbers behind it.

Sources: `src/fencing/saber/` (`spec.py`, `reach.py`, `blade.py`, `mjlab_task.py`, `smoke.py`, `cli.py`), the generated `src/fencing/saber/assets/g1_29dof_saber.xml`, and the run outputs in `~/hcf-data/eval/saber/` (`reach.json`, `smoke.json`, `renders/`). Every number below is copied from those files.

## The saber

One rigid body made of three solid parts. Defaults are in `SaberSpec` (`spec.py`).

| Part | Shape | Size | Mass |
|---|---|---|---|
| Grip (with the printed socket) | cylinder along x | length 0.120 m, radius 0.015 m | 0.070 kg |
| Guard | box | 0.010 (x) × 0.070 (y) × 0.100 (z) m | 0.030 kg |
| Blade (foam over a core) | cylinder along x | length 0.750 m, radius 0.012 m | 0.100 kg |
| **Total** | | **0.88 m overall** | **0.20 kg** |

**Saber frame.** The origin is at the centre of the grip.

- +x runs along the blade, towards the tip.
- +z is the back (false) edge, on the thumb side of the hand.
- −z is the cutting edge.

| Position along +x | x (m) |
|---|---|
| Grip | −0.06 to +0.06 |
| Guard centre | 0.065 |
| Site `guard`, where the blade leaves the guard (the forte) | 0.070 |
| Blade centre | 0.445 |
| Site `blade_tip` | 0.820 |

**Mass properties come from geometry, not geoms.** Each part's inertia is that of a solid cylinder or box, and the three are combined with the parallel-axis theorem (`mass_properties`). The results written to the MJCF:

| Property | Value |
|---|---|
| Mass | 0.2 kg |
| COM | x = 0.23225 m from the grip centre |
| Principal inertia about the COM | (5.2325e-5, 1.39455e-2, 1.39328e-2) kg·m² |

The y and z inertias differ slightly because the guard box is taller than it is wide. ADR 0002 rounds both to 1.39e-2. `tests/test_saber.py::test_mass_properties_agree_with_mujoco_inertia_from_geoms` checks these values against MuJoCo's own inertia-from-geoms.

The visual geoms carry `mass = 0`, and the body has an explicit `<inertial>`, so the geoms never add mass twice.

## How it is mounted in the MJCF

`attach_saber()` adds the body `saber` as a child of `right_wrist_yaw_link`, the last link of the G1's right arm. There is no joint between them, so the saber is welded on.

```xml
<body name="saber" pos="0.1 0 0" quat="0.965926 0 -0.258819 0">
  <inertial pos="0.23225 0 0" mass="0.2" diaginertia="5.2325e-05 0.0139455 0.0139328"/>
  <geom name="saber_grip"  ... contype="0" conaffinity="0" group="2" mass="0"/>
  <geom name="saber_guard" ... contype="0" conaffinity="0" group="2" mass="0"/>
  <geom name="saber_blade" ... contype="0" conaffinity="0" group="2" mass="0"/>
  <site name="guard"     pos="0.07 0 0" group="3"/>
  <site name="blade_tip" pos="0.82 0 0" group="3"/>
</body>
```

- **Position.** The grip centre sits 0.10 m along the wrist link's x axis (the forearm axis), inside the rubber hand's fist. The fist spans x = 0.04 to 0.17 m, measured from the hand mesh.
- **Orientation.** The blade is rotated 30° about −y, which lifts the tip towards +z, the thumb side. Yaw is 0. As a quaternion that is `(cos 15°, 0, −sin 15°, 0)`. A positive `mount_yaw_deg` would swing the tip towards the palm.
- **Blade tip in the wrist frame:** (0.810, 0, 0.410) m, which is 0.10 + 0.82 × (cos 30°, 0, sin 30°). `tests/test_saber.py::test_tip_site_sits_at_the_designed_offset_from_the_wrist` checks this.
- **Collision is off.** Every saber geom has `contype = conaffinity = 0`; the reasons are in ADR 0002. `SaberSpec(collision=True)` adds one capsule, `saber_blade_collision`, over the blade. Its name ends in `_collision` on purpose, so that mjlab's G1 `CollisionCfg` regex gives it the same contact settings as the robot's own capsules.
- **Generated files.**
  - `src/fencing/saber/assets/g1_29dof_saber.xml` is the upstream mjlab G1 plus the saber. Mesh paths are rewritten so plain MuJoCo can load it from the repo.
  - `src/fencing/saber/assets/saber_link_snippet.urdf` is the same saber as a fixed-joint URDF link, for Isaac Lab later. It is untested because Isaac Sim does not run on this WSL2 machine. When importing it, set `merge_fixed_joints=False` or the `blade_tip` frame disappears.

### Motion NPZ compatibility

`right_wrist_yaw_link` is the last body of the G1 tree, so `saber` becomes the last (31st) robot body and every other body keeps its index. NPZ motions built for the plain robot therefore index correctly on the saber robot. The smoke test checks this end to end (see [Smoke test](#smoke-test)): 30 plain bodies, 31 with the saber, 30 in the NPZ. Posed from NPZ frames, the saber robot's tracked bodies land within 7.2e-7 m of the NPZ bodies.

No separate saber NPZ is needed unless a future reward tracks the saber body itself.

## mjlab tasks

`fencing.saber.mjlab_task.register()` registers five tasks without touching `third_party`. It is idempotent. `fencing.train.mjlab_env.register_tasks()` calls it, so eval, export, play and record-obs all see these tasks.

| Task id | Based on |
|---|---|
| `Fencing-Tracking-Flat-G1-Saber` | `Mjlab-Tracking-Flat-Unitree-G1` |
| `Fencing-Tracking-Flat-G1-Saber-No-State-Estimation` | `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` |
| `Fencing-Tracking-Flat-G1-Saber-Entry` | `Fencing-Tracking-Flat-G1-Saber`, plus one reward term (below) |
| `Fencing-Tracking-Flat-G1-Saber-Entry-Joint`, `-Joint05` | `Fencing-Tracking-Flat-G1-Saber-Entry`, plus one more reward term (below) |

Differences from the upstream tasks:

- **Robot spec.** mjlab's own `get_spec()` plus `attach_saber()`. Everything else (actuators, collisions, keyframe) is the upstream `get_g1_robot_cfg()`.
- **Saber mass randomization.**
  - The startup event `saber_mass` uses `dr.pseudo_inertia` on the `saber` body only, so mass and inertia scale together. That models a uniform density change.
  - `pseudo_inertia` scales mass by exp(2α). The α range (−0.255, +0.280) is chosen to give 0.12–0.35 kg (`saber_alpha_range`).
  - The mass is drawn once per env for the whole run and stays constant within an episode.
- **Play configs** drop `saber_mass`, so evaluation uses the nominal 0.20 kg saber.
- **Right wrist action scale.** `right_wrist_pitch_joint` and `right_wrist_yaw_joint` use 0.25 rad per unit of action (`SABER_WRIST_ACTION_SCALE`) instead of upstream's 0.25 × effort / stiffness = 0.075. See [Wrist action scale](#wrist-action-scale) below. Gains, torque limits and the left wrist are unchanged.
- **Blade-tip reward.** A `blade_tip_pos` term, exp(−|tip error|²/0.15²) with weight 1.0, uses the same definition as the eval metric. It was added after `extension_v2` reached 16.8 cm with the wrist fix alone. With the term, `extension_v2_tip` reached 1.8 cm and passes. See [ADR 0003](adr/0003-blade-tip-reward.md).
- **PPO settings** are unchanged from upstream. Runs are logged under the experiment name `g1_saber_tracking`.

**The entry task** (2026-11-11), for the entry from standby ([mtc.md, Entry from standby](mtc.md#entry-from-standby-2026-11-11)). It is `Fencing-Tracking-Flat-G1-Saber` plus one reward term, `feet_pos`. That term is mjlab's own `motion_relative_body_position_error_exp`, restricted to `left_ankle_roll_link` and `right_ankle_roll_link`, with std 0.1 m and weight 1.0 (`FEET_REWARD_*`). Why: `motion_body_pos` averages the squared error over all 14 tracked bodies with std 0.3 m. Two feet 11 cm off cost about 2% of it, and `stand_to_en_garde_v1_tip` learned to shuffle instead of stepping. On the feet alone with std 0.1 m, the same miss costs 0.70 of the new term. Observations, actions, events, terminations, the motion command and every other reward term are the saber task's, so the exported ONNX has the same contract and runs in the unmodified controller. `tests/test_saber_mjlab.py` (mjlab venv) checks this two ways. It compares the two existing tasks' configs against `tests/data/saber_task_cfgs.json`, a snapshot written before the entry task existed. It also checks that the entry task differs from the saber task only by `feet_pos`. Evaluation gates and scoring do not change: the reward is training only.

**The entry joint tasks** (2026-11-12): `Fencing-Tracking-Flat-G1-Saber-Entry-Joint` (weight 1.0) and `-Joint05` (weight 0.5). Each is the entry task plus `joint_pos` = exp(−mean_j (q_ref − q)² / 0.15²) over all joints of the motion command, the same joints and reference as the 0.1 rad joint gate (`joint_position_error_exp`, `JOINT_REWARD_*`, `ENTRY_JOINT_WEIGHTS`). Why: the guard the entry ends in has its CoM about 5 cm outside the G1's double support. A policy that places both feet must therefore depart from the posture somewhere, and with nothing paying for joint angles, `stand_to_en_garde_v1_feet` bent its waist and arms (0.14 rad mean joint error in the guard hold). See [mtc.md, Paid for joint angles](mtc.md#paid-for-joint-angles-stand_to_en_garde_v1_feetjoint05-2026-11-12-recommended). Weight 0.5 gives the recommended entry, `stand_to_en_garde_v1_feetjoint05`: it passes in all five simulators and chains all ten saber motions. Weight 1.0 (`stand_to_en_garde_v1_feetjoint`) has lower joint error but its pelvis does not settle in PhysX. `tests/test_saber_mjlab.py` pins the entry task as it was before these tasks, in its own snapshot `tests/data/saber_entry_task_cfgs.json`, and checks that each joint task differs from the entry task only by `joint_pos`. Train them with `fencing pipeline train motions/stand_to_en_garde_v1.csv --task Fencing-Tracking-Flat-G1-Saber-Entry-Joint05 --tag feetjoint05 --iterations 3000` (the default is 5000; the recommended policy is the 3000-iteration run), or queue them with `fencing pipeline queue motions/stand_to_en_garde_v1.csv --entry-variant feetjoint05` (or `feetjoint`). The queue's default for the entry motion is still `tip` and `feet`.

## Reach analysis

**Question:** with the saber mounted, can the right arm put the blade in the canonical saber positions? And which mount angle and grip offset reach the most?

### Method (`reach.py`)

1. **Stance.** The pelvis is fixed at mjlab's `KNEES_BENT_KEYFRAME`: 0.76 m high, facing +x, with the knees, hips and ankles bent and the left arm in its keyframe pose. The right shoulder (the `right_shoulder_pitch_joint` anchor) is then at (0, −0.100, 0.292) m from the pelvis.
2. **Reach.** The largest shoulder-to-`guard` distance over 20,000 random arm configurations is **0.5717 m**.
3. **Targets.** Each of the 12 canonical positions (below) is a guard position relative to the right shoulder, in units of that reach (1.0 = fully extended), plus a blade direction. The target tip is guard + 0.75 m × direction.
   - The poses are **estimates** written from fencing references and coaching descriptions, scaled to the G1. They were not measured from video.
4. **IK.** Damped least squares on [tip position; blade direction] over the 7 right-arm joints:
   - 300 iterations, damping 0.03, a direction weight of 0.5 m per radian, at most 0.2 rad per step, joints clamped to their ranges.
   - Roll about the blade axis is left free, so edge orientation is not part of the target.
   - Each target gets 32 starts: the zero pose plus 31 random ones. The best result wins.
5. **"Reachable"** means all three of:
   - the tip is within **2 cm** of the target;
   - the blade direction is within **5°**;
   - no self-collision deeper than **5 mm** involves the right arm or the blade. This is checked on a twin model with the blade collision capsule switched on.
6. **Second pass.** The same targets are solved with the 3 waist joints freed as well, because fencers turn the torso.
7. **Mount sweep.** Arm only. Mount pitch 0°, 15°, 30°, 45° and 60°, crossed with a grip-centre offset of 0.10 m or 0.06 m from the wrist-link origin. The shoulder position and reach stay those of the nominal saber, so only the mount changes between rows.

The 12 targets, for a right-handed fencer facing +x (y to the left, z up):

| Target | Kind | Guard offset (× reach) | Blade direction | Description |
|---|---|---|---|---|
| en_garde | guard | (0.55, −0.10, −0.45) | (0.80, 0.10, 0.59) | hand at waist height in front of the right hip, blade forward and about 35° up |
| parry_prime | parry | (0.45, 0.45, 0.00) | (0.25, 0.20, −0.95) | hand at shoulder height in front of the left shoulder, blade hanging down |
| parry_seconde | parry | (0.65, −0.10, −0.25) | (0.50, −0.20, −0.85) | hand at chest height outside, blade down-forward and out |
| parry_tierce | parry | (0.50, −0.25, −0.45) | (0.20, −0.20, 0.96) | hand at waist height outside the right hip, blade near vertical |
| parry_quarte | parry | (0.50, 0.35, −0.45) | (0.20, 0.15, 0.97) | hand at waist height in front of the left hip, blade near vertical |
| parry_quinte | parry | (0.35, −0.15, 0.55) | (0.15, 1.00, 0.15) | hand above and right of the head, blade horizontal over the head |
| parry_sixte | parry | (0.60, −0.20, −0.30) | (0.50, −0.10, 0.86) | hand at chest height outside, blade up and forward |
| cut_head | cut | (0.90, 0.05, 0.15) | (0.90, 0.05, −0.40) | arm extended at shoulder height, blade angled down |
| cut_flank | cut | (0.85, 0.15, −0.20) | (0.80, 0.50, −0.30) | arm extended at chest height, blade sweeping left |
| cut_chest | cut | (0.85, −0.05, −0.05) | (0.85, −0.45, −0.25) | arm extended, blade angled right and down |
| cut_cheek | cut | (0.88, −0.05, 0.12) | (0.85, −0.50, 0.05) | arm extended at shoulder height, blade horizontal, angled right |
| point_thrust | thrust | (0.95, 0.05, 0.00) | (1.00, 0.02, −0.12) | arm fully extended, blade continuing the arm line |

### Mount sweep (arm only)

| Mount pitch | Grip offset | Reachable | Not reachable (tip error, direction error; "coll." means self-collision) |
|---|---|---|---|
| 0° | 0.10 m | 7 / 12 | en garde (4.8 cm, 4.8°), tierce (28.1 cm, 27.6°, coll.), quarte (31.2 cm, 6.9°, coll.), quinte (19.6 cm, 4.3°), sixte (37.2 cm, 1.2°) |
| 15° | 0.10 m | 8 / 12 | tierce (18.3 cm, 24.9°, coll.), quarte (22.3 cm, 23.9°, coll.), quinte (9.8 cm, 6.8°), sixte (10.6 cm, 8.6°) |
| **30°** | **0.10 m** | **10 / 12** | tierce (6.4 cm, 39.0°, coll.), quarte (19.8 cm, 26.2°, coll.) |
| 45° | 0.10 m | 9 / 12 | tierce (12.5 cm, 13.2°), quarte (3.6 cm, 7.3°, coll.), thrust (5.8 cm, 3.5°) |
| 60° | 0.10 m | 9 / 12 | tierce (2.4 cm, 3.4°), quarte (3.2 cm, 5.2°, coll.), thrust (6.8 cm, 3.8°) |
| 0° | 0.06 m | 7 / 12 | tierce (47.3 cm, 18.3°), quarte (23.1 cm, 11.6°, coll.), quinte (6.8 cm, 4.8°), sixte (10.5 cm, 8.7°), thrust (3.3 cm, 0.9°) |
| 15° | 0.06 m | 8 / 12 | tierce (30.1 cm, 15.1°, coll.), quarte (17.2 cm, 21.5°, coll.), sixte (4.5 cm, 4.5°), thrust (3.5 cm, 0.5°) |
| 30° | 0.06 m | 9 / 12 | tierce (26.8 cm, 39.9°, coll.), quarte (16.8 cm, 24.1°, coll.), thrust (4.4 cm, 0.0°) |
| 45° | 0.06 m | 8 / 12 | tierce (7.8 cm, 10.1°), quarte (3.0 cm, 7.6°, coll.), flank (2.0 cm, 0.4°), thrust (6.1 cm, 0.6°) |
| 60° | 0.06 m | 7 / 12 | tierce (2.5 cm, 3.7°), quarte (3.3 cm, 6.0°, coll.), head (3.0 cm, 1.1°), flank (4.8 cm, 1.1°), thrust (9.5 cm, 3.8°) |

**Chosen: 30° pitch, grip centre at 0.10 m.** It is the only setting that reaches 10 of 12. The human-like 15° reaches 8 of 12; it loses quinte and sixte.

- Steeper mounts get closer to the waist-height parries but lose the straight-arm thrust. At 60°, tierce misses by only 2.4 cm. At 45° and 60°, quarte misses by about 3 cm but still collides.
- Moving the grip in to 0.06 m never beats 9 of 12, and it loses the thrust at every pitch.
- These numbers agree with ADR 0002 ("10 of 12 against 8 of 12"; "0.06 m did not help").

### Per target at the chosen mount

| Target | Kind | Arm only | Tip err (cm) | Dir err (°) | Joints at limit | Arm + waist | Joints at limit |
|---|---|---|---|---|---|---|---|
| en_garde | guard | yes | 0.1 | 0.2 | – | yes | – |
| parry_prime | parry | yes | 0.0 | 0.0 | – | yes | – |
| parry_seconde | parry | yes | 0.0 | 0.0 | – | yes | wrist_roll |
| parry_tierce | parry | **no** | 6.4 | 39.0 | shoulder_roll, wrist_pitch, wrist_yaw | yes | waist_pitch |
| parry_quarte | parry | **no** | 19.8 | 26.2 | shoulder_pitch, shoulder_roll, shoulder_yaw, wrist_yaw | yes | wrist_yaw, waist_pitch |
| parry_quinte | parry | yes | 0.0 | 0.0 | wrist_yaw | yes | – |
| parry_sixte | parry | yes | 0.4 | 0.8 | – | yes | shoulder_yaw |
| cut_head | cut | yes | 0.0 | 0.0 | wrist_roll | yes | – |
| cut_flank | cut | yes | 0.0 | 0.0 | – | yes | – |
| cut_chest | cut | yes | 0.0 | 0.0 | – | yes | – |
| cut_cheek | cut | yes | 0.0 | 0.0 | – | yes | – |
| point_thrust | thrust | yes | 0.8 | 0.1 | – | yes | – |

"At limit" means within 0.01 rad of the hard joint limit. With the waist freed, all 12 are reachable (tip and direction errors 0.0).

Two arm-only "yes" poses have a joint at its hard limit: quinte (wrist yaw) and the head cut (wrist roll). Arm-only sixte has the wrist yaw beyond 90 % of its range. In the waist pass, seconde (wrist roll) and sixte (shoulder yaw) touch limits. A tracked motion through these poses has little margin, and mjlab's joint-limit penalty starts at 90 % of the range.

### What is not reachable, and the workaround

**Tierce and quarte with the hand at waist height cannot be reached by the arm alone in a square-on stance, at any mount in the sweep.**

- **Arm only.** The closest arm-only solutions put the elbow and the upper arm (`right_shoulder_yaw_link`) into the torso. For quarte the wrist also hits the left shoulder.
- **Near misses.** At 60° tierce comes within 2.4 cm and 3.4°, just outside the 2 cm tolerance. Quarte is in self-collision at every setting.
- **Renders.** `renders/parry_tierce.png` and `parry_quarte.png` show those closest arm-only poses: the hand is up near the shoulder, not at the hip.

Freeing the waist makes both reachable, but only at the waist's limits:

| Target | waist_yaw (rad) | waist_roll (rad) | waist_pitch (rad) |
|---|---|---|---|
| parry_tierce | 0.056 | 0.456 | **0.52** (at limit) |
| parry_quarte | **1.887** (about 108° of torso turn) | 0.445 | **−0.52** (at limit) |

**Workaround (ADR 0002).** The parry motions should use a **higher hand, a turned torso, or both**, not the textbook waist-height hand with a square-on trunk. Hand-edited wrist keyframes (roadmap 1.4) should aim at the reachable versions.

The 108° waist yaw for quarte comes from IK with no posture cost. It shows that a solution exists, not what a fencer would do. A real quarte would combine some torso turn with a higher hand.

### Payload at en garde

From `reach.json` → `payload`. The arm is in the en garde IK pose, and torques are static (gravity only).

| | Shoulder pitch | Elbow | Wrist pitch | Wrist yaw | COM shift (x, y, z) |
|---|---|---|---|---|---|
| No saber | −2.59 Nm | −0.79 Nm | −0.26 Nm | 0.07 Nm | – |
| Nominal 0.20 kg | −3.42 Nm | −1.31 Nm | −0.60 Nm | 0.31 Nm | (2.4, −0.9, 1.3) mm |
| Heavy 0.35 kg | −4.05 Nm | −1.71 Nm | −0.85 Nm | 0.48 Nm | (4.2, −1.5, 2.3) mm |
| Effort limit | 25 Nm | 25 Nm | 5 Nm | 5 Nm | |

Holding the saber is cheap. **Swinging it is not.** For a fast wrist cut, reach.py assumes 100 rad/s², which is 0 to 10 rad/s in 0.1 s. That is a guess, and the torque scales linearly with it. The inertia of the hand + saber then costs:

- nominal saber: 1.35 Nm at wrist pitch and **3.04 Nm** at wrist yaw;
- heavy saber: 2.25 Nm at wrist pitch and **5.19 Nm** at wrist yaw, **above the 5 Nm limit**.

Fast cuts with the heaviest randomized saber will saturate the wrist yaw motor. That is one more reason to keep the blade light and to use the shoulder and elbow for big cuts.

## Smoke test

`fencing saber smoke` builds and steps the saber task on the GPU (`smoke.py`) and stops at the first check that fails. The last run (`~/hcf-data/eval/saber/smoke.json`) **passed**, on `Fencing-Tracking-Flat-G1-Saber` with `cuda:0` and motion `~/hcf-data/npz/dance1_subject2_mjlab.npz`.

| Check | Result |
|---|---|
| `register()` is idempotent | same ids on two calls |
| Body order: saber robot = plain robot + `saber` at the end; the NPZ matches the plain robot | 30 plain, 31 saber, 30 NPZ |
| Saber robot posed from NPZ frames 0, ⅓, ⅔ and last: tracked bodies land on the NPZ bodies (limit 1e-4 m) | max error 7.18e-7 m |
| Training env, 32 envs × 100 steps with zero actions: rewards finite | yes. There were 222 episode ends, as expected for a zero-action policy that falls |
| Saber mass randomized within 0.12–0.35 kg | min 0.1209, max 0.3186, mean 0.2051 kg |
| Blade tip stays at its mount offset from the right wrist (limit 1e-3 m) | max error 9.3e-7 m |
| The wrist actually moved, so the previous check means something | wrist travel 2.12 m, tip travel 3.88 m (env 0) |

## Renders

`~/hcf-data/eval/saber/renders/` has one PNG per target, front and side views, rendered from the joint angles `reach.json` solved. Unreachable targets show their closest pose. `contact_sheet.png` has all 12. The default render uses the arm-only solutions; `--joint-set arm+waist` renders the waist pass instead.

## Reproduce

All four commands run through `scripts/hcf` (the `fencing` CLI in the repo's venv under `~/hcf-data/venvs/fencing`).

```bash
scripts/hcf saber build-mjcf   # -> src/fencing/saber/assets/g1_29dof_saber.xml + saber_link_snippet.urdf
scripts/hcf saber reach        # -> ~/hcf-data/eval/saber/reach.json (prints the tables and the sweep)
scripts/hcf saber render       # -> ~/hcf-data/eval/saber/renders/*.png (needs reach.json; MUJOCO_GL=egl by default)
scripts/hcf saber smoke        # GPU; re-runs itself in ~/hcf-data/venvs/mjlab if mjlab/torch are missing
```

Useful options:

- `build-mjcf`, `reach` and `render` accept `--mount-pitch DEG`, `--blade-length M` and `--mass KG`. `build-mjcf` also takes `--collision`.
- `reach` takes `--restarts N` (default 32), `--arm-only` (skip the waist pass) and `--no-sweep`.
- `render` takes `--joint-set arm|arm+waist`.
- `smoke` takes `--num-envs` (default 32), `--steps` (default 100), `--device` and `--motion-file`. The default motion is `~/hcf-data/npz/dance1_subject2_mjlab.npz`.

The grip offset in the sweep is not a CLI option. It is `SaberSpec.mount_pos` and `reach.MOUNT_SWEEP`.

## Bladework keyframes (Phase 1.4d)

Video gives the legs and torso but not the wrist or the blade, so the saber line of the Goal 1 bladework motions is keyframed (`fencing.saber.keyframes`, `fencing.saber.recipes`, CLI `fencing motion blade` / `blade-all`; pipeline details in [pipeline.md](pipeline.md#bladework-phase-14d-keyframed-saber-arm)). Each key solves the right arm with the same DLS IK as the reach analysis, but inside the 0.9 soft limits and relative to the **base motion's** shoulder at that frame. Waist yaw/pitch are freed only if the arm alone fails. The arm then follows minimum-jerk moves at ≤ 0.5× motor speed.

Extra targets beyond the reach table: `extension`, plus `parry_tierce_high` and `parry_quarte_high`, the reachable versions of tierce and quarte from the section above (hand at chest height, not waist height).

First run (2026-11-09) on the **synthetic** bases (no video en garde or advance-lunge is published yet). Tip and direction errors are measured on the final motion at each key frame:

| Motion | Key | Joints | Tip err (cm) | Dir err (deg) | Attack/parry move (s) | `motion check` |
|---|---|---|---|---|---|---|
| extension | extension | arm | 0.0 | 0.0 | 0.47 | pass |
| parry_prime | parry_prime | arm | 0.0 | 0.0 | 0.30 | pass |
| parry_seconde | parry_seconde | arm | 0.0 | 0.0 | 0.30 | pass |
| parry_tierce_high | parry_tierce_high | arm | 0.2 | 0.2 | 0.43 | pass |
| parry_quarte_high | parry_quarte_high | arm+waist | 0.0 | 0.0 | 0.30 | pass (not in the Goal 1 set) |
| parry_quinte | parry_quinte | arm | 0.4 | 0.3 | 0.30 | pass |
| parry_sixte | parry_sixte | arm | 1.1 | 1.1 | 0.30 | pass |
| cut_head | cut_head | arm | 0.0 | 0.0 | 0.27 | pass |
| cut_flank | cut_flank | arm | 0.0 | 0.0 | 0.37 | pass |
| cut_chest | cut_chest | arm | 0.0 | 0.0 | 0.33 | pass |
| thrust | point_thrust | arm | 0.0 | 0.0 | 0.37 | pass |
| advance_lunge_cut | cut_head at touchdown (1.70 s) | arm | 0.0 | 0.0 | 0.30 | pass |

The en garde guard keys at the start and end of every motion are within 0.1 cm and 0.1°. No motion has a self-collision frame (blade capsule on, 5 mm depth threshold) or a keyframed joint beyond the soft limits. The fastest keyframed joint reaches 0.45× its motor limit.

Not constrained: the edge orientation (roll about the blade), and the shape of the blade path between keys (joint-space min-jerk, collision-checked frame by frame).

**End hold (default 2 s).** `motion blade` / `blade-all` end with at least this much whole-body stillness (`BladeConfig.end_hold_s`, CLI `--end-hold`, default 2.0 s; the library default is 0 so older scripts reproduce). The hold is counted from the later of the last key's reach and the moment the base motion comes to rest, so a base whose footwork recovery outlasts the last key still gets the full hold. A base that already ends that still is not lengthened. mjlab resamples an env as soon as its reference ends, so a policy only ever practises standing for as long as the reference's own end hold, while the eval holds the last frame for 2 s and counts a fall as a failed recovery. `advance_lunge_cut_v2` (0.5 s hold) recovered in 0.82 of envs; `_v3` with the 2 s hold, 1.00. Details in the [pipeline.md](pipeline.md) known-errors table, row "Policy falls or drifts in the eval's post-motion hold".

**Current set (`*_v3`, 2026-11-10).** These were regenerated on the video `en_garde_v3` base (front foot fixed, 2 s end hold) with `motion blade-all --end-hold 2`; `advance_lunge_cut_v3` is on `advance_lunge_v3`. All pass `motion check`. Training results are in [ADR 0003](adr/0003-blade-tip-reward.md#later-_tip-runs-2026-11-10-weight-and-std-unchanged) and the roadmap (1.5.3 to 1.5.7).

**GVHMR set (2026-11-12).** `motions/recipes/goal1_gvhmr.toml` regenerated the ten static motions on the GVHMR `en_garde_v4` (`extension_v4` ... `cut_head_v5`), with the same scripts, plus `advance_lunge_cut_v6` on `advance_lunge_v5`. All pass `motion check`, all keys are within tolerance, and every static motion starts on `en_garde_v4`'s first frame (parry_quinte_v4 frame 0, the entry's target) and ends on its last frame with the arm back on the guard target. The legs settle by up to 0.055 rad over en garde (v3 set: 0.104), so the end guard key is re-solved. It lands within 0.041 rad of the start pose. `cut_flank`, `cut_chest` and `thrust` end 0.0052 rad from the other seven, which end on the same frame. **`thrust_v4` and `cut_flank_v4` are superseded by `thrust_v5` and `cut_flank_v5`** (next paragraph). `thrust_v4`'s joint-space move swings the tip 0.72 m sideways and down to 7 cm above the floor at 15 m/s (tip path 6.38 m; `thrust_v3`: 1.30 m, lowest 0.93 m). `cut_flank_v4` dips to 0.34 m (v3 0.59 m).

**Why thrust_v4 sweeps the floor, and the blade-line fix (2026-11-12).** The GVHMR fencer stands side-on: `en_garde_v4`'s pelvis yaw is 84° (`en_garde_v3`: 0°), and the shoulder line puts the chest about 79° off the fencing line (v3: about 21°). The blade targets are set along world +x from the right shoulder, so on this body every `point_thrust` solution raises the arm sideways. All 30 valid solutions out of 65 IK starts (warm start + 64 random) have shoulder roll −1.1 to −2.1 rad (`thrust_v3`'s key: +0.12), and none is closer than 3.0 rad to the guard pose in joint space. A joint-space move from the guard to the warm-started solution (roll −1.56, yaw +1.42, wrist roll −1.64) sweeps the tip 3.25 m. The best of the 30 still sweeps it 0.95 m with a 0.19 m sideways swing, against 0.47 m in a straight line. So the IK seed could not fix it: there is no nearby thrust pose to find. The earlier note that the video arm "hangs straight (elbow −0.02 rad)" was wrong. The G1 elbow is bent at 0 rad (99° inner angle) and straight at about 1.4 rad (167°). The keyframed guard arm is the near-straight one (elbow 1.41). `BladeConfig.blade_path = "line"` (opt-in, default `"joint"`) moves the tip along the straight line between two keys, with the blade direction slerped. The arm follows by IK continuation (60 samples), and its two redundant DoF are pulled toward the joint-space interpolation. A new target is solved from the end of that line, and the move is timed so no joint passes the speed cap along the path. `goal1_gvhmr.toml` sets it for `thrust` and `cut_flank`, and later `extension`. The other GVHMR bladework files and every `goal1.toml` output rebuild byte-identical, which was checked by rebuilding them after the change. Measured on the published files:

| motion | tip path (m) | lowest tip (m) | tip path / straight, out and back | largest arm joint change guard→key (rad) | elbow range (rad) | max arm joint speed (rad/s) | max arm joint accel (rad/s²) | forearm–torso clearance (cm) | move (s) | max torque ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| thrust_v3 | 1.303 | 0.928 | 1.38, 1.36 | 1.86 | 1.44 to 1.54 | 9.45 | 76 | 0.2 | 0.37 | – |
| thrust_v4 | 6.384 | 0.069 | 6.79, 6.92 | 2.01 | 1.41 to 1.80 | 10.22 | 83 | 6.6 | 0.37 | 2.74 |
| thrust_v5 | 0.955 | 0.950 | 1.01, 1.01 | 2.01 | −0.23 to 1.42 | 13.75 (elbow) | 229 (elbow) | 6.6 | 0.57 | 0.47 |
| cut_flank_v3 | 2.492 | 0.590 | 1.73, 1.72 | 1.26 | 1.44 to 1.69 | 10.00 | 115 | 0.2 | 0.23 | – |
| cut_flank_v4 | 3.557 | 0.342 | 2.49, 2.51 | 2.12 | 1.41 to 1.88 | 9.76 | 73 | 6.6 | 0.40 | 1.00 |
| cut_flank_v5 | 1.431 | 0.696 | 1.00, 1.00 | 1.79 | −0.35 to 1.42 | 14.58 (elbow) | 258 (elbow) | 5.7 | 0.40 | 0.59 |
| extension_v3 | 0.967 | 1.034 | 1.22, 1.24 | 1.49 | 1.44 to 1.78 | 5.91 | 39 | 0.2 | 0.47 | – |
| extension_v4 | 1.722 | 0.981 | 2.24, 2.19 | 2.71 | 0.46 to 1.42 | 10.14 | 62 | 6.6 | 0.50 | 0.34 |
| extension_v5 | 0.796 | 1.049 | 1.01, 1.00 | 1.82 | −0.24 to 1.42 | 12.33 (elbow, 0.34× motor) | 157 (elbow) | 6.6 | 0.47 | 0.37 |

`extension_v5` was added after the first two (2026-11-12, the same day), by the same recipe change (`blade_path = "line"` on `extension`): `extension_v4` is not swept down, but it took the tip 2.24× the straight distance for the simplest blade action. Clearance is the minimum over the whole motion between the forearm capsules and the torso capsule (below). Acceleration is the second difference of the 30 fps arm joints. Cost of the line path: the arm bends at the elbow during the move (thrust_v5: from 1.41, 167°, to −0.23, 86°, and back out to 1.22, 163°), so the hand draws in and re-extends while the tip travels straight. The thrust takes 0.57 s at the 0.5× motor speed cap (v3 0.37 s), and its fastest joint is now the elbow at 0.44× motor. The tip tracks its line to within 1.3 cm (flank 0.8 cm). `thrust_v5` holds the wrist roll at +1.775 rad, on its soft limit. Neither v5 motion has been trained yet. The torque ratio (`path_torque_ratio`) is computed along the line path; the joint-space formula would not describe that move. The forearm comes no closer to the torso during either move than in the guard. Minimum distances between the primitive collision geoms of `ArmModel.collision_model` (`mj_geomDistance`, frames of both moves): `thrust_v5` forearm to torso 6.6 cm, at the guard; head 27 cm; hand to torso 23.6 cm; saber to torso 32.6 cm. `cut_flank_v5` forearm to torso 5.7 cm, partway through the return, against 6.6 cm in the guard; head 25 cm; hand 21 cm; saber 31 cm. `extension_v5` forearm to torso 6.6 cm (6.8 cm in its first guard), the same as `extension_v4`. Its elbow folds to −0.24 rad like the other two, and its peak arm acceleration is 157 rad/s² against 62 on v4; the torque ratio along the path is 0.37. `thrust_v3` and `cut_flank_v3` keep the forearm 0.2 cm from the torso in their guard. These are capsules and spheres, not the meshes.

**The other GVHMR bladework, move by move (2026-11-12).** Each move is guard → key or key → guard, framed by the sidecar's key frame and `move_s_used`. "Excursion" is the largest max − min of one arm joint over the move. The criterion for "swept like thrust_v4" is a move whose path/straight ratio is above v3's on the same move **and** whose lowest tip is at least 0.10 m below v3's: the tip sweeps down, toward the floor. Per-move cells list the forward move, then the return. The lowest tip is over the whole motion; the two moves' lowest tips differ by at most 0.035 m:

| motion (v3-era / GVHMR) | tip path per move (m) | path / straight per move | lowest tip (m) | excursion (rad, joint) | swept? |
|---|---|---|---|---|---|
| extension v3 / v4 | 0.47, 0.47 / 0.86, 0.84 | 1.22, 1.24 / 2.24, 2.19 | 1.034 / 0.981 | 1.49 shoulder pitch / 2.71 wrist pitch | no (tip 0.05 m lower); v5 anyway, see below |
| parry_prime v3 / v4 | 2.62, 2.66 / 1.99, 2.00 | 2.44, 2.46 / 1.84, 1.84 | 0.303 / 0.302 | 1.98 / 2.07 shoulder pitch | no (ratio lower) |
| parry_seconde v3 / v4 | 2.78, 2.82 / 2.58, 2.60 | 2.78, 2.81 / 2.57, 2.59 | 0.097 / 0.235 | 1.46 / 1.88 elbow | no |
| parry_tierce_high v3 / v4 | 0.95, 0.97 / 0.88, 0.90 | 1.31, 1.33 / 1.21, 1.23 | 1.160 / 1.226 | 2.54 / 2.49 wrist yaw | no |
| parry_quinte v3 / v4 | 0.97, 0.97 / 1.37, 1.39 | 1.08, 1.08 / 1.52, 1.54 | 1.205 / 1.228 | 1.61 / 1.94 shoulder pitch | no (tip stays overhead) |
| parry_sixte v3 / v4 | 0.42, 0.42 / 0.40, 0.41 | 1.01, 1.01 / 1.02, 1.02 | 1.205 / 1.228 | 0.70 / 0.82 shoulder yaw | no |
| cut_head v4 / v5 | 2.24, 2.28 / 1.36, 1.35 | 4.56, 4.51 / 2.72, 2.75 | 0.398 / 0.819 | 1.57 / 2.82 shoulder pitch | no (ratio lower) |
| cut_chest v3 / v4 | 1.53, 1.55 / 1.57, 1.57 | 2.55, 2.55 / 2.60, 2.59 | 0.793 / 0.822 | 1.50 / 2.40 shoulder pitch | no |
| cut_flank v3 / v4 | 1.22, 1.25 / 1.77, 1.77 | 1.73, 1.72 / 2.49, 2.51 | 0.590 / 0.342 | 1.26 / 2.12 wrist pitch | **yes** → v5 |
| thrust v3 / v4 | 0.64, 0.64 / 3.18, 3.20 | 1.38, 1.36 / 6.79, 6.92 | 0.928 / 0.069 | 1.86 / 2.01 wrist pitch | **yes** → v5 |

No further motion meets the criterion. `extension_v4` has the largest detour that does not sweep down: about twice the straight distance, with the wrist pitch swinging 2.7 rad (v3: 1.49 shoulder pitch). It was rebuilt on the line path anyway, as the simplest blade action, and `extension_v5` has path/straight 1.01 and 1.00. Its tip stays 1.05 m up, both moves are on the line (tracking 0.8 cm), and the check passes with no reasons. It starts and ends on `parry_quinte_v4`'s first and last frame exactly. For parry_quinte, a scratch trial used the line out to the parry (1.00) but not on the way back (tracking 6.6 cm > 5 cm, so the return stays in joint space at 1.54). It was not published.

**Where the line path falls back.** A move stays in joint space, and its target is solved exactly as in joint space (same seed, same random state), when the tracked end lands in another solution family (> 0.1 rad), when the tip strays more than 5 cm from the line, or when the base body alone carries a fixed arm's tip more than 3 cm over the move (`LINE_BODY_SHIFT_TOLERANCE_M`). The last case exists because the line is solved on the body at the key's reach frame, which on a lunge would put the tip off the line. The GVHMR guard's sway moves it 0.7-1.7 cm. A line that would start outside the soft joint limits is refused (`track_blade_line` raises) rather than clipped, since clipping would jump the arm on the first frame. `summary["line_moves"]` reports `body_shift_m` with the other checks.

## Wrist action scale

**Problem (2026-11-09).** The first saber policy, `extension_v1` at 1000 iterations, tracked the body well: success 1.00 out of 512, body error 2.4 cm, and every robustness pass at 1.00. But its **blade tip error was 38 cm**, in both mjlab and the CPU MuJoCo sim2sim.

The per-joint breakdown (scratch rollout over `fencing.sim2sim`) shows where the error comes from. The right wrist pitch is off by 0.71 rad on average, the elbow by 0.40, and the wrist yaw by 0.38. Every other joint is within about 0.1 rad.

**Cause.** With the fixed 30° mount (ADR 0002), the en garde guard alone needs about 1 rad at a wrist pitch/yaw joint (−0.83 pitch and −1.17 yaw in the IK solution). The keyframed motions swing these joints over ranges of 1.2–2.5 rad. mjlab maps an action to a PD target as `default + scale × action`, with scale = 0.25 × effort / stiffness. For the 5 Nm wrist motors that is 0.075 rad per unit. Holding 1 rad therefore takes an action of about 13. Swinging 2.5 rad takes a change of about 33, which the action-rate penalty also charges for. At 1000 iterations the policy had not got there.

**What was tried and dropped.** A null-space posture term in the blade IK pulled wrist pitch/yaw toward 0. It used the arm's 2 redundant DoF (tip position plus blade direction fix 5 of 7). The guard pose still needed about 1.1 rad, and the term cost key accuracy, so it was reverted. Where the IK's wrist angles come from is the mount geometry, not the solver.

**Fix.** The saber task widens the right wrist pitch/yaw scale to 0.25, so a 1 rad target is an action of about 4, the same range the other arm joints use. PD gains and the 5 Nm torque limits are unchanged, so the physics is identical. The scale is exported in the ONNX metadata (`action_scale`), which means `hcf sim2sim` and motion_tracking_controller apply it with no extra step.

**Old runs keep their scale.** A checkpoint's actions only mean something under the scale it trained with. So `hcf train eval/export/play/record-obs` and `train run --resume` read the scale from the run's own `params/env.yaml` (`provenance.trained_action_scale`) instead of taking the task's current default. When the two differ they print a note. As a result, `extension_v1_v1` still evaluates and exports with 0.0745.

**Tradeoff.** Exploration noise at the wrist is now about 3× larger in radians early in training. The static gravity load of a 0.35 kg saber at about 0.4 m is roughly 1.4 Nm, well inside 5 Nm, so the angles themselves are fine.

**Open questions.**
- A mount sweep that minimises wrist excursion over the Goal 1 targets, instead of counting reachable targets, might pick a different pitch. This interacts with the hardware socket design.
- The wrist torque during fast cuts (5.19 Nm at 0.35 kg, see Payload) stays a hardware limit either way.

`fencing motion blade` reports the peak wrist pitch/yaw angle of each motion as `wrist_pitch_yaw_max_abs_rad` in its summary.

## Torque demand of keyed moves (2026-11-11)

The Payload section above guessed 100 rad/s² for a fast wrist cut. `keyframes.move_torque_ratio` computes the demand instead. It runs inverse dynamics with the nominal saber along a keyed minimum-jerk move, with the body pinned, and returns peak |torque| / effort limit per arm joint. The blade summary reports the worst as `max_torque_ratio`. Values for the `*_v3` attack and parry moves, as shoulder pitch / wrist pitch / wrist yaw: cut_head 1.94 / 5.20 / 2.71 (0.2 s), cut_flank 0.73 / 1.37 / 1.32, cut_chest 0.87 / 0.95 / 2.07, thrust 0.62 / 0.24 / 0.35, parry_seconde 0.97 / 2.27 / 1.40, parry_prime 1.24 / 1.19 / 1.17.

Every fast move asks the 5 Nm wrists for more than they have, so the reference is not exactly trackable at its fastest instants. The policies cope at about 2x, and cut_head at 5x is where key-frame placement broke down. `cut_head_v4` takes 0.3 s and demands 0.81 / 2.35 / 1.20 ([pipeline.md](pipeline.md), known errors). These are reference demands, not what a policy draws: in sim2sim cut_head_v3 peaked at 22.8 of 25 Nm on the shoulder pitch and at the 5 Nm limit on the wrist pitch.
