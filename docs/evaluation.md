# Goal 1 evaluation

How a tracking checkpoint is scored, in mjlab (`fencing train eval`) and in plain CPU MuJoCo (`fencing sim2sim`). Both use the same metric code (`src/fencing/train/metrics.py`) and the same thresholds (`src/fencing/train/thresholds.py`), so their numbers can be compared directly.

## Commands

```bash
scripts/hcf train export <run> --checkpoint N            # policy.onnx + scene/scene.xml + export.json
scripts/hcf train eval   <run> --checkpoint N --num-envs 1024 [--no-robustness] [--seconds S]
scripts/hcf train play   <run> --video --checkpoint N [--seconds 10]
scripts/hcf sim2sim      <run> --checkpoint N [--video] [--pd explicit] [--motion NPZ]
```

`<run>` is a run dir, its name, or a run name (`dance1_subject2_p12v2`). Everything lands in `$HCF_DATA/eval/<run dir name>/model_<N>/`:

| File | Written by |
|---|---|
| `metrics.json` | `train eval`: nominal pass, robustness passes, thresholds, verdict |
| `policy.onnx`, `scene/`, `export.json` | `train export` |
| `sim2sim.json`, `sim2sim.mp4` | `sim2sim` (the gap vs `metrics.json` is added when that file exists) |
| `play.mp4` | `train play` |

**Tests.** Two commands, both needed. The main suite runs in the fencing venv. It skips the
mjlab task tests (`tests/test_saber_mjlab.py`), because that venv has no torch or mjlab:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/hcf-data/venvs/fencing UV_LINK_MODE=copy uv run --project <repo> pytest -q <repo>/tests
PYTHONPATH=src ~/hcf-data/venvs/mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_saber_mjlab.py   # from <repo>
```

The second command is the only one that checks the training tasks. That includes the check
that the entry task leaves the existing saber tasks exactly as committed
(`tests/data/saber_task_cfgs.json`, [saber.md, mjlab tasks](saber.md#mjlab-tasks)).

Useful options: `--seconds S` scores only the first S seconds (quick checks on long motions); `--hold-seconds` (default 2) sets the recovery hold; `--passes nominal push` picks passes; `--friction-range 0.2 1.5` widens the friction pass; `--threshold nominal.success_rate=0.9` or `--thresholds file.json` override thresholds. All commands exit 0 and print the verdict; read `verdict` in the JSON to gate on it.

## Protocol

Each episode starts on reference frame 0 (mjlab play mode: no root pose or velocity perturbation; mjlab still adds U(-0.1, 0.1) rad joint noise at reset, which is what makes the N envs differ) and runs the **whole motion**, then holds the last reference frame for `--hold-seconds`. mjlab's play config would wrap back to frame 0 at the end, teleporting the robot; eval clamps the command at the last frame instead. Training never sees this hold: mjlab resamples an env as soon as its motion ends, so a policy only learns to stand as long as the reference's own end hold. Reference motions therefore carry an end hold of at least `--hold-seconds` (2 s; see [pipeline.md](pipeline.md#video-derived-motion-known-errors-and-their-fixes)), or recovery is down to luck.

A rollout ends at the first **termination**, using the task's own checks: torso height off the reference by more than 0.25 m, torso tilt (projected-gravity z difference) above 0.8, or an ankle or wrist more than 0.25 m off in height. Metrics are averaged over the steps an env was alive, pairing the post-step robot with the reference frame that step was scored against (as mjlab's `evaluate.py` and the reward do).

Note the difference from mjlab's own `evaluate.py`: it scores a fixed 10 s window (the training episode length) with observation noise on. We score the full motion and report the 10 s figure as `window_success_rate` for comparison. Our `obs_noise` pass is the closest match to mjlab's protocol.

## Metrics

| Key (in `nominal`, and in each robustness pass) | Definition |
|---|---|
| `success_rate` | Share of the N envs with no termination before the last reference frame |
| `window_success_rate` | Same, over the first 10 s only (mjlab `evaluate.py`'s horizon) |
| `joint_angle_error_rad` | Mean absolute joint-angle error over the 29 joints |
| `body_pos_error_m` | Mean position error of the 14 tracked bodies against the reference **re-anchored** to the robot's torso x/y and heading (mjlab's `r_mpkpe`). Pose error with global drift removed |
| `body_pos_error_global_m` | Same in the world frame (mjlab's `mpkpe`); includes drift |
| `per_body_pos_error_m` | `body_pos_error_m` per tracked body |
| `anchor_pos_error_m`, `anchor_ori_error_rad` | World-frame torso position and rotation error (the "root" error; the torso is BeyondMimic's anchor body) |
| `foot_slip_m_s` | Mean horizontal speed of an ankle while it should be planted: the reference foot is within 3 cm of its standing height and moving under 0.15 m/s, *and* the robot foot is down too |
| `recovery_rate` | Share of envs that finish the motion and the hold without termination, ending with the pelvis within 5 cm of the reference final height and moving under 0.2 m/s |
| `blade_tip_error_m` | Saber task only. Mean distance between the robot's blade tip and the reference blade tip (re-anchored like `body_pos_error_m`); `blade_tip_error_global_m` and `blade_tip_error_max_m` too. The reference tip comes from the reference wrist pose plus the saber's fixed offset, since the reference motion has no saber body |
| `blade_tip_error_key_m` | Bladework motions with key frames only (see below). The same tip distance, re-anchored, averaged over the frames within ±2 policy steps of each key frame; `blade_tip_error_key_global_m` is the world-frame variant, and `blade_key_frames` lists the scored key frames (reference frames at 50 Hz). Null when no env was alive at a key, or when the scene has no saber |
| `termination_counts` | Per term, how many envs it ended. Terms that fire on the same step are all counted, so the counts can add up to more than the number of terminated envs |

### Key frames

The roadmap defines the blade-tip error at the **key frame**: the held parry, the moment of the hit, the end of the extension. Bladework motions come from blade keys (`fencing motion blade`, [saber.md](saber.md)), and the key frame is derived from that schedule, not typed by hand. It is every blade key except the rest guard (`en_garde`) and hand-backs to the base arm, at the frame the keyframer *reached* the target (`blade.keys[i].frame` in the sidecar). Each Goal 1 recipe has exactly one: `parry_*` at the parry, `cut_*` and `thrust` at the landing, `extension` at full extension, `advance_lunge_cut` at the cut on lunge touchdown. The rule lives in `src/fencing/saber/scored_keys.py`.

- **Recorded** in the motion's sidecar as `key_frames` (`[{index, target, kind, frame, time_s}]`). New bladework gets it when it is generated or published. Existing sidecars are backfilled from their own `blade.keys` with `scripts/hcf motion key-frames [motions/X_vN.json ...] --write`, which is a dry run without `--write`. With no paths it picks the newest version of every recipe. It rewrites only that field and never touches a CSV.
- **Found** by every scorer the same way: the NPZ the policy tracks (`export.json` `motion_file`, the run's NPZ in `train eval`, or `--motion`) → its `fencing train npz` sidecar (source CSV and sha256) → the CSV's sidecar → `key_frames`. A key at `time_s` is reference frame `round(time_s / 0.02)`, because csv_to_npz samples the motion at t = k·0.02 from 0. If any link is missing (video motions, un-backfilled sidecars, an NPZ converted by hand) there are no key frames, and the report keeps exactly its old keys. If the CSV's sha256 no longer matches the NPZ's record, its key frames are ignored with a warning. When key frames are found, the report carries `key_frames` (sidecar, steps, keys).
- **Window ±2 steps (0.1 s)**. Three reasons: the reference tip is nearly at rest there (min-jerk arrival, then a hold); a 30 fps key lands on the 50 Hz grid only to within a frame; and five samples damp single-step jitter in the one-episode sim2sim and MTC scores. Measured in sim2sim on 2026-11-11, the key frame alone vs ±2 vs ±5 gives 4.01 / 4.15 / 9.53 cm (cut_head_v3), 2.50 / 2.59 / 3.22 (cut_chest_v3), 2.67 / 2.47 / 1.95 (parry_seconde_v3), and 1.88 / 1.84 / 2.00 (advance_lunge_cut_v4). At ±5 a fast cut's arrival leaks in.
- **Re-anchored is the gated variant.** On advance_lunge_cut_v4 the global key error is 14.2 cm against 1.8 re-anchored. The difference is lunge drift of the whole body, which `anchor_pos_error_m` already gates, not blade placement.

Sim2sim key-frame blade error, re-anchored (global), 2026-11-11, checkpoint 2999. The re-scored sim2sim.json files have every pre-existing value unchanged: extension_v3 1.1 (2.0) cm, parry_prime_v3 1.6 (1.3), parry_seconde_v3 2.5 (3.9), parry_sixte_v3 1.6 (2.4), parry_quinte_v3 2.0 (4.0), parry_tierce_high_v3 1.6 (2.1), cut_head_v3 4.2 (5.0), cut_flank_v3 2.7 (3.4), cut_chest_v3 2.6 (7.7), thrust_v3 3.3 (4.8), advance_lunge_cut_v4 1.8 (14.2). The cuts and the thrust are the hardest key frames, and cut_head_v3's key error (4.2 cm) is well above its whole-motion mean (2.6 cm). `cut_head_v4`, with the attack move lengthened from 0.2 s to 0.3 s ([saber.md](saber.md)), brings it to 2.0 (8.0) cm and replaces v3 in the final set.

## Passes

| Pass | What changes from nominal |
|---|---|
| `nominal` | mjlab's play config, without observation noise, pushes, or startup randomization (encoder bias, torso CoM offset, foot friction, saber mass). Foot friction is the model's 0.6 |
| `push` | mjlab's training `push_robot` event: a root velocity kick every 1-3 s (up to 0.5 m/s linear, 0.78 rad/s yaw) |
| `obs_noise` | Training observation noise plus encoder bias |
| `friction` | Training `foot_friction`: per-env friction, uniform in 0.3-1.2 |
| `saber_mass` | Saber task only. Per-env saber mass over the training range, 0.12-0.35 kg |

Each pass reuses mjlab's own event term from the task's training config, so "robust" means robust to exactly what the policy was trained on. Widen the friction range to probe beyond it.

## Thresholds (provisional)

Thresholds are provisional until Phase 1.2 numbers exist on known motions. (The Phase 1.2 numbers now exist: see "Phase 1.2 at the upstream length" below.)

**Open question: the joint gate and the wrists.** `joint_angle_error_rad` is the mean over all 29 joints. For saber motions the three wrist joints are redundant once the blade tip is placed (the tip has its own gate and reward), and the policy picks its own wrist combination. Sim2sim breakdown on 2026-11-10 (mean over all joints / without the 6 wrist joints / wrists only): advance_lunge_cut_v3 0.110 / 0.080 / 0.225, advance_lunge_cut_v2 0.099 / 0.082 / 0.165, extension_v2 0.081 / 0.079 / 0.089, en_garde_v1 0.091 / 0.083 / 0.119. Near-identical runs land about 0.01 apart, so with the wrists in, a 0.10 gate is a coin flip for bladework. Options: gate the non-wrist joints at 0.10 and the wrists separately, or exclude the sword wrist where the blade-tip gate applies. Not changed yet: this is the user's call.

**Phase 1.2 calibration point (2026-11-10).** The upstream recipe, unchanged, on LAFAN1 `dance1_subject2` (`dance1_subject2_p12v2r`, 10000 it): mean joint error 0.108 in mjlab and in sim2sim, which fails the 0.10 gate. Every other gate passes. Without the wrists it is 0.096, with the wrists alone at 0.163. A known-good motion trained with the reference recipe fails the joint gate, so 0.10 over all 29 joints is too tight as a pass/fail line. A non-wrist gate at 0.10 would pass dance1 and `advance_lunge_cut_v3`. Of the runs where I measured the split, only `cut_head_v2` stays over (0.102; its leg offset is fixed in v3, which passes the existing gate). They are defaults in `thresholds.py`, and every one can be overridden. The conclusion that 0.10 is too tight is withdrawn below.

**Phase 1.2 at the upstream length (2026-11-11).** The same recipe on the same motion, resumed from `model_9999` to 30,000 iterations in total (`dance1_subject2_p12v2r30k`, `model_29998`), which is the upstream recipe's own length. The 10,000-iteration run above was a third of that. Every gate passes, in four simulators:

| simulator | joint (rad) | body (cm) | anchor (cm) | slip (m/s) | recovery |
|---|---|---|---|---|---|
| mjlab (1024 episodes) | 0.0928 | 3.27 | 5.84 (0.054 rad) | 0.046 | 1.00 |
| CPU MuJoCo sim2sim | 0.0927 | 3.26 | 5.75 | 0.0456 | 1 |
| CPU PhysX | 0.0954 | 3.52 | 7.08 | 0.0434 | 1 |
| C++ controller (one run, 0 s settle) | 0.0944 | 3.33 | 24.5 | 0.037 | 1 |

In mjlab success is 1.00, push 0.994, observation noise 1.00 and friction 1.00. The C++ controller's anchor error of 24.5 cm only just passes the 25 cm gate: the state estimator's pelvis position error is 23.4 cm on average, the same odometry drift as before, and the anchor orientation error is 0.056 rad. At 10,000 iterations the same run had joint 0.108 (FAIL) in mjlab and sim2sim, and 0.110 with anchor 30.5 cm (FAIL) in the C++ controller. So the earlier reading, that the 0.10 joint gate was too tight as written, was wrong: the miss was run length. At 30,000 iterations the unchanged recipe passes with margin (0.093 against 0.10), and the gate stands as it is. Isaac Lab was still not run (Isaac Sim does not run on WSL2), so the Isaac half of "in mjlab and in Isaac Lab" stays open.

**Update 2026-11-11.** The saber cases turned out to be reference defects, not the gate: `advance_lunge_cut` had two different sword-arm poses for the same guard (v5: joint 0.076, was 0.098-0.110), and the rear wrists of the lunge family held unobserved retargeter angles (`advance_lunge_v4`: 0.081, was 0.099; `lunge_v5`: 0.087, was 0.100). All 16 final Goal 1 policies now pass the unchanged 0.10 gate in mjlab, CPU MuJoCo and the C++ controller, the worst at 0.091. dance1 was the last case over the gate (0.108 at 10,000 iterations); at 30,000 it passes too (0.093, above), so the gate was left as it is.

| Check | Default | Why |
|---|---|---|
| `nominal.success_rate` | >= 0.95 | BeyondMimic's bar: its MuJoCo sim-to-sim ablation counts a hard motion (cartwheel) as solved at 95% success. The paper reports success rates, not error tables |
| `nominal.joint_angle_error_rad` | <= 0.10 | About 6 degrees mean. The policy reads joint targets only as a phase cue, so this is loose on purpose |
| `nominal.body_pos_error_m` | <= 0.05 | 5 cm drift-free body error. Published whole-body trackers on LAFAN1-class motions land around 3-6 cm; the reward's std (0.3 m) is far looser |
| `nominal.anchor_pos_error_m` | <= 0.25 | Global torso drift over a long motion. Matches the termination height threshold |
| `nominal.anchor_ori_error_rad` | <= 0.20 | About 11 degrees of torso rotation error, heading included |
| `nominal.foot_slip_m_s` | <= 0.10 | A planted foot sliding faster than 10 cm/s is visible and would cost balance on the real floor |
| `nominal.recovery_rate` | >= 0.90 | Stable en garde after the motion, most of the time |
| `nominal.blade_tip_error_m` | <= 0.10 | 10 cm of blade-tip error, about a target zone's width. The mean over the whole motion includes the lag through fast swings, so it stays loose. The key-frame check below is the tight one |
| `nominal.blade_tip_error_key_m` | <= 0.05 | Half the whole-motion gate, as planned once key frames existed. At the key the reference tip is at rest, so swing lag is out and what remains is placement. 5 cm is the body-position gate and 2.5x the keyframer's own 2 cm placement tolerance. All 11 current v3/v4 bladework policies pass in sim2sim (1.1-4.2 cm), but cut_head_v3 at 4.2 cm is close to the line. Only listed for motions with key frames: without them the check is left out entirely, so those runs report the same checks and verdict as before. With key frames and no saber in the scene, it is not applicable |
| `robustness.push.success_rate` | >= 0.80 | Pushes are harsher than the other passes |
| `robustness.obs_noise.success_rate`, `.friction.`, `.saber_mass.` | >= 0.90 | Training-range perturbations should barely matter |

The BeyondMimic figure was checked against the paper (arXiv 2508.08241). The tracking-error thresholds are not from that paper, which publishes no per-metric error table. They are judgement calls, to be replaced with what our own policies get on LAFAN1 motions.

A check whose metric is missing is `skipped`, never passed. There are three verdicts:

- **FAIL:** a check failed, or no check ran.
- **PARTIAL:** every check that ran passed, but either an applicable check was skipped (e.g. `--no-robustness`, a crashed pass, or `--hold-seconds 0` so recovery was never scored) or the motion was truncated with `--seconds`. PARTIAL never counts as passing.
- **PASS:** every applicable check ran and passed over the whole motion.

Checks that cannot apply to a run are left out of this. These are the blade-tip errors on a run without a saber, and a robustness pass for a randomization the task doesn't have. The key-frame check on a motion without key frames is not even listed. On a saber run with key frames, a skipped key-frame check (no env alive at the key, or `--seconds` cut it off) makes the run PARTIAL.

## Sim-to-sim

`fencing sim2sim` runs the exported ONNX in **plain CPU MuJoCo** (the `mujoco` Python bindings plus `onnxruntime`, from the fencing venv, with no mjlab or torch) on the exact scene mjlab simulated (`scene/scene.xml`, written by `train export` with mjlab's runtime solver options). It uses only what a deployment has. That is the contract `motion_tracking_controller` reads:

- **ONNX metadata**: joint names and order, default joint positions, action scale, kp/kd, observation term names, scales and clips, anchor body, tracked bodies. mjlab rounds the floats to 3 decimals. Measured on the G1 export, that turns the wrist pitch/yaw action scale 0.0745 into 0.075 (+0.67%) and the 0.4386 joints' scale into 0.439 (+0.10%), so `train export` also attaches `hcf_full_precision` (JSON), and sim2sim uses it when present (`metadata_full_precision` in the report).
- **The reference motion bundled in the ONNX** (`time_step` input gives joint and body references), unless `--motion NPZ` is given.
- **The observation is rebuilt in Python** (`src/fencing/sim2sim/observation.py`), term by term from the robot state. `tests/test_sim2sim_observation.py` checks it against observations recorded from mjlab (`tests/data/mjlab_obs_g1_tracking.npz`, regenerate with `fencing train record-obs`). It must match to 1e-4.
- **Actions** become PD targets `default + scale * a`, applied for `decimation` (4) physics steps of 5 ms, which matches mjlab's 50 Hz policy and 200 Hz physics. `--pd actuator` (default) drives the scene's PD actuators, which is what mjlab simulates. `--pd explicit` computes torques `kp (q* - q) - kd qd` from the metadata gains every physics step, which is closer to deployment code. The scene's gains are checked against the metadata, and sim2sim refuses to run when they differ.
- The metrics are the same as above, with E = 1, a deterministic reset (no joint noise unless `--reset-joint-noise`), and termination checks reimplemented in numpy. `gap_vs_mjlab` reports `mujoco - mjlab` for each nominal metric. Note that the mjlab side averages over N noisy resets, while the MuJoCo side is one episode.

What the gap does and does not cover:

- **Covered**: the physics engine (MuJoCo-Warp on GPU vs MuJoCo on CPU), our observation builder and metadata parsing, ONNX export, action application.
- **Not covered here**: the real deployment stack. That is a separate check: `scripts/mtc_sim2sim.sh` runs the same ONNX in `motion_tracking_controller`'s own C++ controller, state estimator and MuJoCo simulator (ROS 2 Jazzy in a rootless Ubuntu 24.04 rootfs on this machine) and scores it with these metrics and thresholds into `mtc.json`, next to `sim2sim.json`. It covers plain-G1 and saber policies (saber policies run on a plant with the saber welded on), and the C++ reads the 3-decimal metadata, not `hcf_full_precision`. The scorer pairs each recorded state with the frame the controller was tracking, which is one tick (20 ms) behind mjlab's convention. See [mtc.md](mtc.md) for the build, the deterministic start, the scoring conventions and the results.
- **Not available**: Isaac Sim. It does not run on WSL2, so the roadmap's "Isaac Sim vs MuJoCo" gap is measured here as mjlab (MuJoCo-Warp) vs CPU MuJoCo instead. Its physics engine, PhysX, does run here without Kit: see the next section.

### Sim-to-sim in PhysX

`fencing sim2sim --engine physx` runs the same rollout, observation builder, reset and `TrackingScorer` in **CPU PhysX**, through `ovphysx` (Isaac Lab 3.0's OvPhysX backend: PhysX without Omniverse Kit, a pip wheel). Only the plant differs (`src/fencing/sim2sim/physx_plant.py`, `physx_rollout.py`). It writes `sim2sim_physx.json` next to `sim2sim.json` and the authored scene `physx_scene.usda` next to the output JSON (the same directory as `--out` when given), and applies the same thresholds. Checked on WSL2 on 2026-11-11: no Vulkan, no GPU (`OVPHYSX_DISABLE_GPU=1`), about 1-2 s per rollout.

`ovphysx` is not a fencing dependency. It is about 1.7 GB with `warp-lang` and `usd-core`, and its licence is NVIDIA's: the wheel's LICENSE.txt says "By downloading or using ovphysx, you agree to the NVIDIA Omniverse terms", and `ovstage` is NVIDIA-proprietary. There is no accept flag or prompt. It lives in its own venv:

```
uv venv --python 3.12 ~/hcf-data/venvs/physx
uv pip install --python ~/hcf-data/venvs/physx/bin/python ovphysx==0.7.0.74035321 physx-usd-schemas \
    newton-usd-schemas usd-core numpy scipy "mujoco~=3.11.0" onnxruntime pyyaml tqdm pytest
PYTHONPATH=src ~/hcf-data/venvs/physx/bin/python -m fencing.cli sim2sim <export dir> --engine physx
```

`--video` and `--pd explicit` are refused. `tests/test_sim2sim_physx.py` checks the conversions against MuJoCo in the fencing venv. Its plant test (readback and forward kinematics on a two-link model) runs only in the PhysX venv and skips elsewhere.

**The plant.** The USD is authored from the compiled MuJoCo model of the export's `scene.xml`, not from a URDF, so it is the scene mjlab simulated. Read back from PhysX in SI units on the G1 and saber scenes:

- **Matched**: masses (max diff 6e-7 kg, total 33.341 / 33.541 kg), COMs (7e-9 m), inertia tensors (1e-8), joint axes, frames and order (forward kinematics at reference frame 0 within 2.3e-7 m), joint limits (1.6e-7 rad), armature (4e-10), kp and kd from the ONNX metadata (relative 6e-8), effort limit values from the actuator forceranges (exact), zero joint friction and damping, the 33 capsule and sphere colliders, the 4 excluded body pairs, and the saber as a welded link with its own inertia. Foot-ground friction is exactly 0.6 and other body-ground friction 1.0, as in MuJoCo (the feet have priority 1). Gravity, 200 Hz physics, 50 Hz policy.
- **Not matched**: the contact model, since PhysX contacts are rigid with a contact offset and MuJoCo's are soft (solref 0.02, solimp). The solver: PhysX TGS (8 position / 4 velocity iterations, Isaac Lab's G1 values) vs MuJoCo implicitfast with Newton and a pyramidal cone. Joint limits are hard in PhysX and soft in MuJoCo. Self-contacts have friction 1.0 in PhysX but are frictionless in MuJoCo (condim 1). Linear and angular body damping is set to 0 where PhysX's defaults are not. How a drive clamps at its effort limit: the values match, but in a lifted (no-contact) open-loop replay, wrist joints asked for 1.9 to 2.6 times their effort limit drifted 0.011 rad in 20 ms between the engines, against 0.004 rad for unsaturated joints.
- **Velocity convention**: PhysX reads and writes centre-of-mass linear velocity. The adapter converts to the link origin (mjlab's `body_link_lin_vel_w` and the IMU velocimeter). A finite difference over a 1e-4 s step confirms it: 0.003 m/s residual after conversion vs 0.145 m/s without.

**Results** (checkpoint 2999, one deterministic episode, 2026-11-11; newest eval dir per policy, all 16 final policies). MuJoCo is `sim2sim.json`, MTC is `mtc.json`:

| policy | engine | success | recovery | joint (rad) | body (cm) | anchor (cm) | blade (cm) | blade at key (cm) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| en_garde_v3_pipe | MuJoCo | 1 | 1 | 0.077 | 3.8 | 2.3 | - | - | PASS |
|  | PhysX | 1 | 1 | 0.075 | 3.6 | 2.1 | - | - | PASS |
|  | MTC | 1 | 1 | 0.076 | 3.8 | 2.8 | - | - | PASS |
| advance_v1_pipe | MuJoCo | 1 | 1 | 0.078 | 3.1 | 3.3 | - | - | PASS |
|  | PhysX | 1 | 1 | 0.078 | 3.2 | 3.8 | - | - | PASS |
|  | MTC | 1 | 1 | 0.079 | 3.2 | 3.3 | - | - | PASS |
| retreat_v1_pipe | MuJoCo | 1 | 1 | 0.087 | 2.9 | 1.9 | - | - | PASS |
|  | PhysX | 1 | 1 | 0.087 | 2.9 | 2.3 | - | - | PASS |
|  | MTC | 1 | 1 | 0.086 | 2.9 | 2.0 | - | - | PASS |
| extension_v3_tip | MuJoCo | 1 | 1 | 0.074 | 4.2 | 2.2 | 1.3 | 1.1 | PASS |
|  | PhysX | 1 | 1 | 0.073 | 3.6 | 3.7 | 1.4 | 0.8 | PASS |
|  | MTC | 1 | 1 | 0.074 | 4.2 | 2.4 | 1.5 | 1.5 | PASS |
| lunge_v5_pipe | MuJoCo | 1 | 1 | 0.087 | 4.0 | 5.2 | - | - | PASS |
|  | PhysX | 1 | 1 | 0.089 | 3.7 | 6.2 | - | - | PASS |
|  | MTC | 1 | 1 | 0.088 | 3.9 | 5.8 | - | - | PASS |
| parry_prime_v3_tip | MuJoCo | 1 | 1 | 0.075 | 4.2 | 1.9 | 1.5 | 1.6 | PASS |
|  | PhysX | 1 | 1 | 0.074 | 3.8 | 2.6 | 1.5 | 2.7 | PASS |
|  | MTC | 1 | 1 | 0.074 | 4.1 | 2.1 | 1.8 | 2.0 | PASS |
| parry_seconde_v3_tip | MuJoCo | 1 | 1 | 0.091 | 4.4 | 2.6 | 2.5 | 2.5 | PASS |
|  | PhysX | 1 | 1 | 0.086 | 4.4 | 2.9 | 2.4 | 4.5 | PASS |
|  | MTC | 1 | 1 | 0.089 | 4.4 | 2.0 | 2.5 | 2.8 | PASS |
| parry_tierce_high_v3_tip | MuJoCo | 1 | 1 | 0.090 | 4.0 | 2.6 | 1.1 | 1.6 | PASS |
|  | PhysX | 1 | 1 | 0.091 | 3.8 | 2.7 | 1.0 | 1.7 | PASS |
|  | MTC | 1 | 1 | 0.089 | 3.8 | 2.8 | 1.1 | 1.7 | PASS |
| parry_quinte_v3_tip | MuJoCo | 1 | 1 | 0.072 | 4.6 | 2.8 | 1.6 | 2.0 | PASS |
|  | PhysX | 1 | 1 | 0.072 | 4.3 | 2.5 | 1.5 | 1.1 | PASS |
|  | MTC | 1 | 1 | 0.073 | 4.6 | 2.7 | 1.7 | 1.5 | PASS |
| parry_sixte_v3_tip | MuJoCo | 1 | 1 | 0.073 | 4.2 | 3.3 | 1.1 | 1.6 | PASS |
|  | PhysX | 1 | 1 | 0.071 | 3.9 | 4.9 | 1.2 | 1.4 | PASS |
|  | MTC | 1 | 1 | 0.072 | 4.1 | 3.5 | 1.2 | 1.6 | PASS |
| cut_head_v4_tip | MuJoCo | 1 | 1 | 0.088 | 4.5 | 1.8 | 2.4 | 2.0 | PASS |
|  | PhysX | 1 | 1 | 0.083 | 4.3 | 2.5 | 2.1 | 3.1 | PASS |
|  | MTC | 1 | 1 | 0.088 | 4.5 | 1.6 | 2.4 | 2.9 | PASS |
| cut_head_v3_tip (superseded by v4) | MuJoCo | 1 | 1 | 0.084 | 3.9 | 2.3 | 2.6 | 4.2 | PASS |
|  | PhysX | 1 | 1 | 0.079 | 3.6 | 3.1 | 2.6 | **6.1** | **FAIL** |
|  | MTC | 1 | 1 | 0.084 | 3.8 | 2.2 | 2.7 | 4.4 | PASS |
| cut_flank_v3_tip | MuJoCo | 1 | 1 | 0.068 | 4.2 | 2.4 | 1.8 | 2.7 | PASS |
|  | PhysX | 1 | 1 | 0.067 | 4.0 | 2.4 | 1.8 | 2.8 | PASS |
|  | MTC | 1 | 1 | 0.068 | 4.1 | 1.9 | 2.0 | 2.3 | PASS |
| cut_chest_v3_tip | MuJoCo | 1 | 1 | 0.078 | 4.0 | 2.2 | 1.6 | 2.6 | PASS |
|  | PhysX | 1 | 1 | 0.075 | 3.7 | 2.4 | 1.6 | 2.8 | PASS |
|  | MTC | 1 | 1 | 0.077 | 3.9 | 1.9 | 1.6 | 1.9 | PASS |
| thrust_v3_tip | MuJoCo | 1 | 1 | 0.079 | 4.6 | 2.3 | 1.7 | 3.3 | PASS |
|  | PhysX | 1 | 1 | 0.078 | 4.5 | 3.1 | 1.7 | 1.8 | PASS |
|  | MTC | 1 | 1 | 0.078 | 4.4 | 2.0 | 1.5 | 1.9 | PASS |
| advance_lunge_v4_pipe | MuJoCo | 1 | 1 | 0.081 | 3.1 | 4.8 | - | - | PASS |
|  | PhysX | 1 | 1 | 0.079 | 3.2 | 7.5 | - | - | PASS |
|  | MTC | 1 | 1 | 0.080 | 3.1 | 6.8 | - | - | PASS |
| advance_lunge_cut_v5_tip | MuJoCo | 1 | 1 | 0.074 | 3.0 | 6.5 | 1.8 | 2.1 | PASS |
|  | PhysX | 1 | 1 | 0.074 | 3.2 | 7.8 | 2.0 | 2.3 | PASS |
|  | MTC | 1 | 1 | 0.076 | 3.1 | 8.0 | 2.0 | 1.5 | PASS |

**Entry candidates** (`stand_to_en_garde_v1`, the standby → guard entry for the saber chains,
2026-11-12; [mtc.md, Paid for joint angles](mtc.md#paid-for-joint-angles-stand_to_en_garde_v1_feetjoint05-2026-11-12-recommended)).
The motion has no key frame. MTC is the held start (`mtc.json`):

| policy | engine | success | recovery | joint (rad) | body (cm) | anchor (cm) | blade (cm) | blade at key (cm) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| stand_to_en_garde_v1_tip | MuJoCo | 1 | 1 | 0.056 | 3.1 | 5.6 | 1.2 | - | PASS |
|  | PhysX | 1 | 1 | 0.057 | 3.0 | 7.8 | 1.1 | - | PASS |
|  | MTC | 1 | 1 | 0.056 | 3.2 | 6.3 | 1.3 | - | PASS |
| stand_to_en_garde_v1_feet | MuJoCo | 1 | 1 | 0.0986 | 3.3 | 5.1 | 1.3 | - | PASS |
|  | PhysX | 1 | 1 | 0.0987 | 3.3 | 3.8 | 1.3 | - | PASS |
|  | MTC | 1 | 1 | 0.0992 | 3.3 | 4.9 | 1.4 | - | PASS |
| stand_to_en_garde_v1_feet5k (model 4998) | MuJoCo | 1 | 1 | **0.1000** | 3.1 | 4.6 | 1.5 | - | **FAIL** |
|  | PhysX | 1 | 1 | **0.1012** | 3.2 | 3.8 | 1.5 | - | **FAIL** |
|  | MTC | 1 | 1 | **0.1003** | 3.1 | 4.7 | 1.5 | - | **FAIL** |
| stand_to_en_garde_v1_feetjoint | MuJoCo | 1 | 1 | 0.031 | 2.6 | 6.0 | 1.1 | - | PASS |
|  | PhysX | 1 | **0** | 0.032 | 2.4 | 5.5 | 1.2 | - | **FAIL** |
|  | MTC | 1 | 1 | 0.032 | 2.6 | 6.3 | 1.1 | - | PASS |
| stand_to_en_garde_v1_feetjoint05 (recommended) | MuJoCo | 1 | 1 | 0.044 | 2.8 | 4.9 | 1.0 | - | PASS |
|  | PhysX | 1 | 1 | 0.044 | 2.7 | 3.4 | 1.2 | - | PASS |
|  | MTC | 1 | 1 | 0.044 | 2.8 | 5.1 | 1.2 | - | PASS |

v1_feetjoint is the only policy here or above that fails in PhysX on recovery: its pelvis is
still moving at 0.277 m/s at the end of the post-motion hold (0.468 m/s at 4 s; recovery needs
under 0.2), and with `--hold-seconds 8` it terminates at 13.72 s on `ee_body_pos`. In CPU MuJoCo
it holds 8 s. Its PhysX foot slip is 0.056 m/s against 0.015 for v1_feetjoint05. The cause was
not isolated. v1_feet5k's failures are the joint gate, by 0.00002-0.0012 rad.

All 16 final policies pass in PhysX, as they do in MuJoCo and MTC, now that `cut_head_v4` has replaced `cut_head_v3`. With v3 it was 15 of 16. Joint errors are within 0.005 rad and body errors within 0.6 cm of MuJoCo's, with no terminations and full recovery. The one failure was cut_head_v3's key-frame blade placement, 6.1 cm against the 5 cm gate; v4 is at 3.1 cm in PhysX (2.0 MuJoCo, 2.9 MTC), at the cost of body error rising from 3.9 to 4.5 cm against its 5 cm gate. The v3 failure is kept in the table because it shows what the cross-check found. It is not solver noise: PhysX gives 5.8 / 6.1 / 6.4 / 6.4 cm at 4/1, 8/4, 16/4 and 32/8 iterations (6.1 without self-collision), while MuJoCo gives 4.15-4.24 cm across actuator and explicit PD and three reset-noise seeds. cut_head_v3 was already the closest to the line (above), and its margin holds in the MuJoCo engines only. The anchor error is also higher in PhysX for the dynamic motions (up to 2.7 cm above MuJoCo's, on advance_lunge_v4), which points at contact and foot placement. That is a hypothesis and has not been isolated. For cut_head_v3_tip the engines first diverge at the first foot touchdown (rigid vs soft contact; left ankle pitch differs by 0.071 rad after the first policy step on the ground, 0.004 rad lifted). At the key step the sword arm differs by about 0.03 rad and the torso anchor by 1.2 cm, which moves the re-anchored reference tip by about 3 cm. Step-0 observations agree to 1.3e-7, so no ordering, sign or frame error was found. Without a policy, the plants agree closely: holding reference frame 0 under PD, both fall forward on the same path (pelvis height 0.540 vs 0.521 m at 0.6 s).

Not covered: GPU PhysX (not tried, because a training run held the GPU), more than one episode per engine. ovphysx's tensor-binding API, which the plant uses, is deprecated since 0.6 in favour of `PhysX.read`/`write`. It is still present in 0.7.0 and pinned above.

## Rendering on this machine

`MUJOCO_GL=egl` works under WSL2 but resolves to Mesa **llvmpipe** (CPU rendering), about 0.25 s per 640x480 frame with shadows. Both `train play` and `sim2sim --video` turn shadows off and render at 25 fps (every other 50 Hz policy step). The reference is drawn as a translucent green ghost. MP4s are written by ffmpeg (`~/.local/bin/ffmpeg`, H.264), with OpenCV's mp4v as the fallback.

llvmpipe competes for CPU. On a saturated machine (training plus other jobs) a renderer running at `nice -n 19` gets almost no CPU: a 10 s clip stalled for 25 minutes once. Render short clips (`--seconds 2`, `--video-seconds 2`) while training runs, or render without `nice` when the machine is idle. Metrics never depend on rendering. Run `sim2sim` without `--video` for the numbers; it takes about 30 s for 20 s of motion.
