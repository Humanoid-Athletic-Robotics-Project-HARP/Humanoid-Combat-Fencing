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

Sim2sim key-frame blade error, re-anchored (global), 2026-11-11, checkpoint 2999. The re-scored sim2sim.json files have every pre-existing value unchanged: extension_v3 1.1 (2.0) cm, parry_prime_v3 1.6 (1.3), parry_seconde_v3 2.5 (3.9), parry_sixte_v3 1.6 (2.4), parry_quinte_v3 2.0 (4.0), parry_tierce_high_v3 1.6 (2.1), cut_head_v3 4.2 (5.0), cut_flank_v3 2.7 (3.4), cut_chest_v3 2.6 (7.7), thrust_v3 3.3 (4.8), advance_lunge_cut_v4 1.8 (14.2). The cuts and the thrust are the hardest key frames, and cut_head's key error (4.2 cm) is well above its whole-motion mean (2.6 cm).

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

Thresholds are provisional until Phase 1.2 numbers exist on known motions.

**Open question: the joint gate and the wrists.** `joint_angle_error_rad` is the mean over all 29 joints. For saber motions the three wrist joints are redundant once the blade tip is placed (the tip has its own gate and reward), and the policy picks its own wrist combination. Sim2sim breakdown on 2026-11-10 (mean over all joints / without the 6 wrist joints / wrists only): advance_lunge_cut_v3 0.110 / 0.080 / 0.225, advance_lunge_cut_v2 0.099 / 0.082 / 0.165, extension_v2 0.081 / 0.079 / 0.089, en_garde_v1 0.091 / 0.083 / 0.119. Near-identical runs land about 0.01 apart, so with the wrists in, a 0.10 gate is a coin flip for bladework. Options: gate the non-wrist joints at 0.10 and the wrists separately, or exclude the sword wrist where the blade-tip gate applies. Not changed yet: this is the user's call.

**Phase 1.2 calibration point (2026-11-10).** The upstream recipe, unchanged, on LAFAN1 `dance1_subject2` (`dance1_subject2_p12v2r`, 10000 it): mean joint error 0.108 in mjlab and in sim2sim, which fails the 0.10 gate. Every other gate passes. Without the wrists it is 0.096, with the wrists alone at 0.163. A known-good motion trained with the reference recipe fails the joint gate, so 0.10 over all 29 joints is too tight as a pass/fail line. A non-wrist gate at 0.10 would pass dance1 and `advance_lunge_cut_v3`. Of the runs where I measured the split, only `cut_head_v2` stays over (0.102; its leg offset is fixed in v3, which passes the existing gate). They are defaults in `thresholds.py`, and every one can be overridden.

**Update 2026-11-11.** The saber cases turned out to be reference defects, not the gate: `advance_lunge_cut` had two different sword-arm poses for the same guard (v5: joint 0.076, was 0.098-0.110), and the rear wrists of the lunge family held unobserved retargeter angles (`advance_lunge_v4`: 0.081, was 0.099; `lunge_v5`: 0.087, was 0.100). All 16 final Goal 1 policies now pass the unchanged 0.10 gate in mjlab, CPU MuJoCo and the C++ controller, the worst at 0.091. The gate is still open for dance1 only (0.108, upstream recipe on upstream data), so it was left as it is.

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
- **Not available**: Isaac Sim. It does not run on WSL2, so the roadmap's "Isaac Sim vs MuJoCo" gap is measured here as mjlab (MuJoCo-Warp) vs CPU MuJoCo instead.

## Rendering on this machine

`MUJOCO_GL=egl` works under WSL2 but resolves to Mesa **llvmpipe** (CPU rendering), about 0.25 s per 640x480 frame with shadows. Both `train play` and `sim2sim --video` turn shadows off and render at 25 fps (every other 50 Hz policy step). The reference is drawn as a translucent green ghost. MP4s are written by ffmpeg (`~/.local/bin/ffmpeg`, H.264), with OpenCV's mp4v as the fallback.

llvmpipe competes for CPU. On a saturated machine (training plus other jobs) a renderer running at `nice -n 19` gets almost no CPU: a 10 s clip stalled for 25 minutes once. Render short clips (`--seconds 2`, `--video-seconds 2`) while training runs, or render without `nice` when the machine is idle. Metrics never depend on rendering. Run `sim2sim` without `--video` for the numbers; it takes about 30 s for 20 s of motion.
