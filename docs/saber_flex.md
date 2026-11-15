# Bending blade

The saber in [saber.md](saber.md) is one rigid body. A real foam practice saber bends: stop a swing and the tip keeps going, then wobbles. This page covers an optional bending blade, a command that reruns the existing saber policies with it, and what that showed.

Nothing changes unless you ask for it. The default `SaberSpec` builds exactly the committed rigid saber, and training refuses a bending one.

## How to turn it on

**From the command line**, as a MuJoCo model file you can open or simulate:

```bash
scripts/hcf saber build-mjcf --flex-segments 1 --flex-f1 5 --flex-f1-edge 12.5
# -> ~/hcf-data/eval/flex/g1_29dof_saber_bending.xml (the committed rigid asset is not touched)
```

**In Python**, anywhere the code builds the saber:

```python
from fencing.saber.spec import SaberSpec, g1_saber_model

saber = SaberSpec(flex_segments=1, flex_f1_flatwise_hz=5.0, flex_f1_edgewise_hz=12.5)
model = g1_saber_model(saber)  # the G1 with a bending blade; SaberSpec() is the rigid one
```

**To test the trained saber policies with it**: `scripts/hcf saber flex …` below.

**Training** with it is not possible yet: `Fencing-Tracking-Flat-G1-Saber` refuses a bending saber (see the last sections).

Numbers: `flex_segments` = blade pieces (0 = rigid); `--flex-f1` / `flex_f1_flatwise_hz` = how many times per second the blade wobbles towards its flat side; `--flex-f1-edge` / `flex_f1_edgewise_hz` = towards its edge (0 = same). Until a practice saber is measured, 5 and 12.5 Hz are estimates.

## Commands

```bash
scripts/hcf saber flex validate --policy cut_head_v3              # rigid saber: must equal the committed sim2sim.json
scripts/hcf saber flex sweep --policy all --segments 3 --f1 3 5 7 10 15
scripts/hcf saber flex measured --policy all --frames 180 --fps 240 --frames-edge 96 --blade-mass-g 120
scripts/hcf saber flex report $HCF_DATA/eval/flex/<tag>.json      # markdown table
scripts/hcf saber flex <command> --help
```

- Results go to `$HCF_DATA/eval/flex/<tag>.json`.
- `--policy all` runs 13 committed exports: the saber policies of [hardware_tests.md](hardware_tests.md), `cut_head_v3`, and the entry policy `stand_to_en_garde_v1_feetjoint05`.
- A full sweep takes about 3 minutes on a laptop CPU.
- It needs the G1 meshes from `scripts/setup_third_party.sh`. Exported scenes use `meshdir="assets"`, which is not committed, so the command points them at mjlab's copy of the same meshes.

## Model (`src/fencing/saber/flex.py`)

- **Handle.** Grip and guard stay one body, `saber`, welded to `right_wrist_yaw_link` where `attach_saber` puts them.
- **Blade.** It is split into N equal pieces, `saber_blade_0` … `saber_blade_{N-1}`.
  - Each piece hangs from the previous one by two spring hinges with no actuator: `saber_flex_{i}_y` and `saber_flex_{i}_z`.
  - There is no twist.
  - The `blade_tip` site sits on the last piece.
- **Directions.** In the saber frame ([saber.md](saber.md)), +x runs along the blade and −z is the cutting edge.
  - The `_y` hinges bend the blade towards an edge ("edgewise"), the stiff direction of a flat blade.
  - The `_z` hinges bend it towards a flat side ("flatwise"), the floppy direction.
- **Stiffness.** Each direction is set by one number: the first wobble frequency of the blade clamped at the guard, in Hz. The hinge stiffness that gives exactly that frequency is solved from the chain's mass matrix, as an eigenvalue problem.
- **Damping.** It is stiffness-proportional and set by a damping ratio. The default 0.05 makes each swing about 27 % smaller than the last.
- **Checks** (`tests/test_saber_flex.py`):
  - the first frequency is exact;
  - the gravity sag is within 2 % of beam theory;
  - the second mode is within 5 % of beam theory (6.5× the first at 3 pieces, against 6.27×);
  - total mass, balance point and the straight-blade tip equal the rigid saber's.
- **Not modelled:** twist, taper (the blade is the uniform 24 mm rod of `SaberSpec`), air drag on the foam, and contact.

Hinge values for the nominal 0.20 kg saber with damping 0.05, per hinge:

| Pieces | First wobble | Stiffness (N·m/rad) | Damping (N·m·s/rad) | Tip sag, blade level |
|---|---|---|---|---|
| 3 | 5 Hz | 24.0 | 0.077 | 1.5 cm |
| 3 | 12.5 Hz | 150.3 | 0.191 | 0.2 cm |
| 2 | 5 Hz | 20.4 | 0.065 | 1.5 cm |
| 2 | 12.5 Hz | 127.3 | 0.162 | 0.2 cm |

## The switch in detail

| `SaberSpec` field | Default | Meaning |
|---|---|---|
| `flex_segments` | 0 | Number of blade pieces. 0 is the rigid blade. |
| `flex_f1_flatwise_hz` | 0 | First wobble towards the flat side. Required when pieces > 0. |
| `flex_f1_edgewise_hz` | 0 | First wobble towards the edge. 0 means the same as flatwise. |
| `flex_damping_ratio` | 0.05 | Damping ratio of each direction's first mode. |

The four fields are `repr=False`, so `repr(SaberSpec)`, which the task snapshots in `tests/data/` store, does not change.

With `flex_segments = 0` the built model is identical to the committed one: same counts and the same physical-parameter hash as `tests/test_saber_mjlab.py` uses, and the same `to_xml()`.

| Code | With `flex_segments > 0` |
|---|---|
| `attach_saber`, `g1_saber_spec`, `g1_saber_model`, `write_mjcf` | Build it. They refuse a blade that would blow up at the model's timestep, and refuse `collision=True`. |
| Training (`mjlab_task.saber_g1_robot_cfg`) | Refuses. The policy inputs would include the blade hinges (the real robot cannot sense them), the tracking command expects the 29 robot joints, and the blade-tip reward assumes a straight blade. |
| Keyframer, reach analysis, blade path (`keyframes`, `reach`, `blade`) | Refuse. They plan with the straight blade. |
| URDF snippet | Refuses (rigid saber only). |
| `fencing saber build-mjcf` | `--flex-segments N --flex-f1 HZ [--flex-f1-edge HZ] [--flex-damping R]` writes a bending model to `$HCF_DATA/eval/flex/` (no URDF snippet). |
| `fencing saber reach / render` | Rigid only. |

### Stability

MuJoCo integrates joint springs explicitly. A stiff spring on a light piece therefore blows up at a large timestep, and MuJoCo then resets the simulation with only a warning.

Measured at a 5 ms timestep with `cut_head_v4`, `parry_prime_v3` and `cut_chest_v3` (2026-11-14):

| Pieces | Ran at | Blew up at |
|---|---|---|
| 2 | 12.5 Hz | 12.9 Hz |
| 3 | 8 Hz | 9 Hz |
| 5 | 6 Hz | 7 Hz |

`flex.check_stable` simulates the clamped blade, watches its energy, and adds a 5 % margin. The stiffer direction sets the limit. It allows:

| Timestep | 1 piece | 2 pieces | 3 pieces | 5 pieces |
|---|---|---|---|---|
| 5 ms | ≤ 30.4 Hz | ≤ 12.3 Hz | ≤ 7.6 Hz | ≤ 6.3 Hz |
| 2.5 ms | | ≤ 24.6 Hz | ≤ 15.3 Hz | |

## What `fencing saber flex` measures

It reruns committed saber exports (`results/policies/<run>/model_2999`, unchanged ONNX) in CPU MuJoCo with the blade swapped in.

It copies the default path of `fencing.sim2sim.rollout.rollout` (scene PD, no start noise, no video) instead of changing sim2sim. A test runs both on the rigid saber and requires identical metrics, so a change to `rollout.py` that the copy misses fails it. Folding this into `hcf sim2sim` as `--flex-*` options would remove the copy, but that is a change to `rollout.py`, left to its owner.

It adds three measurements:

- **The real tip.** This is the `blade_tip` site on the last piece. It is scored like `blade_tip_error_m` and `blade_tip_error_key_m` (same re-anchored reference tip, same ±2-step key window) and checked against the same gates in `train/thresholds.py`: 10 cm overall, 5 cm at key frames. The repo metric puts the tip at a fixed offset from the wrist, so it cannot see bending.
- **Right wrist pitch and yaw torque**, against the 5 N·m limit.
- **The largest bend.**

Details that keep the comparison fair:

- The policy's inputs are built from the ONNX contract's 29 joints by name, so it never sees the hinges.
- The physics timestep comes from the blade's highest mode (0.2–0.5 ms here); the policy step stays 20 ms.
- The rigid baseline is re-run at the same timestep.
- The blade settles into its gravity sag before the motion starts.
- `validate` (2026-11-14): on the rigid saber, all 13 policies match their committed `sim2sim.json` within 1e-16.

## Results (2026-11-14)

Key-moment real-tip error in cm (gate 5 cm, **bold** = fail). The blade has 3 pieces, the same stiffness in both directions, and damping 0.05. "Rigid" is the committed rigid saber at 5 ms.

| Policy | Rigid | 3 Hz | 5 Hz | 7 Hz | 10 Hz | 15 Hz |
|---|---|---|---|---|---|---|
| extension_v3 | 1.1 | **7.2** | 2.6 | 1.7 | 1.2 | 1.0 |
| parry_sixte_v3 | 1.6 | **8.6** | 3.3 | 2.1 | 1.8 | 1.7 |
| parry_tierce_high_v3 | 1.6 | **25.9** | **5.5** | 2.7 | 1.9 | 1.7 |
| parry_quinte_v3 | 2.0 | **28.6** | **5.2** | 1.7 | 1.3 | 1.7 |
| thrust_v3 | 3.3 | **15.9** | **5.1** | 3.6 | 2.9 | 2.6 |
| advance_lunge_cut_v5 | 2.1 | **19.5** | **7.6** | 4.3 | 3.1 | 2.5 |
| cut_flank_v3 | 2.7 | **39.7** | **14.1** | **7.8** | 4.7 | 3.6 |
| cut_chest_v3 | 2.6 | **40.4** | **16.9** | **8.9** | **5.2** | 3.8 |
| parry_seconde_v3 | 2.5 | **43.9** | **18.4** | **10.0** | **6.4** | 4.9 |
| parry_prime_v3 | 1.6 | **52.6** | **21.6** | **11.0** | **6.0** | 3.9 |
| cut_head_v4 | 2.0 | **39.7** | **21.5** | **10.7** | **5.8** | 3.9 |
| cut_head_v3 | 4.2 | **41.4** | **20.4** | **12.1** | **7.9** | **6.1** |
| fails | 0 / 12 | 12 / 12 | 10 / 12 | 6 / 12 | 5 / 12 | 1 / 12 |

Other results from the same runs:

- No policy fell.
- Joint and body errors stayed within 0.002 rad and 0.12 cm of the rigid saber at the same timestep.
- A damping ratio from 0.02 to 0.15 barely changed the results.
- The wrist already sits at its 5 N·m limit with the rigid saber, and the bending blade changed the time spent at the limit by at most 1.3 percentage points.
- The entry policy passes at every stiffness; it has no key frame.

**Flat blade, both directions.** The moves bend the blade both ways. Flatwise bending dominates at the key frame for prime, seconde and the chest cut.

The table below gives the flatwise / edgewise frequency of each case (3 pieces) and the policies that fail the 5 cm key gate:

| Flatwise / edgewise | Fails | Failing policies |
|---|---|---|
| 4 / 10 Hz | 8 / 12 | the six below, plus parry_sixte_v3 (6.1 cm) and parry_quinte_v3 (6.0 cm) |
| 5 / 12.5 Hz | 6 / 12 | parry_prime_v3, parry_seconde_v3, cut_chest_v3, cut_flank_v3, cut_head_v3, cut_head_v4 (9.0–15.1 cm) |
| 6.3 / 15.7 Hz | 6 / 12 | the same six (5.9–9.9 cm) |
| Blade rolled 90° in the hand (12.5 / 5 Hz) | 8 / 12 | adds parry_tierce_high_v3 and advance_lunge_cut_v5 |

So keep the mount orientation of [saber.md](saber.md).

**Number of pieces.** In the 5 / 12.5 Hz case:

- 2 pieces at a fine timestep match 3 pieces within 0.4 cm.
- At 5 ms, 1 and 2 pieces give the same pass/fail as 3 for every policy, and under-read the fast moves by up to 2.6 cm.

**Reading.** The rigid saber stands in for the real one only if the practice saber's floppy direction wobbles at about 15 Hz or more. The estimates, which are not measurements, are all lower:

- a steel sabre: 5–9 Hz, from the bend test in the [FIE material rules](https://static.fie.org/uploads/40/204157-book%20material%20August%202026%20ang.pdf) (sabre: 200 g at the tip, 4–7 cm of bend) on a tapered blade, with the taper checked against steel foil blades measured at 6.0–6.8 Hz ([Redmond & Redmond 2026](https://emerginginvestigators.org/articles/26-007));
- a foam sabre with a 15 × 6 mm fibreglass core ([FakeSteel Polish sabre](https://www.fakesteel.cz/products/polish-sabre), 500 g): 4–6 Hz flatwise and 10–16 Hz edgewise, assuming a fibreglass stiffness of 25–40 GPa.

**Caveats.**

- These policies were trained with the rigid saber.
- The saber's size and mass are `SaberSpec`'s nominal values.
- The model has not yet been compared with a real blade.

## Measuring a practice saber

About 15 minutes with a phone that records slow motion, a clamp, a kitchen scale and a tape measure.

1. **Weigh it.** Weigh the whole saber, and the blade alone if it comes apart. Measure the blade length from the guard to the tip.
2. **Clamp it.** Clamp the grip and guard to a table edge with the blade horizontal and a flat side facing down. Put bright tape on the tip.
3. **Film the flatwise wobble.** Film side-on in slow motion. Pull the tip down about 5 cm and let go; do this 3 times.
   - Count the frames from one lowest point to the 5th lowest point after it, and average the three counts.
   - Use the capture rate (e.g. 240 fps), not the playback rate.
   - Wobble = 5 × fps / frames.
4. **Film the edgewise wobble.** Turn the saber 90°, so an edge faces down, and repeat (`--frames-edge`).
5. **Damping (optional).** Note the tip's height at the first peak and at the peak 3 cycles later (`--decay A_FIRST A_LATER 3`).
6. **Run it.** Run `scripts/hcf saber flex measured …` with those numbers. It prints both frequencies and the damping, then a table of every policy: the bending blade against a rigid saber of the same mass and length.

`--bend LOAD_G DEFLECTION_CM FREE_LENGTH_CM` adds a bend-test cross-check. It assumes a uniform blade, so it reads low for a tapered one. Trust the video.

## Training with the bending blade (not done)

Training refuses a bending saber until these exist:

- **Observations.** Restrict the joint observations to the 29 motor joints. mjlab's defaults include every joint, so the hinges would leak into the policy and the ONNX contract.
- **Tracking command.** It must handle the extra joints: the motion command assumes the 29 joints of the motion file.
- **Blade-tip reward.** It should score the real tip ([ADR 0003](adr/0003-blade-tip-reward.md) defines it on the rigid offset).
- **Mass randomization.** It must cover the blade pieces: `dr.pseudo_inertia` acts on the `saber` body only, which with the switch on is just the grip and guard.
- **Pieces and timestep.** Choose them from the measured stiffness (see Stability).
- **A new ADR** ([training/README.md](../training/README.md)).

Also open: the keyframer plans the arm for a straight blade, so a policy that tracks those arm angles is pushed to move its wrist as if the blade were rigid. One option is to plan the arm with the blade's dynamics.

## Files

| File | Contents |
|---|---|
| `src/fencing/saber/flex.py` | Model, stability check, scene editing |
| `src/fencing/saber/flex_eval.py` | `fencing saber flex` |
| `src/fencing/saber/spec.py` | The `flex_*` fields and `attach_saber` |
| `src/fencing/saber/mjlab_task.py` | The training refusal |
| `src/fencing/saber/cli.py` | `fencing saber flex`, and the `--flex-*` options of `build-mjcf` |
| `tests/test_saber_flex.py` | Tests |
