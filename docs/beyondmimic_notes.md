# BeyondMimic notes

Answers to the reading questions in the [week plan](plans/2026-11-09-week.md#beyondmimic-reading-questions). Paper: [arXiv 2508.08241](https://arxiv.org/abs/2508.08241). Code: `third_party/mjlab/src/mjlab/tasks/tracking/` (primary) and `third_party/whole_body_tracking/` (original).

Where to look in the code:

| Question | mjlab file |
|---|---|
| Observations, actions | `tasks/tracking/tracking_env_cfg.py`, `mdp/observations.py` |
| Reward, anchor body | `mdp/rewards.py`, `mdp/commands.py` |
| Adaptive sampling | `mdp/commands.py` |
| Domain randomization | `tracking_env_cfg.py` (events) |
| PPO settings | `tasks/tracking/config/g1/` |

## 1. Policy inputs and outputs

_TODO_

## 2. Tracking reward and the anchor body

_TODO_

## 3. Adaptive sampling

_TODO_

## 4. Domain randomization

_TODO_

## 5. Why one recipe works, and where it fails

_TODO_

## 6. What guided diffusion adds (skim)

_TODO_

## 7. Training time and GPU

_TODO_

## 8. Reference motion format, and does GMR's output match? ✅

**Yes, it matches.** Checked against the pinned commits on 2026-11-08.

- Both stacks' `csv_to_npz.py` read a headerless CSV: `x,y,z, qx,qy,qz,qw, <29 joint angles>`. That is the root quaternion in **xyzw**, Unitree's LAFAN1 convention. They convert to wxyz internally, interpolate to 50 fps, run forward kinematics, and upload an NPZ (joint pos/vel plus per-body pose/velocity) to the W&B registry.
- GMR's `unitree_g1` model has the **identical 29-joint order** to the joint list in both `csv_to_npz.py` scripts. Checked programmatically: mjlab, whole_body_tracking and GMR all agree.
- GMR saves a pickle (`root_pos`, `root_rot` in xyzw, `dof_pos`, `fps`). GMR's own `scripts/batch_gmr_pkl_to_csv.py` writes exactly the CSV above. **No converter needed.**
- Gotcha: the NPZ is engine-specific (MuJoCo and PhysX order bodies differently). Convert each CSV separately for mjlab and for Isaac Lab ([ADR 0001](adr/0001-training-stack.md)).
