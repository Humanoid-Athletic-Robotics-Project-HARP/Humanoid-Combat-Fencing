# ADR 0003: Saber task adds a blade-tip position reward to the BeyondMimic recipe

- **Status:** accepted. Validated on `extension_v2_tip` and on every `*_v3` bladework run trained so far (below).
- **Date:** 2026-11-09

## Context

The roadmap (Phase 1.5) says to fix the data first and change the training recipe only as a last resort, recorded as an ADR. This is one of those changes.

**What happened.** `extension_v2` was the first saber run with the wider right-wrist action scale ([saber.md, Wrist action scale](../saber.md#wrist-action-scale)). At 3000 iterations it passed every check except one:

- success 1.00 of 1024
- joint error 0.095 rad
- body error 4.1 cm
- slip 0.047 m/s
- push, noise, friction and saber mass all 1.00
- **blade tip 16.8 cm**, against a gate of 10 cm. sim2sim agrees.

The widening helped: extension_v1 had a 38 cm tip error. But it did not get the policy through the gate.

**Diagnosis.** A per-joint rollout in CPU MuJoCo showed the right wrist yaw off by 0.30 rad on average, with no actuator saturating (peak 2.8 of 5 Nm). The en_garde_v1 policy on the same video base has the same right hip yaw and ankle roll offsets (about 0.28 rad). Those come from the base motion and not from the saber, and they do not affect the tip.

The cause is the reward. BeyondMimic's tracking reward has no term for the blade. It tracks the wrist link with an orientation std of about 0.4 rad, so a 0.3 rad wrist error costs about 0.04 of `motion_body_ori`. The 0.75 m blade turns that same error into roughly 20 cm at the tip. The policy has no reason to be precise where precision matters most for fencing.

## Decision

The saber task (`Fencing-Tracking-Flat-G1-Saber` and its no-state-estimation twin) adds one reward term, `blade_tip_pos`. It is defined in `fencing.saber.mjlab_task.blade_tip_position_error_exp`.

- **Formula:** `exp(-|tip_ref - tip_robot|² / std²)`. The tip sits at the fixed saber offset `tip_in_wrist` in the `right_wrist_yaw_link` frame.
  - The reference wrist pose is the motion command's re-anchored `body_pos/quat_relative_w`, the same frame `motion_body_pos` uses.
  - This is exactly the eval metric `blade_tip_error_m`, so the reward and the gate measure the same thing. A test checks the two agree.
- **Weight 1.0** matches the upstream `motion_body_*` terms.
- **std 0.15 m** puts the steep part of the Gaussian around the gate. The term scores 0.89 at 5 cm, 0.64 at 10 cm and 0.29 at 16.8 cm.
- Every other part of the recipe is unchanged: observations, PPO settings, the other rewards and the terminations. The term is also present in the play config, so an evaluation logs it.

## Result

`extension_v2_tip` (3000 it, same motion and seed settings as `extension_v2_pipe`) gave mjlab **PASS** and sim2sim **PASS**:

| | `extension_v2_pipe` (no tip term) | `extension_v2_tip` |
|---|---|---|
| Blade tip | 16.8 cm | **1.8 cm** (sim2sim 1.7) |
| Body position | 4.1 cm | 4.9 cm (gate 5.0) |
| Joint angle | 0.095 rad | 0.080 rad |
| Foot slip | 0.047 m/s | 0.058 m/s |
| Robustness (push, noise, friction, saber mass) | 1.00 | 1.00 |

The tradeoff predicted below shows up as body error: the policy gives up 8 mm on the tracked bodies, which leaves only 1 mm of margin. If later bladework runs fail on body error, lower the tip weight to about 0.5 before changing anything else.

### Later `_tip` runs (2026-11-10, weight and std unchanged)

All runs below are 3000 iterations. "Robustness" is push, noise, friction and saber mass. The v3 rows use the `en_garde_v3` base (front foot fixed) with a 2 s end hold, so the base motion changed too, not only the reward.

| Run | mjlab / sim2sim | Blade tip | Joint (rad) | Body (cm) | Robustness |
|---|---|---|---|---|---|
| `parry_quinte_v2_tip` | PASS / PASS | 2.5 | 0.089 | 4.0 | 1.00 |
| `cut_head_v2_tip` | FAIL / FAIL | — | 0.106 (joint) | — | — |
| `advance_lunge_cut_v2_tip` | FAIL / PASS | 2.5 | 0.099 | 3.2 | recovery 0.82 (end-hold mismatch, see pipeline.md) |
| `advance_lunge_cut_v3_tip` | FAIL / FAIL | 1.9 | 0.110 (joint) | 3.7 | 1.00 |
| `advance_lunge_cut_v4_tip` (rear wrist neutralised, see pipeline.md) | PASS / PASS | 2.1 | 0.098 | 3.2 | 1.00 |
| `extension_v3_tip` | PASS / PASS | 1.4 | 0.075 | 4.2 | 1.00 |
| `cut_head_v3_tip` | PASS / PASS | 2.7 | 0.085 | 4.0 | 1.00 |
| `parry_quinte_v3_tip` | PASS / PASS | 1.7 | 0.072 | 4.5 | 1.00 |
| `cut_flank_v3_tip` | PASS / PASS | 1.9 | 0.068 | 4.2 | 1.00 |
| `cut_chest_v3_tip` | PASS / PASS | 1.7 | 0.078 | 4.0 | 1.00 |
| `thrust_v3_tip` | PASS / PASS | 1.8 | 0.080 | 4.6 | 1.00 |
| `parry_prime_v3_tip` | PASS / PASS | 1.7 | 0.075 | 4.2 | 1.00 |
| `parry_seconde_v3_tip` | PASS / PASS | 2.6 | 0.091 | 4.5 | 1.00 |
| `parry_sixte_v3_tip` | PASS / PASS | 1.2 | 0.073 | 4.2 | 1.00 |
| `parry_tierce_high_v3_tip` | PASS / PASS | 1.2 | 0.090 | 4.0 | 1.00 |

Every `_tip` run tracks the blade inside 3 cm. On average the body-error cost looks smaller than in the first comparison: the v3 runs fall between 4.0 and 4.6 cm. No run has failed on body error, so the weight stays at 1.0.

Two runs fail, both on the mean joint gate:
- `cut_head_v2` because of the old base's leg offset.
- `advance_lunge_cut_v3` because of its wrists. Without the six wrist joints it sits at 0.080, against wrists alone at 0.225. See [evaluation.md](../evaluation.md), "the joint gate and the wrists".

The term lets the policy choose its own redundant wrist combination, so a mean over all joints that includes the wrists now measures something this ADR deliberately stopped asking for.

## Consequences

- **Runs before and after this change train under different recipes.** Runs tagged `_tip` (from 2026-11-09 19:39) have the term. `extension_v1_v1` and `extension_v2_pipe` do not. Each run's `params/env.yaml` records its reward set.
- **The policy can now give up some joint accuracy to hit the blade position.** If joint or body error rises above its gate in the `_tip` runs, look at the weight and std first.
- **The std is a judgement call.** If `extension_v2_tip` still fails on the tip, raise the weight before lowering the std. A narrow Gaussian gives almost no gradient from 17 cm away.
- **`parry_quinte_v2_pipe` was stopped** at about 220 iterations under the old recipe and requeued under the new one.
- **The hardware path is unaffected.** Reward terms don't reach the ONNX policy or `motion_tracking_controller`.
