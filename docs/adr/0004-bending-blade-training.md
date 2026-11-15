# ADR 0004: A separate saber task trains with a bending blade

- **Status:** proposed. One test run (`cut_head_v4_bending_v1`, below); no committed policy uses this task yet.
- **Date:** 2026-11-15

## Context

The saber policies are trained with a straight, rigid blade. A real saber bends. [saber_flex.md](../saber_flex.md) reruns the committed policies on a spring-hinged blade with the stiffness of a foam practice saber (estimated, not measured yet): 6 of the 12 bladework policies miss the 5 cm key-frame tip gate (parry prime, parry seconde, chest cut, flank cut and both head cuts), by 2-4×. The other 6 pass and none fall. Only a much stiffer blade (about 15 Hz or more towards the flat side) would let them all pass.

The roadmap allows recipe changes only as a last resort, written down as an ADR. This is one: the robot model, the policy inputs, a reward term and a randomization event change for the fast moves to pass on a bending blade.

Related G1 work with a held implement does not simulate bending. [A G1 tennis system](https://arxiv.org/abs/2603.12686) mounts the racket on a 3D-printed wrist connector, as ADR 0002 does, and randomizes the racket's mass (×0.8-1.2) and centre of mass. [RPG](https://arxiv.org/abs/2604.21355) deploys sword swings on a real G1 but does not describe a weapon model.

## Decision

A new task, `Fencing-Tracking-Flat-G1-Saber-Bending` (`fencing.saber.mjlab_flex_task`), trains with the bending blade. The existing saber tasks stay exactly as they are (their snapshot tests pass unchanged), so committed runs and their recipe are untouched. Differences from `Fencing-Tracking-Flat-G1-Saber`:

- **Robot.** The blade is one piece on two spring hinges at the guard (`SaberSpec.flex_*`, [saber_flex.md](../saber_flex.md)). One piece because it stays stable at the training timestep (5 ms) over the whole randomization range below; two pieces blow up above 12.3 Hz. One and two pieces gave the same pass/fail on the eval gates.
- **Policy inputs: the 29 robot joints only.** The robot has no sensor on the blade, so the policy never sees the hinges. Its input size is the saber task's.
- **Critic inputs: plus the blade hinge angles and speeds.** The critic is used only during training and is not exported. The BeyondMimic critic already gets values the robot can't measure (exact body positions and orientations).
- **Motion command.** It compares and resets the 29 robot joints. The blade starts each episode straight and still.
- **Blade-tip reward on the real tip.** `blade_tip_pos` keeps [ADR 0003](0003-blade-tip-reward.md)'s weight, std and reference tip (the straight blade's tip on the reference wrist: where the tip should be does not depend on the blade). The robot's tip is the `blade_tip` site at the end of the bending blade instead of the fixed offset from the wrist.
- **Randomization.** The startup event `saber_mass`:
  - scales the whole saber's mass and inertia (handle and blade) by one factor per env over the saber task's range, 0.12-0.35 kg;
  - picks each direction's wobble at the nominal mass on its own: 4-6 Hz towards the flat side, 10-16 Hz towards the edge (the foam practice saber estimates);
  - adjusts the hinge damping so the damping ratio stays 0.05.

  A lighter saber wobbles faster (frequency ~ 1/√mass), so the actual wobble spans 3.0-7.7 Hz flat and 7.6-20.7 Hz edge. The policy can't see the blade and has no memory of past inputs, so randomization makes it tolerant of the blades in this range, not adaptive to one of them.
- **Export.** The ONNX metadata lists the 29 robot joints, so the policy file reads like a saber-task policy (`fencing train export` and the checkpoints saved during training).

### Not chosen

- **Change the existing saber task.** It would change the recipe of every committed run and its snapshot.
- **Keep the rigid blade and randomize the saber's centre of mass**, as the tennis system does with its racket. That is cheaper, but it doesn't reproduce the tip lagging behind the wrist, which is what fails the gate.
- **Randomize each blade piece's mass separately** (`dr.pseudo_inertia` on several bodies). No real saber has a randomly uneven blade.
- **Give the policy a short history of its inputs**, so it could feel the wobble through the wrist and adapt. Possible later; it changes the policy inputs, so it would need its own ADR.

## Result

`cut_head_v4_bending_v1` (2026-11-15): `cut_head_v4`'s motion and recipe (3000 iterations, 4096 envs), trained on an RTX 4060 laptop (1 h 50 min). Compared with the committed `cut_head_v4` by `scripts/hcf saber flex measured` on the nominal 0.2 kg saber, the blade in 3 pieces at a 0.2 ms timestep (finer than the training blade), and on a rigid saber of the same mass. Key tip = real blade tip at the key frames (gate 5 cm):

| Blade (flat / edge wobble) | Committed `cut_head_v4` | `cut_head_v4_bending_v1` |
|---|---|---|
| Rigid | **3.2 cm** PASS | 7.3 cm FAIL |
| Floppy (4 / 10 Hz) | 22.4 cm FAIL | 9.2 cm FAIL |
| Nominal (5 / 12.5 Hz) | 15.1 cm FAIL | **3.7 cm** PASS |
| Stiff (6.3 / 15.7 Hz) | 9.6 cm FAIL | **3.0 cm** PASS |

The other gates barely move: joint 0.090 rad (committed 0.089), body 4.7 cm (4.5), whole-motion real tip 1.6-2.1 cm on the bending blades (committed 2.9-5.1). Neither falls. The new policy also bends the blade less (largest tip deflection 12 cm on the nominal blade, against 16 cm). `scripts/hcf sim2sim` on its export: PASS (joint 0.091, body 4.7 cm).

What this shows, for one motion and one training run:

- **Training with the blade fixes the key-frame miss in the middle and stiff end of the training range**, with the reference motion still planned for a straight blade. For this motion the tip reward could pull the wrist far enough despite the body-tracking terms.
- **It does not reach the floppy end** (4 / 10 Hz): better than the committed policy but still twice the gate.
- **It loses the rigid blade** (7.3 cm): the policy leads and brakes for a bend that does not come. It is tuned to bending blades, so the real saber's stiffness has to be measured before policies are retrained for it, and a range that should also cover stiffer blades has to include them.
- Not shown: other motions, other training runs (luck), the C++ controller, the real saber.

## Consequences

- **Bending-blade runs have a different recipe.** Their `params/env.yaml` records the task; the saber task's runs are unaffected.
- **`fencing train evaluate` measures the tip from the fixed wrist offset.** For bending-blade policies it reports the straight blade's tip, so score them with `scripts/hcf saber flex` (the real tip) until evaluate reads the `blade_tip` site.
- **The exported scene contains the blade hinges** (the nominal blade). `scripts/hcf sim2sim` runs it and so simulates the bending blade, but scores the tip at the fixed wrist offset like evaluate. `motion_tracking_controller` has not been run on it.
- **The ranges are estimates.** Once a practice saber is measured, narrow the wobble and mass ranges to the measured values plus their uncertainty, and retrain.
- **The keyframer still plans the arm for a straight blade.** The policy is rewarded on the tip, so it can move its wrist differently from the reference, but the joint and body terms pull it towards the straight-blade arm motion. Planning the arm with the blade's dynamics is open.
