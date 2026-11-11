# ADR 0002: Saber is a fixed mount on the right wrist, not held by a Dex3 hand

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Phase 1.3 has to settle how the G1 holds the saber before any saber motion is trained.

- **The model we train on has no fingers.** mjlab's G1 (`asset_zoo/robots/unitree_g1/xmls/g1.xml`, 29 DoF) ends each arm at `*_wrist_yaw_link`, which carries a static rubber-hand mesh and one collision capsule. The Dex3-1 three-finger hand (7 DoF per hand) is a separate hardware option with a separate model. Training a grasping hand would mean a different robot model, 14 more actuated joints to either learn or lock, and contact-rich grasp physics, none of which BeyondMimic-style tracking is set up for.
- **A grasp is the weakest link under impact.** A cut or a parry puts a torque on the hilt. A position-controlled finger grip can slip or rotate in the hand, and then the blade-tip error is undefined. A bolted mount has one fixed, measurable transform from the wrist to the tip.
- **The wrist actuators are small.** Wrist pitch and yaw are 4010 motors with a 5 Nm effort limit (`g1_constants.py`). Wrist roll is a 5020 with 25 Nm. Every gram on the blade, and every centimetre it sits from the wrist axis, costs torque.
- **G1 is about 0.75 times an adult fencer's height** (1.32 m against roughly 1.75 m), and its arm is shorter than that ratio suggests (about 0.57 m from shoulder to guard).

## Decision

1. **Fixed mount on `right_wrist_yaw_link`.** On the hardware, a printed socket replaces the rubber hand, bolts to the wrist flange, and clamps the saber grip. In simulation the saber is one rigid body called `saber`, welded to that link with no joint.
2. **Mount pose.** The grip centre sits 0.10 m along the forearm axis from the `right_wrist_yaw_link` origin, inside where the rubber hand's fist is. The blade points along the forearm axis, **pitched 30° towards the thumb side** (+z of the hand). The cutting edge faces the little-finger side (−z). This is the fencer's handshake grip, with the blade continuing the forearm and angled up.
   - 30° rather than the human-like 15° because it reached more of the canonical positions in the mount sweep: 10 of 12 against 8 of 12 for the same targets. Details are in [docs/saber.md](../saber.md). Moving the grip closer to the wrist (0.06 m) did not help.
3. **Training weapon:** a foam or padded sport saber.

   | | Nominal | Randomization range | FIE saber limit |
   |---|---|---|---|
   | Blade length | 0.75 m | fixed | ≤ 0.88 m |
   | Overall length | 0.88 m (grip 0.12 + guard 0.01 + blade 0.75) | fixed | ≤ 1.05 m |
   | Mass | 0.20 kg (blade 0.10, guard 0.03, grip + socket 0.07) | 0.12–0.35 kg | ≤ 0.50 kg |

   A 0.75 m blade is 85 % of the FIE maximum. It is still clearly a saber, and it has less inertia about the small wrist motors than a full-length blade. The mass covers bare foam trainers (about 0.10–0.25 kg) plus the printed socket.
4. **Mass properties come from geometry:** the grip is a solid cylinder, the guard a box and the blade a cylinder, combined with the parallel-axis theorem. The result is mass 0.20 kg, COM 0.232 m from the grip centre and principal inertia (5.2e-5, 1.39e-2, 1.39e-2) kg·m². A test checks this against MuJoCo's own inertia-from-geoms.
5. **Collision is off for Goal 1.** The saber's geoms are visual only (`contype = conaffinity = 0`). The reasons:
   - The reference motions are retargeted from video without a weapon, so the blade will sometimes pass through the robot's own legs or torso in the reference. Self-contact would fight the tracking reward. It would also trip the task's `self_collisions` penalty, because that sensor watches the whole pelvis subtree, saber included.
   - A foam blade is compliant, and rigid MuJoCo contact would misrepresent it.
   - Goal 1 has no opponent or target to hit.

   `SaberSpec(collision=True)` adds a `saber_blade_collision` capsule for Goal 2, when hitting a dummy matters.
6. **Frames:** site `blade_tip` at the end of the blade and site `guard` at the base of the blade, both on the `saber` body. The blade-tip error metric uses `blade_tip`.
7. **Saber mass is randomized in training.** The startup event `saber_mass` uses `dr.pseudo_inertia` on the `saber` body only, so mass and inertia scale together. That models a change in density, not a point mass. Play and evaluation configs use the nominal saber.

## Consequences

- One task id carries the weapon: `Fencing-Tracking-Flat-G1-Saber` (plus `-No-State-Estimation`), registered by `fencing.saber.mjlab_task.register()`.
- The saber is the last body of the tree, so **plain-robot NPZs work unchanged**: every tracked body keeps its index (verified to < 1e-6 m in the smoke test). No separate saber NPZ is needed unless a future reward tracks the blade body itself.
- Tierce and quarte with the hand at waist height are **not reachable** with the arm alone in a square-on stance, whatever the mount angle (see docs/saber.md). The parry motions must use a higher hand, a turned torso, or both. Hand-edited wrist keyframes (roadmap 1.4) should aim at the reachable versions.
- With the 30° mount the guard needs about 1 rad at the wrist pitch/yaw joints. That is out of reach for upstream's 0.075 rad/unit wrist action scale, as the first extension policy showed with a 38 cm blade tip error. The saber task therefore uses 0.25 rad/unit for the right wrist pitch/yaw (docs/saber.md, "Wrist action scale"). A future mount sweep should score wrist excursion, not only reachability.
- The fixed mount gives up any finger action: no beats with the fingers, no changing the grip. That is acceptable for Goals 1–2. If Goal 3 needs finer point control, revisit the Dex3 option with a hand model.
- **Hardware to-do:** measure the real wrist flange (bolt pattern, rubber-hand removal) and design the socket. The 0.10 m grip offset and the 30° pitch are the parameters to reproduce. Re-measure the real mass and update `SaberSpec` if it falls outside 0.12–0.35 kg.
- Isaac Sim is not available on this machine (WSL2). A URDF snippet with the same numbers is generated (`src/fencing/saber/assets/saber_link_snippet.urdf`), but it is untested. When it is imported, disable fixed-joint merging so the `blade_tip` frame survives.
