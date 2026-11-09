# ADR 0001: Training stack is mjlab, with Isaac Lab kept as a second simulator

- **Status:** accepted
- **Date:** 2026-11-08

## Context

The roadmap was written around BeyondMimic's `whole_body_tracking` on Isaac Sim 4.5 / Isaac Lab v2.1.0. On 2026-10-16 its authors archived that repository. Its README now says BeyondMimic motion tracking "lives on in [mjlab](https://github.com/mujocolab/mjlab)", an Isaac Lab-style manager API running on MuJoCo-Warp. One of BeyondMimic's authors (Qiayuan Liao) is an mjlab author. mjlab ships the tracking task as `Mjlab-Tracking-Flat-Unitree-G1`.

| | `whole_body_tracking` + Isaac Lab 2.1.0 | mjlab |
|---|---|---|
| Maintained | No (archived, frozen) | Yes (v1.6.0, active) |
| Install | Isaac Sim 4.5 + Isaac Lab + conda | `uv sync` |
| Training | Linux + NVIDIA | Linux + NVIDIA (WSL2 also works) |
| Playback / eval on the Mac | No | Yes (CPU) |
| Matches the paper's reported numbers | Yes, the code the paper used | Same recipe, different physics engine |

## Decision

1. **mjlab is the primary stack.** All Goal 1 motions are trained and evaluated with `Mjlab-Tracking-Flat-Unitree-G1` first.
2. **`whole_body_tracking` + Isaac Lab v2.1.0 stays pinned** as a second simulator and as the reference implementation. It is used for:
   - Phase 1.2: one LAFAN1 motion trained in both stacks, to check our setup against the paper's numbers.
   - The cross-simulator check, and debugging when an mjlab policy behaves oddly.
3. **The CSV is the shared source of truth.** Each stack builds its own NPZ (`<motion>_mjlab`, `<motion>_isaac`), because the two engines number bodies differently.
4. **Sim-to-sim gate** (roadmap rule "Isaac Sim and MuJoCo"): an mjlab-trained policy must also run in `motion_tracking_controller`'s MuJoCo launch. That runs the real deployment C++/ONNX code against CPU MuJoCo. For motions that pass, also train and evaluate in Isaac Lab when practical.

## Consequences

- Faster setup, and the Mac becomes useful for evaluation and data work.
- Two NPZ conversions per motion. The registry naming convention has to be followed strictly.
- The GPU machine should be native Ubuntu 22.04 so the Isaac half still works.
- Open check for Phase 1.2: does mjlab's ONNX export load in `motion_tracking_controller` unchanged? If not, write a small adapter, or deploy Isaac-trained policies only, and record the outcome as a new ADR.
- The roadmap's "Isaac Sim + MuJoCo" wording now means "mjlab (MuJoCo-Warp) + deployment MuJoCo, and Isaac Sim where practical".
