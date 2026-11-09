# Training

Run notes only. **No checkpoints, NPZ or ONNX files in git.** Those live in W&B.

- Every run gets a row in [`docs/log.md`](../docs/log.md).
- Longer notes for a run (what was tried, plots, why it failed) go in `training/<date>_<motion>_<stack>.md`.
- The recipe is the upstream default (`Mjlab-Tracking-Flat-Unitree-G1` / `Tracking-Flat-G1-v0`). Changing it takes an ADR in `docs/adr/`.

Commands: [`docs/setup.md`](../docs/setup.md#commands-by-phase).
