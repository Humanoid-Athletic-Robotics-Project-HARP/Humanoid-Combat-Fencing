# Run log

One row per training run, evaluation or environment change. Newest at the bottom.

- **Stack:** `mjlab`, `isaac` (Isaac Lab / whole_body_tracking), `mtc` (motion_tracking_controller sim-to-sim), `hw` (hardware)
- **Data:** motion CSV and version, e.g. `en_garde_v1`. For LAFAN1, the clip name
- **Commit:** this repo's short SHA at run time
- **Result:** pass / fail plus the key numbers (tracking error, success rate out of N, foot slip). Failures and their likely cause matter as much as passes

## Environments

| Date | Machine | GPU | Driver | CUDA | Notes |
|---|---|---|---|---|---|
| 2026-11-08 | MacBook (Apple Silicon) | none | — | — | mjlab v1.6.0 + GMR via uv; CSV→NPZ and kinematic replay verified on CPU |

## Runs

| Date | Stack | Motion | Data | Commit | W&B | Result | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
