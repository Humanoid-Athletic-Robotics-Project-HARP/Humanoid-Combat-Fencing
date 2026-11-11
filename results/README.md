# Artifacts

Results copied out of `$HCF_DATA` (`~/hcf-data` on the training PC) so they travel with the repo. Refresh with:

```bash
python3 scripts/collect_results.py            # --dry-run to count first
```

The script only adds or updates files. It never deletes anything here and never touches `$HCF_DATA`. The code still reads and writes `$HCF_DATA`; this folder is a copy, not a working directory.

## What is here

| Folder | Contents |
|---|---|
| `policies/<run>/model_<N>/` | One evaluated checkpoint: `policy.onnx` (what `motion_tracking_controller` loads), `export.json`, `metrics.json` (mjlab eval), `sim2sim.json` (CPU MuJoCo), `mtc.json` (C++ controller sim, where it was run), `scene/scene.xml`, and any renders |
| `policies/<run>/training/` | The run's `params/` (agent and env config), `run.json`, mjlab's `git/*.diff`, the tensorboard event file, the **last** checkpoint (`model_<N>.pt`), and `task.txt` with the task folder name |
| `videos/` | The 3D renders of the final policies, one MP4 per run: a CPU-MuJoCo rollout of the exported ONNX (640x480, 25 fps), robot in white with the saber, reference motion as a translucent green ghost. Each is also at `policies/<run>/model_<N>/sim2sim.mp4` |
| `SUMMARY.md`, `summary.csv` | Generated table of every score: one row per checkpoint and simulator |
| `policies/saber/` | Phase 1.3 saber checks: wrist reach (`reach.json`), the smoke test, and renders of each blade position |
| `pipeline_summaries/` | One JSON per `fencing pipeline train` invocation |
| `motion_npz/` | The reference motions as mjlab reads them (50 Hz NPZ) with their provenance JSON |
| `pose/` | MediaPipe output per video or clip: `mediapipe_raw.npz`, `keypoints_world.npz`, `pose.json` |
| `retarget/` | GMR retarget output (`.pkl`, stats, raw CSVs) |
| `motion_work/` | Cleanup and blade-keyframe intermediates, with contact sheets of the robot |
| `video_screening/` | `manifest.jsonl`, `shortlist.json`, `search_state.json` from `hcf scrape` / `hcf screen` (links and scores, no video) |
| `mtc/models/` | The saber plant given to `motion_tracking_controller` (`g1_saber.xml`, `g1_saber.urdf`) |
| `mtc/recordings/` | Per run: the recorded states (`bag.npz`), `extra.json`, and the launch logs (gzipped) |
| `training_logs/` | Every training, pipeline, pose, retarget and setup log, gzipped |

Run names start with the date and time the run began, so the newest version of a motion is the last one in a listing. Verdicts and the story behind each run are in [`docs/log.md`](../docs/log.md) and [`docs/roadmap.md`](../docs/roadmap.md).

## What is left out, and why

| Left out | Why |
|---|---|
| Raw videos (`videos/raw`, 2.2 GB) and pose overlay PNGs | Other people's footage. The repo keeps links only ([`data/video_log.md`](../data/video_log.md)) |
| LAFAN1 data, the `dance1_subject2` motion NPZ and that run's ONNX | LAFAN1 is CC BY-NC-ND. The NPZ is a derivative, and the ONNX bundles the reference motion. Its metrics, config and weights-only checkpoint are here |
| GVHMR code and checkpoints (5.2 GB), body-model files | Their own licences; fetched by `scripts/setup_linux_gpu.sh` or by hand ([`docs/setup.md`](../docs/setup.md)) |
| Intermediate checkpoints (`model_500.pt` ... , 1.4 GB) | Only the last checkpoint of each run is kept |
| G1 mesh STLs inside each exported `scene/` (about 20 MB per export) | Identical upstream files repeated in every export; `scene.xml` is kept, and `train export` rewrites the meshes |
| Python venvs (29 GB), the ROS rootfs for `motion_tracking_controller` (3.9 GB), cloned third-party source | Rebuilt by the setup scripts ([`docs/setup.md`](../docs/setup.md), [`docs/mtc.md`](../docs/mtc.md)) |
| MTC debug recordings outside `mtc/runs` (`rec_dance1*`, `run1`, `run2`) | One-off probes from bringing the controller up |

No file here is over GitHub's 100 MB limit, and the script skips any that would be.
