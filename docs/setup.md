# Setup

Everything needed to work through Goal 1. The stack choice is explained in [ADR 0001](adr/0001-training-stack.md). In short, **mjlab** is the main training stack and the original **Isaac Lab** BeyondMimic code is pinned next to it as a second simulator and as the reference implementation.

## Which machine does what

| Task | Mac (Apple Silicon) | Linux GPU machine |
|---|:-:|:-:|
| Collect and tag clips (`data/video_log.md`) | ✅ | ✅ |
| GVHMR (video → SMPL-X) | ⚠️ Colab / HF demo only | ✅ |
| GMR (SMPL-X → G1) and kinematic replay | ✅ | ✅ |
| Motion cleanup + `scripts/replay_csv.sh` | ✅ | ✅ |
| CSV → NPZ (mjlab) | ✅ (CPU) | ✅ |
| **Training** (mjlab or Isaac Lab) | ❌ | ✅ |
| Policy playback / evaluation (mjlab) | ✅ (CPU) | ✅ |
| Isaac Sim (Isaac Lab 2.1.0 stack) | ❌ | ✅ |
| Sim-to-sim + hardware (`motion_tracking_controller`, ROS 2 Jazzy) | ❌ | ✅ (Docker or Ubuntu 24.04) |

**GPU machine: native Ubuntu 22.04 with an NVIDIA RTX GPU.** This is the most compatible choice:

- Isaac Sim 4.5 needs GLIBC ≥ 2.34 (pip install) and does not support WSL2 or macOS. Ubuntu 22.04 is the version Isaac Lab 2.1 was tested on. *WSL2 checked 2026-11-11 on the RTX 5080 PC:* headless Isaac Sim still starts Kit through Vulkan, and WSL2 exposes no NVIDIA Vulkan driver to Linux (`/usr/lib/wsl/lib` has CUDA, NVML and OptiX only; Mesa falls back to llvmpipe). Upstream closed this as not planned ([IsaacLab #3792](https://github.com/isaac-sim/IsaacLab/issues/3792)); Docker inside WSL2 fails the same way. The untried route on that PC is **native Windows 11**, which Isaac Sim 4.5 and Isaac Lab 2.1 support (`isaaclab.bat --install`, Python 3.10, long paths enabled, cu128 nightly torch, a clone path without spaces). It was not installed: it needs roughly 20–50 GB on a system drive with 70 GB free, an EULA acceptance, and `whole_body_tracking` is untested on Windows.
- mjlab only needs CUDA 12.4+. It also runs on WSL2, but the Isaac half of the stack would not.
- `motion_tracking_controller` needs ROS 2 **Jazzy**, which targets Ubuntu **24.04**. On a 22.04 machine, run it in a `ros:jazzy` Docker container. *Unverified:* whether Isaac Sim 4.5 runs on 24.04. Check NVIDIA's 4.5 system requirements before choosing 24.04 for the whole machine.
- RTX 50-series (Blackwell) GPUs: Isaac Lab 2.1 needs a PyTorch nightly instead of 2.5.1:
  `pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`.
- Aim for 24 GB+ of VRAM to train with the default 4096 environments. With less, lower `--env.scene.num-envs`.

## Upstream code

All pinned as git submodules in `third_party/` (`git submodule status` shows the exact commits):

| Submodule | Pin | Role |
|---|---|---|
| `mjlab` | v1.6.0 | **Primary.** BeyondMimic tracking task `Mjlab-Tracking-Flat-Unitree-G1` on MuJoCo-Warp |
| `whole_body_tracking` | final commit (archived 2026-10-16) | Original BeyondMimic on Isaac Lab 2.1.0, task `Tracking-Flat-G1-v0` |
| `IsaacLab` | v2.1.0 | Needed by `whole_body_tracking`. Not fetched by default (`update = none`) |
| `motion_tracking_controller` | main @ 742924f | C++/ONNX inference: MuJoCo sim-to-sim and real G1 deployment |
| `GMR` | master @ 25f56db | Retargets SMPL-X (from GVHMR) onto the G1 |
| `GVHMR` | main @ ee960bb | Video → world-frame SMPL-X human motion |

```bash
scripts/setup_third_party.sh         # Mac / any machine
scripts/setup_third_party.sh --gpu   # also Isaac Lab (Linux GPU machine)
```

## Mac setup

```bash
scripts/setup_mac.sh
```

This installs `uv`, fetches the submodules, runs `uv sync` in `third_party/mjlab` and `third_party/GMR`, downloads the LAFAN1 G1 CSVs to `data/external/lafan1/` (gitignored, CC BY-NC-ND), and runs smoke tests.

**macOS notes**

- GMR's interactive viewer uses `mujoco.viewer.launch_passive`, which on macOS must run under `mjpython`. Use `uv run mjpython scripts/vis_robot_motion.py ...` instead of `uv run python ...`.
- Pass `--device cpu` to mjlab scripts. `scripts/replay_csv.sh` picks the device for you.

## Linux GPU machine setup

```bash
git clone --recurse-submodules https://github.com/Humanoid-Athletic-Robotics-Project-HARP/Humanoid-Combat-Fencing.git
cd Humanoid-Combat-Fencing
scripts/setup_linux_gpu.sh check      # GPU / driver / CUDA / GLIBC; paste the output line into docs/log.md
scripts/setup_linux_gpu.sh            # all stages: mjlab isaaclab wbt gvhmr gmr
```

Stages can be run on their own (`scripts/setup_linux_gpu.sh mjlab gmr`). The script follows each upstream project's install docs at the pinned commit but **has not yet been run end to end**. Expect to fix things the first time and commit the fixes.

Smoke tests once installed:

```bash
# mjlab: dummy agent on the tracking task with a LAFAN1 motion
cd third_party/mjlab && uv run play Mjlab-Tracking-Flat-Unitree-G1 --agent zero --motion-file /path/to/motion.npz
# Isaac Lab: its own sample task (Phase 1.1 "done when")
conda activate env_isaaclab && cd third_party/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless --max_iterations 50
```

## Manual steps (licences and logins, cannot be scripted)

| Step | What | Where it goes |
|---|---|---|
| **W&B** | Create a W&B **organization** (team). `wandb login`. Under Registry, create a collection named **Motions** with artifact type "All Types". Set `export WANDB_ENTITY=<org>` (the org, not your username) | env var on every machine |
| **SMPL-X** | Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and download the models | GMR: `third_party/GMR/assets/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.pkl`. GVHMR: `third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_{GENDER}.npz` |
| **SMPL** | Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) | `third_party/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_{GENDER}.pkl` |
| **GVHMR checkpoints** | `gvhmr`, `hmr2`, `vitpose`, `yolo` (and optionally `dpvo`) from the [GVHMR Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | `third_party/GVHMR/inputs/checkpoints/` (layout in `third_party/GVHMR/docs/INSTALL.md`) |
| **Isaac Sim EULA** | Accept on first launch | — |

GMR's SMPL-X loader: if you use the `.pkl` models, GMR's README says to change `ext` in `smplx/body_models.py` from `npz` to `pkl`.

SMPL, SMPL-X, GVHMR and LAFAN1 are all **non-commercial research only**. None of them are committed to this repo.

## Commands by phase

Paths are relative to the submodule named in each block.

### 1.2 Prove the pipeline on a known motion

Use one LAFAN1 motion, for example `dance1_subject2` (BeyondMimic's own example). Each stack builds its own NPZ (see the warning below).

```bash
# mjlab (third_party/mjlab)
uv run -m mjlab.scripts.csv_to_npz --input-file ../../data/external/lafan1/g1/dance1_subject2.csv \
  --output-name dance1_subject2_mjlab --input-fps 30 --output-fps 50
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name $WANDB_ENTITY-org/wandb-registry-motions/dance1_subject2_mjlab \
  --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path $WANDB_ENTITY/mjlab/<run-id>   # works on the Mac too

# Isaac Lab (third_party/whole_body_tracking, conda env env_isaaclab)
python scripts/csv_to_npz.py --input_file ../../data/external/lafan1/g1/dance1_subject2.csv --input_fps 30 \
  --output_name dance1_subject2_isaac --headless
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name $WANDB_ENTITY-org/wandb-registry-motions/dance1_subject2_isaac \
  --headless --logger wandb --log_project_name fencing --run_name dance1_isaac
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path=<run-path>
```

> [!WARNING]
> **The NPZ files from the two stacks are not interchangeable.** MuJoCo and PhysX number bodies in different orders, so a mismatched NPZ maps tracking targets to the wrong bodies and training will not converge. Always convert each CSV twice and suffix the registry name: `<motion>_mjlab`, `<motion>_isaac`. The **CSV** is shared and is the source of truth.

Check the registry name format against what the W&B UI shows for your org. The `-org/wandb-registry-motions/` form comes from the BeyondMimic README.

### 1.4 Video → G1 motion CSV

The `fencing` CLI wraps the whole chain (WSL/Linux, venvs under `$HCF_DATA/venvs`):

```bash
scripts/hcf screen <yt_id>              # people tracking, fencer track, usable ranges (pose venv, GPU)
scripts/hcf pose <yt_id>                # usable ranges -> clips <yt_id>_<start_ms>-<end_ms> (pose venv, CPU)
scripts/hcf retarget <yt_id>            # clips -> retarget/csv/<clip_id>.csv + .json (gmr venv, CPU)
scripts/hcf motion check --raw ~/hcf-data/retarget/csv/<clip_id>.csv
scripts/hcf motion render ~/hcf-data/retarget/csv/<clip_id>.csv --png /tmp/<clip_id>.png
scripts/hcf motion batch                # dry run of the cleanup over retarget/csv (add --publish to publish)
```

`fencing pose` uses GVHMR only when `$HCF_DATA/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` and `.../smpl/SMPL_NEUTRAL.pkl` exist (licence-gated, see the table above; that branch is untested). Otherwise it uses the licence-free MediaPipe fallback and says so once. Useful flags: `--max-seconds` (clip length cap, default 20), `--height` (fencer height, default 1.75 m), `--hfov` / `--pitch-deg` (camera), `--reprocess` (redo the post-processing from cached MediaPipe output), `--force`. Check `pose/<clip_id>/overlay.png` before retargeting.

The upstream commands the wrapper follows, for running a stage by hand:

```bash
# 1. GVHMR (third_party/GVHMR, conda env gvhmr). -s = static camera, skips visual odometry
python tools/demo/demo.py --video=/path/to/clip.mp4 -s
#    → outputs/demo/<clip>/hmr4d_results.pt  (+ overlay video for the visual check)
#    No GPU? Use the GVHMR Colab or HuggingFace Space demo linked in its README and download hmr4d_results.pt.

# 2. GMR (third_party/GMR): SMPL-X → G1 (pkl: root_pos, root_rot xyzw, dof_pos, fps)
uv run scripts/gvhmr_to_robot.py --gvhmr_pred_file /path/to/hmr4d_results.pt --robot unitree_g1 \
  --save_path /path/to/out/<clip>.pkl --record_video

# 3. pkl → BeyondMimic CSV (writes <folder>/csv/<clip>.csv, downsampled to 30 fps)
uv run scripts/batch_gmr_pkl_to_csv.py --folder /path/to/out

# 4. Cleanup (Phase 1.4e), then kinematic replay to check it (repo root, Mac or Linux)
scripts/replay_csv.sh motions/en_garde_v1.csv
```

**Format check (week-plan question 8): the formats match.** GMR's `unitree_g1` model (`g1_mocap_29dof.xml`) has the same 29-joint order as the CSV order that both `csv_to_npz.py` scripts expect. `batch_gmr_pkl_to_csv.py` writes `x,y,z,qx,qy,qz,qw,<29 joints>`, which is the Unitree/BeyondMimic convention. No custom converter is needed. This was checked against the pinned commits on 2026-11-08.

### 1.5 Train and evaluate

Same commands as 1.2, with `motions/<motion>_vN.csv` as input. Never change the recipe without an ADR.

On the WSL2 GPU PC, the repo CLI chains every stage. Data lives in `$HCF_DATA` (default `~/hcf-data`). There is no W&B (`WANDB_MODE=disabled`).

```bash
scripts/hcf pipeline status                      # per Goal 1 motion: published CSV, runs, latest verdicts
scripts/hcf pipeline --dry-run train --goal1      # print the plan without running it
scripts/hcf pipeline train motions/en_garde_v1.csv --iterations 3000
#   one motion after another: train (foreground) -> export -> eval -> sim2sim,
#   then a row in docs/log.md and a summary in $HCF_DATA/eval/pipeline/<ts>.json.
#   --task auto picks Fencing-Tracking-Flat-G1-Saber for bladework motions.
scripts/hcf pipeline data                        # (scrape) -> screen -> pose -> retarget -> motion batch
scripts/hcf train eval <run> --checkpoint N      # single stages; see docs/evaluation.md
scripts/hcf sim2sim <run> --checkpoint N [--video]
```

Training is CPU-latency-bound here (see `docs/log.md`, Environments). Two runs fit on the 16 GB card at once. Other heavy CPU jobs slow both runs a lot.

### Sim-to-sim and hardware (1.5, 1.6)

`third_party/motion_tracking_controller`, ROS 2 Jazzy workspace (see its README):

```bash
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=~/policies/en_garde.onnx   # sim-to-sim
# real.launch.py: hardware only, run by people only, under safety/checklist.md
```

Both stacks export an ONNX file with metadata (joint order, gains) and a `forward(obs, time_step)` that returns the reference. **Unverified:** that mjlab's export (`mjlab/tasks/tracking/rl/runner.py`) loads in `motion_tracking_controller` unchanged. Check this in Phase 1.2 with the LAFAN1 policy before relying on it.
