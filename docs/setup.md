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

### GVHMR venv

`scripts/setup_linux_gpu.sh gvhmr` builds a uv venv at `$HCF_DATA/venvs/gvhmr` (default `~/hcf-data`), not a conda env. It needs no sudo. What it does and why:

- **torch 2.7.1+cu128**, because older wheels have no kernels for RTX 50-series GPUs. Then `src/fencing/pose/gvhmr_requirements.txt` (adds `yacs` and `ninja`, which upstream lacks), GVHMR itself (editable), and `chumpy` (needs `--no-build-isolation`).
- **pytorch3d v0.7.9 built with its CUDA ops.** `demo.py` imports `pytorch3d.renderer` and `pytorch3d.ops.knn` when it loads, and both need the compiled `pytorch3d._C`. It also renders on the GPU, so a CPU-only build is not enough. The build needs a CUDA 12.8 compiler, and no CUDA 12 pip wheel ships `nvcc`. So the script unpacks NVIDIA's own redist tarballs (nvcc 12.8.61, cccl 12.8.55, cudart 12.8.57, sha256-checked) into `$HCF_DATA/cuda-12.8`, and takes the cuBLAS/cuSPARSE/etc. headers from the venv's `nvidia-*` wheels.
- **gcc 14 if the system gcc is newer.** CUDA 12.8 rejects gcc 15, and nvcc 12.9 fails on gcc 15's C++ headers too. On such a system (Ubuntu 26.04) the script unpacks Ubuntu's gcc-14 packages into `$HCF_DATA/gcc-14` with `apt-get download` + `dpkg -x`.
- **A one-line header fix if glibc is 2.41 or newer.** That glibc declares `sinpi`, `cospi` and `rsqrt` `noexcept`, and CUDA 12.8's `crt/math_functions.h` does not, so every `.cu` file fails to compile. The script adds the matching `noexcept` to the six declarations in its private copy (the original is kept as `math_functions.h.orig`).
- **`torch.load` back to its old default.** torch 2.6 and later default to `weights_only=True`, which refuses the YOLO, HMR2 and GVHMR checkpoints. The file `hcf_torch_load_compat.pth` in that venv's site-packages sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. That turns off torch's safe default for every `torch.load` call that does not pass `weights_only`, in every Python process started from this venv and in their child processes. It covers the model checkpoints and also the cached `preprocess/*.pt` files GVHMR writes under its output folder. A pickle can run code when it is loaded, so only load checkpoints and caches you made or trust, and only put files from the upstream projects in `inputs/checkpoints`. Deleting that file undoes the change.
- Symlinks `third_party/GVHMR/inputs` to `$HCF_DATA/gvhmr/inputs` (GVHMR reads some paths from its package root and some from the working directory), and creates the empty `body_models/smplx` and `body_models/smpl` folders.

*Checked 2026-11-12 on the RTX 5080 PC (WSL2, Ubuntu 26.04, gcc 15.2, glibc 2.43):* without the SMPL/SMPL-X files, `demo.py -s` runs all of preprocessing and stops at the first body model load. Preprocessing is YOLOv8 tracking, ViTPose-H 2D keypoints and HMR2 image features, and it needs no body model. The stop is `assert smplx_path.exists()` in `hmr4d/utils/body_model/smplx_lite.py` for `body_models/smplx/SMPLX_NEUTRAL.npz`. The assert prints no file name; the traceback goes through `EnDecoder`. Everything after that point needs the body models:

| Stage | Needs |
|---|---|
| Person tracking (YOLOv8x), ViTPose-H, HMR2 features | `checkpoints/{yolo,vitpose,hmr2}` only |
| GVHMR network (`EnDecoder` builds the SMPL-X skeleton when the model is created) | `SMPLX_NEUTRAL.npz` + `checkpoints/gvhmr` |
| In-camera and global render videos | `SMPLX_NEUTRAL.npz` + `SMPL_NEUTRAL.pkl` |

For a 5 s, 30 fps, 1280×720 clip, preprocessing took 26–28 s (about 5.5 s per second of video, including about 12 s of model loading). Tracking was 3 s, ViTPose 7 s and HMR2 3.5 s. GPU memory peaked about 4 GB above what was already in use (torch: 3.2 GB allocated, 3.7 GB reserved, plus the CUDA context), and held there until the process exited. The GVHMR network and rendering were not reached, so their memory is unmeasured. Check free memory before starting it next to mjlab trainings.

`fencing pose` does not call `demo.py` as a script, because the demo tracks the largest person, which in a fencing clip can be the wrong one. `fencing.pose.gvhmr_worker` imports the demo's own functions and gives it the pipeline's fencer boxes instead (docs/pipeline.md, "GVHMR branch"). Phase 1 runs now, without the body models:

```bash
scripts/hcf pose --all --existing --gvhmr-preprocess   # every clip with a pose/<clip_id>/ folder; cached, re-runs skip done clips
```

This was run on 2026-11-12 over all 64 clips that had a pose folder (the four published sources first): 64 of 64 finished, no errors. That is 638 s of video in 2850 s, about 4.5 s per second of video, one clip at a time, with mjlab trainings on the same GPU. On 30 of the clips YOLO sees two or more people. On 4 of them the person GVHMR's demo would have picked (largest summed box area) is the other fencer or the coach, not the pipeline's fencer; the chosen boxes are on the fencer in all 4 (checked on the overlays). The fencer check fails on 3 clips, which will stay `fallback_mediapipe`:

| Clip | Why it fails | What the overlay shows |
|---|---|---|
| `cLmKzPFfkew_44000-53000` | YOLO boxes on 46 % of frames, keypoints agree on 79 % | One person. The screen track box stays at the en garde position during lunges, so the YOLO box of the lunging fencer does not overlap it enough and the stale track box is used |
| `rKRYHsJDnJc_18033-23467` | keypoints agree on 69 % of frames | Crowded hall. The chosen boxes follow the screen's fencer, but MediaPipe's landmarks jump to the person beside them on part of the clip, so the existing fallback pose is partly wrong too |
| `8GwT1fx7_RM_7925-10594` | YOLO boxes on 0 % of frames | No person: the screen tracked a fencer drawn in a logo. This clip has no `pose.json` |

All other 61 clips pass, with YOLO boxes on at least 98 % of frames, at least 97 % of frames agreeing, and a median centroid distance of at most 0.023 box heights.

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
| **SMPL-X** | Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/), download "SMPL-X v1.1" (`models_smplx_v1_1.zip`) and take `models/smplx/SMPLX_NEUTRAL.npz` | `$HCF_DATA/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` (`third_party/GVHMR/inputs` links there). `fencing pose` and `fencing retarget` both read this one file. GMR's own scripts (run by hand) want theirs in `third_party/GMR/assets/body_models/smplx/` |
| **SMPL** | Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/), download "SMPL for Python" v1.1.0 and take the neutral model, `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` | `$HCF_DATA/gvhmr/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` (renamed). Only GVHMR's render videos use it (`--gvhmr-render`) |
| **GVHMR checkpoints** | `gvhmr`, `hmr2`, `vitpose`, `yolo` (and optionally `dpvo`) from the [GVHMR Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | `$HCF_DATA/gvhmr/inputs/checkpoints/` (layout in `third_party/GVHMR/docs/INSTALL.md`) |
| **Isaac Sim EULA** | Accept on first launch | — |

GMR's SMPL-X loader, only for GMR's own scripts: if you use the `.pkl` models, GMR's README says to change `ext` in `smplx/body_models.py` from `npz` to `pkl`. `fencing retarget` passes the `.npz` and needs no change.

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

`fencing pose` uses GVHMR when `$HCF_DATA/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` exists (licence-gated, see the table above). Otherwise it uses the licence-free MediaPipe fallback and says so once. A bare `<yt_id>` covers the clips planned from screen.json plus the clips of that video that already have a `pose/<clip_id>/` folder; `--existing` takes only the latter. Useful flags: `--max-seconds` (clip length cap, default 20), `--height` (fencer height, default 1.75 m), `--hfov` / `--pitch-deg` (camera), `--reprocess` (redo the post-processing from cached MediaPipe output), `--gvhmr-preprocess` (GVHMR phase 1 only, see below), `--gvhmr-render` (GVHMR's mesh videos, needs `SMPL_NEUTRAL.pkl`), `--force`. Check `pose/<clip_id>/overlay.png` (and `gvhmr/fencer_check.png` for GVHMR) before retargeting.

#### When the body models arrive

Phase 1 of GVHMR (person boxes, ViTPose, HMR2 features) is already cached for every clip made so far (see "GVHMR venv" above). After the two files are in place (table above, exact names):

```bash
ls ~/hcf-data/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
   ~/hcf-data/gvhmr/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
scripts/hcf pose <yt_id> --existing --gvhmr-render   # only the GVHMR network; pose_method -> gvhmr
scripts/hcf retarget <yt_id>                         # GVHMR clips again; old CSVs in retarget/csv/previous/
# every clip made so far:
scripts/hcf pose --all --existing
scripts/hcf retarget --all
```

Without `--existing`, `pose <yt_id>` also makes any new clips screen.json plans for that video (MediaPipe and the full GVHMR run). The first time, look at `pose/<clip_id>/gvhmr/<clip_id>/<clip_id>_3_incam_global_horiz.mp4` (mesh on the video, and in the world) for one clip, and check GPU memory: the network's peak is not measured yet. A clip whose GVHMR keypoints are not on MediaPipe's fencer stays `fallback_mediapipe` and says why in `pose.json`.

The upstream commands the wrapper follows, for running a stage by hand:

```bash
# 1. GVHMR (gvhmr venv; run from $HCF_DATA/gvhmr, because it finds inputs/checkpoints relative to
#    the working directory). -s = static camera, skips visual odometry
env -C ~/hcf-data/gvhmr PYTHONPATH="$PWD/third_party/GVHMR" ~/hcf-data/venvs/gvhmr/bin/python \
  "$PWD/third_party/GVHMR/tools/demo/demo.py" --video=/path/to/clip.mp4 --output_root=/path/to/out -s
#    → /path/to/out/<clip>/hmr4d_results.pt  (+ overlay video for the visual check)
#    By hand, demo.py tracks the LARGEST person, not the fencer: on two-fencer clips use
#    `scripts/hcf pose` (fencing.pose.gvhmr_worker), which gives it the screen's fencer boxes.
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
