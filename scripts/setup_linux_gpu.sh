#!/usr/bin/env bash
# Linux + NVIDIA GPU setup (target: Ubuntu 22.04, native, not WSL2).
#
#   scripts/setup_linux_gpu.sh            # all stages
#   scripts/setup_linux_gpu.sh mjlab gmr  # only the named stages
#
# Stages:
#   check     GPU, driver, CUDA, GLIBC; prints a line to paste into docs/log.md
#   mjlab     primary training stack (uv)
#   isaaclab  Isaac Sim 4.5 + Isaac Lab v2.1.0 in conda env "env_isaaclab"
#   wbt       BeyondMimic whole_body_tracking into env_isaaclab (+ Unitree description files)
#   gvhmr     GVHMR in conda env "gvhmr" (checkpoints are a manual download)
#   gmr       GMR (uv)
#
# Requires: conda (Miniconda/Anaconda) for isaaclab/wbt/gvhmr; sudo for apt packages.
# Written against the upstream install docs at the pinned commits; not yet run end to end.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$REPO_ROOT/third_party"
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(check mjlab isaaclab wbt gvhmr gmr)

need_conda() {
  command -v conda >/dev/null || { echo "conda not found: install Miniconda first"; exit 1; }
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}
env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

stage_check() {
  grep -qi microsoft /proc/version 2>/dev/null && \
    echo "WARNING: WSL2 detected. mjlab works here; Isaac Sim 4.5 is not supported on WSL2."
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvidia-smi | grep -o "CUDA Version: [0-9.]*"
  ldd --version | head -1   # Isaac Sim pip install needs GLIBC >= 2.34 (Ubuntu 22.04 has 2.35)
  echo
  echo "For docs/log.md:  $(date +%F) | env | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*')"
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX 50"; then
    echo "NOTE: RTX 50-series GPU. Isaac Lab 2.1 needs a PyTorch nightly (cu128) instead of 2.5.1; see docs/setup.md."
  fi
}

stage_mjlab() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  "$REPO_ROOT/scripts/setup_third_party.sh"
  (cd "$TP/mjlab" && uv sync && uv run list-envs | grep Tracking)
}

stage_isaaclab() {
  need_conda
  sudo apt-get install -y cmake build-essential
  "$REPO_ROOT/scripts/setup_third_party.sh" --gpu
  env_exists env_isaaclab || conda create -y -n env_isaaclab python=3.10
  conda activate env_isaaclab
  pip install --upgrade pip
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
  (cd "$TP/IsaacLab" && ./isaaclab.sh --install)
  echo "Isaac Lab installed. First Isaac Sim launch asks you to accept the EULA and takes several minutes."
}

stage_wbt() {
  need_conda
  conda activate env_isaaclab
  local assets="$TP/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets"
  if [[ ! -d "$assets/unitree_description" ]]; then
    curl -L -o /tmp/unitree_description.tar.gz \
      https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
    tar -xzf /tmp/unitree_description.tar.gz -C "$assets/"
    rm /tmp/unitree_description.tar.gz
  fi
  python -m pip install -e "$TP/whole_body_tracking/source/whole_body_tracking"
}

stage_gvhmr() {
  need_conda
  env_exists gvhmr || conda create -y -n gvhmr python=3.10
  conda activate gvhmr
  (cd "$TP/GVHMR" && pip install -r requirements.txt && pip install -e . && mkdir -p inputs/checkpoints outputs)
  echo "GVHMR installed. Checkpoints and SMPL/SMPL-X models are manual downloads: see docs/setup.md, step GVHMR."
}

stage_gmr() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  (cd "$TP/GMR" && uv sync)
}

for s in "${STAGES[@]}"; do
  echo "================ $s"
  "stage_$s"
done
