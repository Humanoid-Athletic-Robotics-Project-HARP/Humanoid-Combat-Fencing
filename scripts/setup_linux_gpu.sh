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
#   gvhmr     GVHMR in uv venv $HCF_DATA/venvs/gvhmr, pytorch3d built with CUDA (checkpoints are a manual download)
#   gmr       GMR (uv)
#
# Requires: conda (Miniconda/Anaconda) for isaaclab/wbt; sudo for apt packages (not for gvhmr).
# Written against the upstream install docs at the pinned commits; not yet run end to end.
set -euo pipefail
shopt -s inherit_errexit  # so set -e also applies inside $(...) (the gvhmr helpers run there)

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

# CUDA 12.8 build toolchain for pytorch3d's compiled ops, without sudo or a system CUDA:
# nvcc/cccl/cudart from NVIDIA's redist tarballs (12.8.0 components, the same CUDA as the torch
# wheel), plus the library headers that already ship in the venv's nvidia-* wheels.
# Prints the CUDA_HOME it built. No pip wheel of CUDA 12 contains nvcc itself (only ptxas).
gvhmr_cuda_home() {
  local data="$1" site="$2" cuda="$1/cuda-12.8"
  local base=https://developer.download.nvidia.com/compute/cuda/redist
  local -a parts=(
    "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.61-archive.tar.xz 145f8779bd56bdfa214447e5cb1b3a206ec1b7398da460e257f2898fd8604c54"
    "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.55-archive.tar.xz dce4f2e7720d4432ab0861ede2243f9cbd46bc675008932bc9dcdb871fc7d60b"
    "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz 5bd3ac35ea8e8ab880e595d5054ee373abf6d9e53dcb8cef0a5c75358dbc0ae2"
  )
  # Complete means the marker exists. An older install without one counts as complete only if
  # nvcc plus the cudart and cccl (cub, thrust) headers are all there; it gets the marker.
  if [[ ! -e "$cuda/.complete" && -x "$cuda/bin/nvcc" && -e "$cuda/include/cuda_runtime.h" \
        && -e "$cuda/include/cub/cub.cuh" && -e "$cuda/include/thrust/version.h" ]]; then
    touch "$cuda/.complete" || return 1
  fi
  if [[ ! -e "$cuda/.complete" ]]; then
    local tmp part rel sha stage="$cuda.staging"
    tmp="$(mktemp -d)" || return 1
    rm -rf "$stage"
    mkdir -p "$stage" || return 1
    for part in "${parts[@]}"; do
      read -r rel sha <<<"$part"
      curl -fsSL -o "$tmp/part.tar.xz" "$base/$rel" || return 1
      echo "$sha  $tmp/part.tar.xz" | sha256sum -c --quiet || return 1
      tar -xJf "$tmp/part.tar.xz" -C "$stage" --strip-components=1 || return 1
    done
    rm -rf "$tmp" "$cuda"
    mv "$stage" "$cuda" || return 1
    touch "$cuda/.complete" || return 1
  fi
  ln -sfn lib "$cuda/lib64"
  local d h
  for d in cublas cusparse cusolver curand cufft nvtx nvjitlink cuda_nvrtc; do
    for h in "$site/nvidia/$d/include/"*; do
      [[ "$(basename "$h")" == __* ]] && continue
      [[ -e "$cuda/include/$(basename "$h")" ]] || ln -s "$h" "$cuda/include/"
    done
  done
  # glibc >= 2.41 declares sinpi/cospi/rsqrt noexcept and CUDA 12.8's math_functions.h does
  # not, which fails every .cu compile. Match glibc (later CUDA releases do the same).
  local glibc mf="$cuda/include/crt/math_functions.h"
  glibc="$(ldd --version | head -1 | awk '{print $NF}')"
  if [[ "$(printf '%s\n' 2.41 "$glibc" | sort -V | head -1)" == 2.41 ]] && ! grep -q _HCF_GLIBC_NOEXCEPT "$mf"; then
    cp "$mf" "$mf.orig"
    sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(rsqrtf?|sinpif?|cospif?)\((double|float) x\));/\1 _HCF_GLIBC_NOEXCEPT;/' "$mf"
    sed -i '0,/^#define __MATH_FUNCTIONS_H__/s//&\n\/* hcf: glibc >= 2.41 declares these noexcept; match it (scripts\/setup_linux_gpu.sh gvhmr). *\/\n#if defined(__cplusplus)\n#define _HCF_GLIBC_NOEXCEPT noexcept(true)\n#else\n#define _HCF_GLIBC_NOEXCEPT\n#endif/' "$mf"
  fi
  echo "$cuda"
}

# CUDA 12.8 accepts gcc <= 14. On a newer system gcc (Ubuntu 26.04 has 15), unpack Ubuntu's own
# gcc-14 packages into HCF_DATA (no sudo) and print its bin dir; print nothing if gcc is fine.
gvhmr_host_gcc() {
  local data="$1" gcc14="$1/gcc-14"
  (( $(gcc -dumpversion | cut -d. -f1) <= 14 )) && return 0
  # An install without the marker is complete only if the compiler binaries are all there.
  if [[ ! -e "$gcc14/.complete" && -x "$gcc14/usr/bin/g++-14" && -x "$gcc14/usr/bin/gcc-14" \
        && -x "$gcc14/usr/bin/cpp-14" ]]; then
    touch "$gcc14/.complete" || return 1
  fi
  if [[ ! -e "$gcc14/.complete" ]]; then
    local tmp deb stage="$gcc14.staging"
    tmp="$(mktemp -d)" || return 1
    (cd "$tmp" && apt-get download gcc-14 g++-14 cpp-14 gcc-14-base libgcc-14-dev libstdc++-14-dev \
      gcc-14-x86-64-linux-gnu g++-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu >&2) || return 1
    rm -rf "$stage"
    mkdir -p "$stage" || return 1
    for deb in "$tmp"/*.deb; do dpkg -x "$deb" "$stage" || return 1; done
    rm -rf "$tmp" "$gcc14"
    mv "$stage" "$gcc14" || return 1
    touch "$gcc14/.complete" || return 1
  fi
  echo "$gcc14/usr/bin"
}

stage_gvhmr() {
  # uv venv at $HCF_DATA/venvs/gvhmr, where src/fencing/paths.py looks for it.
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  local data="${HCF_DATA:-$HOME/hcf-data}"
  local venv="$data/venvs/gvhmr" py="$data/venvs/gvhmr/bin/python"
  [[ -x "$py" ]] || uv venv --python 3.10 "$venv"
  # torch 2.7.1+cu128 is the oldest torch with kernels for RTX 50-series (sm_120). Do not downgrade.
  uv pip install --python "$py" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$py" -r "$REPO_ROOT/src/fencing/pose/gvhmr_requirements.txt"
  uv pip install --python "$py" --no-deps -e "$TP/GVHMR"
  # chumpy unpickles SMPL .pkl files; its setup.py imports pip, so it needs the venv's own build tools.
  uv pip install --python "$py" pip==26.3 setuptools wheel
  uv pip install --python "$py" --no-build-isolation chumpy==0.70
  local site cuda gccbin
  site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  # torch >= 2.6 defaults torch.load to weights_only=True; GVHMR (written for torch 2.3) and
  # ultralytics 8.2 load full pickles. Restore the old default in this venv only. This trusts
  # every checkpoint under inputs/checkpoints: only put files from the upstream projects there.
  echo 'import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")' >"$site/hcf_torch_load_compat.pth"

  # pytorch3d with its compiled ops: demo.py imports pytorch3d.renderer and pytorch3d.ops.knn,
  # which need pytorch3d._C, and it renders on the GPU.
  cuda="$(gvhmr_cuda_home "$data" "$site")"
  gccbin="$(gvhmr_host_gcc "$data")"
  [[ -d "$data/src/pytorch3d" ]] || git clone --depth 1 --branch v0.7.9 \
    https://github.com/facebookresearch/pytorch3d.git "$data/src/pytorch3d"
  local -a build_env=(PATH="$venv/bin:$cuda/bin:$PATH" CUDA_HOME="$cuda" CUB_HOME="$cuda/include"
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    MAX_JOBS=8)
  [[ -n "$gccbin" ]] && build_env+=(CC="$gccbin/gcc-14" CXX="$gccbin/g++-14")
  if "$py" -c "from pytorch3d import _C" 2>/dev/null; then
    echo "pytorch3d compiled ops already importable; skipping build"
  else
    env "${build_env[@]}" nice -n 19 "$py" -m pip install --no-build-isolation --no-deps --force-reinstall \
      "$data/src/pytorch3d"
  fi
  "$py" -c 'import torch; from pytorch3d import _C; from pytorch3d.ops import knn_points; p = torch.rand(1, 64, 3, device="cuda"); knn_points(p, p, K=2); print("pytorch3d CUDA ops OK")'

  # GVHMR reads inputs/checkpoints both from the cwd and from its package root.
  mkdir -p "$data/gvhmr/inputs/checkpoints/body_models/"{smplx,smpl}
  [[ -e "$TP/GVHMR/inputs" ]] || ln -s "$data/gvhmr/inputs" "$TP/GVHMR/inputs"
  echo "GVHMR installed. Checkpoints and SMPL/SMPL-X models are manual downloads into $data/gvhmr/inputs/checkpoints: see docs/setup.md, step GVHMR."
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
