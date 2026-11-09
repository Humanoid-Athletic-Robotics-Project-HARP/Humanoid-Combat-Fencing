#!/usr/bin/env bash
# Fetch the pinned upstream code in third_party/.
#
#   scripts/setup_third_party.sh          # everything except Isaac Lab (Mac, or any machine)
#   scripts/setup_third_party.sh --gpu    # also Isaac Lab v2.1.0 (Linux GPU machine only)
#
# Pinned commits live in the superproject (git submodule status). To move a pin,
# check out the new commit inside the submodule and commit the submodule change here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SUBMODULES=(
  third_party/mjlab                       # primary training stack (BeyondMimic tracking task)
  third_party/whole_body_tracking         # original BeyondMimic on Isaac Lab 2.1.0 (archived upstream)
  third_party/motion_tracking_controller  # BeyondMimic sim-to-sim (MuJoCo) and real deployment, ROS 2
  third_party/GMR                         # human motion -> G1 retargeting
  third_party/GVHMR                       # video -> SMPL-X human motion
)

if [[ "${1:-}" == "--gpu" ]]; then
  # IsaacLab is marked `update = none` in .gitmodules so a plain clone skips it;
  # --checkout overrides that.
  git -C "$REPO_ROOT" submodule update --init --depth 1 --checkout third_party/IsaacLab
fi

git -C "$REPO_ROOT" submodule update --init --depth 1 "${SUBMODULES[@]}"
git -C "$REPO_ROOT" submodule status
