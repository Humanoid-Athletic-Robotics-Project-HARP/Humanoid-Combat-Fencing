#!/usr/bin/env bash
# Mac (Apple Silicon) setup: everything that does not need an NVIDIA GPU.
#
#   - mjlab   : CSV -> NPZ conversion and policy playback / evaluation (CPU)
#   - GMR     : SMPL-X -> G1 retargeting and kinematic replay
#   - LAFAN1  : Unitree-retargeted G1 reference motions for Phase 1.2
#
# Not set up here (needs CUDA): Isaac Sim / Isaac Lab, training, local GVHMR.
# For GVHMR without a GPU, use its Colab or HuggingFace demo (see docs/setup.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v uv >/dev/null || { echo "Installing uv"; brew install uv; }

"$REPO_ROOT/scripts/setup_third_party.sh"

echo "== mjlab"
(cd "$REPO_ROOT/third_party/mjlab" && uv sync)
echo "== GMR"
(cd "$REPO_ROOT/third_party/GMR" && uv sync)

echo "== LAFAN1 (G1 CSVs, CC BY-NC-ND 4.0, kept out of git)"
uvx --from huggingface_hub hf download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset --include "g1/*" --local-dir "$REPO_ROOT/data/external/lafan1"

echo "== smoke tests"
(cd "$REPO_ROOT/third_party/mjlab" && uv run list-envs | grep Tracking)
(cd "$REPO_ROOT/third_party/GMR" && uv run python -c \
  "import mujoco; from general_motion_retargeting.params import ROBOT_XML_DICT as d; \
m = mujoco.MjModel.from_xml_path(str(d['unitree_g1'])); print('GMR G1 model ok, nq =', m.nq)")

cat <<'EOF'

Done. Manual steps that cannot be scripted (licences / logins):
  1. SMPL-X body models -> third_party/GMR/assets/body_models/smplx/  (register at smpl-x.is.tue.mpg.de)
  2. wandb login, and create the "Motions" registry (docs/setup.md, step W&B)
EOF
