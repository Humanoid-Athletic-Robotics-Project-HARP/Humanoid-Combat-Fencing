#!/usr/bin/env bash
# Kinematic replay (no physics) of a G1 motion CSV, locally, without W&B.
# Works on the Mac (CPU) and on the GPU machine. This is the Phase 1.4 "plays back
# properly" check: watch the video for foot sliding, feet through the floor, drift.
#
#   scripts/replay_csv.sh motions/en_garde_v1.csv          # input fps defaults to 30
#   scripts/replay_csv.sh some_clip.csv 50
#
# Writes outputs/replay/<name>.mp4 and outputs/replay/<name>.npz (mjlab body ordering).
# CSV format: x,y,z, qx,qy,qz,qw, 29 joint angles (Unitree / BeyondMimic convention).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
FPS="${2:-30}"
NAME="$(basename "$CSV" .csv)"
OUT="$REPO_ROOT/outputs/replay"
WORK="$(mktemp -d)"
mkdir -p "$OUT"

cols=$(head -1 "$CSV" | awk -F, '{print NF}')
[[ "$cols" == 36 ]] || { echo "expected 36 columns (7 root + 29 joints), got $cols"; exit 1; }

DEVICE=cpu
command -v nvidia-smi >/dev/null && DEVICE=cuda:0
[[ "$(uname)" == Linux ]] && export MUJOCO_GL="${MUJOCO_GL:-egl}"

# csv_to_npz always writes ./motion.mp4 and /tmp/motion.npz, then uploads to W&B;
# WANDB_MODE=disabled turns the upload into a no-op.
(cd "$WORK" && WANDB_MODE=disabled uv run --project "$REPO_ROOT/third_party/mjlab" \
  -m mjlab.scripts.csv_to_npz --input-file "$CSV" --output-name "$NAME" \
  --input-fps "$FPS" --output-fps 50 --device "$DEVICE" --render True 2>&1 \
  | grep -v -e "^Module " -e "it/s\]")

mv "$WORK/motion.mp4" "$OUT/$NAME.mp4"
cp /tmp/motion.npz "$OUT/$NAME.npz"
rm -rf "$WORK"
echo "video: $OUT/$NAME.mp4"
echo "npz:   $OUT/$NAME.npz"
