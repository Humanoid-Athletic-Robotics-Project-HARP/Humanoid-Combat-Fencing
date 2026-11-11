#!/usr/bin/env bash
# Sim-to-sim through the deployment stack: motion_tracking_controller's C++ controller, its
# state estimator and its MuJoCo simulator (ROS 2 Jazzy in the rootless rootfs, docs/mtc.md),
# scored with the same metrics, thresholds and verdict as `fencing sim2sim`.
#
#   scripts/mtc_sim2sim.sh <eval-or-run-dir> [--seconds S] [--hold-seconds H] [--settle-seconds T]
#                          [--checkpoint N] [--keep] [--no-hold]
#
# The robot is held still in the reference motion's first frame until walking_controller first
# commands torque (scripts/mtc/hcf_mtc_hold); --no-hold starts it as upstream does, unheld in
# unitree_description's pose, where the result depends on how fast the controller activates.
#
# The plant follows the policy (fencing.sim2sim.mtc_plant): a saber policy (export.json task id
# and the export's mjlab scene, which must agree) runs on unitree_description's G1 with the
# training saber welded to the right wrist, regenerated into $HCF_DATA/mtc/models each run; a
# plain one on unitree_description's G1 unchanged. If the two sources disagree, or neither
# exists, it refuses to run.
#
# Writes <eval dir>/mtc.json and prints `[mtc] ...` lines like sim2sim's. The ROS recording
# goes to $HCF_DATA/mtc/runs/<eval dir name>/ (bag, logs, bag.npz); --keep keeps the bag.
# $MTC_LAUNCH_ARGS is passed to the launch (e.g. "spawn_delay:=3", which activates the controller
# late on purpose to check the start does not depend on it). CPU only. Needs $HCF_DATA/mtc (built by docs/mtc.md); fails if it is missing.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HCF_DATA="${HCF_DATA:-$HOME/hcf-data}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HCF_DATA/venvs/fencing}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
MTC="$HCF_DATA/mtc"
ROOTFS="$MTC/noble"

[[ $# -ge 1 ]] || { sed -n '6,7p' "$0"; exit 2; }
TARGET="$1"; shift
SCORE_ARGS=(); CHECKPOINT=(); KEEP=0; HOLD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds|--hold-seconds|--settle-seconds) SCORE_ARGS+=("$1" "$2"); shift 2 ;;
    --checkpoint) CHECKPOINT=(--checkpoint "$2"); shift 2 ;;
    --keep) KEEP=1; shift ;;
    --no-hold) HOLD=0; shift ;;
    *) echo "[mtc] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$MTC/enter.sh" || ! -d "$ROOTFS/root/colcon_ws/install/motion_tracking_controller" ]]; then
  echo "[mtc] FAIL: no motion_tracking_controller build at $MTC (enter.sh + noble/root/colcon_ws)." >&2
  echo "[mtc]       Build it with docs/mtc.md; this check cannot run without it." >&2
  exit 2
fi

PYM=(nice -n 15 uv run --quiet --frozen --project "$REPO_ROOT" python -m)
PY=("${PYM[@]}" fencing.sim2sim.mtc)
# Command substitution, not `read < <(...)`: a failing planner must stop the run here.
PLAN="$("${PY[@]}" "$TARGET" "${CHECKPOINT[@]}" "${SCORE_ARGS[@]}" --plan)"
read -r EXPORT_DIR RECORD_S <<<"$PLAN"
[[ -f "$EXPORT_DIR/policy.onnx" ]] || { echo "[mtc] FAIL: no policy.onnx in $EXPORT_DIR" >&2; exit 2; }
case "$EXPORT_DIR" in
  "$HCF_DATA"/*) ;;
  *) echo "[mtc] FAIL: $EXPORT_DIR is outside \$HCF_DATA, which is the only host path the rootfs sees" >&2; exit 2 ;;
esac
NAME="$(basename "$(dirname "$EXPORT_DIR")")_$(basename "$EXPORT_DIR")"
mkdir -p "$MTC/runs"
RUN="$MTC/runs/$NAME"
IN_ROOTFS() { echo "/hostdata/${1#"$HCF_DATA"/}"; }

PLANT_LINE="$("${PYM[@]}" fencing.sim2sim.mtc_plant "$EXPORT_DIR" --models-dir "$MTC/models")"
read -r PLANT PLANT_URDF PLANT_MJCF <<<"$PLANT_LINE"
MJCF_ARGS=(); MODEL_ARGS=""
if [[ "$PLANT" == g1_saber ]]; then
  MJCF_ARGS=(--mtc-mjcf "$PLANT_MJCF")
  MODEL_ARGS="$(IN_ROOTFS "$PLANT_URDF") $(IN_ROOTFS "$PLANT_MJCF")"
  echo "[mtc] saber policy: plant $PLANT_MJCF + $PLANT_URDF"
fi

echo "[mtc] $EXPORT_DIR/policy.onnx in motion_tracking_controller + mujoco_sim_ros2, ${RECORD_S} s past activation"
# The rootfs gets a fresh copy of the recorder scripts each run (they live in this repo).
rm -rf "$ROOTFS/root/hcf"
cp -r "$REPO_ROOT/scripts/mtc" "$ROOTFS/root/hcf"
START_QPOS=-
if [[ $HOLD -eq 1 ]]; then
  "${PY[@]}" "$EXPORT_DIR" "${MJCF_ARGS[@]}" --write-start-qpos "$MTC/runs/$NAME.start_qpos.txt" >/dev/null
  START_QPOS="$(IN_ROOTFS "$MTC/runs/$NAME.start_qpos.txt")"
fi
# A stale mtc.json must not outlive a run that fails before scoring.
rm -f "$EXPORT_DIR/mtc.json"
MAX_ATTEMPTS=3
for ATTEMPT in $(seq "$MAX_ATTEMPTS"); do
  # Never score a previous run's bag: the recorder clears it too, but not if enter.sh itself fails.
  rm -rf "$RUN"
  STATUS=0
  # enter.sh starts the rootfs with an empty environment: forward the launch arguments explicitly.
  "$MTC/enter.sh" "MTC_LAUNCH_ARGS='${MTC_LAUNCH_ARGS:-}' nice -n 15 bash /root/hcf/record.sh $(IN_ROOTFS "$EXPORT_DIR")/policy.onnx $(IN_ROOTFS "$RUN") $RECORD_S $START_QPOS $MODEL_ARGS" \
    >"$MTC/runs/$NAME.log" 2>&1 || STATUS=$?
  # 0 = recorded the whole span; 2 = wall-clock cap (scored, reported incomplete). Anything else
  # (hold plugin build, bag conversion, enter.sh) means this recording cannot be trusted.
  if [[ $STATUS -ne 0 && $STATUS -ne 2 ]]; then
    echo "[mtc] FAIL: recorder exited $STATUS; not scoring. See $MTC/runs/$NAME.log and $RUN/*.log" >&2
    tail -5 "$MTC/runs/$NAME.log" >&2 || true
    exit 1
  fi
  if [[ ! -f "$RUN/bag.npz" ]]; then
    echo "[mtc] FAIL: recorder exited $STATUS but wrote no $RUN/bag.npz" >&2
    exit 1
  fi
  [[ $STATUS -eq 0 ]] || echo "[mtc] WARNING: recorder hit its wall-clock cap; scoring what was recorded"
  RELEASE=null
  if [[ $HOLD -eq 1 ]]; then
    RELEASE="$(grep -oE 'hcf_mtc_hold: released at sim time [0-9.]+' "$RUN/launch.log" | grep -oE '[0-9.]+$' | head -1 || true)"
    if [[ -z "$RELEASE" ]]; then
      echo "[mtc] FAIL: the start hold never released the robot (no release line in $RUN/launch.log)" >&2
      exit 1
    fi
  fi
  [[ $KEEP -eq 1 ]] || rm -rf "$RUN/bag"

  printf '{"recorder_status": %d, "recording_dir": "%s", "start_hold": %s, "hold_release_sim_s": %s, "plant_urdf": "%s", "attempt": %d}\n' \
    "$STATUS" "$RUN" "$([[ $HOLD -eq 1 ]] && echo true || echo false)" "$RELEASE" \
    "${PLANT_URDF:-unitree_description robot.xacro}" "$ATTEMPT" >"$RUN/extra.json"
  SCORE=0
  "${PY[@]}" "$EXPORT_DIR" "${MJCF_ARGS[@]}" "${SCORE_ARGS[@]}" --recording "$RUN/bag.npz" --extra "$RUN/extra.json" || SCORE=$?
  # 3 = INVALID (fencing.sim2sim.mtc.EXIT_INVALID): messages dropped under CPU load; record again.
  if [[ $SCORE -eq 3 && $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
    echo "[mtc] attempt $ATTEMPT INVALID (dropped messages); recording again"
    continue
  fi
  exit $SCORE
done
