#!/usr/bin/env bash
# Sim-to-sim through the deployment stack: motion_tracking_controller's C++ controller, its
# state estimator and its MuJoCo simulator (ROS 2 Jazzy in the rootless rootfs, docs/mtc.md),
# scored with the same metrics, thresholds and verdict as `fencing sim2sim`.
#
#   scripts/mtc_sim2sim.sh <eval-or-run-dir> [--seconds S] [--hold-seconds H] [--settle-seconds T]
#                          [--checkpoint N] [--keep] [--no-hold] [--standby [--switch-after S] [--entry <entry dir>]]
#
# The robot is held still in the reference motion's first frame until walking_controller first
# commands torque (scripts/mtc/hcf_mtc_hold); --no-hold starts it as upstream does, unheld in
# unitree_description's pose, where the result depends on how fast the controller activates.
#
# --standby starts the way the real robot does (docs/mtc.md, "Start from standby"):
# standby_controller active and the policy inactive, as upstream's real.launch.py; the robot is
# held in unitree_description's default pose on the floor only until standby's first command;
# --switch-after S (default 10) sim seconds later the operator's R1 + A switch starts the policy.
# It writes <eval dir>/mtc_standby.json and records under $HCF_DATA/mtc/runs_standby/, so it never
# touches mtc.json or $HCF_DATA/mtc/runs/ (the held results).
#
# --entry <eval-or-run-dir> (with --standby) is the chained start (docs/mtc.md, "Entry from
# standby"): that entry policy runs as walking_controller (R1 + A from standby), and the target
# runs as walking_controller1 (scripts/mtc/mujoco_custom.launch.py chain_policy_path:=), switched
# to with the operator's R1 + X once the entry reference and a 2 s hold have played. Writes
# <eval dir>/mtc_entry.json and records under $HCF_DATA/mtc/runs_entry/.
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
SCORE_ARGS=(); CHECKPOINT=(); KEEP=0; HOLD=1; STANDBY=(); SWITCH_AFTER=10; ENTRY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds|--hold-seconds|--settle-seconds) SCORE_ARGS+=("$1" "$2"); shift 2 ;;
    --checkpoint) CHECKPOINT=(--checkpoint "$2"); shift 2 ;;
    --keep) KEEP=1; shift ;;
    --no-hold) HOLD=0; shift ;;
    --standby) STANDBY=(--standby); shift ;;
    --switch-after) SWITCH_AFTER="$2"; shift 2 ;;
    --entry) ENTRY=(--entry "$2"); shift 2 ;;
    *) echo "[mtc] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ${#STANDBY[@]} -gt 0 && $HOLD -eq 0 ]]; then
  # Without the hold the robot can fall before standby is even active: not the test --standby is.
  echo "[mtc] --standby needs the start hold (it is released by standby's first command); drop --no-hold" >&2
  exit 2
fi

if [[ ${#ENTRY[@]} -gt 0 && ${#STANDBY[@]} -eq 0 ]]; then
  echo "[mtc] --entry needs --standby: the chained start begins under standby_controller" >&2
  exit 2
fi
if [[ ${#ENTRY[@]} -gt 0 && ! "$SWITCH_AFTER" =~ ^[0-9]+$ ]]; then
  echo "[mtc] --switch-after must be a non-negative integer of seconds with --entry (got '$SWITCH_AFTER')" >&2
  exit 2
fi

if [[ ! -x "$MTC/enter.sh" || ! -d "$ROOTFS/root/colcon_ws/install/motion_tracking_controller" ]]; then
  echo "[mtc] FAIL: no motion_tracking_controller build at $MTC (enter.sh + noble/root/colcon_ws)." >&2
  echo "[mtc]       Build it with docs/mtc.md; this check cannot run without it." >&2
  exit 2
fi

PYM=(nice -n 15 uv run --quiet --frozen --project "$REPO_ROOT" python -m)
PY=("${PYM[@]}" fencing.sim2sim.mtc)
# Command substitution, not `read < <(...)`: a failing planner must stop the run here.
PLAN="$("${PY[@]}" "$TARGET" "${CHECKPOINT[@]}" "${SCORE_ARGS[@]}" "${STANDBY[@]}" "${ENTRY[@]}" --plan)"
# OUT_JSON / RUNS_DIR: mtc.json + runs, mtc_standby.json + runs_standby, or mtc_entry.json +
# runs_entry (fencing.sim2sim.mtc.run_outputs). With --entry also the entry's export dir and the
# sim seconds from the entry's activation to the R1 + X switch.
read -r EXPORT_DIR RECORD_S OUT_JSON RUNS_DIR ENTRY_DIR CHAIN_S <<<"$PLAN"
[[ -n "$RUNS_DIR" ]] || { echo "[mtc] FAIL: planner printed no output names: $PLAN" >&2; exit 2; }
for DIR in "$EXPORT_DIR" ${ENTRY_DIR:+"$ENTRY_DIR"}; do
  [[ -f "$DIR/policy.onnx" ]] || { echo "[mtc] FAIL: no policy.onnx in $DIR" >&2; exit 2; }
  case "$DIR" in
    "$HCF_DATA"/*) ;;
    *) echo "[mtc] FAIL: $DIR is outside \$HCF_DATA, which is the only host path the rootfs sees" >&2; exit 2 ;;
  esac
done
NAME="$(basename "$(dirname "$EXPORT_DIR")")_$(basename "$EXPORT_DIR")"
RUNS="$MTC/$RUNS_DIR"
mkdir -p "$RUNS"
RUN="$RUNS/$NAME"
IN_ROOTFS() { echo "/hostdata/${1#"$HCF_DATA"/}"; }

PLANT_LINE="$("${PYM[@]}" fencing.sim2sim.mtc_plant "$EXPORT_DIR" --models-dir "$MTC/models")"
read -r PLANT PLANT_URDF PLANT_MJCF <<<"$PLANT_LINE"
MJCF_ARGS=(); MODEL_ARGS=""
if [[ "$PLANT" == g1_saber ]]; then
  MJCF_ARGS=(--mtc-mjcf "$PLANT_MJCF")
  [[ ${#STANDBY[@]} -gt 0 ]] && STANDBY+=(--urdf "$PLANT_URDF")
  MODEL_ARGS="$(IN_ROOTFS "$PLANT_URDF") $(IN_ROOTFS "$PLANT_MJCF")"
  echo "[mtc] saber policy: plant $PLANT_MJCF + $PLANT_URDF"
fi

echo "[mtc] $EXPORT_DIR/policy.onnx in motion_tracking_controller + mujoco_sim_ros2, ${RECORD_S} s past activation"
# The rootfs gets a fresh copy of the recorder scripts each run (they live in this repo).
rm -rf "$ROOTFS/root/hcf"
cp -r "$REPO_ROOT/scripts/mtc" "$ROOTFS/root/hcf"
START_QPOS=-
if [[ $HOLD -eq 1 ]]; then
  # With --standby: the model's default standing pose, not the reference's first frame.
  "${PY[@]}" "$EXPORT_DIR" "${MJCF_ARGS[@]}" "${STANDBY[@]}" --write-start-qpos "$RUNS/$NAME.start_qpos.txt" >/dev/null
  START_QPOS="$(IN_ROOTFS "$RUNS/$NAME.start_qpos.txt")"
fi
STANDBY_ENV=""
[[ ${#STANDBY[@]} -gt 0 ]] && STANDBY_ENV="MTC_STANDBY_SWITCH_S='$SWITCH_AFTER'"
FIRST_POLICY="$EXPORT_DIR/policy.onnx"
if [[ -n "${ENTRY_DIR:-}" ]]; then
  # The entry is the policy R1 + A starts; the target waits as walking_controller1 for R1 + X.
  FIRST_POLICY="$ENTRY_DIR/policy.onnx"
  STANDBY_ENV="$STANDBY_ENV MTC_CHAIN_POLICY='$(IN_ROOTFS "$EXPORT_DIR")/policy.onnx' MTC_CHAIN_SWITCH_S='$CHAIN_S'"
  STANDBY_ENV="$STANDBY_ENV MTC_WALL_CAP_S='$(( (SWITCH_AFTER + CHAIN_S + RECORD_S) * 4 + 120 ))'"
  echo "[mtc] chained start: standby -> $ENTRY_DIR (R1 + A after ${SWITCH_AFTER} s) -> target (R1 + X after ${CHAIN_S} s)"
fi
# A stale score must not outlive a run that fails before scoring (only this start's own JSON).
rm -f "$EXPORT_DIR/$OUT_JSON"
MAX_ATTEMPTS=3
for ATTEMPT in $(seq "$MAX_ATTEMPTS"); do
  # Never score a previous run's bag: the recorder clears it too, but not if enter.sh itself fails.
  rm -rf "$RUN"
  STATUS=0
  # enter.sh starts the rootfs with an empty environment: forward the launch arguments explicitly.
  "$MTC/enter.sh" "MTC_LAUNCH_ARGS='${MTC_LAUNCH_ARGS:-}' $STANDBY_ENV nice -n 15 bash /root/hcf/record.sh $(IN_ROOTFS "$FIRST_POLICY") $(IN_ROOTFS "$RUN") $RECORD_S $START_QPOS $MODEL_ARGS" \
    >"$RUNS/$NAME.log" 2>&1 || STATUS=$?
  # 0 = recorded the whole span; 2 = wall-clock cap (scored, reported incomplete). Anything else
  # (hold plugin build, bag conversion, enter.sh, standby switch refused) means this recording
  # cannot be trusted.
  if [[ $STATUS -ne 0 && $STATUS -ne 2 ]]; then
    echo "[mtc] FAIL: recorder exited $STATUS; not scoring. See $RUNS/$NAME.log and $RUN/*.log" >&2
    tail -5 "$RUNS/$NAME.log" >&2 || true
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

  printf '{"recorder_status": %d, "recording_dir": "%s", "start_hold": %s, "hold_release_sim_s": %s, "plant_urdf": "%s", "attempt": %d, "start_from_standby": %s, "standby_switch_after_s": %s, "chain_entry_dir": %s, "chain_switch_after_s": %s}\n' \
    "$STATUS" "$RUN" "$([[ $HOLD -eq 1 ]] && echo true || echo false)" "$RELEASE" \
    "${PLANT_URDF:-unitree_description robot.xacro}" "$ATTEMPT" \
    "$([[ ${#STANDBY[@]} -gt 0 ]] && echo true || echo false)" "$([[ ${#STANDBY[@]} -gt 0 ]] && echo "$SWITCH_AFTER" || echo null)" \
    "$([[ -n "${ENTRY_DIR:-}" ]] && echo "\"$ENTRY_DIR\"" || echo null)" "${CHAIN_S:-null}" \
    >"$RUN/extra.json"
  SCORE=0
  "${PY[@]}" "$EXPORT_DIR" "${MJCF_ARGS[@]}" "${SCORE_ARGS[@]}" "${STANDBY[@]}" "${ENTRY[@]}" --recording "$RUN/bag.npz" \
    --out "$EXPORT_DIR/$OUT_JSON" --extra "$RUN/extra.json" || SCORE=$?
  # 3 = INVALID (fencing.sim2sim.mtc.EXIT_INVALID): messages dropped under CPU load; record again.
  if [[ $SCORE -eq 3 && $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
    echo "[mtc] attempt $ATTEMPT INVALID (dropped messages); recording again"
    continue
  fi
  exit $SCORE
done
