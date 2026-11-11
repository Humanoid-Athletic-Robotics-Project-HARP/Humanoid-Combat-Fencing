"""`fencing retarget`: pose clips -> raw Unitree G1 motion CSVs with GMR (Phase 1.4d).

    fencing retarget --all                          # every pose/<clip_id>/pose.json without a CSV yet
    fencing retarget IpM7citb5bs_12000-19500        # by clip_id
    fencing retarget IpM7citb5bs                    # every clip of one video
    fencing retarget --all --force --height 1.80

Outputs (docs/pipeline.md): retarget/<clip_id>.pkl (GMR pickle: root_pos, root_rot xyzw,
dof_pos, fps), retarget/csv/<clip_id>.csv (headerless 36 columns, root quat xyzw, z up,
30 fps, G1 29-DoF order) and retarget/csv/<clip_id>.json (sidecar for `fencing motion`).

Input: pose.json's pose_method picks the branch. `gvhmr` clips go through GMR's own SMPL-X
loader (needs the licence-gated model, untested here); `fallback_mediapipe` clips go
keypoints_world.npz -> fencing.pose.joint_frames -> GMR frame by frame. The worker runs in
the `gmr` venv; logs in logs/retarget/<clip_id>.log. No mirroring: left-handed sources keep
their handedness_guess and `fencing motion` mirrors them.
"""

from __future__ import annotations

import argparse
import json
import time

from fencing import paths
from fencing.pose import gvhmr
from fencing.pose.venvs import WorkerError, eprint, run_in_venv
from fencing.retarget.convert import build_sidecar

CSV_DIR = paths.RETARGET / "csv"


def pose_clips() -> list[str]:
    return sorted(p.parent.name for p in paths.POSE.glob("*/pose.json"))


def resolve_clips(tokens: list[str]) -> list[str]:
    """clip_ids as given, or every clip of a yt_id."""
    available = pose_clips()
    out: list[str] = []
    for token in tokens:
        if (paths.POSE / token / "pose.json").exists():
            out.append(token)
            continue
        matches = [c for c in available if c.startswith(f"{token}_")]
        if not matches:
            eprint(f"[retarget] {token}: no pose output (run `fencing pose {token}` first)")
        out += matches
    return list(dict.fromkeys(out))


def choose_method(pose_summary: dict, requested: str) -> str:
    """'gvhmr' or 'fallback' for the worker."""
    if requested != "auto":
        return requested
    return "gvhmr" if pose_summary.get("pose_method") == "gvhmr" else "fallback"


def retarget_one(clip_id: str, args) -> dict:
    pose_dir = paths.POSE / clip_id
    out_csv = CSV_DIR / f"{clip_id}.csv"
    out_json = out_csv.with_suffix(".json")
    if out_csv.exists() and out_json.exists() and not args.force:
        eprint(f"[retarget] {clip_id}: done already ({out_csv})")
        return json.loads(out_json.read_text())
    summary = json.loads((pose_dir / "pose.json").read_text())
    method = choose_method(summary, args.method)
    height = args.height or summary.get("human_height_m")
    if args.height:
        summary["human_height_m"] = args.height
    stats_path = paths.RETARGET / f"{clip_id}.stats.json"
    log = paths.LOGS / "retarget" / f"{clip_id}.log"
    cmd = [
        "-m", "fencing.retarget.gmr_worker", "--pose-dir", str(pose_dir), "--method", method,
        "--out-pkl", str(paths.RETARGET / f"{clip_id}.pkl"), "--out-csv", str(out_csv), "--out-stats", str(stats_path),
    ]  # fmt: skip
    if height and method == "fallback":
        cmd += ["--height", str(height)]
    if method == "gvhmr":
        cmd += ["--body-models", str(gvhmr.BODY_MODELS)]
    eprint(f"[retarget] {clip_id}: GMR ({method}, log {log})")
    run_in_venv(
        "gmr",
        cmd,
        log,
        extra_env={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
    )
    stats = json.loads(stats_path.read_text())
    if method == "gvhmr" and not args.height:
        summary["human_height_m"] = stats.get("human_height_m")
    sidecar = build_sidecar(summary, stats, notes=f"retargeted with GMR smplx_to_g1 ({method} input)")
    out_json.write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fencing retarget", description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="*", help="clip_ids or yt_ids (all clips of that video)")
    parser.add_argument("--all", action="store_true", help=f"every clip with a pose.json under {paths.POSE}")
    parser.add_argument("--method", choices=["auto", "fallback", "gvhmr"], default="auto", help="input branch (default: from pose.json)")
    parser.add_argument("--height", type=float, help="override the human height (default: pose.json)")
    parser.add_argument("--force", action="store_true", help="redo clips that already have a CSV")
    args = parser.parse_args(argv)

    tokens = list(args.clips) + (pose_clips() if args.all else [])
    if not tokens:
        parser.error(f"name clip_ids / yt_ids or pass --all (pose outputs live in {paths.POSE})")
    paths.ensure_dirs()
    clips = resolve_clips(tokens)
    failures = 0
    for clip_id in clips:
        start = time.time()
        try:
            sidecar = retarget_one(clip_id, args)
        except (FileNotFoundError, WorkerError, ValueError, KeyError, OSError) as exc:
            failures += 1
            eprint(f"[retarget] {clip_id}: FAILED: {exc}")
            continue
        r = sidecar.get("retarget", {})
        print(
            f"{clip_id}: {r.get('frames')} frames @ {sidecar['fps']} fps, hand {sidecar['handedness_guess']}, "
            f"IK err {r.get('ik_error_median')}, at-limit {r.get('joint_limit_frac')} -> {CSV_DIR / (clip_id + '.csv')}"
            f"  [{time.time() - start:.0f}s]",
            flush=True,
        )
    if clips:
        eprint(f"[retarget] {len(clips) - failures}/{len(clips)} clips ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
