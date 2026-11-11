"""GMR (SMPL-X -> Unitree G1 IK) on one pose clip. Runs inside the `gmr` venv, launched by
`fencing retarget`:

    python -m fencing.retarget.gmr_worker --pose-dir pose/<clip_id> --method fallback \
        --height 1.75 --out-pkl retarget/<clip_id>.pkl --out-csv retarget/csv/<clip_id>.csv \
        --out-stats retarget/<clip_id>.stats.json

--method fallback: keypoints_world.npz -> fencing.pose.joint_frames -> per-frame dicts fed to
    GeneralMotionRetargeting.retarget(). Constructing GMR with src_human="smplx" loads only the
    robot MJCF and the IK config, never the SMPL-X body model, so this needs no licensed files.
--method gvhmr: hmr4d_results.pt through GMR's own SMPL-X loader (needs the SMPL-X model in
    --body-models; untested on this machine, the model files are licence-gated).

No mirroring here: a left-handed fencer stays left-handed (that is `fencing motion`'s job).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from fencing.motion.io import JOINT_NAMES
from fencing.retarget.convert import (
    csv_array_to_pickle_dict,
    despike_joints,
    make_continuous,
    qpos_to_csv_array,
    resample_csv_array,
)

# The first frame starts from the robot's zero pose; give the IK enough iterations to settle
# before the per-frame warm start takes over.
FIRST_FRAME_REPEATS = 30


def fallback_frames(pose_dir: Path) -> tuple[list[dict], float]:
    from fencing.pose.joint_frames import compute_joint_frames, frames_to_gmr_dicts

    with np.load(pose_dir / "keypoints_world.npz") as data:
        keypoints = np.asarray(data["keypoints"], dtype=np.float64)
        fps = float(data["fps"])
    positions, quats = compute_joint_frames(keypoints)
    return frames_to_gmr_dicts(positions, quats), fps


def gvhmr_frames(pose_dir: Path, body_models: Path) -> tuple[list[dict], float, float]:
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(str(pose_dir / "hmr4d_results.pt"), str(body_models))
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    return frames, float(fps), float(height)


def check_joint_order(gmr) -> None:
    import mujoco as mj

    names = [mj.mj_id2name(gmr.model, mj.mjtObj.mjOBJ_JOINT, gmr.model.dof_jntid[i]) for i in range(6, gmr.model.nv)]
    if tuple(names) != JOINT_NAMES:
        raise RuntimeError(f"GMR G1 joint order differs from fencing.motion.io.JOINT_NAMES: {names}")


def retarget_frames(gmr, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    qpos, errors = [], []
    for i, frame in enumerate(frames):
        for _ in range(FIRST_FRAME_REPEATS if i == 0 else 1):
            q = gmr.retarget({k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in frame.items()})
        qpos.append(q)
        errors.append(gmr.error2() if gmr.use_ik_match_table2 else gmr.error1())
    return np.asarray(qpos), np.asarray(errors)


def run(args: argparse.Namespace) -> None:
    from general_motion_retargeting import GeneralMotionRetargeting

    height = args.height
    if args.method == "gvhmr":
        frames, fps, gvhmr_height = gvhmr_frames(args.pose_dir, args.body_models)
        height = height or gvhmr_height
    else:
        frames, fps = fallback_frames(args.pose_dir)
    if len(frames) < 2:
        raise RuntimeError(f"only {len(frames)} frame(s) in {args.pose_dir}")

    gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1", actual_human_height=height, verbose=False)
    check_joint_order(gmr)
    start = time.time()
    qpos, errors = retarget_frames(gmr, frames)
    elapsed = time.time() - start

    array = qpos_to_csv_array(qpos)
    array[:, 3:7] = make_continuous(array[:, 3:7])
    array = resample_csv_array(array, fps)
    array[:, 7:], despiked = despike_joints(array[:, 7:])
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as fh:
        pickle.dump(csv_array_to_pickle_dict(array, 30.0), fh)
    np.savetxt(args.out_csv, array, delimiter=",", fmt="%.6f")

    stats = {
        "frames": int(len(array)),
        "source_fps": round(fps, 3),
        "human_height_m": height,
        "ik_error_median": round(float(np.median(errors)), 4),
        "ik_error_p95": round(float(np.percentile(errors, 95)), 4),
        "ik_seconds": round(elapsed, 1),
        "joint_limit_frac": joint_limit_fraction(gmr, array[:, 7:]),
        "despiked_joint_samples": despiked,
    }
    args.out_stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[gmr] {len(array)} frames, IK error median {stats['ik_error_median']}, {1000 * elapsed / len(frames):.0f} ms/frame -> {args.out_csv}", flush=True)


def joint_limit_fraction(gmr, dof: np.ndarray, margin: float = 0.02) -> float:
    """Fraction of (frame, joint) values within `margin` rad of a joint limit (IK saturation)."""
    lo = gmr.model.jnt_range[1:, 0]
    hi = gmr.model.jnt_range[1:, 1]
    near = (dof <= lo + margin) | (dof >= hi - margin)
    return round(float(near.mean()), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["fallback", "gvhmr"], required=True)
    parser.add_argument("--height", type=float, help="human height for GMR's scale table (gvhmr default: from betas)")
    parser.add_argument("--body-models", type=Path, help="folder holding smplx/SMPLX_NEUTRAL.npz (gvhmr only)")
    parser.add_argument("--out-pkl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-stats", type=Path, required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
