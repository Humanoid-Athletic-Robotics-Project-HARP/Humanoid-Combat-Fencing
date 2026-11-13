"""GMR (SMPL-X -> Unitree G1 IK) on one pose clip. Runs inside the `gmr` venv, launched by
`fencing retarget`:

    python -m fencing.retarget.gmr_worker --pose-dir pose/<clip_id> --method fallback \
        --height 1.75 --out-pkl retarget/<clip_id>.pkl --out-csv retarget/csv/<clip_id>.csv \
        --out-stats retarget/<clip_id>.stats.json

--method fallback: keypoints_world.npz -> fencing.pose.joint_frames -> per-frame dicts fed to
    GeneralMotionRetargeting.retarget(). Constructing GMR with src_human="smplx" loads only the
    robot MJCF and the IK config, never the SMPL-X body model, so this needs no licensed files.
--method gvhmr: hmr4d_results.pt (GVHMR's smpl_params_global, 30 fps, y up) through SMPL-X
    (--body-models/smplx/SMPLX_NEUTRAL.npz, licence-gated) and GMR's
    get_gvhmr_data_offline_fast. GMR's own load_gvhmr_pred_file is not used: it pads the 10
    GVHMR betas to 16 but builds the model with smplx's default num_betas=10, so lbs() would
    fail on 26 shape components against 20 blend shapes (10 shape + 10 expression). Here the
    model gets as many betas as GVHMR wrote, averaged over the clip instead of frame 0.
    The z-up result is then shifted so the floor is at z = 0 under every frame (a planted
    SMPL-X foot joint at the fallback's 3 cm ball-of-foot height) and the pelvis starts at
    x = y = 0, the same world the fallback keypoints are in. GVHMR's global height drifts
    0.14-0.49 m over a broadcast clip, so the floor is tracked per frame, not one constant.

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

from fencing.motion.contacts import estimate_ground_track
from fencing.motion.io import JOINT_NAMES
from fencing.pose.joint_frames import BALL_OF_FOOT_HEIGHT_M
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
GVHMR_FPS = 30.0  # GVHMR works on the clip cut at 30 fps (fencing.pose.gvhmr)
GVHMR_POSE_DIMS = {"body_pose": 63, "global_orient": 3, "transl": 3}
FOOT_JOINTS = ("left_foot", "right_foot")


def fallback_frames(pose_dir: Path) -> tuple[list[dict], float]:
    from fencing.pose.joint_frames import compute_joint_frames, frames_to_gmr_dicts

    with np.load(pose_dir / "keypoints_world.npz") as data:
        keypoints = np.asarray(data["keypoints"], dtype=np.float64)
        fps = float(data["fps"])
    positions, quats = compute_joint_frames(keypoints)
    return frames_to_gmr_dicts(positions, quats), fps


def gvhmr_smplx_params(smpl_params_global: dict) -> dict:
    """GVHMR's smpl_params_global (numpy arrays) -> the smplx_data dict GMR's
    get_gvhmr_data_offline_fast reads. Betas are averaged over the clip."""
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in smpl_params_global.items()}
    missing = [k for k in (*GVHMR_POSE_DIMS, "betas") if k not in arrays]
    if missing:
        raise KeyError(f"smpl_params_global lacks {missing} (has {sorted(arrays)})")
    frames = len(arrays["body_pose"])
    for key, dim in GVHMR_POSE_DIMS.items():
        if arrays[key].shape != (frames, dim):
            raise ValueError(f"smpl_params_global[{key!r}] has shape {arrays[key].shape}, expected ({frames}, {dim})")
    betas = arrays["betas"]
    if betas.ndim == 2:
        betas = betas.mean(axis=0)
    return {
        "pose_body": arrays["body_pose"],
        "betas": betas,
        "root_orient": arrays["global_orient"],
        "trans": arrays["transl"],
        "mocap_frame_rate": np.array(GVHMR_FPS),
    }


def height_from_betas(betas: np.ndarray) -> float:
    """GMR's rule (load_gvhmr_pred_file): 1.66 m + 0.1 m per unit of the first beta."""
    return float(1.66 + 0.1 * np.asarray(betas).reshape(-1)[0])


def ground_frames(
    frames: list[dict], foot_height: float = BALL_OF_FOOT_HEIGHT_M, fps: float = GVHMR_FPS
) -> tuple[list[dict], np.ndarray]:
    """Shift z-up GMR frames so the floor is at z = 0 under every frame and the pelvis starts
    at x = y = 0. The floor is `fencing motion`'s first-round ground track (the lower foot's
    minimum over ~1 s, smoothed): it follows slow drift of the whole body but not a step or a
    short jump. Returns (frames, the subtracted (T, 3) shift)."""
    feet = np.array([[frame[j][0][2] for j in FOOT_JOINTS] for frame in frames])
    floor = estimate_ground_track(feet, fps) - foot_height
    start = np.asarray(frames[0]["pelvis"][0], dtype=np.float64)
    shift = np.column_stack([np.full(len(frames), start[0]), np.full(len(frames), start[1]), floor])
    shifted = [
        {k: (np.asarray(p, dtype=np.float64) - shift[i], q) for k, (p, q) in frame.items()} for i, frame in enumerate(frames)
    ]
    return shifted, shift


def gvhmr_frames(pose_dir: Path, body_models: Path) -> tuple[list[dict], float, float, np.ndarray]:
    import smplx
    import torch
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast

    # Our own GVHMR output (fencing.pose.gvhmr); it also holds non-tensor intermediate outputs.
    pred = torch.load(pose_dir / "hmr4d_results.pt", map_location="cpu", weights_only=False)
    params = gvhmr_smplx_params({k: v.numpy() for k, v in pred["smpl_params_global"].items()})
    n = len(params["pose_body"])
    body_model = smplx.create(str(body_models), "smplx", gender="neutral", use_pca=False, num_betas=len(params["betas"]))
    output = body_model(
        betas=torch.tensor(params["betas"]).float().view(1, -1),
        global_orient=torch.tensor(params["root_orient"]).float(),
        body_pose=torch.tensor(params["pose_body"]).float(),
        transl=torch.tensor(params["trans"]).float(),
        left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
        jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
        return_full_pose=True,
    )  # fmt: skip
    frames, fps = get_gvhmr_data_offline_fast(params, body_model, output, tgt_fps=int(GVHMR_FPS))
    frames, shift = ground_frames(frames)
    return frames, float(fps), height_from_betas(params["betas"]), shift


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
    extra_stats = {}
    if args.method == "gvhmr":
        if args.body_models is None:
            raise ValueError("--method gvhmr needs --body-models (the folder holding smplx/SMPLX_NEUTRAL.npz)")
        frames, fps, gvhmr_height, shift = gvhmr_frames(args.pose_dir, args.body_models)
        height = height or gvhmr_height
        extra_stats = {
            "betas_height_m": round(gvhmr_height, 3),
            "ground_shift_m": [round(float(v), 4) for v in (shift[0, 0], shift[0, 1], np.median(shift[:, 2]))],
            "ground_drift_m": round(float(np.ptp(shift[:, 2])), 4),
        }
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
        **extra_stats,
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
