#!/usr/bin/env python3
"""Where an entry policy put its feet in an MTC recording (docs/mtc.md, "Entry from standby").

    PYTHONPATH=src ~/hcf-data/venvs/fencing/bin/python scripts/entry_foot_placement.py \
        <entry export dir> <bag.npz> [--activation activation_t]

Not a gate: foot placement is measured, not thresholded. For each foot it prints:

  * end xy error: the foot's xy displacement from the first policy tick to 1 s into the end hold,
    minus the reference's own displacement over the same frames. A held run starts the robot on
    the reference, so the robot and the reference start from the same stance.
  * lift: the highest the foot rose above its starting height while the reference ran, next to
    the reference's own lift.
  * end stance: front (right) foot minus back (left) foot, in the anchor's heading frame, against
    the reference's last frame.

`--activation` names the recording's activation-times key for the entry: `activation_t` (held
and standby runs, and the entry in a chained run). For the motion in a chained run use
`activation1_t` with the motion's export dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from fencing.sim2sim import mtc, mtc_plant
from fencing.sim2sim import rollout as sim
from fencing.sim2sim.observation import MujocoState
from fencing.sim2sim.rollout import FOOT_BODIES

END_HOLD_PROBE_S = 1.0


def heading(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def in_heading_frame(vector_xy, yaw: float) -> np.ndarray:
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * vector_xy[0] - s * vector_xy[1], s * vector_xy[0] + c * vector_xy[1]])


def foot_placement(export_dir: Path, bag: Path, activation_key: str = "activation_t") -> dict:
    export = sim.load_export(export_dir)
    policy = sim.OnnxPolicy(export.onnx)
    contract = policy.contract
    model = mujoco.MjModel.from_xml_path(str(export.scene_xml))
    state = MujocoState(model, contract)
    recording = mtc.load_recording(bag)
    replay = mtc.Replayer(model, state, recording,
                          *mtc.site_in_body(mtc_plant.MODELS_DIR / mtc_plant.SABER_MJCF_NAME))
    bodies = list(contract.body_names)
    foot_ids = [state.body_ids[bodies.index(name)] for name in FOOT_BODIES]
    foot_ref = [bodies.index(name) for name in FOOT_BODIES]
    anchor = bodies.index(contract.anchor_body_name)
    frames = int(export.info["motion_frames"])
    reference = policy.reference(frames)
    period = float(export.info.get("step_dt", 0.02))

    start = float(recording[activation_key][0])
    replay.set_state(start)
    feet_start = replay.data.xpos[foot_ids].copy()
    lift = np.zeros(2)
    for k in range(frames):
        replay.set_state(start + period * k)
        lift = np.maximum(lift, replay.data.xpos[foot_ids][:, 2] - feet_start[:, 2])
    replay.set_state(start + period * (frames - 1) + END_HOLD_PROBE_S)
    feet_end = replay.data.xpos[foot_ids].copy()
    yaw = heading(state.anchor_pose(replay.data)[1])

    ref_start, ref_end = reference.body_pos_w[0][foot_ref], reference.body_pos_w[frames - 1][foot_ref]
    error = np.linalg.norm((feet_end - feet_start)[:, :2] - (ref_end - ref_start)[:, :2], axis=1)
    ref_lift = (reference.body_pos_w[:, foot_ref, 2] - ref_start[:, 2]).max(axis=0)
    stance = in_heading_frame(feet_end[1, :2] - feet_end[0, :2], yaw)
    ref_stance = in_heading_frame(ref_end[1, :2] - ref_end[0, :2], heading(reference.body_quat_w[frames - 1, anchor]))
    return {"end_xy_error_m": {"left": float(error[0]), "right": float(error[1])},
            "lift_m": {"left": float(lift[0]), "right": float(lift[1])},
            "reference_lift_m": {"left": float(ref_lift[0]), "right": float(ref_lift[1])},
            "end_stance_m": stance.round(4).tolist(), "reference_end_stance_m": ref_stance.round(4).tolist()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--activation", default="activation_t")
    args = parser.parse_args(argv)
    r = foot_placement(args.export_dir, args.bag, args.activation)
    e, lift, ref = r["end_xy_error_m"], r["lift_m"], r["reference_lift_m"]
    print(f"{args.export_dir.parent.name}: end xy error L {100 * e['left']:.1f} R {100 * e['right']:.1f} cm | "
          f"lift L {100 * lift['left']:.1f} R {100 * lift['right']:.1f} cm (reference {100 * ref['left']:.1f} "
          f"{100 * ref['right']:.1f}) | stance {r['end_stance_m']} vs reference {r['reference_end_stance_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
