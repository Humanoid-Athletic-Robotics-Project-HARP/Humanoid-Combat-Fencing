"""`fencing sim2sim`: run an exported ONNX policy in plain CPU MuJoCo (no mjlab, no torch).

    fencing sim2sim <export-dir | run-dir | run name> [--checkpoint N] [--motion NPZ]
                    [--seconds S] [--hold-seconds 2] [--pd actuator|explicit] [--video]
                    [--engine mujoco|physx]

The export dir is what `fencing train export` writes (policy.onnx, scene/scene.xml,
export.json). Given a run instead, we use paths.EVAL/<run dir>/<checkpoint>/. The
observation is rebuilt in Python from the robot state and the reference motion (the one
bundled in the ONNX unless --motion is given), driven only by the ONNX metadata.

Writes <export dir>/sim2sim.json with the Goal 1 metrics (E=1) and, when mjlab's
metrics.json for the same checkpoint exists, the mjlab-vs-MuJoCo gap.

`--engine physx` runs the same rollout and scoring in CPU PhysX (fencing.sim2sim.physx_rollout)
as an independent-engine cross-check and writes <export dir>/sim2sim_physx.json. It needs
ovphysx, which is not a dependency of this package (see docs/evaluation.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fencing.train import naming, thresholds

GAP_METRICS = ("success_rate", "joint_angle_error_rad", "body_pos_error_m", "body_pos_error_global_m",
               "anchor_pos_error_m", "anchor_ori_error_rad", "foot_slip_m_s", "recovery_rate",
               "blade_tip_error_m", "blade_tip_error_key_m")
ENGINES = ("mujoco", "physx")
SIMULATORS = {"mujoco": "mujoco (CPU, python bindings)", "physx": "physx (ovphysx CPU, no Kit, TGS)"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing sim2sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="export dir, run dir, run dir name or run name")
    parser.add_argument("--checkpoint", help="with a run: iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--motion", type=Path, help="reference NPZ (default: the motion bundled in the ONNX)")
    parser.add_argument("--seconds", type=float, help="score only the first S seconds of the motion")
    parser.add_argument("--hold-seconds", type=float, default=2.0,
                        help="hold the last reference frame this long to measure recovery (0: skip)")
    parser.add_argument("--pd", choices=("actuator", "explicit"), default="actuator",
                        help="actuator: the scene's PD actuators (what mjlab simulates); explicit: torques "
                             "kp(q*-q)-kd*qd from the metadata each physics step (deployment-style)")
    parser.add_argument("--reset-joint-noise", type=float, default=0.0,
                        help="uniform joint noise at reset (mjlab play uses 0.1); default 0 = deterministic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="also write sim2sim.mp4 (EGL offscreen)")
    parser.add_argument("--video-seconds", type=float, default=10.0, help="length of the video (default 10)")
    parser.add_argument("--engine", choices=ENGINES, default="mujoco",
                        help="mujoco (default) or physx: the same rollout and scoring in CPU PhysX (ovphysx, "
                             "separate venv; see docs/evaluation.md), written to sim2sim_physx.json")
    parser.add_argument("--out", type=Path,
                        help="output JSON (default: <export dir>/sim2sim.json, sim2sim_physx.json with physx)")
    parser.add_argument("--thresholds", type=Path, help="JSON file overriding docs/evaluation.md thresholds")
    parser.add_argument("--threshold", action="append", metavar="KEY=VALUE", help="override one threshold")
    return parser


def default_output(export_dir: Path, engine: str) -> Path:
    """Each engine has its own report, so a PhysX run never overwrites the MuJoCo one."""
    return export_dir / ("sim2sim.json" if engine == "mujoco" else f"sim2sim_{engine}.json")


def engine_option_errors(args: argparse.Namespace) -> list[str]:
    if args.engine != "physx":
        return []
    errors = []
    if args.video:
        errors.append("--engine physx does not render video (drop --video)")
    if args.pd != "actuator":
        errors.append("--engine physx uses PhysX's implicit joint drives (--pd actuator only)")
    return errors


def resolve_export_dir(target: str, checkpoint: str | None) -> Path:
    candidate = Path(target).expanduser()
    if (candidate / "policy.onnx").is_file():
        if checkpoint:
            raise ValueError("--checkpoint only applies when the target is a run, not an export dir")
        return candidate.resolve()
    run_dir = naming.resolve_run_dir(target)
    return naming.eval_dir(run_dir, naming.resolve_checkpoint(run_dir, checkpoint))


def mjlab_gap(sim2sim_nominal: dict, metrics_json: Path) -> dict | None:
    """sim2sim minus mjlab for the nominal metrics both report."""
    if not metrics_json.is_file():
        return None
    report = json.loads(metrics_json.read_text())
    mjlab_nominal = report.get("nominal") or {}  # None when eval ran without the nominal pass
    gap = {}
    for key in GAP_METRICS:
        a, b = mjlab_nominal.get(key), sim2sim_nominal.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            gap[key] = {"mjlab": a, "mujoco": b, "diff": b - a}
    # Both sides must score the same span of the motion for the diff to mean anything.
    return {"metrics_json": str(metrics_json), "mjlab_seconds": report.get("seconds"),
            "mjlab_num_envs": mjlab_nominal.get("num_envs"), "metrics": gap}


def summary_line(nominal: dict) -> str:
    def fmt(key, scale=1.0, unit=""):
        value = nominal.get(key)
        return "n/a" if value is None else f"{value * scale:.3g}{unit}"

    line = (f"success {fmt('success_rate')} | joint {fmt('joint_angle_error_rad')} rad | "
            f"body {fmt('body_pos_error_m', 100, ' cm')} | anchor {fmt('anchor_pos_error_m', 100, ' cm')} | "
            f"slip {fmt('foot_slip_m_s')} m/s | recovery {fmt('recovery_rate')} | "
            f"blade {fmt('blade_tip_error_m', 100, ' cm')}")
    if "blade_key_frames" in nominal:
        line += f" | blade at key {fmt('blade_tip_error_key_m', 100, ' cm')}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    errors = engine_option_errors(args)
    if errors:
        parser.error("; ".join(errors))
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from fencing.sim2sim import rollout as sim

    export_dir = resolve_export_dir(args.target, args.checkpoint)
    export = sim.load_export(export_dir)
    policy = sim.OnnxPolicy(export.onnx)
    reference = sim.reference_from_npz(args.motion, policy.contract.body_names) if args.motion else None
    key_frames = sim.export_key_frames(export, args.motion, reference.num_frames if reference else None)
    config = sim.RolloutConfig(
        seconds=args.seconds, hold_seconds=args.hold_seconds, pd_mode=args.pd,
        reset_joint_noise=args.reset_joint_noise, seed=args.seed,
        video=export_dir / "sim2sim.mp4" if args.video else None, video_seconds=args.video_seconds,
        stop_on_termination=True, key_frames=key_frames.steps if key_frames else None,
    )
    print(f"[sim2sim] {export.onnx} on {export.scene_xml} (engine={args.engine}, pd={args.pd})", flush=True)
    if args.engine == "physx":
        from fencing.sim2sim.physx_rollout import SCENE_USD_NAME, rollout_physx

        out_dir = (args.out or default_output(export_dir, args.engine)).parent
        result = rollout_physx(export, reference, config, policy, usd_path=out_dir / SCENE_USD_NAME)
    else:
        result = sim.rollout(export, reference, config, policy)
    checks = thresholds.check(result, thresholds.load_thresholds(args.thresholds, args.threshold))
    nominal_checks = [c for c in checks if c.metric.startswith("nominal.")]
    report = {
        "kind": "sim2sim",
        "export_dir": str(export_dir),
        "onnx": str(export.onnx),
        "scene_xml": str(export.scene_xml),
        "run_dir": export.info.get("run_dir"),
        "checkpoint": export.info.get("checkpoint"),
        "iteration": export.info.get("iteration"),
        "task": export.info.get("task"),
        "simulator": SIMULATORS[args.engine],
        **result,
        "gap_vs_mjlab": mjlab_gap(result["nominal"], export_dir / "metrics.json"),
        "thresholds": [c.__dict__ for c in nominal_checks],
        "verdict": thresholds.verdict(nominal_checks, thresholds.not_applicable(result),
                                      thresholds.incomplete_reasons(result)),
    }
    if key_frames:
        report["key_frames"] = key_frames.report()
    out = args.out or default_output(export_dir, args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if result["terminated_at_step"] is not None:
        print(f"[sim2sim] terminated at {result['terminated_at_s']:.2f} s ({', '.join(result['termination_terms'])})")
    print(f"[sim2sim] {summary_line(result['nominal'])}")
    if report["gap_vs_mjlab"]:
        gaps = ", ".join(f"{k} {v['diff']:+.3g}" for k, v in report["gap_vs_mjlab"]["metrics"].items())
        print(f"[sim2sim] gap vs mjlab ({args.engine} - mjlab): {gaps}")
        if report["gap_vs_mjlab"]["mjlab_seconds"] != args.seconds:
            print(f"[sim2sim] WARNING: metrics.json scored --seconds {report['gap_vs_mjlab']['mjlab_seconds']}, "
                  f"this run --seconds {args.seconds}; the gap compares different spans")
    for check in nominal_checks:
        print(f"[sim2sim]   {check.line()}")
    if result["video"]:
        print(f"[sim2sim] video: {result['video']}")
    print(f"[sim2sim] {report['verdict']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
