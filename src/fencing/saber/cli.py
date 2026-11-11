"""`fencing saber <command>`: saber model, wrist-reach analysis, renders and the GPU smoke test.

    fencing saber build-mjcf   # write src/fencing/saber/assets/g1_29dof_saber.xml (+ URDF snippet)
    fencing saber reach        # IK over the canonical parries/cuts -> ~/hcf-data/eval/saber/reach.json
    fencing saber render       # PNGs of those poses -> ~/hcf-data/eval/saber/renders/
    fencing saber smoke        # build + step the mjlab saber task on the GPU (mjlab venv)

build-mjcf, reach and render run in the fencing venv. smoke needs mjlab + torch; if they
are not importable here it re-runs itself with ~/hcf-data/venvs/mjlab/bin/python.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from fencing.paths import EVAL, REPO_ROOT, venv_python

SABER_EVAL = EVAL / "saber"


def _saber_from_args(args: argparse.Namespace):
    from fencing.saber.spec import NOMINAL_SABER

    saber = NOMINAL_SABER
    if getattr(args, "mount_pitch", None) is not None:
        saber = replace(saber, mount_pitch_deg=args.mount_pitch)
    if getattr(args, "blade_length", None) is not None:
        saber = replace(saber, blade_length=args.blade_length)
    if getattr(args, "mass", None) is not None:
        saber = saber.with_total_mass(args.mass)
    if getattr(args, "collision", False):
        saber = replace(saber, collision=True)
    return saber


def _add_saber_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mount-pitch", type=float, help="blade angle above the forearm axis, deg")
    parser.add_argument("--blade-length", type=float, help="blade length, m")
    parser.add_argument("--mass", type=float, help="total saber mass, kg (parts scale together)")


def cmd_build_mjcf(args: argparse.Namespace) -> int:
    import mujoco

    from fencing.saber.spec import mass_properties, tip_in_hand, write_mjcf, write_urdf_snippet

    saber = _saber_from_args(args)
    path = write_mjcf(args.out, saber)
    model = mujoco.MjModel.from_xml_path(str(path))  # prove plain MuJoCo loads the file
    snippet = write_urdf_snippet(args.urdf_out, saber)
    props = mass_properties(saber)
    print(f"wrote {path} (nbody={model.nbody}, nq={model.nq}) and {snippet}")
    print(f"saber: {saber.total_length:.3f} m, {props.mass:.3f} kg, COM {props.com[0]:.3f} m "
          f"from grip centre, tip in wrist frame {tip_in_hand(saber).round(3).tolist()}")
    return 0


def cmd_reach(args: argparse.Namespace) -> int:
    from fencing.saber.reach import markdown_table, run_reach_analysis

    saber = _saber_from_args(args)
    report = run_reach_analysis(saber, restarts=args.restarts, include_waist=not args.arm_only,
                                mount_sweep=() if args.no_sweep else _default_sweep())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "reach.json"
    out.write_text(json.dumps(report, indent=2))
    print(markdown_table(report))
    print("\nmount sweep (arm only):")
    for label, row in report["mount_pitch_sweep"].items():
        print(f"  {label}: {row['num_reachable']}/{len(report['targets'])} reachable; "
              f"not: {', '.join(row['unreachable']) or '-'}")
    print(f"\npayload at en garde: {json.dumps(report['payload']['static_torque_nm'])}")
    print(f"wrist cut torque estimate: {json.dumps(report['payload']['wrist_cut_torque_nm'])}")
    print(f"COM shift (mm): {json.dumps(report['payload']['com_shift_mm'])}")
    print(f"wrote {out}")
    return 0


def _default_sweep():
    from fencing.saber.reach import MOUNT_SWEEP

    return MOUNT_SWEEP


def cmd_render(args: argparse.Namespace) -> int:
    from fencing.saber.render import render_poses

    reach_json = args.reach_json
    if not reach_json.exists():
        print(f"{reach_json} not found; run `fencing saber reach` first", file=sys.stderr)
        return 1
    saber = _saber_from_args(args)
    for path in render_poses(reach_json, args.out_dir, saber, joint_set=args.joint_set):
        print(f"wrote {path}")
    return 0


def cmd_smoke(args: argparse.Namespace, raw_argv: list[str]) -> int:
    try:
        import mjlab  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        python = venv_python("mjlab")
        if not python.exists():
            print(f"mjlab not importable and {python} missing; set up the mjlab venv",
                  file=sys.stderr)
            return 1
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p)
        return subprocess.call([str(python), "-m", "fencing.saber.smoke", *raw_argv], env=env)
    from fencing.saber.smoke import main as smoke_main

    return smoke_main(raw_argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing saber", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    from fencing.saber.spec import DEFAULT_MJCF_OUT, DEFAULT_URDF_SNIPPET_OUT

    build = sub.add_parser("build-mjcf", help="write the G1 + saber MJCF and URDF snippet")
    _add_saber_options(build)
    build.add_argument("--collision", action="store_true", help="add the blade collision capsule")
    build.add_argument("--out", type=Path, default=DEFAULT_MJCF_OUT)
    build.add_argument("--urdf-out", type=Path, default=DEFAULT_URDF_SNIPPET_OUT)

    reach = sub.add_parser("reach", help="IK reachability of parries and cuts")
    _add_saber_options(reach)
    reach.add_argument("--out-dir", type=Path, default=SABER_EVAL)
    reach.add_argument("--restarts", type=int, default=32)
    reach.add_argument("--arm-only", action="store_true", help="skip the arm+waist pass")
    reach.add_argument("--no-sweep", action="store_true", help="skip the mount-angle sweep")

    render = sub.add_parser("render", help="offscreen PNGs of the reach poses")
    _add_saber_options(render)
    render.add_argument("--reach-json", type=Path, default=SABER_EVAL / "reach.json")
    render.add_argument("--out-dir", type=Path, default=SABER_EVAL / "renders")
    render.add_argument("--joint-set", choices=("arm", "arm+waist"), default="arm")

    sub.add_parser("smoke", help="GPU smoke test of the mjlab saber task (args: see --help)",
                   add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "smoke":
        # Forward everything after "smoke" untouched; fencing.saber.smoke owns those options.
        return cmd_smoke(argparse.Namespace(), argv[1:])
    args = build_parser().parse_args(argv)
    handlers = {"build-mjcf": cmd_build_mjcf, "reach": cmd_reach, "render": cmd_render}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
