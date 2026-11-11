"""Print docs/mtc.md's results tables from the mtc.json and sim2sim.json files on disk.

    python -m fencing.sim2sim.mtc_table <eval dir>...

One MTC row and one Python row per eval dir (`<run>/model_<N>`), so the doc cannot drift from
the data: regenerate the tables with this rather than editing numbers by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HEADER = ("| motion (eval dir) | source | joint (rad) | body (cm) | blade tip (cm) | anchor (cm) | slip (m/s) "
          "| success | recovery | verdict |\n|---|---|---|---|---|---|---|---|---|---|")
# (key, scale, decimals) in the order of HEADER's metric columns.
COLUMNS = (("joint_angle_error_rad", 1, 3), ("body_pos_error_m", 100, 2), ("blade_tip_error_m", 100, 2),
           ("anchor_pos_error_m", 100, 2), ("foot_slip_m_s", 1, 3))


def metric_cells(nominal: dict) -> list[str]:
    cells = ["n/a" if nominal.get(k) is None else f"{nominal[k] * scale:.{digits}f}" for k, scale, digits in COLUMNS]
    return cells + [f"{nominal[k]:g}" if nominal.get(k) is not None else "n/a" for k in ("success_rate", "recovery_rate")]


def _motion(eval_dir: Path) -> str:
    # Run dirs are named <date>_<time>_<motion>.
    return eval_dir.parent.name.split("_", 2)[-1]


def rows(eval_dir: Path) -> list[str]:
    eval_dir = Path(eval_dir)
    label = f"{_motion(eval_dir)} (`{eval_dir.parent.name}/{eval_dir.name}`)"
    out = []
    mtc_json = eval_dir / "mtc.json"
    if mtc_json.is_file():
        report = json.loads(mtc_json.read_text())
        out.append(f"| {label} | MTC | " + " | ".join(metric_cells(report["nominal"])) + f" | {report['verdict']} |")
    else:
        out.append(f"| {label} | MTC | no mtc.json |||||||| |")
    sim2sim_json = eval_dir / "sim2sim.json"
    if sim2sim_json.is_file():
        report = json.loads(sim2sim_json.read_text())
        out.append("| | Python | " + " | ".join(metric_cells(report["nominal"])) + f" | {report['verdict']} |")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fencing.sim2sim.mtc_table", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(HEADER)
    for eval_dir in args.eval_dirs:
        print("\n".join(rows(eval_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
