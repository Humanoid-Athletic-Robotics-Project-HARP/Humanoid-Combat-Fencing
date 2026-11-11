#!/usr/bin/env python3
"""Copy the results that live under $HCF_DATA (~/hcf-data) into the repo's results/ folder.

    python3 scripts/collect_results.py [--dry-run]

Safe to re-run: a file is copied only when its size or mtime differs. Nothing is ever
deleted from results/, and nothing under $HCF_DATA is changed.

What is copied, and what is left out on purpose, is listed in results/README.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HCF_DATA", Path.home() / "hcf-data"))
OUT = REPO / "results"

# LAFAN1 is CC BY-NC-ND: no derivatives may be redistributed. The motion NPZ is a derivative,
# and the exported ONNX bundles the reference motion, so both stay out. Metrics, configs and
# the weights-only checkpoint are kept.
LAFAN_RUN = re.compile(r"dance1|lafan", re.IGNORECASE)
GITHUB_FILE_LIMIT = 95 * 1024 * 1024


class Collector:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.copied = 0
        self.copied_bytes = 0
        self.kept = 0
        self.skipped: list[str] = []

    def copy(self, src: Path, dst: Path, *, gz: bool = False) -> None:
        if gz:
            dst = dst.with_name(dst.name + ".gz")
        st = src.stat()
        if not gz and st.st_size > GITHUB_FILE_LIMIT:
            self.skipped.append(f"{src} ({st.st_size / 1e6:.0f} MB, over GitHub's file limit)")
            return
        if dst.exists():
            dt = dst.stat()
            same_time = abs(dt.st_mtime - st.st_mtime) < 2
            if same_time and (gz or dt.st_size == st.st_size):
                self.kept += 1
                return
        self.copied += 1
        self.copied_bytes += st.st_size
        if self.dry_run:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if gz:
            with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
            os.utime(dst, (st.st_atime, st.st_mtime))
        else:
            shutil.copy2(src, dst)

    def tree(self, src: Path, dst: Path, *, skip=lambda p: False, gz=lambda p: False) -> None:
        if not src.exists():
            return
        for path in sorted(src.rglob("*")):
            if path.is_file() and not path.is_symlink() and not skip(path):
                self.copy(path, dst / path.relative_to(src), gz=gz(path))


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def collect(c: Collector) -> None:
    # Evaluated policies: ONNX, export contract, mjlab / sim2sim / MTC scores, scene XML, renders.
    # The G1 mesh STLs are the same upstream files in every export (about 20 MB each time).
    for run in sorted((DATA / "eval").glob("*")):
        if run.name == "pipeline" or not run.is_dir():
            continue
        lafan = bool(LAFAN_RUN.search(run.name))
        c.tree(
            run,
            OUT / "policies" / run.name,
            skip=lambda p, lafan=lafan: p.suffix.lower() == ".stl" or (lafan and p.suffix == ".onnx"),
        )
    c.tree(DATA / "eval" / "pipeline", OUT / "pipeline_summaries")

    # The rendered rollouts again, flat and named by run, so they are easy to find and play.
    for video in sorted((DATA / "eval").glob("*/model_*/sim2sim.mp4")):
        run_name = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", video.parents[1].name)
        if not LAFAN_RUN.search(run_name):
            c.copy(video, OUT / "videos" / f"{run_name}.mp4")

    # Training runs: config, provenance, tensorboard curve, the last checkpoint.
    for run in sorted((DATA / "runs" / "rsl_rl").glob("*/*")):
        if not run.is_dir():
            continue
        dst = OUT / "policies" / run.name / "training"
        lafan = bool(LAFAN_RUN.search(run.name))
        checkpoints = sorted(run.glob("model_*.pt"), key=checkpoint_number)
        last = checkpoints[-1] if checkpoints else None
        has_exported_policy = any((DATA / "eval" / run.name).glob("model_*/policy.onnx"))
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pt" and path != last:
                continue  # intermediate checkpoints: 1.4 GB, reproducible from the config
            if path.suffix == ".onnx" and (lafan or has_exported_policy):
                continue  # the same network is already there as model_N/policy.onnx
            c.copy(path, dst / path.relative_to(run))
        if not c.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "task.txt").write_text(run.parent.name + "\n")

    # Reference motions as mjlab reads them (50 Hz NPZ + provenance).
    c.tree(DATA / "npz", OUT / "motion_npz", skip=lambda p: bool(LAFAN_RUN.search(p.name)))

    # Video -> pose -> retarget -> cleanup intermediates. Overlay PNGs are video frames of
    # real people, so they stay out like the videos themselves.
    c.tree(
        DATA / "pose",
        OUT / "pose",
        skip=lambda p: p.suffix in {".lock", ".png", ".jpg", ".mp4"},
    )
    c.tree(DATA / "retarget", OUT / "retarget")
    c.tree(DATA / "motion", OUT / "motion_work")
    for name in ("manifest.jsonl", "search_state.json", "shortlist.json"):
        if (DATA / "videos" / name).exists():
            c.copy(DATA / "videos" / name, OUT / "video_screening" / name)

    # motion_tracking_controller sim-to-sim: plant models, recordings, launch logs.
    c.tree(DATA / "mtc" / "models", OUT / "mtc" / "models")
    c.tree(
        DATA / "mtc" / "runs",
        OUT / "mtc" / "recordings",
        skip=lambda p: bool(LAFAN_RUN.search(str(p))) and p.suffix == ".npz",
        gz=lambda p: p.suffix == ".log",
    )

    # Every training, pipeline, pose and setup log, gzipped.
    c.tree(DATA / "logs", OUT / "training_logs", gz=lambda p: True)


SUMMARY_METRICS = [
    ("success_rate", "success", 1.0, "{:.2f}"),
    ("recovery_rate", "recovery", 1.0, "{:.2f}"),
    ("joint_angle_error_rad", "joint_rad", 1.0, "{:.3f}"),
    ("body_pos_error_m", "body_cm", 100.0, "{:.1f}"),
    ("anchor_pos_error_m", "anchor_cm", 100.0, "{:.1f}"),
    ("foot_slip_m_s", "slip_m_s", 1.0, "{:.3f}"),
    ("blade_tip_error_m", "tip_cm", 100.0, "{:.1f}"),
    ("blade_tip_error_key_m", "tip_key_cm", 100.0, "{:.1f}"),
]
SUMMARY_SOURCES = [("mjlab", "metrics.json"), ("sim2sim", "sim2sim.json"), ("mtc", "mtc.json")]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}") from error


def summary_rows() -> list[dict[str, str]]:
    """One row per evaluated checkpoint and simulator, read from results/policies."""
    rows = []
    for model_dir in sorted((OUT / "policies").glob("*/model_*")):
        run = model_dir.parent.name
        export = load_json(model_dir / "export.json") or {}
        motion = Path(export.get("motion_file", "")).stem.removesuffix("_mjlab")
        for source, filename in SUMMARY_SOURCES:
            report = load_json(model_dir / filename)
            if report is None:
                continue
            nominal = report.get("nominal", {})
            row = {
                "run": run,
                "checkpoint": model_dir.name,
                "motion": motion,
                "task": report.get("task") or export.get("task", ""),
                "simulator": source,
                "verdict": report.get("verdict", ""),
            }
            for key, column, scale, fmt in SUMMARY_METRICS:
                value = nominal.get(key)
                row[column] = fmt.format(value * scale) if isinstance(value, (int, float)) else ""
            robustness = report.get("robustness", {})
            for name in ("push", "obs_noise", "friction", "saber_mass"):
                value = robustness.get(name, {}).get("success_rate")
                row[f"robust_{name}"] = f"{value:.2f}" if isinstance(value, (int, float)) else ""
            rows.append(row)
    return rows


def write_summary() -> int:
    rows = summary_rows()
    if not rows:
        return 0
    columns = list(rows[0])
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    shown = [c for c in columns if c not in {"checkpoint", "task"} and not c.startswith("robust_")]
    lines = [
        "# Results summary",
        "",
        "Generated by `scripts/collect_results.py` from the score files under `policies/`. Do not edit by hand.",
        "One row per evaluated checkpoint and simulator: `mjlab` (GPU training simulator, 1024 episodes),",
        "`sim2sim` (plain CPU MuJoCo, one episode), `mtc` (`motion_tracking_controller`'s C++ controller and",
        "MuJoCo sim, one episode). Thresholds and metric definitions: [`docs/evaluation.md`](../docs/evaluation.md).",
        "Robustness pass rates and the task are in [`summary.csv`](summary.csv). Empty cell = not measured.",
        "",
        "| " + " | ".join(shown) + " |",
        "|" + "---|" * len(shown),
    ]
    lines += ["| " + " | ".join(row[c] for c in shown) + " |" for row in rows]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count what would be copied")
    args = parser.parse_args()
    c = Collector(args.dry_run)
    collect(c)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {c.copied} files ({c.copied_bytes / 1e6:.0f} MB before compression), {c.kept} already up to date")
    for line in c.skipped:
        print("skipped:", line)
    if not args.dry_run:
        print(f"summary: {write_summary()} rows -> {OUT / 'SUMMARY.md'} and summary.csv")


if __name__ == "__main__":
    main()
