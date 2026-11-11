"""`fencing pipeline`: run the Goal 1 stages end to end, unattended.

    fencing pipeline data [--scrape 40] [--publish]     # (scrape) -> screen -> pose -> retarget -> motion batch
    fencing pipeline train motions/en_garde_v1.csv motions/advance_v1.csv [--iterations 5000]
    fencing pipeline train --goal1 [--iterations 5000]  # the newest version of every Goal 1 motion
    fencing pipeline all --scrape 40 --iterations 5000  # data (with --publish), then train --goal1
    fencing pipeline status                             # what each Goal 1 motion has: CSV, run, verdicts

Every stage is the ordinary `fencing <stage>` command run as a subprocess, so each keeps its
own venv handling (`fencing train` re-executes itself in the mjlab venv) and one failing stage
or motion never stops the rest. Training is a sequential queue on the one GPU: for each motion
train (foreground) -> export -> eval -> sim2sim, then a row is appended to docs/log.md and the
results to paths.EVAL/pipeline/<timestamp>.json.

Task choice (`--task auto`): bladework motions (extension, parry_*, cut_*, thrust,
advance_lunge_cut) train on the saber task `Fencing-Tracking-Flat-G1-Saber` (ADR 0002); the
footwork motions (en_garde, advance, retreat, lunge, advance_lunge) on the plain G1, as the
roadmap trains motions 1-4 without the saber first.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from fencing import paths

SABER_TASK = "Fencing-Tracking-Flat-G1-Saber"

# Roadmap Goal 1 order (docs/roadmap.md, Phase 1.5): each motion should pass before the next.
GOAL1_MOTIONS = (
    "en_garde", "advance", "retreat", "extension", "lunge",
    "parry_prime", "parry_seconde", "parry_tierce_high", "parry_quinte", "parry_sixte",
    "cut_head", "cut_flank", "cut_chest", "thrust",
    "advance_lunge", "advance_lunge_cut",
)
BLADE_PREFIXES = ("extension", "parry_", "cut_", "thrust", "advance_lunge_cut")
_VERSIONED = re.compile(r"^(?P<motion>[a-z0-9_]+?)_v(?P<version>\d+)$")


def log(message: str) -> None:
    print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def motion_base(name: str) -> str:
    """`parry_quinte_v3` -> `parry_quinte`; a name without a version is returned as is."""
    match = _VERSIONED.match(name)
    return match.group("motion") if match else name


def task_for(motion: str) -> str:
    return SABER_TASK if motion_base(motion).startswith(BLADE_PREFIXES) else paths.TRACKING_TASK


def latest_versions(motions_dir: Path, names: tuple[str, ...] = GOAL1_MOTIONS) -> dict[str, Path]:
    """Newest `<name>_v<N>.csv` in motions_dir for each name that has one (numeric version order)."""
    found: dict[str, tuple[int, Path]] = {}
    for csv in motions_dir.glob("*_v*.csv"):
        match = _VERSIONED.match(csv.stem)
        if not match or match.group("motion") not in names:
            continue
        motion, version = match.group("motion"), int(match.group("version"))
        if motion not in found or version > found[motion][0]:
            found[motion] = (version, csv)
    return {name: found[name][1] for name in names if name in found}


def fencing_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "fencing.cli", *args]


def run_stage(name: str, args: list[str], dry_run: bool = False) -> bool:
    """Run one `fencing ...` command; True on exit 0. Output streams straight through."""
    log(f"{name}: {' '.join(args)}")
    if dry_run:
        return True
    start = time.time()
    code = subprocess.run(fencing_cmd(*args)).returncode
    log(f"{name}: exit {code} after {time.time() - start:.0f}s")
    return code == 0


# ------------------------------------------------------------------ data


def cmd_data(args: argparse.Namespace) -> int:
    ok = True
    if args.scrape:
        ok &= run_stage("scrape", ["scrape", "run", "--max-downloads", str(args.scrape)], args.dry_run)
    if not args.skip_screen:
        ok &= run_stage("screen", ["screen", "--all"], args.dry_run)
    pose = ["pose", "--all"] + (["--max-clips", str(args.max_clips)] if args.max_clips else [])
    ok &= run_stage("pose", pose, args.dry_run)
    ok &= run_stage("retarget", ["retarget", "--all"], args.dry_run)
    batch = ["motion", "batch", str(paths.RETARGET / "csv")] + (["--publish"] if args.publish else [])
    ok &= run_stage("motion", batch, args.dry_run)
    if args.blade:
        ok &= run_stage("blade", blade_args(args.publish), args.dry_run)
    return 0 if ok else 1


def blade_args(publish: bool) -> list[str]:
    """Regenerate the bladework motions on the newest video-derived bases, if there are any."""
    bases = latest_versions(paths.MOTIONS_DIR, ("en_garde", "advance_lunge"))
    cmd = ["motion", "blade-all"]
    if "en_garde" in bases:
        cmd += ["--base-en-garde", str(bases["en_garde"])]
    if "advance_lunge" in bases:
        cmd += ["--base-advance-lunge", str(bases["advance_lunge"])]
    return cmd + (["--publish"] if publish else [])


# ------------------------------------------------------------------ train


def run_dirs_named(run_name: str) -> list[Path]:
    from fencing.train.naming import run_name_from_dir

    return sorted(d for d in paths.RUNS_LOG_ROOT.glob("*/*") if d.is_dir() and run_name_from_dir(d) == run_name)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def git_short_sha() -> str:
    result = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)
    sha = result.stdout.strip() or "?"
    dirty = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return sha + ("+dirty" if dirty else "")


def summarize_result(result: dict) -> str:
    """One `Result` cell for docs/log.md from a pipeline result record."""
    parts = []
    metrics = result.get("metrics") or {}
    nominal = metrics.get("nominal") or {}
    if metrics:
        parts.append(f"mjlab {metrics.get('verdict', '?')}")
    if nominal:
        def shown(key: str, fmt: str, scale: float = 1.0, unit: str = "") -> str:
            value = nominal.get(key)  # None when every env fell before scoring or no foot planted
            return "n/a" if value is None else f"{scale * value:{fmt}}{unit}"

        parts.append(f"success {shown('success_rate', '.2f')} of {nominal.get('num_episodes', '?')}")
        parts.append(f"joint {shown('joint_angle_error_rad', '.3f', unit=' rad')}")
        parts.append(f"body {shown('body_pos_error_m', '.1f', 100, ' cm')}")
        parts.append(f"slip {shown('foot_slip_m_s', '.3f', unit=' m/s')}")
        if nominal.get("blade_tip_error_m") is not None:
            parts.append(f"blade tip {100 * nominal['blade_tip_error_m']:.1f} cm")
    sim2sim = result.get("sim2sim") or {}
    if sim2sim:
        parts.append(f"sim2sim {sim2sim.get('verdict', '?')}")
    failed = [s for s, ok in (result.get("stages") or {}).items() if not ok]
    if failed:
        parts.append("failed stages: " + ", ".join(failed))
    return "; ".join(parts) or "no result"


def append_log_row(result: dict, log_md: Path = paths.DOCS_DIR / "log.md") -> None:
    """Add one row to the Runs table of docs/log.md (the project rule: log every run)."""
    run_dir = Path(result.get("run_dir") or "?")
    row = "| {date} | mjlab | {motion} | {data} | {commit} | — (local tensorboard: `{run}`) | {res} | {notes} |".format(
        date=datetime.now().strftime("%Y-%m-%d"),
        motion=motion_base(result["motion"]),
        data=result["motion"],
        commit=result.get("commit", "?"),
        run=run_dir.name,
        res=summarize_result(result),
        notes=f"`fencing pipeline train`, task {result['task']}, {result['iterations']} it, {result['num_envs']} envs",
    )
    text = log_md.read_text(encoding="utf-8")
    placeholder = "| | | | | | | | |"
    if placeholder in text:
        text = text.replace(placeholder, row, 1)
    else:
        text = text.rstrip("\n") + "\n" + row + "\n"
    log_md.write_text(text, encoding="utf-8")


def train_one(csv: Path, args: argparse.Namespace) -> dict:
    from fencing.train.naming import eval_dir, motion_name_from_path, resolve_checkpoint
    from fencing.train.naming import run_name as make_run_name

    motion = motion_name_from_path(csv)
    task = args.task if args.task != "auto" else task_for(motion)
    run_name = make_run_name(motion, args.tag)
    result: dict = {"motion": motion, "csv": str(csv), "task": task, "run_name": run_name,
                    "iterations": args.iterations, "num_envs": args.num_envs, "commit": git_short_sha(),
                    "started": datetime.now().isoformat(timespec="seconds"), "stages": {}}
    before = set(run_dirs_named(run_name))
    train = ["train", "run", str(csv), "--name", run_name, "--task", task, "--num-envs", str(args.num_envs),
             "--iterations", str(args.iterations), "--foreground"]
    result["stages"]["train"] = run_stage(f"train {motion}", train, args.dry_run)
    if args.dry_run:
        return result
    new_dirs = [d for d in run_dirs_named(run_name) if d not in before]
    if not new_dirs:
        log(f"train {motion}: no new run dir for {run_name}; skipping evaluation")
        return result
    run_dir = new_dirs[-1]
    result["run_dir"] = str(run_dir)
    try:
        checkpoint = resolve_checkpoint(run_dir, None)
    except FileNotFoundError as exc:
        log(f"train {motion}: {exc}")
        return result
    result["checkpoint"] = checkpoint.name
    ckpt = str(checkpoint.name)
    result["stages"]["export"] = run_stage(f"export {motion}", ["train", "export", str(run_dir), "--checkpoint", ckpt])
    result["stages"]["eval"] = run_stage(
        f"eval {motion}", ["train", "eval", str(run_dir), "--checkpoint", ckpt, "--num-envs", str(args.eval_envs)])
    result["stages"]["sim2sim"] = run_stage(f"sim2sim {motion}", ["sim2sim", str(run_dir), "--checkpoint", ckpt])
    out = eval_dir(run_dir, checkpoint)
    result["metrics"] = read_json(out / "metrics.json")
    result["sim2sim"] = {k: v for k, v in read_json(out / "sim2sim.json").items()
                         if k in ("verdict", "nominal", "gap_vs_mjlab", "terminated_at_s")}
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    return result


def cmd_train(args: argparse.Namespace) -> int:
    csvs = [Path(p).expanduser().resolve() for p in args.motions]
    if args.goal1:
        csvs += list(latest_versions(paths.MOTIONS_DIR).values())
    csvs = list(dict.fromkeys(csvs))
    if not csvs:
        log(f"nothing to train: name motion CSVs or pass --goal1 (looked in {paths.MOTIONS_DIR})")
        return 2
    missing = [c for c in csvs if not c.is_file()]
    if missing:
        log("missing motion files: " + ", ".join(map(str, missing)))
        return 2
    log("queue: " + ", ".join(c.stem for c in csvs))
    out_dir = paths.EVAL / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    results = []
    for csv in csvs:
        result = train_one(csv, args)
        results.append(result)
        if not args.dry_run:
            summary_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
            if not args.no_log:  # failed runs too: the log records failures as well as passes
                append_log_row(result)
            log(f"{result['motion']}: {summarize_result(result)}")
    ok = all(all(r["stages"].values()) for r in results)
    if not args.dry_run:
        log(f"summary -> {summary_path}")
    return 0 if ok else 1


# ------------------------------------------------------------------ status


def motion_status(motion: str, csv: Path) -> dict:
    from fencing.train.naming import list_checkpoints

    runs = sorted((d for d in paths.RUNS_LOG_ROOT.glob("*/*") if d.is_dir() and d.name.split("_", 2)[-1].startswith(csv.stem)),
                  key=lambda d: d.name)
    status = {"motion": motion, "csv": csv.name, "task": task_for(motion), "runs": len(runs)}
    if runs:
        run = runs[-1]
        ckpts = list_checkpoints(run)
        status["latest_run"] = run.name
        status["latest_checkpoint"] = ckpts[-1].name if ckpts else None
        if ckpts:
            ev = paths.EVAL / run.name / ckpts[-1].stem
            status["mjlab"] = read_json(ev / "metrics.json").get("verdict")
            status["sim2sim"] = read_json(ev / "sim2sim.json").get("verdict")
    return status


def cmd_status(args: argparse.Namespace) -> int:
    versions = latest_versions(paths.MOTIONS_DIR)
    for motion in GOAL1_MOTIONS:
        if motion not in versions:
            print(f"{motion:18s} no published CSV")
            continue
        s = motion_status(motion, versions[motion])
        print(f"{motion:18s} {s['csv']:26s} task={'saber' if s['task'] == SABER_TASK else 'g1':5s} runs={s['runs']} "
              f"ckpt={s.get('latest_checkpoint')} mjlab={s.get('mjlab')} sim2sim={s.get('sim2sim')}")
    return 0


# ------------------------------------------------------------------ cli


def _add_train_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--iterations", type=int, default=5000, help="PPO iterations per motion (default 5000)")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--eval-envs", type=int, default=1024, help="episodes per eval pass (default 1024)")
    p.add_argument("--task", default="auto", help="task id, or auto (saber task for bladework motions)")
    p.add_argument("--tag", default="pipe", help="run version tag: run name <motion>_<tag> (default pipe)")
    p.add_argument("--no-log", action="store_true", help="don't append rows to docs/log.md")


def _add_data_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scrape", type=int, default=0, metavar="N", help="first scrape and download up to N new videos")
    p.add_argument("--skip-screen", action="store_true")
    p.add_argument("--max-clips", type=int, default=0, help="pose: at most this many clips per video")
    p.add_argument("--publish", action="store_true", help="publish passing motions to motions/ (new versions)")
    p.add_argument("--blade", action="store_true", help="also regenerate bladework motions on video bases")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the stage commands without running them")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="(scrape) -> screen -> pose -> retarget -> motion batch")
    _add_data_options(p)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("train", help="train -> export -> eval -> sim2sim for each motion, one after another")
    p.add_argument("motions", nargs="*", help="motion CSVs")
    p.add_argument("--goal1", action="store_true", help="add the newest version of every Goal 1 motion in motions/")
    _add_train_options(p)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("all", help="data --publish --blade, then train --goal1")
    _add_data_options(p)
    _add_train_options(p)
    p.set_defaults(func=cmd_all)
    p = sub.add_parser("status", help="per Goal 1 motion: published CSV, runs, latest verdicts")
    p.set_defaults(func=cmd_status)
    return parser


def cmd_all(args: argparse.Namespace) -> int:
    args.publish = True
    args.blade = True
    data_ok = cmd_data(args) == 0
    args.motions, args.goal1 = [], True
    return 0 if (cmd_train(args) == 0 and data_ok) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
