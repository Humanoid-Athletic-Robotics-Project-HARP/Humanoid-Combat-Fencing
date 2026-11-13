"""`fencing pipeline`: run the Goal 1 stages end to end, unattended.

    fencing pipeline data [--scrape 40] [--publish]     # (scrape) -> screen -> pose -> retarget -> motion batch
    fencing pipeline train motions/en_garde_v1.csv motions/advance_v1.csv [--iterations 5000]
    fencing pipeline train --goal1 [--iterations 5000]  # the newest version of every Goal 1 motion
    fencing pipeline all --scrape 40 --iterations 5000  # data (with --publish), then train --goal1
    fencing pipeline status                             # what each Goal 1 motion has: CSV, run, verdicts
    fencing pipeline queue --goal1 [--mtc] [--dry-run]  # two trainings at a time, resumable, verdict table

Every stage is the ordinary `fencing <stage>` command run as a subprocess, so each keeps its
own venv handling (`fencing train` re-executes itself in the mjlab venv) and one failing stage
or motion never stops the rest. `train` is a sequential queue on the one GPU: for each motion
train (foreground) -> export -> eval -> sim2sim, then a row is appended to docs/log.md and the
results to paths.EVAL/pipeline/<timestamp>.json.

`queue` runs the same chain (sim2sim with --video) for at most two motions at a time, each
motion's output in paths.LOGS/queue/<timestamp>/<label>.log. It is resumable: a run of the
same motion file (name and version, any tag) on the same task (run.json), whose last checkpoint
is >= --iterations - 1 and whose metrics.json and sim2sim.json parse with a verdict, makes the
motion skipped; a finished run without those is only evaluated. With --mtc,
scripts/mtc_sim2sim.sh then runs one motion at a time, after every training is done (the MTC run
is CPU-timed and invalid under load), skipping evals with a valid (not INVALID) mtc.json. A
failed or INVALID MTC run makes the queue exit 1 and shows in the verdict table.

Task choice (`--task auto`): bladework motions (extension, parry_*, cut_*, thrust,
advance_lunge_cut) train on the saber task `Fencing-Tracking-Flat-G1-Saber` (ADR 0002); the
footwork motions (en_garde, advance, retreat, lunge, advance_lunge) on the plain G1, as the
roadmap trains motions 1-4 without the saber first. stand_to_en_garde has no default in code.
Four tasks have been trained for it: the saber task (run tag `tip`),
`Fencing-Tracking-Flat-G1-Saber-Entry` (the saber task plus the entry reward, tag `feet`), and that
task plus a joint-angle reward at weight 1.0 or 0.5 (`-Entry-Joint`, `-Entry-Joint05`; tags
`feetjoint`, `feetjoint05`). The simulation evidence recommends `feetjoint05`: it alone passes in
every simulator and chains all ten saber motions (docs/mtc.md, "Paid for joint angles").
`queue` trains tip and feet unless --entry-variant names one; `train` needs an explicit --task.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fencing import paths

SABER_TASK = "Fencing-Tracking-Flat-G1-Saber"
ENTRY_TASK = "Fencing-Tracking-Flat-G1-Saber-Entry"  # fencing.saber.mjlab_task.ENTRY_TASK_ID (needs mjlab to import)
ENTRY_MOTIONS = ("stand_to_en_garde",)
# The two stand_to_en_garde variants `queue` trains by default (run tag -> task), docs/log.md.
ENTRY_VARIANTS = {"tip": SABER_TASK, "feet": ENTRY_TASK}
# The entry task plus a joint-angle reward (fencing.saber.mjlab_task.ENTRY_JOINT_TASK_IDS), trained
# only when --entry-variant names one; "both" stays tip + feet (docs/mtc.md, "Paid for joint angles").
ENTRY_JOINT_VARIANTS = {"feetjoint": "Fencing-Tracking-Flat-G1-Saber-Entry-Joint",
                        "feetjoint05": "Fencing-Tracking-Flat-G1-Saber-Entry-Joint05"}
ENTRY_VARIANT_TASKS = {**ENTRY_VARIANTS, **ENTRY_JOINT_VARIANTS}
# Recommended by simulation only (docs/mtc.md); named in messages, never applied as a default.
RECOMMENDED_ENTRY_VARIANT = "feetjoint05"
MAX_GPU_TRAININGS = 2
MTC_SCRIPT = paths.REPO_ROOT / "scripts" / "mtc_sim2sim.sh"

# Roadmap Goal 1 order (docs/roadmap.md, Phase 1.5): each motion should pass before the next.
GOAL1_MOTIONS = (
    "en_garde", "advance", "retreat", "extension", "lunge",
    "parry_prime", "parry_seconde", "parry_tierce_high", "parry_quinte", "parry_sixte",
    "cut_head", "cut_flank", "cut_chest", "thrust",
    "advance_lunge", "advance_lunge_cut",
)
QUEUE_MOTIONS = GOAL1_MOTIONS + ENTRY_MOTIONS  # `queue --goal1`: the 17 motions of motions/recipes/goal1.toml
BLADE_PREFIXES = ("extension", "parry_", "cut_", "thrust", "advance_lunge_cut")
_VERSIONED = re.compile(r"^(?P<motion>[a-z0-9_]+?)_v(?P<version>\d+)$")


def log(message: str) -> None:
    print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def motion_base(name: str) -> str:
    """`parry_quinte_v3` -> `parry_quinte`; a name without a version is returned as is."""
    match = _VERSIONED.match(name)
    return match.group("motion") if match else name


def task_for(motion: str) -> str:
    base = motion_base(motion)
    if base in ENTRY_MOTIONS:
        recommended = ENTRY_VARIANT_TASKS[RECOMMENDED_ENTRY_VARIANT]
        raise ValueError(f"{motion}: no default task; one is not chosen in code. The recommended entry "
                         f"is {recommended} (tag {RECOMMENDED_ENTRY_VARIANT}, docs/mtc.md). Pass --task, "
                         "or use `pipeline queue`, which trains tip and feet unless --entry-variant names one")
    return SABER_TASK if base.startswith(BLADE_PREFIXES) else paths.TRACKING_TASK


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


def run_stage(name: str, args: list[str], dry_run: bool = False, log_path: Path | None = None) -> bool:
    """Run one `fencing ...` command; True on exit 0. Output streams straight through, or into log_path."""
    log(f"{name}: {' '.join(args)}")
    if dry_run:
        return True
    start = time.time()
    if log_path is None:
        code = subprocess.run(fencing_cmd(*args)).returncode
    else:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ {shlex.join(fencing_cmd(*args))}\n")
            fh.flush()
            code = subprocess.run(fencing_cmd(*args), stdout=fh, stderr=subprocess.STDOUT).returncode
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


def train_args(csv: Path, run_name: str, task: str, args: argparse.Namespace) -> list[str]:
    return ["train", "run", str(csv), "--name", run_name, "--task", task, "--num-envs", str(args.num_envs),
            "--iterations", str(args.iterations), "--foreground"]


def export_args(run_dir: Path | str, ckpt: str) -> list[str]:
    return ["train", "export", str(run_dir), "--checkpoint", ckpt]


def eval_args(run_dir: Path | str, ckpt: str, args: argparse.Namespace) -> list[str]:
    return ["train", "eval", str(run_dir), "--checkpoint", ckpt, "--num-envs", str(args.eval_envs)]


def sim2sim_args(run_dir: Path | str, ckpt: str, video: bool = False) -> list[str]:
    return ["sim2sim", str(run_dir), "--checkpoint", ckpt] + (["--video"] if video else [])


def new_result(csv: Path, args: argparse.Namespace, task: str | None = None, run_name: str | None = None) -> dict:
    from fencing.train.naming import motion_name_from_path
    from fencing.train.naming import run_name as make_run_name

    motion = motion_name_from_path(csv)
    task = task or (args.task if args.task != "auto" else task_for(motion))
    return {"motion": motion, "csv": str(csv), "task": task, "run_name": run_name or make_run_name(motion, args.tag),
            "iterations": args.iterations, "num_envs": args.num_envs, "commit": git_short_sha(),
            "started": datetime.now().isoformat(timespec="seconds"), "stages": {}}


def train_one(csv: Path, args: argparse.Namespace, log_path: Path | None = None, task: str | None = None,
              run_name: str | None = None) -> dict:
    result = new_result(csv, args, task, run_name)
    motion, run_name = result["motion"], result["run_name"]
    before = set(run_dirs_named(run_name))
    train = train_args(csv, run_name, result["task"], args)
    result["stages"]["train"] = run_stage(f"train {motion}", train, args.dry_run, log_path)
    if args.dry_run:
        return result
    new_dirs = [d for d in run_dirs_named(run_name) if d not in before]
    if not new_dirs:
        log(f"train {motion}: no new run dir for {run_name}; skipping evaluation")
        return result
    return evaluate_run(result, new_dirs[-1], args, log_path)


def evaluate_run(result: dict, run_dir: Path, args: argparse.Namespace, log_path: Path | None = None) -> dict:
    """export -> eval -> sim2sim of the run's latest checkpoint; fills result in place and returns it."""
    from fencing.train.naming import eval_dir, resolve_checkpoint

    motion = result["motion"]
    result["run_dir"] = str(run_dir)
    try:
        checkpoint = resolve_checkpoint(run_dir, None)
    except FileNotFoundError as exc:
        log(f"train {motion}: {exc}")
        return result
    result["checkpoint"] = checkpoint.name
    ckpt = str(checkpoint.name)
    result["stages"]["export"] = run_stage(f"export {motion}", export_args(run_dir, ckpt), log_path=log_path)
    result["stages"]["eval"] = run_stage(f"eval {motion}", eval_args(run_dir, ckpt, args), log_path=log_path)
    result["stages"]["sim2sim"] = run_stage(f"sim2sim {motion}", sim2sim_args(run_dir, ckpt, getattr(args, "video", False)),
                                            log_path=log_path)
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
    if args.task == "auto":
        try:
            for csv in csvs:
                task_for(csv.stem)
        except ValueError as exc:
            log(str(exc))
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


# ------------------------------------------------------------------ queue


@dataclass
class QueueItem:
    csv: Path
    motion: str  # file stem, e.g. en_garde_v4
    task: str
    run_name: str  # what a new run is called: <motion>_<tag>
    state: str  # "train" | "evaluate" (trained, not evaluated) | "done" (evaluated)
    run_dir: Path | None = None
    checkpoint: Path | None = None
    tag: str = ""
    label: str = ""  # the motion, or <motion>_<tag> for a stand_to_en_garde variant

    def __post_init__(self) -> None:
        self.label = self.label or self.motion


def runs_of_motion(motion: str) -> list[Path]:
    """Run dirs of this motion file (name and version) under any tag: run name `<motion>_<tag>`."""
    from fencing.train.naming import run_name_from_dir

    return sorted(d for d in paths.RUNS_LOG_ROOT.glob("*/*")
                  if d.is_dir() and run_name_from_dir(d).startswith(f"{motion}_"))


def evaluated(out: Path) -> bool:
    """metrics.json and sim2sim.json both parse and carry a verdict (a half-written file is not an eval)."""
    return all(isinstance(read_json(out / name).get("verdict"), str) for name in ("metrics.json", "sim2sim.json"))


def mtc_valid(out: Path) -> bool:
    """mtc.json parses with a judged verdict; INVALID (dropped messages under load) is not a result."""
    verdict = read_json(out / "mtc.json").get("verdict")
    return isinstance(verdict, str) and verdict != "INVALID"


def _sha256(path: Path) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def queue_item(csv: Path, args: argparse.Namespace, task: str | None = None, tag: str | None = None,
               label: str = "") -> QueueItem:
    """Where a motion stands. Runs of this motion file under any tag count when their run.json names
    the same task (and, where recorded, the same CSV sha256) and the last checkpoint reaches
    --iterations - 1. The newest evaluated one makes it done; else the newest finished one is to be
    evaluated; else it trains. A run stopped short is not resumed: the motion trains again.
    """
    from fencing.train.naming import checkpoint_iteration, eval_dir, list_checkpoints, motion_name_from_path
    from fencing.train.naming import run_name as make_run_name

    motion = motion_name_from_path(csv)
    task = task or (args.task if args.task != "auto" else task_for(motion))
    tag = tag or args.tag
    item = QueueItem(csv, motion, task, make_run_name(motion, tag), "train", tag=tag, label=label)
    csv_sha = _sha256(csv)
    trained: tuple[Path, Path] | None = None
    for run_dir in sorted(runs_of_motion(motion), key=lambda d: d.name, reverse=True):
        meta = read_json(run_dir / "run.json")
        if meta.get("task") != task:  # another task, or no run.json (task unknown)
            continue
        if meta.get("motion_csv_sha256") and csv_sha and meta["motion_csv_sha256"] != csv_sha:
            continue
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints or checkpoint_iteration(checkpoints[-1]) + 1 < args.iterations:
            continue
        if evaluated(eval_dir(run_dir, checkpoints[-1])):
            item.state, item.run_dir, item.checkpoint = "done", run_dir, checkpoints[-1]
            return item
        trained = trained or (run_dir, checkpoints[-1])
    if trained:
        item.state, (item.run_dir, item.checkpoint) = "evaluate", trained
    return item


def queue_items(csvs: list[Path], args: argparse.Namespace) -> list[QueueItem]:
    """One item per motion; stand_to_en_garde as one item per variant (--entry-variant) unless --task is given."""
    items = []
    for csv in csvs:
        if args.task == "auto" and motion_base(csv.stem) in ENTRY_MOTIONS:
            variant = getattr(args, "entry_variant", "both")
            for tag in (ENTRY_VARIANTS if variant == "both" else [variant]):
                items.append(queue_item(csv, args, ENTRY_VARIANT_TASKS[tag], tag, label=f"{csv.stem}_{tag}"))
        else:
            items.append(queue_item(csv, args))
    return items


def refresh(item: QueueItem, args: argparse.Namespace) -> QueueItem:
    return queue_item(item.csv, args, item.task, item.tag or None, item.label)


def planned_stages(item: QueueItem, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """The `fencing ...` stages the queue will run for an item (placeholders for a run dir not made yet)."""
    if item.state == "done":
        return []
    run_dir = str(item.run_dir) if item.run_dir else f"<new run dir of {item.run_name}>"
    ckpt = item.checkpoint.name if item.checkpoint else f"model_{args.iterations - 1}.pt"
    stages = [("train", train_args(item.csv, item.run_name, item.task, args))] if item.state == "train" else []
    return stages + [("export", export_args(run_dir, ckpt)), ("eval", eval_args(run_dir, ckpt, args)),
                     ("sim2sim", sim2sim_args(run_dir, ckpt, video=True))]


def mtc_cmd(run_dir: Path | str, checkpoint: Path | str) -> list[str]:
    match = re.match(r"model_(\d+)\.pt$", Path(checkpoint).name)
    iteration = match.group(1) if match else str(checkpoint)
    return ["bash", str(MTC_SCRIPT), str(run_dir), "--checkpoint", iteration]


def needs_mtc(item: QueueItem) -> bool:
    from fencing.train.naming import eval_dir

    return not (item.state == "done" and mtc_valid(eval_dir(item.run_dir, item.checkpoint)))


def run_queue(items: list[QueueItem], work, max_parallel: int = MAX_GPU_TRAININGS, on_result=None) -> list[dict]:
    """work(item) -> result for every item not done, at most max_parallel at once (one GPU, two trainings)."""
    if not 1 <= max_parallel <= MAX_GPU_TRAININGS:
        raise ValueError(f"max_parallel must be 1..{MAX_GPU_TRAININGS} (one GPU), got {max_parallel}")

    def safe(item: QueueItem) -> dict:
        try:
            result = work(item)
        except Exception as exc:  # one motion failing must not stop the queue; the failure is reported
            log(f"{item.label}: {type(exc).__name__}: {exc}")
            result = {"motion": item.motion, "csv": str(item.csv), "task": item.task, "stages": {"queue": False},
                      "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("label", item.label)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(safe, item) for item in items if item.state != "done"]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result:
                on_result(result)
    return results


def execute_queue(items: list[QueueItem], work, mtc_work=None, max_parallel: int = MAX_GPU_TRAININGS,
                  on_result=None) -> tuple[list[dict], list[dict]]:
    """Trainings (max_parallel at once), then, if mtc_work is given, MTC one item at a time."""
    results = run_queue(items, work, max_parallel, on_result)
    mtc_results = [mtc_work(item) for item in items] if mtc_work is not None else []
    return results, mtc_results


def queue_failures(results: list[dict], mtc_results: list[dict]) -> list[str]:
    """Labels of what failed: a training chain with a failed stage, or an MTC run that failed or was INVALID."""
    failed = [r["label"] for r in results if not all(r.get("stages", {}).values())]
    return failed + [f"{r['label']} mtc" for r in mtc_results if not r.get("ok", True)]


def _verdict(path: Path) -> str:
    return read_json(path).get("verdict") or "-"


def verdict_rows(items: list[QueueItem], results: list[dict], args: argparse.Namespace,
                 mtc_results: list[dict] = ()) -> list[dict]:
    """One row per item from what is on disk now (so skipped motions show their earlier verdicts)."""
    from fencing.train.naming import eval_dir

    by_label = {r["label"]: r for r in results}
    mtc_by_label = {r["label"]: r for r in mtc_results}
    rows = []
    for item in items:
        now = refresh(item, args)
        result = by_label.get(item.label)
        if result is None:
            this_run = "skipped (evaluated before)" if item.state == "done" else "-"
        else:
            failed = [s for s, ok in result.get("stages", {}).items() if not ok]
            this_run = ("failed: " + ", ".join(failed)) if failed else ("evaluated" if item.state == "evaluate" else "trained")
        row = {"motion": item.label, "task": {SABER_TASK: "saber", ENTRY_TASK: "entry", **{task: "entry+joint" for task in ENTRY_JOINT_VARIANTS.values()}}.get(item.task, "g1"),
               "run": now.run_dir.name if now.run_dir else "-",
               "checkpoint": now.checkpoint.name if now.checkpoint else "-", "mjlab": "-", "sim2sim": "-", "mtc": "-",
               "this_run": this_run}
        if now.run_dir and now.checkpoint:
            out = eval_dir(now.run_dir, now.checkpoint)
            row.update(mjlab=_verdict(out / "metrics.json"), sim2sim=_verdict(out / "sim2sim.json"),
                       mtc=_verdict(out / "mtc.json"))
        mtc = mtc_by_label.get(item.label)
        if mtc and not mtc.get("ok", True):
            row["mtc"] = f"{row['mtc'] if row['mtc'] != '-' else 'failed'} ({mtc['mtc']})"
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    columns = ("motion", "task", "mjlab", "sim2sim", "mtc", "checkpoint", "this_run", "run")
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows]) for c in columns}
    lines = ["  ".join(f"{c:{widths[c]}}" for c in columns)]
    lines += ["  ".join(f"{str(r[c]):{widths[c]}}" for c in columns) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def _state_text(item: QueueItem) -> str:
    if item.state == "done":
        return f"skip, evaluated: {item.run_dir.name}/{item.checkpoint.name}"
    if item.state == "evaluate":
        return f"trained, evaluate {item.run_dir.name}/{item.checkpoint.name}"
    return "train"


def _queue_csvs(args: argparse.Namespace) -> list[Path]:
    csvs = [Path(p).expanduser().resolve() for p in args.motions]
    if args.goal1:
        csvs += list(latest_versions(paths.MOTIONS_DIR, QUEUE_MOTIONS).values())
    return list(dict.fromkeys(csvs))


def cmd_queue(args: argparse.Namespace) -> int:
    csvs = _queue_csvs(args)
    if not csvs:
        log(f"nothing to queue: name motion CSVs or pass --goal1 (looked in {paths.MOTIONS_DIR})")
        return 2
    missing = [c for c in csvs if not c.is_file()]
    if missing:
        log("missing motion files: " + ", ".join(map(str, missing)))
        return 2
    items = queue_items(csvs, args)
    if args.dry_run:
        for item in items:
            print(f"# {item.label} ({item.task}): {_state_text(item)}")
            for _, stage in planned_stages(item, args):
                print(shlex.join(fencing_cmd(*stage)))
        print(f"# at most {args.max_parallel} of the motions above at once; each motion's stages run in order")
        if args.mtc:
            print("# then MTC, one at a time, after every training (not where a valid mtc.json exists):")
            for item in filter(needs_mtc, items):
                run_dir = item.run_dir or f"<new run dir of {item.run_name}>"
                print(shlex.join(mtc_cmd(run_dir, item.checkpoint or f"model_{args.iterations - 1}.pt")))
        return 0

    log_dir = paths.LOGS / "queue" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    args.video = True
    log(f"queue: {', '.join(f'{i.label} ({i.state})' for i in items)}; logs -> {log_dir}")

    def work(item: QueueItem) -> dict:
        log_path = log_dir / f"{item.label}.log"
        if item.state == "train":
            return train_one(item.csv, args, log_path, item.task, item.run_name)
        return evaluate_run(new_result(item.csv, args, item.task, item.run_name), item.run_dir, args, log_path)

    def on_result(result: dict) -> None:  # runs in this thread (as_completed), so docs/log.md writes never race
        if not args.no_log and "error" not in result:
            append_log_row(result)
        log(f"{result['label']}: {summarize_result(result)}")

    def mtc_work(item: QueueItem) -> dict:
        from fencing.train.naming import eval_dir

        now = refresh(item, args)
        if now.state != "done":
            return {"label": item.label, "mtc": "not evaluated", "ok": False}
        if not needs_mtc(now):
            return {"label": item.label, "mtc": "skipped: valid mtc.json", "ok": True}
        cmd = mtc_cmd(now.run_dir, now.checkpoint)
        log(f"mtc {item.label}: {shlex.join(cmd)}")
        with open(log_dir / f"{item.label}.mtc.log", "a", encoding="utf-8") as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        ok = code == 0 and mtc_valid(eval_dir(now.run_dir, now.checkpoint))
        log(f"mtc {item.label}: exit {code}" + ("" if ok else " (failed or INVALID)"))
        return {"label": item.label, "mtc": f"exit {code}", "ok": ok}

    results, mtc_results = execute_queue(items, work, mtc_work if args.mtc else None, args.max_parallel, on_result)
    rows = verdict_rows(items, results, args, mtc_results)
    table = format_table(rows)
    (log_dir / "summary.json").write_text(json.dumps({"results": results, "mtc": mtc_results, "verdicts": rows},
                                                     indent=2, default=str) + "\n")
    (log_dir / "verdicts.txt").write_text(table + "\n")
    print(table)
    log(f"summary -> {log_dir / 'summary.json'}")
    failures = queue_failures(results, mtc_results)
    if failures:
        log("failed: " + ", ".join(failures))
    return 1 if failures else 0


# ------------------------------------------------------------------ cli


def _add_train_options(p: argparse.ArgumentParser, iterations: int = 5000) -> None:
    p.add_argument("--iterations", type=int, default=iterations, help=f"PPO iterations per motion (default {iterations})")
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
    p = sub.add_parser("queue", help="train -> export -> eval -> sim2sim --video, two motions at a time, "
                                     "resumable; then optional MTC one at a time; verdict table")
    p.add_argument("motions", nargs="*", help="motion CSVs")
    p.add_argument("--goal1", action="store_true",
                   help="add the newest version of every Goal 1 motion in motions/ (with stand_to_en_garde)")
    p.add_argument("--max-parallel", type=int, default=MAX_GPU_TRAININGS, choices=range(1, MAX_GPU_TRAININGS + 1),
                   help=f"motions trained at once (default and most {MAX_GPU_TRAININGS}: one GPU)")
    p.add_argument("--mtc", action="store_true", help="then scripts/mtc_sim2sim.sh, one motion at a time, after all training")
    p.add_argument("--entry-variant", choices=("both", *ENTRY_VARIANT_TASKS), default="both",
                   help="stand_to_en_garde: tip (saber task), feet (entry task), both (default: tip + feet), or "
                        "feetjoint / feetjoint05 (entry task + joint_pos 1.0 / 0.5; docs/mtc.md names the recommended entry)")
    _add_train_options(p, iterations=3000)  # what every Goal 1 policy was trained with
    p.set_defaults(func=cmd_queue)
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
