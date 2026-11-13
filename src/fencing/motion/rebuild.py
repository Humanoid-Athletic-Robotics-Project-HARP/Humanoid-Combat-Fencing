"""`fencing motion rebuild`: rebuild published motions from the retarget CSVs with a recipe file.

    fencing motion rebuild [--recipe motions/recipes/goal1.toml] [--only lunge cut_head]
                           [--out DIR] [--compare] [--from-published en_garde_v1] [--publish]

A recipe (motions/recipes/*.toml, format in the file's header) lists, per motion, a source (a
retarget CSV, or another motion of the recipe) and the steps that turn it into the published
motion. Almost every step is one existing function (`OPS`): `clean` is fencing.motion.pipeline's
clean_motion, `blade` fencing.saber.recipes, `entry` fencing.motion.entry, `append_end_hold`
segment.hold_pose, the other fixes fencing.motion.fixes. The exception is `rear_arm_pose`, new
code here (`Rebuild._apply`): lunge_v3's rear arm was set by hand (the mean arm pose of
en_garde_v2, held in every frame) and no function did it; it reproduces lunge_v3 exactly.

Outputs go to a scratch directory (default HCF_DATA/motion/rebuild/<recipe>/): <motion>.csv,
<motion>.report.json (steps, `motion check` report, comparison) and checkpoints/<name>.csv.
motions/ is only written with --publish, and then only as new versions (publish_motion: next
free _vN, created exclusively), each after its own check passes and after everything it is
built on was published or found unchanged. A motion whose output is byte-identical to any of its
published versions is not published again; what is built on it names that existing file.

--from-published NAME starts from the published motions/NAME.csv instead of rebuilding up to
that checkpoint (or that final motion). It separates "the steps after NAME are reproducible"
from "the input of NAME still exists", and is for verification only: it cannot be combined
with --publish.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from fencing import paths
from fencing.motion.io import JOINT_NAMES, Motion, read_csv, read_sidecar, write_csv, write_sidecar

RECIPE_DIR = paths.MOTIONS_DIR / "recipes"
DEFAULT_RECIPE = RECIPE_DIR / "goal1.toml"
# The recipe of the current Goal 1 set (GVHMR retargets). `motion blade-all` takes each bladework
# recipe's blade_path from it, so the two cannot disagree about how a motion's blade moves.
GVHMR_RECIPE = RECIPE_DIR / "goal1_gvhmr.toml"
REBUILD_DIR = paths.DATA / "motion" / "rebuild"  # scratch outputs (never committed)
SCHEMA = 1
IDENTICAL_TOL = 1e-9
COLUMNS = ("root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw", *JOINT_NAMES)
FOOT_INDEX = {"left": 0, "right": 1}  # fencing.motion.kinematics foot order

# op -> (allowed keys, required keys). Every step may also carry `checkpoint`.
OPS: dict[str, tuple[frozenset, frozenset]] = {
    "clean": (frozenset({"label", "frames", "config"}), frozenset({"label"})),
    "set_foot_heading": (frozenset({"foot", "target_yaw_deg"}), frozenset({"foot"})),
    "append_end_hold": (frozenset({"seconds"}), frozenset({"seconds"})),
    "rear_arm_pose": (frozenset({"side", "pose_from"}), frozenset({"side", "pose_from"})),
    "clear_arm_torso_collision": (frozenset({"side"}), frozenset({"side"})),
    "neutralize_wrists": (frozenset({"sides", "axes"}), frozenset()),
    "blade": (frozenset({"recipe", "cut", "hit_at_s", "config"}), frozenset({"recipe"})),
    "entry": (frozenset({"frame"}), frozenset()),
}
# Ops that make a motion from its source rather than change one: first step only, and they need
# this kind of source.
FIRST_STEP_OPS = {"clean": "clip", "blade": "motion", "entry": "motion"}
MOTION_KEYS = frozenset({"source", "steps", "reproduces", "allow_warn", "tuned_on", "tied_to_mediapipe"})


class RecipeError(ValueError):
    """The recipe file is malformed or inconsistent."""


# ---------------------------------------------------------------- recipe


@dataclass
class Step:
    op: str
    params: dict
    checkpoint: str | None = None

    def describe(self) -> str:
        p = self.params
        if self.op == "clean":
            return f"clean[{p['frames'][0]}:{p['frames'][1]}]" if "frames" in p else "clean[whole clip]"
        if self.op == "blade":
            return f"blade {p['recipe']}"
        return self.op


@dataclass
class MotionRecipe:
    name: str
    source_kind: str  # "clip" | "motion"
    source: str  # clip id or motion name
    steps: list[Step]
    reproduces: str | None = None
    allow_warn: bool = False
    tuned_on: str | None = None
    tied_to_mediapipe: list[str] = field(default_factory=list)


@dataclass
class Recipe:
    path: Path
    motions: dict[str, MotionRecipe]

    @property
    def checkpoints(self) -> dict[str, str]:
        """checkpoint name -> the motion whose steps produce it."""
        return {s.checkpoint: m.name for m in self.motions.values() for s in m.steps if s.checkpoint}

    def owner(self, ref: str) -> str:
        """The motion a reference (motion name or checkpoint name) belongs to."""
        if ref in self.motions:
            return ref
        owner = self.checkpoints.get(ref)
        if owner is None:
            raise RecipeError(f"{self.path}: {ref!r} is neither a motion nor a checkpoint of this recipe")
        return owner

    def position(self, ref: str) -> tuple[str, int, str]:
        """(motion, number of steps run, last op) of a motion or checkpoint reference: where it is taken."""
        owner = self.owner(ref)
        steps = self.motions[owner].steps
        index = len(steps) if ref == owner else next(i + 1 for i, s in enumerate(steps) if s.checkpoint == ref)
        return owner, index, steps[index - 1].op

    def dependencies(self, name: str) -> list[str]:
        m = self.motions[name]
        refs = [m.source] if m.source_kind == "motion" else []
        refs += [s.params["pose_from"] for s in m.steps if "pose_from" in s.params]
        return list(dict.fromkeys(self.owner(r) for r in refs))

    def order(self, only: list[str] | None = None) -> list[str]:
        """Dependency order (file order where free); `only` adds what those motions need."""
        unknown = [n for n in only or [] if n not in self.motions]
        if unknown:
            raise RecipeError(f"unknown motions {unknown}; the recipe has {', '.join(self.motions)}")
        wanted = list(only) if only else list(self.motions)
        done: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise RecipeError(f"{self.path}: dependency cycle through {name!r}")
            visiting.add(name)
            for dep in self.dependencies(name):
                visit(dep)
            visiting.discard(name)
            done.append(name)

        for name in wanted:
            visit(name)
        return done


def _parse_step(where: str, raw: dict) -> Step:
    if not isinstance(raw, dict) or "op" not in raw:
        raise RecipeError(f"{where}: every step needs an `op`")
    op = raw["op"]
    if op not in OPS:
        raise RecipeError(f"{where}: unknown op {op!r}; known: {', '.join(OPS)}")
    allowed, required = OPS[op]
    params = {k: v for k, v in raw.items() if k not in ("op", "checkpoint")}
    extra = sorted(set(params) - allowed)
    missing = sorted(required - set(params))
    if extra:
        raise RecipeError(f"{where}: op {op!r} does not take {extra}; it takes {sorted(allowed)}")
    if missing:
        raise RecipeError(f"{where}: op {op!r} needs {missing}")
    frames = params.get("frames")
    if frames is not None and not (isinstance(frames, list) and len(frames) == 2
                                   and all(isinstance(v, int) for v in frames) and 0 <= frames[0] < frames[1]):
        raise RecipeError(f"{where}: frames must be [start, end] with 0 <= start < end, got {frames!r}")
    for key in ("foot", "side"):
        if key in params and params[key] not in FOOT_INDEX:
            raise RecipeError(f"{where}: {key} must be 'left' or 'right', got {params[key]!r}")
    checkpoint = raw.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise RecipeError(f"{where}: checkpoint must be a name, got {checkpoint!r}")
    return Step(op, params, checkpoint)


def load_recipe(path: Path | str = DEFAULT_RECIPE) -> Recipe:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"{path}: not valid TOML ({exc})") from exc
    if data.get("schema") != SCHEMA:
        raise RecipeError(f"{path}: schema must be {SCHEMA}, got {data.get('schema')!r}")
    from fencing.motion.publish import PublishError, check_motion_name

    motions: dict[str, MotionRecipe] = {}
    for name, raw in (data.get("motions") or {}).items():
        where = f"{path.name}: motions.{name}"
        try:
            check_motion_name(name, allow_unknown=True)
        except PublishError as exc:
            raise RecipeError(f"{where}: {exc}") from None
        extra = sorted(set(raw) - MOTION_KEYS)
        if extra:
            raise RecipeError(f"{where}: unknown keys {extra}; known: {sorted(MOTION_KEYS)}")
        kind, _, source = str(raw.get("source", "")).partition(":")
        if kind not in ("clip", "motion") or not source:
            raise RecipeError(f"{where}: source must be 'clip:<clip_id>' or 'motion:<name>', got {raw.get('source')!r}")
        steps = [_parse_step(f"{where}.steps[{i}]", s) for i, s in enumerate(raw.get("steps") or [])]
        if not steps:
            raise RecipeError(f"{where}: no steps")
        for i, step in enumerate(steps):
            need = FIRST_STEP_OPS.get(step.op)
            if need and (i != 0 or kind != need):
                raise RecipeError(f"{where}.steps[{i}]: {step.op!r} must be the first step of a motion with a {need}: source")
        motions[name] = MotionRecipe(name, kind, source, steps, raw.get("reproduces"), bool(raw.get("allow_warn", False)),
                                     raw.get("tuned_on"), list(raw.get("tied_to_mediapipe") or []))
    if not motions:
        raise RecipeError(f"{path}: no [motions.<name>] tables")
    recipe = Recipe(path, motions)
    seen: dict[str, str] = {}
    for m in motions.values():
        for s in m.steps:
            if not s.checkpoint:
                continue
            if s.checkpoint in motions or s.checkpoint in seen:
                raise RecipeError(f"{path}: checkpoint name {s.checkpoint!r} is used twice")
            seen[s.checkpoint] = m.name
    for m in motions.values():
        if m.source_kind == "motion" and m.source not in motions:
            raise RecipeError(f"{path}: motions.{m.name} is built on {m.source!r}, which the recipe does not define")
        recipe.dependencies(m.name)  # every pose_from resolves
    recipe.order()  # no cycles
    return recipe


def clean_config(values: dict):
    """CleanConfig from recipe keys: unknown keys refused; `false` on a non-bool field is None (off)."""
    from fencing.motion.pipeline import CleanConfig

    known = {f.name: f for f in fields(CleanConfig) if f.name not in ("contact", "thresholds")}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise RecipeError(f"unknown clean config keys {unknown}; known: {sorted(known)}")
    converted = {}
    for key, value in values.items():
        is_bool_field = isinstance(known[key].default, bool)
        converted[key] = None if (value is False and not is_bool_field) else value
    return CleanConfig(**converted)


def blade_config(values: dict):
    from fencing.saber.keyframes import BladeConfig

    known = {f.name for f in fields(BladeConfig) if f.name != "saber"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise RecipeError(f"unknown blade config keys {unknown}; known: {sorted(known)}")
    return BladeConfig(**values)


def blade_paths(recipe: Recipe) -> dict[str, str]:
    """Bladework recipe name (a blade step's `recipe`) -> the blade_path its config sets, for the steps
    that set one. Two steps of one bladework recipe that disagree are refused."""
    found: dict[str, str] = {}
    for m in recipe.motions.values():
        for step in m.steps:
            if step.op != "blade" or "blade_path" not in (step.params.get("config") or {}):
                continue
            name, value = step.params["recipe"], step.params["config"]["blade_path"]
            if found.setdefault(name, value) != value:
                raise RecipeError(f"{recipe.path}: blade recipe {name!r} sets blade_path {found[name]!r} and {value!r}")
    return found


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_columns(side: str) -> list[int]:
    return [i for i, n in enumerate(JOINT_NAMES)
            if n.startswith(f"{side}_") and any(k in n for k in ("shoulder", "elbow", "wrist"))]


# ---------------------------------------------------------------- comparison


def compare_csv(rebuilt: Path, published: Path) -> dict:
    """identical (bytes or every value within 1e-9) / differs (max abs and where) / shape / missing."""
    out = {"published": published.name}
    if not published.is_file():
        return {**out, "status": "missing"}
    if rebuilt.read_bytes() == published.read_bytes():
        return {**out, "status": "identical", "bytes_identical": True, "max_abs": 0.0}
    a = np.loadtxt(rebuilt, delimiter=",", ndmin=2)
    b = np.loadtxt(published, delimiter=",", ndmin=2)
    if a.shape != b.shape:
        return {**out, "status": "shape", "shape": list(a.shape), "published_shape": list(b.shape)}
    diff = np.abs(a - b)
    worst = float(diff.max())
    frame, column = np.unravel_index(int(diff.argmax()), diff.shape)
    status = "identical" if worst <= IDENTICAL_TOL else "differs"
    return {**out, "status": status, "bytes_identical": False, "max_abs": worst,
            "worst_column": COLUMNS[column], "worst_frame": int(frame)}


def compare_text(result: dict) -> str:
    name = result["published"]
    status = result["status"]
    if status == "missing":
        return f"vs {name}: not in motions/"
    if status == "shape":
        return f"vs {name}: shape {tuple(result['shape'])} vs {tuple(result['published_shape'])}"
    if status == "identical":
        return f"vs {name}: identical" + ("" if result.get("bytes_identical") else f" (max abs {result['max_abs']:.1e})")
    return f"vs {name}: max abs {result['max_abs']:.3g} ({result['worst_column']}, frame {result['worst_frame']})"


# ---------------------------------------------------------------- rebuild


@dataclass
class Outcome:
    name: str
    csv: Path | None = None
    steps: list[dict] = field(default_factory=list)
    quality: dict | None = None
    error: str | None = None
    source_csv: str | None = None
    source_sha256: str | None = None
    pose_method: str | None = None
    anchored: str | None = None
    compare: list[dict] = field(default_factory=list)
    retune: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)  # clean/blade/entry results kept for publishing

    def record(self) -> dict:
        return {"motion": self.name, "csv": str(self.csv) if self.csv else None, "error": self.error,
                "source_csv": self.source_csv, "source_sha256": self.source_sha256, "pose_method": self.pose_method,
                "anchored": self.anchored,
                "steps": self.steps, "quality": self.quality, "compare": self.compare,
                "re_choose_on_this_data": self.retune}


class Rebuild:
    def __init__(self, recipe: Recipe, out_dir: Path, csv_dir: Path | None = None, motions_dir: Path | None = None,
                 anchors: tuple[str, ...] = (), compare: bool = False):
        self.recipe = recipe
        self.out_dir = Path(out_dir)
        self.csv_dir = Path(csv_dir) if csv_dir else paths.RETARGET / "csv"
        self.motions_dir = Path(motions_dir) if motions_dir else paths.MOTIONS_DIR
        if self.out_dir.resolve().is_relative_to(self.motions_dir.resolve()):
            raise RecipeError(f"output dir {self.out_dir} is inside {self.motions_dir}; published motions are "
                              "written only by --publish, as new versions")
        known = set(recipe.checkpoints) | {m.reproduces for m in recipe.motions.values() if m.reproduces}
        unknown = [a for a in anchors if a not in known]
        if unknown:
            raise RecipeError(f"--from-published {unknown}: not a checkpoint or `reproduces` name of {recipe.path.name}")
        self.anchors = set(anchors)
        self.do_compare = compare
        self.outputs: dict[str, Path] = {}  # motion or checkpoint name -> CSV written by this run
        self.outcomes: dict[str, Outcome] = {}

    # ------------------------------------------------------------ driving

    def run(self, only: list[str] | None = None, on_outcome=None) -> list[Outcome]:
        for name in self.recipe.order(only):
            outcome = self.build(name)
            self.outcomes[name] = outcome
            if on_outcome:
                on_outcome(outcome)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_sidecar(self.out_dir / "summary.json", {"recipe": str(self.recipe.path),
                                                      "motions": [o.record() for o in self.outcomes.values()]})
        return list(self.outcomes.values())

    def build(self, name: str) -> Outcome:
        m = self.recipe.motions[name]
        outcome = Outcome(name)
        failed = [d for d in self.recipe.dependencies(name) if self.outcomes.get(d) is None or self.outcomes[d].error]
        if failed:
            outcome.error = f"not built: {', '.join(failed)} failed"
            return outcome
        try:
            motion = self._run_steps(m, outcome)
        except Exception as exc:  # one motion failing must not stop the rest; the error is reported
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome
        outcome.csv = self.out_dir / f"{name}.csv"
        if not (outcome.anchored and outcome.anchored == m.reproduces):
            # Publish writes this object, not a read-back of the CSV: write_csv normalizes the root
            # quaternion, so re-writing a read-back CSV can move a 6th decimal, and the published file
            # would then differ from the scratch CSV that `unchanged` and the dependants are judged on.
            outcome.extras["written"] = motion
            motion = self._round_trip(motion, outcome.csv)
        self.outputs[name] = outcome.csv
        from fencing.motion.quality import quality_report

        outcome.quality = quality_report(motion)  # what `fencing motion check` reports
        if self.do_compare:
            outcome.compare = self._comparisons(m, outcome)
        if m.tuned_on and outcome.pose_method and outcome.pose_method != m.tuned_on:
            outcome.retune = list(m.tied_to_mediapipe)
        write_sidecar(outcome.csv.with_suffix(".report.json"), outcome.record())
        return outcome

    # ------------------------------------------------------------ steps

    def _run_steps(self, m: MotionRecipe, outcome: Outcome) -> Motion:
        start, motion = self._anchor(m, outcome)
        if motion is None:
            if m.source_kind == "clip":
                raw_csv = self.csv_dir / f"{m.source}.csv"
                if not raw_csv.is_file():
                    raise FileNotFoundError(f"retarget CSV {raw_csv} not found")
                outcome.source_csv = str(raw_csv)
                outcome.source_sha256 = sha256_file(raw_csv)
                outcome.pose_method = (read_sidecar(raw_csv) or {}).get("pose_method")
                motion = read_csv(raw_csv)
            else:
                outcome.source_csv = str(self.outputs[m.source])
                outcome.source_sha256 = sha256_file(self.outputs[m.source])
                outcome.pose_method = self.outcomes[m.source].pose_method
                motion = read_csv(self.outputs[m.source])
        for step in m.steps[start:]:
            motion, info = self._apply(m, step, motion, outcome)
            entry = {"op": step.op, **step.params, "info": info}
            if step.checkpoint:
                path = self.out_dir / "checkpoints" / f"{step.checkpoint}.csv"
                motion = self._round_trip(motion, path)
                self.outputs[step.checkpoint] = path
                # A recipe name for this point of the chain, not a published file (it is not published).
                entry["recipe_checkpoint"] = step.checkpoint
                entry["checkpoint_sha256"] = sha256_file(path)
            outcome.steps.append(entry)
        return motion

    def _anchor(self, m: MotionRecipe, outcome: Outcome) -> tuple[int, Motion | None]:
        """(first step to run, starting motion) when --from-published names a point of this motion."""
        if m.reproduces in self.anchors:
            published = self.motions_dir / f"{m.reproduces}.csv"
            outcome.anchored = m.reproduces
            target = self.out_dir / f"{m.name}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(published, target)
            return len(m.steps), read_csv(target)
        for index in range(len(m.steps) - 1, -1, -1):
            checkpoint = m.steps[index].checkpoint
            if checkpoint in self.anchors:
                published = self.motions_dir / f"{checkpoint}.csv"
                path = self.out_dir / "checkpoints" / f"{checkpoint}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(published, path)
                self.outputs[checkpoint] = path
                outcome.anchored = checkpoint
                return index + 1, read_csv(path)
        return 0, None

    def _apply(self, m: MotionRecipe, step: Step, motion: Motion, outcome: Outcome) -> tuple[Motion, dict]:
        from fencing.motion import fixes

        p = step.params
        if step.op == "clean":
            from fencing.motion.pipeline import clean_motion

            config = clean_config(p.get("config") or {})
            frames = tuple(p["frames"]) if "frames" in p else None
            result = clean_motion(motion, config, label=p["label"], frames=frames)
            outcome.extras["clean"] = (result, config)
            return result.motion, {"segment": result.segment, "hand": result.hand, "hand_source": result.hand_source,
                                   "mirrored": result.mirrored, "verdict": result.report["verdict"],
                                   "steps": result.steps}
        if step.op == "set_foot_heading":
            return fixes.set_foot_heading(motion, FOOT_INDEX[p["foot"]], math.radians(p.get("target_yaw_deg", 0.0)))
        if step.op == "append_end_hold":
            from fencing.motion.segment import hold_pose

            out, _ = hold_pose(motion, 0.0, p["seconds"], 0.0)
            return out, {"seconds": p["seconds"], "appended_frames": out.n_frames - motion.n_frames}
        if step.op == "rear_arm_pose":
            columns = arm_columns(p["side"])
            pose_file = self.outputs[p["pose_from"]]
            owner, after_step, after_op = self.recipe.position(p["pose_from"])
            pose = read_csv(pose_file).joint_pos[:, columns].mean(axis=0)
            joint_pos = motion.joint_pos.copy()
            change = float(np.abs(joint_pos[:, columns] - pose).max())
            joint_pos[:, columns] = pose
            source = {"recipe_ref": p["pose_from"], "motion": owner, "after_step": after_step, "after_op": after_op,
                      "file": str(pose_file), "sha256": sha256_file(pose_file)}
            return motion.replace(joint_pos=joint_pos), {"pose_source": source, "statistic": "mean",
                                                         "joints": [JOINT_NAMES[c] for c in columns],
                                                         "pose_rad": pose.tolist(), "max_change_rad": change}
        if step.op == "clear_arm_torso_collision":
            return fixes.clear_arm_torso_collision(motion, p["side"])
        if step.op == "neutralize_wrists":
            return fixes.neutralize_wrists(motion, tuple(p.get("sides", ("left", "right"))),
                                           tuple(p.get("axes", ("pitch", "yaw"))))
        if step.op == "blade":
            from fencing.saber import recipes
            from fencing.saber.keyframes import script_summary

            base = recipes.load_base(recipes.base_kind(p["recipe"]), self.outputs[m.source])
            script = recipes.recipe_script(p["recipe"], base, p.get("cut", "cut_head"), p.get("hit_at_s"))
            result = recipes.run_script(m.name, base, script, blade_config(p.get("config") or {}))
            outcome.extras["blade"] = result
            s = result.blade.summary
            return result.blade.motion, {"script": script_summary(result.blade.script), "keys": recipes.key_table(result),
                                         "blade_ok": result.blade_ok, "keys_off_target": s["keys_off_target"],
                                         "collision_frames": s["collision_frames"]}
        if step.op == "entry":
            from fencing.motion import entry

            result = entry.build_entry(motion, frame=p.get("frame", 0))
            outcome.extras["entry"] = result
            return result.motion, {"frame": p.get("frame", 0), **result.report}
        raise RecipeError(f"no implementation for op {step.op!r}")  # load_recipe rejects unknown ops first

    @staticmethod
    def _round_trip(motion: Motion, path: Path) -> Motion:
        """Write the CSV and continue from what was written (the format rounds to 1e-6)."""
        write_csv(motion, path)
        return read_csv(path)

    def _comparisons(self, m: MotionRecipe, outcome: Outcome) -> list[dict]:
        results = []
        for step in m.steps:
            if step.checkpoint and step.checkpoint in self.outputs:
                results.append(compare_csv(self.outputs[step.checkpoint], self.motions_dir / f"{step.checkpoint}.csv"))
        if m.reproduces:
            results.append(compare_csv(outcome.csv, self.motions_dir / f"{m.reproduces}.csv"))
        return results

    # ------------------------------------------------------------ publish

    def newest_published(self, name: str) -> Path | None:
        from fencing.motion.publish import existing_versions

        versions = existing_versions(name, self.motions_dir)
        return self.motions_dir / f"{name}_v{versions[-1]}.csv" if versions else None

    def publish(self, allow_warn_all: bool = False) -> dict[str, PublishStatus]:
        """Publish every built motion in dependency order: a new version, `unchanged` (byte-identical to
        a published version, newest first, nothing written), or refused with the reason."""
        from fencing.motion.publish import identical_file

        if self.anchors:
            raise RecipeError("a --from-published rebuild is for verification only; it is never published")
        # motion -> the file in motions/ it now is (new or unchanged). A dependant was built on the
        # scratch CSV, which for an unchanged motion has the same bytes as this file, so it is
        # equally built on this file and its provenance names it.
        resolved: dict[str, PublishStatus] = {}
        status: dict[str, PublishStatus] = {}
        for name, outcome in self.outcomes.items():
            m = self.recipe.motions[name]
            missing = [d for d in self.recipe.dependencies(name) if d not in resolved]
            verdict = (outcome.quality or {}).get("verdict")
            allowed = ("pass", "warn") if (m.allow_warn or allow_warn_all) else ("pass",)
            same = None if outcome.error or outcome.csv is None else identical_file(outcome.csv, name, self.motions_dir)
            if outcome.error:
                status[name] = PublishStatus("refused", reason=outcome.error)
            elif missing:
                status[name] = PublishStatus("refused", reason=f"{', '.join(missing)} was not published")
            elif same is not None:
                status[name] = resolved[name] = PublishStatus("unchanged", same)
            elif verdict not in allowed:
                reasons = "; ".join((outcome.quality or {}).get("reasons", []))
                status[name] = PublishStatus("refused", reason=f"check {verdict} ({reasons})")
            elif "blade" in outcome.extras and not outcome.extras["blade"].blade_ok:
                status[name] = PublishStatus("refused", reason="blade keys off target or colliding")
            else:
                status[name] = resolved[name] = PublishStatus("new", self._publish_one(m, outcome, resolved))
        return status

    def _rebuild_record(self, m: MotionRecipe, outcome: Outcome, resolved: dict[str, PublishStatus]) -> dict:
        import copy

        record = {"recipe": _repo_relative(self.recipe.path), "motion": m.name, "reproduces": m.reproduces,
                  "source": f"{m.source_kind}:{m.source}", "pose_method": outcome.pose_method}
        if m.source_kind == "clip":
            record.update(source_csv=outcome.source_csv, source_sha256=outcome.source_sha256)
        else:
            base = resolved[m.source]
            record.update(base_file=base.path.name, base_status=base.kind, base_sha256=sha256_file(base.path))
        steps = copy.deepcopy(outcome.steps)
        for step in steps:
            pose = (step.get("info") or {}).get("pose_source")
            if pose:
                owner = resolved[pose["motion"]]
                pose.update(motion_published_as=owner.path.name, motion_publish_status=owner.kind)
        record["steps"] = steps
        return record

    def _publish_one(self, m: MotionRecipe, outcome: Outcome, resolved: dict[str, PublishStatus]) -> Path:
        from fencing.motion.publish import Provenance, publish_motion
        from fencing.motion.quality import summary_line

        motion = outcome.extras["written"]  # the object written to outcome.csv (see build)
        rebuild = self._rebuild_record(m, outcome, resolved)
        note = (f"rebuilt by `fencing motion rebuild` ({_repo_relative(self.recipe.path)}, motion {m.name}; "
                f"{outcome.pose_method or 'unknown'} retarget CSV"
                + (f"; recipe written to reproduce {m.reproduces}" if m.reproduces else "") + ")")
        if "blade" in outcome.extras:
            from fencing.saber import recipes

            result = outcome.extras["blade"]
            base_path = resolved[m.source].path
            result.base.path = _repo_relative(base_path)
            result.base.source = recipes._describe_source(base_path, result.base.motion)
            prov = recipes.provenance_for(result)
            prov.notes = f"{prov.notes}; {note}"
            return publish_motion(motion, m.name, prov, motions_dir=self.motions_dir,
                                  record={**result.record(), "rebuild": rebuild})
        if "entry" in outcome.extras:
            result = outcome.extras["entry"]
            frame = m.steps[0].params.get("frame", 0)
            target = resolved[m.source].path.name
            prov = Provenance(f"synthetic: standby_controller default_position -> {target} frame {frame}",
                              "whole motion", "R", "fencing.motion.entry (leg IK, CoM over the stance sole)",
                              f"{summary_line(outcome.quality)}; {note}")
            record = {"entry": {"target": target, "frame": frame, **result.report}, "quality": outcome.quality,
                      "rebuild": rebuild}
            return publish_motion(motion, m.name, prov, motions_dir=self.motions_dir, record=record,
                                  allow_unknown_name=True)
        prov = self._clean_chain_provenance(m, outcome, note, resolved)
        record = {"quality": outcome.quality, "rebuild": rebuild}
        if "clean" in outcome.extras:
            from fencing.motion.pipeline import config_dict

            result, config = outcome.extras["clean"]
            record.update({"config": config_dict(config), "hand": result.hand, "hand_source": result.hand_source,
                           "mirrored": result.mirrored, "segment": result.segment, "raw_quality": result.raw_report})
        return publish_motion(motion, m.name, prov, motions_dir=self.motions_dir, record=record,
                              allow_unknown_name=True)

    def _clean_chain_provenance(self, m: MotionRecipe, outcome: Outcome, note: str,
                                resolved: dict[str, PublishStatus]):
        from fencing.motion.publish import Provenance
        from fencing.motion.quality import summary_line

        if m.source_kind == "clip":
            source = (f"{m.source} ({outcome.pose_method or 'unknown'} retarget, "
                      f"sha256 {(outcome.source_sha256 or '')[:12]})")
        else:
            source = resolved[m.source].path.name
        segment, hand, cleanup = "whole clip", "R", []
        if "clean" in outcome.extras:
            from fencing.motion.pipeline import provenance_for

            result, config = outcome.extras["clean"]
            base = provenance_for(result, source, config)
            segment, hand, cleanup = base.segment, base.hand, [base.cleanup]
        run_steps = m.steps[len(m.steps) - len(outcome.steps):]  # all of them unless anchored (never published)
        infos = {id(s): e.get("info") or {} for s, e in zip(run_steps, outcome.steps)}
        for step in m.steps:
            p = step.params
            if step.op == "set_foot_heading":
                cleanup.append(f"{p['foot']} foot heading -> {p.get('target_yaw_deg', 0.0):g} deg")
            elif step.op == "append_end_hold":
                cleanup.append(f"end hold +{p['seconds']:g} s")
            elif step.op == "rear_arm_pose":
                # Name the published file and the point of the chain, never the recipe's checkpoint
                # name: that names a file of the old data, and the pose came from this run's chain.
                pose = infos[id(step)]["pose_source"]
                owner = resolved[pose["motion"]]
                cleanup.append(f"{p['side']} arm -> mean pose of {pose['motion']} after step {pose['after_step']} "
                               f"({pose['after_op']}), sha256 {pose['sha256'][:12]}; {pose['motion']} published as "
                               f"{owner.path.name}" + (" (unchanged)" if owner.kind == "unchanged" else ""))
            elif step.op == "clear_arm_torso_collision":
                cleanup.append(f"{p['side']} arm cleared of torso")
            elif step.op == "neutralize_wrists":
                cleanup.append("wrist pitch/yaw -> neutral")
        if segment != "whole clip":
            segment += f" + holds; {len(outcome.steps)} recipe steps"
        return Provenance(source, segment, hand, ", ".join(cleanup), f"{summary_line(outcome.quality)}; {note}")


@dataclass
class PublishStatus:
    kind: str  # "new" (written now) | "unchanged" (identical to a published version, nothing written) | "refused"
    path: Path | None = None
    reason: str | None = None

    def __str__(self) -> str:
        if self.kind == "new":
            return str(self.path)
        if self.kind == "unchanged":
            return f"unchanged: {self.path.name}"
        return f"not published: {self.reason}"


def _repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(paths.REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- report text


def outcome_lines(outcome: Outcome, recipe: Recipe) -> list[str]:
    from fencing.motion.quality import summary_line

    m = recipe.motions[outcome.name]
    if outcome.error:
        return [f"{outcome.name}: FAILED {outcome.error}"]
    source = m.source if m.source_kind == "motion" else f"{m.source}, {outcome.pose_method or 'pose method unknown'}"
    lines = [f"{outcome.name} -> {outcome.csv}  [{m.source_kind} {source}]"]
    shown = [s.describe() for s in m.steps]
    if outcome.anchored:
        lines.append(f"  from published {outcome.anchored}.csv; steps after it: "
                     + (" > ".join(s["op"] for s in outcome.steps) or "none"))
    else:
        lines.append("  steps: " + " > ".join(shown))
    for step in outcome.steps:
        info = step.get("info") or {}
        if step["op"] == "clean":
            seg = info.get("segment") or {}
            where = f"frames {seg['start']}-{seg['end']}" if seg else "whole clip"
            lines.append(f"  clean: {where}, hand {info.get('hand')} ({info.get('hand_source')}), "
                         f"mirrored {info.get('mirrored')}")
        if step["op"] == "blade":
            lines.append(f"  blade: [{info.get('script')}] {info.get('keys')}; on target {info.get('blade_ok')}")
    quality = outcome.quality or {}
    reasons = quality.get("reasons") or []
    lines.append(f"  check: {quality.get('verdict')} {summary_line(quality)}" + (f" [{'; '.join(reasons)}]" if reasons else ""))
    for result in outcome.compare:
        lines.append("  " + compare_text(result))
    if outcome.retune:
        lines.append(f"  RE-CHOOSE on {outcome.pose_method} data (recipe tuned on {m.tuned_on}):")
        lines.extend(f"    - {text}" for text in outcome.retune)
    return lines


def summary_table(outcomes: list[Outcome], recipe: Recipe, publish_status: dict | None = None) -> str:
    rows = [("motion", "check", "vs published", "published")]
    for o in outcomes:
        reproduces = recipe.motions[o.name].reproduces
        final = next((c for c in o.compare if c["published"] == f"{reproduces}.csv"), None)
        if final is None:
            vs = "-"
        elif final["status"] == "identical":
            vs = "identical"
        elif final["status"] == "differs":
            vs = f"max abs {final['max_abs']:.3g}"
        else:
            vs = final["status"]
        check = "FAILED" if o.error else (o.quality or {}).get("verdict", "?")
        pub = publish_status.get(o.name, "") if publish_status else ""
        rows.append((o.name, check, vs, str(pub)))
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    return "\n".join(f"{r[0]:{widths[0]}}  {r[1]:{widths[1]}}  {r[2]:{widths[2]}}  {r[3]}".rstrip() for r in rows)
