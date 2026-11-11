"""Goal 1 bladework references: blade scripts on top of en garde / advance-lunge bases.

    extension, parry_<name>, cut_<target>, thrust   base: en garde (static footwork)
    advance_lunge_cut                                base: advance-lunge, cut lands at touchdown

Bases: an explicit CSV, or "auto" = the newest published REPO/motions/<kind>_vN.csv (video
derived), falling back to fencing.motion.synthetic (marked SYNTHETIC in provenance).
Everything is re-runnable with another base, so the bladework can be regenerated once
video en garde / advance-lunge motions are published.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fencing import paths
from fencing.motion.io import Motion, read_csv, write_csv, write_sidecar
from fencing.motion.publish import Provenance, existing_versions, publish_motion
from fencing.motion.quality import quality_report, summary_line
from fencing.saber.keyframes import BladeConfig, BladeKey, BladeResult, apply_blade_script, script_summary
from fencing.saber.scored_keys import key_frames_from_blade_keys

BLADE_DIR = paths.DATA / "motion" / "blade"  # unpublished outputs (never committed)

SYNTHETIC_SCRIPTS = {
    "en_garde": [("en_garde", 1.0)],
    "advance_lunge": [("en_garde", 0.8), ("advance", 0.6), ("lunge", 1.2), ("en_garde", 0.8)],
}

# name -> (blade target, requested move s, hold s). Attack moves ask for 0.2 s and are
# stretched by the keyframer to the fastest the speed cap allows.
STATIC_RECIPES = {
    "extension": ("extension", 0.45, 0.8),
    "parry_prime": ("parry_prime", 0.3, 0.6),
    "parry_seconde": ("parry_seconde", 0.3, 0.6),
    "parry_tierce_high": ("parry_tierce_high", 0.3, 0.6),
    "parry_quarte_high": ("parry_quarte_high", 0.3, 0.6),
    "parry_quinte": ("parry_quinte", 0.3, 0.6),
    "parry_sixte": ("parry_sixte", 0.3, 0.6),
    "cut_head": ("cut_head", 0.2, 0.3),
    "cut_flank": ("cut_flank", 0.2, 0.3),
    "cut_chest": ("cut_chest", 0.2, 0.3),
    "cut_cheek": ("cut_cheek", 0.2, 0.3),
    "thrust": ("point_thrust", 0.2, 0.3),
}
LUNGE_RECIPES = ("advance_lunge_cut",)
RECIPES = tuple(STATIC_RECIPES) + LUNGE_RECIPES
GOAL1_RECIPES = ("extension", "parry_prime", "parry_quinte", "parry_tierce_high", "parry_sixte",
                 "parry_seconde", "cut_head", "cut_flank", "cut_chest", "thrust", "advance_lunge_cut")

GUARD_HOLD_S = 0.6
RECOVER_S = 0.5


@dataclass
class BaseMotion:
    motion: Motion
    kind: str  # en_garde | advance_lunge | custom
    synthetic: bool
    source: str  # clip id / file / synthetic description, for provenance
    path: str | None


@dataclass
class RecipeResult:
    name: str
    base: BaseMotion
    blade: BladeResult
    quality: dict

    @property
    def blade_ok(self) -> bool:
        s = self.blade.summary
        return not s["keys_off_target"] and s["collision_frames"] == 0

    def record(self) -> dict:
        blade = self.blade.record()
        return {
            "motion": self.name,
            "base": {"kind": self.base.kind, "synthetic": self.base.synthetic, "source": self.base.source,
                     "path": self.base.path},
            "blade": blade,
            # The instants the blade-tip error is scored at (docs/evaluation.md).
            "key_frames": key_frames_from_blade_keys(blade["keys"], self.blade.motion.fps),
            "quality": self.quality,
        }


# ---------------------------------------------------------------- bases


def base_kind(recipe: str) -> str:
    return "advance_lunge" if recipe in LUNGE_RECIPES else "en_garde"


def newest_published(kind: str, motions_dir: Path = paths.MOTIONS_DIR) -> Path | None:
    versions = existing_versions(kind, motions_dir)
    return motions_dir / f"{kind}_v{versions[-1]}.csv" if versions else None


def load_base(kind: str, source: str | Path | None = "auto") -> BaseMotion:
    """source: a CSV path, "synthetic", or "auto" (newest published <kind>, else synthetic)."""
    if source in (None, "auto"):
        found = newest_published(kind)
        source = found if found is not None else "synthetic"
    if source == "synthetic":
        from fencing.motion.synthetic import synthesize

        if kind not in SYNTHETIC_SCRIPTS:
            raise ValueError(f"no synthetic base for {kind!r}; pass a CSV")
        from fencing.motion.fixes import remove_floor_penetration

        motion, spans = synthesize(SYNTHETIC_SCRIPTS[kind])
        # The synthetic lunge's toe dips ~5 mm below the floor; the cleanup's own guard fixes it.
        motion, _ = remove_floor_penetration(motion)
        phases = ", ".join(f"{label} {(e - s) / motion.fps:.2f}s" for s, e, label in spans)
        return BaseMotion(motion, kind, True, f"SYNTHETIC (fencing.motion.synthetic: {phases}; no video)", None)
    path = Path(source)
    motion = read_csv(path)
    return BaseMotion(motion, kind, False, _describe_source(path, motion), str(path))


def _describe_source(path: Path, motion: Motion) -> str:
    """Clip id from a published record (<name>_vN.json), a cleanup report or a sidecar, else the file name."""
    report_path = path.with_suffix(".report.json")  # written by `fencing motion clean`
    if report_path.exists():
        try:
            source = json.loads(report_path.read_text()).get("source")
        except json.JSONDecodeError:
            source = None
        if source:
            return f"{Path(source).stem} via {path.name}"
    record_path = path.with_suffix(".json")
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            record = {}
        clip = (record.get("provenance") or {}).get("source_clip") or record.get("clip_id")
        if clip:
            return f"{clip} via {path.name}"
    return path.name


# ---------------------------------------------------------------- lunge timing


def lunge_timing(motion: Motion) -> tuple[float, float]:
    """(front-foot touchdown, recovery end) of the lunge, in seconds.

    Touchdown: the first front-foot landing once the stance has widened 90 % of the way to
    its widest, before the recovery. Earlier landings belong to an earlier step (the advance).
    The widest stance itself is not the touchdown: with the back foot dragged forward during
    the flight (advance_lunge_v2) it comes mid-flight, 0.45 s before the front foot lands.
    Recovery end: first frame after the peak where the stance is back within 15 % of the
    starting width. Falls back to geometric thresholds when contacts are ambiguous.
    """
    from fencing.motion.segment import line_features

    f = line_features(motion, hand="R")
    fps = motion.fps
    stance = f.stance
    peak = int(np.argmax(stance))
    contact = f.contacts[:, f.front]
    start_width = float(np.median(stance[: max(1, int(0.3 * fps))]))
    wide = int(np.argmax(stance >= start_width + 0.9 * (stance[peak] - start_width)))
    back = np.flatnonzero(stance[peak:] <= start_width + 0.15 * (stance[peak] - start_width))
    recovery = peak + int(back[0]) if back.size else motion.n_frames - 1 - int(round(0.3 * fps))
    landings = [t for t in range(max(1, wide), recovery) if contact[t] and not contact[t - 1]]
    touchdown = landings[0] if landings else wide
    return touchdown / fps, recovery / fps


# ---------------------------------------------------------------- scripts


def recipe_script(name: str, base: BaseMotion, cut: str = "cut_head", hit_at_s: float | None = None) -> list[BladeKey]:
    if name in STATIC_RECIPES:
        target, move_s, hold_s = STATIC_RECIPES[name]
        return [BladeKey("en_garde", 0.0, GUARD_HOLD_S), BladeKey(target, move_s, hold_s),
                BladeKey("en_garde", RECOVER_S, GUARD_HOLD_S)]
    if name == "advance_lunge_cut":
        touchdown, recovery = lunge_timing(base.motion)
        hit = touchdown if hit_at_s is None else hit_at_s
        # Hand before foot: the cut starts in the lunge's extension and lands at touchdown;
        # held through the lunge, back to guard by the end of the recovery.
        recover_at = max(recovery, hit + 0.3 + RECOVER_S)
        return [BladeKey("en_garde", 0.0, 0.0), BladeKey(cut, 0.3, 0.3, at_s=hit),
                BladeKey("en_garde", RECOVER_S, GUARD_HOLD_S, at_s=recover_at)]
    raise KeyError(f"unknown recipe {name!r}; known: {', '.join(RECIPES)}")


def load_script_file(path: str | Path) -> list[BladeKey]:
    """JSON list of {"action", "move_s", "hold_s", "at_s"} objects."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: blade script must be a non-empty JSON list of keys")
    return [BladeKey(**item) for item in data]


# ---------------------------------------------------------------- run / write / publish


def run_script(name: str, base: BaseMotion, script: list[BladeKey], config: BladeConfig = BladeConfig()) -> RecipeResult:
    blade = apply_blade_script(base.motion, script, config, tail_s=config.end_hold_s)
    blade.motion.meta = {**blade.motion.meta, "recipe": name, "base_synthetic": base.synthetic}
    return RecipeResult(name, base, blade, quality_report(blade.motion))


def run_recipe(name: str, base: BaseMotion, config: BladeConfig = BladeConfig(), cut: str = "cut_head",
               hit_at_s: float | None = None) -> RecipeResult:
    return run_script(name, base, recipe_script(name, base, cut, hit_at_s), config)


def write_outputs(result: RecipeResult, out: Path) -> Path:
    write_csv(result.blade.motion, out)
    write_sidecar(out.with_suffix(".report.json"), result.record())
    return out


def provenance_for(result: RecipeResult) -> Provenance:
    s = result.blade.summary
    body = "SYNTHETIC footwork (no video)" if result.base.synthetic else "video-derived footwork"
    joints = "right arm + waist yaw/pitch" if s["waist_used"] else "right arm"
    cfg = s["config"]
    cleanup = (f"bladework keyframes [{script_summary(result.blade.script)}]; {joints} IK per key, "
               f"min-jerk, speed <= {cfg['velocity_fraction']:g}x motor, soft limits {cfg['soft_limit_factor']:g}")
    if cfg.get("end_hold_s"):
        cleanup += f", end hold {cfg['end_hold_s']:g} s"
    notes = (f"legs/root/left arm: {body}; saber arm: hand-keyframed (not from video); "
             f"key tip err max {100 * s['max_tip_error_m']:.1f} cm, dir {s['max_angle_error_deg']:.1f} deg; "
             f"{summary_line(result.quality)}")
    return Provenance(result.base.source, f"whole base ({result.base.motion.n_frames} frames) + bladework",
                      "R", cleanup, notes)


def publish(result: RecipeResult, allow_warn: bool = False, motions_dir: Path = paths.MOTIONS_DIR) -> Path | None:
    """Publish if the quality verdict passes (or warns, with allow_warn) and every key is on target."""
    allowed = ("pass", "warn") if allow_warn else ("pass",)
    if result.quality["verdict"] not in allowed or not result.blade_ok:
        return None
    return publish_motion(result.blade.motion, result.name, provenance_for(result), motions_dir=motions_dir,
                          record=result.record())


def key_table(result: RecipeResult) -> str:
    rows = []
    for k in result.blade.keys:
        if k.action == "base":
            continue
        rows.append(f"{k.action}@{k.time_s:.2f}s({k.joint_set}) tip {100 * k.tip_error_m:.1f}cm "
                    f"dir {k.angle_error_deg:.1f}deg move {k.move_s_used:.2f}s")
    return "; ".join(rows)
