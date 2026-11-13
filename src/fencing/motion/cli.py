"""`fencing motion ...`: clean, segment, mirror, check, render and publish G1 motion CSVs.

    fencing motion clean RAW.csv --label lunge [--segment auto|A:B] [--publish]
    fencing motion batch [DIR] [--publish] [--labels lunge advance]
    fencing motion segment RAW.csv
    fencing motion check motions/lunge_v1.csv [--raw]
    fencing motion render a.csv [--compare b.csv] [--png out.png] [--mp4 out.mp4]
    fencing motion mirror IN.csv OUT.csv | normalize IN.csv OUT.csv | trim IN.csv OUT.csv
    fencing motion publish CLEAN.csv --name lunge --source CLIP_ID
    fencing motion blade BASE.csv|auto|synthetic --script parry_quinte [--publish] [--render]
    fencing motion blade-all [--base-en-garde X.csv] [--base-advance-lunge Y.csv] [--publish]
    fencing motion blade-list
    fencing motion key-frames [motions/cut_head_v3.json ...] [--write]
    fencing motion rebuild [--recipe motions/recipes/goal1.toml] [--only NAME ...] [--compare] [--publish]
"""

from __future__ import annotations

import os

# Must be set before anything imports mujoco; EGL on this machine is Mesa llvmpipe (CPU).
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from fencing import paths  # noqa: E402

PUBLISHABLE = ("pass", "warn")


def _parse_frames(text: str) -> tuple[int, int]:
    try:
        start, end = (int(v) for v in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"segment must be START:END frame numbers, got {text!r}") from None
    if not 0 <= start < end:
        raise argparse.ArgumentTypeError(f"segment {text!r}: need 0 <= START < END")
    return start, end


def _print_json(data) -> None:
    from fencing.motion.io import _json_default

    print(json.dumps(data, indent=2, default=_json_default))


def _add_clean_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("cleanup options")
    g.add_argument("--hand", choices=["R", "L"], help="sword hand of the source fencer (default: sidecar, else detect)")
    g.add_argument("--no-mirror", action="store_true", help="keep left-handed motions left-handed")
    g.add_argument("--heading", choices=["line", "pelvis", "feet"], default="line", help="what faces +x: the fencing line (default), the start pelvis yaw or the start feet")
    g.add_argument("--no-straighten", action="store_true", help="keep the root path and heading as estimated (no fencing-line straightening)")
    g.add_argument("--lateral-keep", type=float, default=0.25, help="share of the sideways root offset straightening keeps (default 0.25)")
    g.add_argument("--time-scale", type=float, default=1.0, help="slow down by this factor (2 = half speed)")
    g.add_argument("--smooth-hz", type=float, default=8.0, help="jitter low-pass cutoff; 0 disables")
    g.add_argument("--velocity-fraction", type=float, default=0.5, help="joint speed cap as a fraction of the motor limit; 0 disables")
    g.add_argument("--max-joint-speed", type=float, help="extra global joint speed cap, rad/s")
    g.add_argument("--no-drift", action="store_true", help="skip vertical drift removal")
    g.add_argument("--no-slide", action="store_true", help="skip foot-slide removal")
    g.add_argument("--hold", type=float, default=0.5, help="static hold added at both ends, s (0 disables)")


def _config_from(args):
    from fencing.motion.pipeline import CleanConfig

    return CleanConfig(
        hand=args.hand,
        mirror_left_handed=not args.no_mirror,
        heading=args.heading,
        straighten=not args.no_straighten,
        lateral_keep=args.lateral_keep,
        time_scale=args.time_scale,
        smooth_hz=args.smooth_hz or None,
        velocity_fraction=args.velocity_fraction or None,
        max_joint_speed=args.max_joint_speed,
        drift_cutoff_hz=None if args.no_drift else 0.3,
        fix_slide=not args.no_slide,
        hold_s=args.hold,
    )


def _clip_id(raw) -> str:
    sidecar = raw.meta.get("sidecar") or {}
    return str(sidecar.get("clip_id") or Path(raw.meta.get("source_path", "motion")).stem)


def _write_clean_output(result, out_csv: Path, extra: dict) -> Path:
    from fencing.motion.io import write_csv, write_sidecar

    write_csv(result.motion, out_csv)
    write_sidecar(out_csv.with_suffix(".report.json"), {**extra, **result.record()})
    return out_csv


def _maybe_publish(result, label: str, clip: str, config, allow_warn: bool, render: bool) -> Path | None:
    from fencing.motion.pipeline import config_dict, provenance_for
    from fencing.motion.publish import publish_if_new, publish_message

    verdict = result.report["verdict"]
    allowed = ("pass", "warn") if allow_warn else ("pass",)
    if verdict not in allowed:
        print(f"  not published: verdict {verdict} ({'; '.join(result.report['reasons'])})", file=sys.stderr)
        return None
    prov = provenance_for(result, clip, config)
    path, is_new = publish_if_new(result.motion, label, prov, record={**result.record(), "config": config_dict(config)})
    print(f"  {publish_message(path, is_new)}")
    if render and is_new:
        _render_pair(None, path, path.with_suffix(".png"))
    return path


def _render_pair(before, after_path: Path, png: Path) -> None:
    from fencing.motion.io import read_csv
    from fencing.motion.render import contact_sheet

    rows = ([("raw", before)] if before is not None else []) + [("clean", read_csv(after_path))]
    contact_sheet(rows, png)
    print(f"  contact sheet {png}")


# ---------------------------------------------------------------- commands


def cmd_clean(args) -> int:
    from fencing.motion.io import read_csv
    from fencing.motion.pipeline import CLEAN_DIR, clean_motion

    raw = read_csv(args.raw, fps=args.fps)
    config = _config_from(args)
    frames = auto = None
    if args.segment == "auto":
        auto = True
    elif args.segment != "all":
        try:
            frames = _parse_frames(args.segment)
        except argparse.ArgumentTypeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    result = clean_motion(raw, config, label=args.label, frames=frames, auto_segment=bool(auto))
    clip = _clip_id(raw)
    out = Path(args.out) if args.out else CLEAN_DIR / f"{clip}__{args.label}.csv"
    _write_clean_output(result, out, {"source": str(args.raw), "label": args.label})
    print(f"cleaned -> {out}")
    print(f"  raw:   {_short(result.raw_report)}")
    print(f"  clean: {_short(result.report)}")
    if args.render:
        before = raw if result.segment is None else raw.slice(result.segment["start"], result.segment["end"])
        _render_pair(before, out, out.with_suffix(".png"))
    if args.publish:
        published = _maybe_publish(result, args.label, clip, config, args.allow_warn, render=False)
        if published is None:
            return 1
    return 0 if result.report["verdict"] in PUBLISHABLE else 1


def _short(report: dict) -> str:
    from fencing.motion.quality import summary_line

    line = summary_line(report)
    return line + (f" [{'; '.join(report['reasons'])}]" if report.get("reasons") else "")


def cmd_batch(args) -> int:
    from fencing.motion.io import read_csv
    from fencing.motion.pipeline import CLEAN_DIR, clean_motion, segments_table
    from fencing.motion.segment import find_segments

    source = Path(args.dir)
    files = sorted(source.glob("*.csv"))
    if not files:
        print(f"no raw CSVs in {source}", file=sys.stderr)
        return 1
    config = _config_from(args)
    summary = []
    for csv_path in files:
        print(f"{csv_path.name}:")
        try:
            raw = read_csv(csv_path)
            segments, feats = find_segments(raw, hand=config.hand)
        except Exception as exc:  # one bad clip must not stop a batch of hundreds
            print(f"  skipped: {exc}", file=sys.stderr)
            summary.append({"file": csv_path.name, "error": str(exc)})
            continue
        clip = _clip_id(raw)
        print(f"  hand {feats.hand} ({feats.hand_confidence:.2f}); segments: {segments_table(segments, raw.fps)}")
        for seg in segments:
            if seg.confidence < args.min_confidence or (args.labels and seg.label not in args.labels):
                continue
            try:
                result = clean_motion(raw, config, label=seg.label, frames=(seg.start, seg.end))
            except Exception as exc:
                print(f"  {seg.label}[{seg.start}:{seg.end}] failed: {exc}", file=sys.stderr)
                summary.append({"file": csv_path.name, "segment": [seg.start, seg.end], "error": str(exc)})
                continue
            result.segment.update({"confidence": seg.confidence, "label": seg.label})
            out = CLEAN_DIR / f"{clip}__{seg.label}_{seg.start}-{seg.end}.csv"
            _write_clean_output(result, out, {"source": str(csv_path), "label": seg.label})
            print(f"  {seg.label}[{seg.start}:{seg.end}] conf {seg.confidence:.2f} -> {out.name}: {_short(result.report)}")
            entry = {"file": csv_path.name, "segment": [seg.start, seg.end], "label": seg.label, "out": str(out), "verdict": result.report["verdict"]}
            if args.publish:
                published = _maybe_publish(result, seg.label, clip, config, args.allow_warn, render=False)
                entry["published"] = str(published) if published else None
            summary.append(entry)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    (CLEAN_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {CLEAN_DIR / 'batch_summary.json'}")
    return 0


def cmd_segment(args) -> int:
    from fencing.motion.io import read_csv
    from fencing.motion.pipeline import segments_table
    from fencing.motion.segment import find_segments

    raw = read_csv(args.csv, fps=args.fps)
    segments, feats = find_segments(raw, hand=args.hand)
    data = {
        "file": str(args.csv),
        "hand": feats.hand,
        "hand_confidence": feats.hand_confidence,
        "fencing_line_xy": feats.line.tolist(),
        "segments": segments_table(segments, raw.fps),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2))
    _print_json(data)
    return 0


def cmd_check(args) -> int:
    from fencing.motion.io import MotionFormatError, read_csv
    from fencing.motion.quality import quality_report

    worst = 0
    reports = {}
    for path in args.csv:
        try:
            motion = read_csv(path, fps=args.fps)
            report = quality_report(motion, floor_at_zero=not args.raw)
        except MotionFormatError as exc:
            report = {"verdict": "fail", "reasons": [str(exc)], "format_ok": False}
        reports[str(path)] = report
        worst = max(worst, {"pass": 0, "warn": 0, "fail": 1}[report["verdict"]])
        print(f"{path}: {_short(report)}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
    else:
        _print_json(reports if len(reports) > 1 else next(iter(reports.values())))
    return worst


def cmd_render(args) -> int:
    from fencing.motion.io import read_csv
    from fencing.motion.render import contact_sheet, render_video

    rows = [(Path(args.csv).stem, read_csv(args.csv, fps=args.fps))]
    for other in args.compare or []:
        rows.append((Path(other).stem, read_csv(other, fps=args.fps)))
    if not args.png and not args.mp4:
        args.png = str(paths.REPO_ROOT / "outputs" / "replay" / f"{rows[0][0]}.png")
    if args.png:
        print(f"contact sheet: {contact_sheet(rows, args.png, columns=args.columns, azimuth=args.azimuth, saber=args.saber)}")
    if args.mp4:
        print(f"video: {render_video(rows, args.mp4, azimuth=args.azimuth, saber=args.saber)}")
    return 0


def cmd_mirror(args) -> int:
    from fencing.motion.io import read_csv, write_csv
    from fencing.motion.normalize import mirror_motion

    write_csv(mirror_motion(read_csv(args.input, fps=args.fps)), args.output)
    print(f"mirrored -> {args.output}")
    return 0


def cmd_normalize(args) -> int:
    from fencing.motion.io import read_csv, write_csv
    from fencing.motion.kinematics import forward_kinematics
    from fencing.motion.normalize import canonicalize, initial_heading

    motion = read_csv(args.input, fps=args.fps)
    fk = forward_kinematics(motion) if args.heading in ("feet", "line") else None
    out, info = canonicalize(motion, initial_heading(motion, fk, args.heading))
    write_csv(out, args.output)
    print(f"normalized ({info}) -> {args.output}")
    return 0


def cmd_trim(args) -> int:
    from fencing.motion.io import read_csv, write_csv
    from fencing.motion.segment import hold_pose

    motion = read_csv(args.input, fps=args.fps)
    end = args.end if args.end is not None else motion.n_frames
    out, _ = hold_pose(motion.slice(args.start, end), args.hold, args.hold, args.ease)
    write_csv(out, args.output)
    print(f"trimmed [{args.start}:{end}] + {args.hold} s holds -> {args.output} ({out.n_frames} frames)")
    return 0


def cmd_publish(args) -> int:
    from fencing.motion.io import read_csv
    from fencing.motion.publish import Provenance, publish_if_new, publish_message
    from fencing.motion.quality import quality_report, summary_line

    motion = read_csv(args.csv, fps=args.fps)
    report = quality_report(motion)
    if report["verdict"] == "fail" and not args.force:
        print(f"refusing to publish a failing motion ({'; '.join(report['reasons'])}); --force overrides", file=sys.stderr)
        return 1
    prov = Provenance(args.source, args.segment, args.hand_note, args.cleanup, args.notes or summary_line(report))
    path, is_new = publish_if_new(motion, args.name, prov, record={"quality": report, "input": str(args.csv)})
    print(publish_message(path, is_new))
    return 0


def cmd_entry(args) -> int:
    """Standby pose -> a motion's first frame, held (docs/mtc.md, "Entry from standby")."""
    from fencing.motion import entry
    from fencing.motion.io import read_csv, write_csv
    from fencing.motion.publish import Provenance, publish_if_new, publish_message
    from fencing.motion.quality import quality_report, summary_line

    target = read_csv(args.target)
    result = entry.build_entry(target, frame=args.frame)
    report = quality_report(result.motion)
    record = {"entry": {"target": Path(args.target).name, "frame": args.frame, **result.report}, "quality": report}
    out = Path(args.out) if args.out else paths.DATA / "motion" / "entry" / f"{args.name}.csv"
    write_csv(result.motion, out)
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    phases = ", ".join(f"{p['label']} {p['start_s']:.2f}-{p['end_s']:.2f}" for p in result.report["phases"])
    print(f"{args.name} -> {out}  ({result.report['duration_s']:.2f} s: {phases})")
    print(f"  check: {report['verdict']} {summary_line(report)}")
    if not args.publish:
        return 0
    if report["verdict"] != "pass":
        print(f"refusing to publish: quality {report['verdict']} ({'; '.join(report['reasons'])})", file=sys.stderr)
        return 1
    prov = Provenance(f"synthetic: standby_controller default_position -> {Path(args.target).name} frame {args.frame}",
                      "whole motion", "R", "fencing.motion.entry (leg IK, CoM over the stance sole)",
                      summary_line(report))
    path, is_new = publish_if_new(result.motion, args.name, prov, record=record, allow_unknown_name=True)
    print(publish_message(path, is_new))
    return 0


# ---------------------------------------------------------------- bladework (Phase 1.4d)


def _blade_config(args, recipe: str | None = None):
    from fencing.motion import rebuild
    from fencing.saber.keyframes import BladeConfig

    blade_path = args.blade_path
    if blade_path == "recipe":  # the recipe says which motions move the tip on the line (thrust, cut_flank, extension)
        blade_path = rebuild.blade_paths(rebuild.load_recipe(rebuild.GVHMR_RECIPE)).get(recipe, "joint")
    return BladeConfig(velocity_fraction=args.velocity_fraction, heading=args.heading, allow_waist=not args.no_waist,
                       end_hold_s=args.end_hold, blade_path=blade_path)


def _run_blade(name: str, base, args, script, recipe: str | None = None) -> int:
    """Run one blade script, write HCF_DATA/motion/blade/<name>.csv, optionally render + publish."""
    from fencing.motion.publish import identical_version, publish_message
    from fencing.saber import recipes

    result = recipes.run_script(name, base, script, _blade_config(args, recipe))
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else recipes.BLADE_DIR
    out = Path(args.out) if getattr(args, "out", None) else out_dir / f"{name}.csv"
    recipes.write_outputs(result, out)
    s = result.blade.summary
    print(f"{name} -> {out}  [base: {result.base.source}]")
    print(f"  keys: {recipes.key_table(result)}")
    print(f"  blade: collision frames {s['collision_frames']}, beyond soft limit {s['controlled_joints_beyond_soft_limit'] or '-'}, "
          f"max speed {s['controlled_max_speed_fraction_of_motor']:.2f}x motor, hold drift {100 * s['max_hold_drift_m']:.1f} cm")
    print(f"  check: {_short(result.quality)}")
    if args.render:
        from fencing.motion.render import contact_sheet

        keys = [k.frame for k in result.blade.keys]
        frames = sorted({0, *keys, *[max(0, (a + b) // 2) for a, b in zip([0] + keys, keys)], result.blade.motion.n_frames - 1})
        png = contact_sheet([(name, result.blade.motion)], out.with_suffix(".png"), azimuth=args.azimuth, saber=True, frames=frames)
        print(f"  contact sheet {png}")
    ok = result.quality["verdict"] in PUBLISHABLE and result.blade_ok
    if args.publish:
        same = identical_version(result.blade.motion, name, paths.MOTIONS_DIR)
        if same is not None:  # publish_motion would write nothing; say so rather than "published"
            print(f"  {publish_message(same, False)}")
            return 0 if ok else 1
        published = recipes.publish(result, allow_warn=args.allow_warn)
        if published is None:
            print(f"  not published: verdict {result.quality['verdict']}, keys off target {s['keys_off_target'] or '-'}, "
                  f"collision frames {s['collision_frames']}", file=sys.stderr)
            return 1
        print(f"  published {published}")
    return 0 if ok else 1


def cmd_blade(args) -> int:
    from fencing.saber import recipes

    if args.script in recipes.RECIPES:
        base = recipes.load_base(recipes.base_kind(args.script), args.base)
        script = recipes.recipe_script(args.script, base, args.cut, args.hit_at)
        return _run_blade(args.name or args.script, base, args, script, recipe=args.script)
    path = Path(args.script)
    if not path.exists():
        print(f"error: --script {args.script!r} is neither a recipe ({', '.join(recipes.RECIPES)}) nor a JSON file", file=sys.stderr)
        return 2
    if not args.name or args.base in ("auto", "synthetic"):
        print("error: a JSON script needs --name and an explicit base CSV", file=sys.stderr)
        return 2
    return _run_blade(args.name, recipes.load_base("custom", args.base), args, recipes.load_script_file(path))


def cmd_blade_all(args) -> int:
    from fencing.saber import recipes

    names = args.only or list(recipes.GOAL1_RECIPES)
    unknown = [n for n in names if n not in recipes.RECIPES]
    if unknown:
        print(f"error: unknown recipes {unknown}; known: {', '.join(recipes.RECIPES)}", file=sys.stderr)
        return 2
    sources = {"en_garde": args.base_en_garde, "advance_lunge": args.base_advance_lunge}
    bases = {}
    worst = 0
    for name in names:
        kind = recipes.base_kind(name)
        if kind not in bases:
            bases[kind] = recipes.load_base(kind, sources[kind])
        script = recipes.recipe_script(name, bases[kind], args.cut, args.hit_at)
        worst = max(worst, _run_blade(name, bases[kind], args, script, recipe=name))
    return worst


def cmd_blade_list(args) -> int:
    from fencing.saber import recipes
    from fencing.saber.keyframes import TARGET_ALIASES, blade_targets

    print("recipes (base):")
    for name in recipes.RECIPES:
        detail = recipes.STATIC_RECIPES.get(name)
        what = f"guard > {detail[0]} (move {detail[1]:g} s, hold {detail[2]:g} s) > guard" if detail else "cut lands at lunge touchdown"
        print(f"  {name:20s} ({recipes.base_kind(name)}) {what}")
    print("blade targets:")
    for target in blade_targets().values():
        print(f"  {target.name:20s} {target.kind:7s} {target.description}")
    print("aliases: " + ", ".join(f"{a} -> {b}" for a, b in TARGET_ALIASES.items()))
    return 0


def cmd_key_frames(args) -> int:
    """Record each bladework sidecar's scored key frames (derived from its blade keys)."""
    from fencing.saber import scored_keys

    sidecars = [Path(p).with_suffix(".json") for p in args.sidecars] or scored_keys.newest_bladework_sidecars()
    for sidecar in sidecars:
        record = json.loads(sidecar.read_text())
        keys = scored_keys.with_key_frames(record)["key_frames"]
        changed = scored_keys.backfill(sidecar, write=args.write)
        shown = ", ".join(f"{k['target']}@{k['time_s']:.3f}s (frame {k['frame']})" for k in keys) or "none"
        state = ("written" if args.write else "would write") if changed else "up to date"
        print(f"{sidecar.name}: {shown} [{state}]")
    if not args.write:
        print("dry run: pass --write to update the sidecars (no CSV is touched)")
    return 0


def cmd_rebuild(args) -> int:
    """Every motion of a recipe from the retarget CSVs (docs/pipeline.md, "Rebuilding every motion and policy")."""
    from fencing.motion import rebuild

    if args.publish and args.from_published:
        print("error: --from-published is for verification only and cannot be combined with --publish", file=sys.stderr)
        return 2
    try:
        recipe = rebuild.load_recipe(args.recipe)
        out = Path(args.out) if args.out else rebuild.REBUILD_DIR / Path(args.recipe).stem
        job = rebuild.Rebuild(recipe, out, csv_dir=args.csv_dir, anchors=tuple(args.from_published or ()),
                              compare=args.compare)
        recipe.order(args.only)  # unknown names fail before any work
    except rebuild.RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def show(outcome) -> None:
        print("\n".join(rebuild.outcome_lines(outcome, recipe)), flush=True)

    outcomes = job.run(args.only, on_outcome=show)
    status = job.publish(allow_warn_all=args.allow_warn) if args.publish else None
    print(f"\noutputs: {out} (summary.json, <motion>.report.json)")
    print(rebuild.summary_table(outcomes, recipe, status))
    failed = [o.name for o in outcomes if o.error or (o.quality or {}).get("verdict") == "fail"]
    unpublished = [n for n, s in (status or {}).items() if s.kind == "refused"]
    if not args.publish:
        print("scratch only: nothing in motions/ was written (--publish writes new versions)")
    elif not any(s.kind == "new" for s in status.values()):
        print("nothing new to publish: every motion built is identical to a published version in motions/")
    return 1 if failed or unpublished else 0


def _add_blade_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("bladework options")
    g.add_argument("--velocity-fraction", type=float, default=0.5, help="joint speed cap as a fraction of the motor limit (default 0.5)")
    g.add_argument("--heading", choices=["x", "pelvis"], default="x", help="target axes: world +x fencing line (default) or per-frame pelvis yaw")
    g.add_argument("--no-waist", action="store_true", help="arm only; never turn the waist to reach a target")
    g.add_argument("--end-hold", type=float, default=2.0,
                   help="static hold after the last key, s (default 2.0: the eval holds the last frame 2 s, and "
                        "training only teaches standing as long as the reference does)")
    g.add_argument("--blade-path", choices=["recipe", "joint", "line"], default="recipe",
                   help="how the arm moves between keys: joint space, the tip on the straight line, or recipe "
                        "(default: what motions/recipes/goal1_gvhmr.toml sets for the recipe; else joint)")
    g.add_argument("--cut", default="cut_head", help="advance_lunge_cut: which cut lands at touchdown (default cut_head)")
    g.add_argument("--hit-at", type=float, help="advance_lunge_cut: time the cut lands, s (default: detected front-foot touchdown)")
    g.add_argument("--render", action="store_true", help="contact sheet PNG with the saber, frames at the keys")
    g.add_argument("--azimuth", type=float, default=135.0, help="render camera azimuth, deg (default 135: front-right)")
    g.add_argument("--publish", action="store_true", help="publish to motions/ if the check passes and every key is on target")
    g.add_argument("--allow-warn", action="store_true", help="publish on a 'warn' verdict too")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing motion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, help="override input fps (default: sidecar, else 30)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clean", help="run the whole cleanup chain on one raw CSV")
    p.add_argument("raw", type=Path)
    p.add_argument("--label", required=True, help="motion name, e.g. lunge (also the published name)")
    p.add_argument("--segment", default="all", help="'all' (whole clip, default), 'auto' (best auto-detected LABEL segment) or START:END frames")
    p.add_argument("--out", help="output CSV (default HCF_DATA/motion/clean/<clip>__<label>.csv)")
    p.add_argument("--publish", action="store_true", help="also publish to motions/ if the verdict passes")
    p.add_argument("--allow-warn", action="store_true", help="publish on a 'warn' verdict too")
    p.add_argument("--render", action="store_true", help="write a before/after contact sheet PNG next to the output")
    _add_clean_options(p)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("batch", help="segment + clean every raw CSV in a directory")
    p.add_argument("dir", nargs="?", default=str(paths.RETARGET / "csv"))
    p.add_argument("--labels", nargs="*", help="only these labels")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--allow-warn", action="store_true")
    _add_clean_options(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("segment", help="find and label single-motion segments")
    p.add_argument("csv", type=Path)
    p.add_argument("--hand", choices=["R", "L"])
    p.add_argument("--json", help="also write the table to this file")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("check", help="validate + quality report (exit 1 on fail)")
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--raw", action="store_true", help="floor not at z=0 yet (estimate it)")
    p.add_argument("--json", help="write reports here instead of stdout")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("render", help="kinematic replay to contact-sheet PNG and/or MP4 (no mjlab, no W&B)")
    p.add_argument("csv")
    p.add_argument("--compare", nargs="*", help="more CSVs rendered as extra rows / side-by-side")
    p.add_argument("--png")
    p.add_argument("--mp4")
    p.add_argument("--columns", type=int, default=8)
    p.add_argument("--saber", action="store_true", help="draw the training saber on the right wrist")
    p.add_argument("--azimuth", type=float, default=90.0, help="camera azimuth, deg (90: side view, +x left-to-right)")
    p.set_defaults(func=cmd_render)

    for name, func, helptext in (
        ("mirror", cmd_mirror, "mirror left <-> right handed"),
        ("normalize", cmd_normalize, "start at the origin facing +x"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        if name == "normalize":
            p.add_argument("--heading", choices=["line", "pelvis", "feet"], default="line")
        p.set_defaults(func=func)

    p = sub.add_parser("trim", help="cut [start:end) and add static holds at both ends")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--hold", type=float, default=0.5)
    p.add_argument("--ease", type=float, default=0.25)
    p.set_defaults(func=cmd_trim)

    p = sub.add_parser("publish", help="publish an already-clean CSV as motions/<name>_v<N>.csv")
    p.add_argument("csv", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--source", required=True, help="source clip id / video_log.md #")
    p.add_argument("--segment", default="whole clip")
    p.add_argument("--hand-note", default="R")
    p.add_argument("--cleanup", default="manual")
    p.add_argument("--notes")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("entry", help="entry motion: upstream standby pose -> a motion's first frame, held")
    p.add_argument("target", type=Path, help="the motion to enter (its frame --frame is the end pose)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--name", default="stand_to_en_garde", help="motion name for output/publish")
    p.add_argument("--out", help="output CSV (default HCF_DATA/motion/entry/<name>.csv)")
    p.add_argument("--publish", action="store_true", help="publish as motions/<name>_v<N>.csv")
    p.set_defaults(func=cmd_entry)

    p = sub.add_parser("blade", help="keyframe the saber arm over a base motion (Phase 1.4d)")
    p.add_argument("base", help="base motion CSV, 'auto' (newest published base, else synthetic) or 'synthetic'")
    p.add_argument("--script", required=True, help="recipe name (see blade-list) or a JSON file of keys [{action, move_s, hold_s, at_s}]")
    p.add_argument("--name", help="motion name for output/publish (default: the recipe name)")
    p.add_argument("--out", help="output CSV (default HCF_DATA/motion/blade/<name>.csv)")
    _add_blade_options(p)
    p.set_defaults(func=cmd_blade)

    p = sub.add_parser("blade-all", help="every Goal 1 bladework recipe (extension, parries, cuts, thrust, advance_lunge_cut)")
    p.add_argument("--base-en-garde", default="auto", help="en garde base CSV, 'auto' or 'synthetic' (default auto)")
    p.add_argument("--base-advance-lunge", default="auto", help="advance-lunge base CSV, 'auto' or 'synthetic' (default auto)")
    p.add_argument("--only", nargs="*", help="only these recipes")
    p.add_argument("--out-dir", help="output directory (default HCF_DATA/motion/blade)")
    _add_blade_options(p)
    p.set_defaults(func=cmd_blade_all)

    p = sub.add_parser("blade-list", help="list bladework recipes and blade targets")
    p.set_defaults(func=cmd_blade_list)

    p = sub.add_parser("key-frames", help="record the scored blade key frames in bladework sidecars (dry run "
                                          "unless --write)")
    p.add_argument("sidecars", nargs="*", help="motion sidecars or CSVs (default: newest version of every recipe)")
    p.add_argument("--write", action="store_true", help="update the sidecars (only the key_frames field)")
    p.set_defaults(func=cmd_key_frames)

    from fencing.motion.rebuild import DEFAULT_RECIPE

    p = sub.add_parser("rebuild", help="rebuild every motion of a recipe from the retarget CSVs (scratch unless --publish)")
    p.add_argument("--recipe", default=str(DEFAULT_RECIPE), help="recipe TOML (default motions/recipes/goal1.toml)")
    p.add_argument("--only", nargs="*", help="only these motions (and what they are built on)")
    p.add_argument("--out", help="output dir (default HCF_DATA/motion/rebuild/<recipe name>)")
    p.add_argument("--csv-dir", help="retarget CSV dir (default HCF_DATA/retarget/csv)")
    p.add_argument("--compare", action="store_true", help="diff each checkpoint and final motion against the published file it names")
    p.add_argument("--from-published", nargs="*", metavar="NAME",
                   help="start from motions/NAME.csv instead of rebuilding up to that checkpoint (verification only)")
    p.add_argument("--publish", action="store_true", help="publish each motion whose check passes as a new version in motions/")
    p.add_argument("--allow-warn", action="store_true", help="publish on a 'warn' verdict for every motion (the recipe's allow_warn sets it per motion)")
    p.set_defaults(func=cmd_rebuild)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
