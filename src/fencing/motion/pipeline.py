"""The whole cleanup chain with sensible defaults: raw retarget CSV -> publishable motion.

Order matters:
  segment (raw)  -> handedness / mirror -> canonical placement -> straighten the fencing line
  -> time scale -> jitter smoothing -> joint-limit clamp -> vertical drift -> ground alignment
  -> plant still hovering feet -> foot-slide removal -> joint-speed clamp -> limit clamp
  -> floor-penetration guard -> static holds at both ends -> plant still feet again (the end
  hold of a clip cut mid-landing) -> canonical placement again -> quality report.
Straightening turns the body about the pelvis, which moves planted feet, so it comes before
the foot-slide fix that re-plants them. Smoothing comes before the contact-based fixes so
jitter does not break contact detection, and the speed clamp after them because IK can add speed. The penetration guard
runs last among the geometric fixes since every earlier step can move the feet slightly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from fencing import paths
from fencing.motion import fixes
from fencing.motion.contacts import ContactParams, contacts_from_fk
from fencing.motion.io import Motion, check_motion
from fencing.motion.kinematics import forward_kinematics, load_g1
from fencing.motion.normalize import canonicalize, detect_handedness, initial_heading, mirror_motion
from fencing.motion.publish import Provenance
from fencing.motion.quality import QualityThresholds, quality_report, summary_line
from fencing.motion.segment import Segment, SegmentParams, best_segment, find_segments, hold_pose

CLEAN_DIR = paths.DATA / "motion" / "clean"  # unpublished batch outputs (never committed)


@dataclass
class CleanConfig:
    hand: str | None = None  # "R" / "L"; None = sidecar handedness_guess, else detect
    mirror_left_handed: bool = True
    heading: str = "line"  # what faces +x: "line" (fencing line, pipeline.md), "pelvis" or "feet"
    straighten: bool = True  # root path onto one line, slow heading drift removed
    lateral_keep: float = 0.25  # share of the sideways root offset straighten keeps
    yaw_cutoff_hz: float = 0.5  # heading changes slower than this are drift
    time_scale: float = 1.0  # > 1 slows the motion down ("slow the lunge down first")
    smooth_hz: float | None = 8.0
    joint_limit_margin_rad: float = 0.02
    drift_cutoff_hz: float | None = 0.3
    ground_smoothing_hz: float | None = 1.5
    fix_slide: bool = True
    slide_blend_s: float = 0.15
    contact_rounds: int = 3  # detect contacts -> drift/ground/slide, repeated (3: straightening moves planted feet)
    plant_hover_m: float | None = 0.02  # a still foot higher than this is put on the floor; None = off
    velocity_fraction: float | None = 0.5  # of the motor no-load limit
    max_joint_speed: float | None = None  # extra global cap, rad/s
    hold_s: float = 0.5
    end_hold_s: float | None = None  # None = hold_s; longer for clips cut mid-recovery
    ease_s: float = 0.25
    contact: ContactParams = field(default_factory=ContactParams)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)


@dataclass
class CleanResult:
    motion: Motion
    steps: list[dict]
    report: dict
    raw_report: dict
    hand: str
    hand_source: str
    mirrored: bool
    segment: dict | None

    def record(self) -> dict:
        return {
            "hand": self.hand,
            "hand_source": self.hand_source,
            "mirrored": self.mirrored,
            "segment": self.segment,
            "steps": self.steps,
            "quality": self.report,
            "raw_quality": self.raw_report,
        }


def select_segment(raw: Motion, label: str, frames: tuple[int, int] | None, hand: str | None) -> tuple[Motion, dict | None]:
    """Cut the requested piece out of a long raw clip (explicit frames, or best auto match)."""
    if frames is not None:
        start, end = frames
        return raw.slice(start, end), {"start": start, "end": end, "label": label, "source": "manual"}
    segments, _ = find_segments(raw, SegmentParams(), hand=hand)
    seg = best_segment(segments, label)
    if seg is None:
        found = ", ".join(f"{s.label}[{s.start}:{s.end}]" for s in segments) or "none"
        raise ValueError(f"no {label!r} segment found in {raw.meta.get('source_path', 'motion')} (found: {found})")
    return raw.slice(seg.start, seg.end), {**seg.as_dict(), "source": "auto"}


def resolve_hand(motion: Motion, config: CleanConfig) -> tuple[str, str]:
    if config.hand is not None:
        if config.hand not in ("R", "L"):
            raise ValueError(f"hand must be 'R' or 'L', got {config.hand!r}")
        return config.hand, "config"
    guess = (motion.meta.get("sidecar") or {}).get("handedness_guess")
    if guess in ("R", "L"):
        return guess, "sidecar"
    fk = forward_kinematics(motion)
    n = max(2, min(motion.n_frames, int(round(0.5 * motion.fps))))
    detected = detect_handedness(fk, np.arange(n))
    return detected.hand, f"detected (confidence {detected.confidence:.2f})"


def clean_motion(
    raw: Motion,
    config: CleanConfig = CleanConfig(),
    label: str | None = None,
    frames: tuple[int, int] | None = None,
    auto_segment: bool = False,
) -> CleanResult:
    check_motion(raw, label=str(raw.meta.get("source_path", "raw motion")))
    g1 = load_g1()
    steps: list[dict] = []

    def log(name: str, info: dict) -> None:
        steps.append({"step": name, **info})

    raw_report = quality_report(raw, config.thresholds, floor_at_zero=False)
    segment = None
    motion = raw
    if frames is not None or auto_segment:
        if label is None:
            raise ValueError("segment selection needs a label (e.g. --label lunge)")
        motion, segment = select_segment(raw, label, frames, config.hand)
        log("segment", {k: v for k, v in segment.items() if k != "features"})

    hand, hand_source = resolve_hand(motion, config)
    mirrored = hand == "L" and config.mirror_left_handed
    if mirrored:
        motion = mirror_motion(motion, g1)
        log("mirror", {"from": "L", "to": "R", "hand_source": hand_source})

    motion, info = _canonical(motion, config.heading)
    log("canonicalize", info)
    if config.straighten:
        motion, info = fixes.straighten_line(motion, lateral_keep=config.lateral_keep, yaw_cutoff_hz=config.yaw_cutoff_hz)
        log("straighten", info)
        if not info.get("skipped"):
            motion, info = _canonical(motion, config.heading)  # the line is cleaner now
            log("canonicalize_straight", info)
    if abs(config.time_scale - 1.0) > 1e-9:
        motion, info = fixes.time_scale(motion, config.time_scale)
        log("time_scale", info)
    if config.smooth_hz:
        motion, info = fixes.smooth_motion(motion, config.smooth_hz)
        log("smooth", info)
    motion, info = fixes.clamp_joint_limits(motion, config.joint_limit_margin_rad, g1)
    log("joint_limits", info)

    for round_index in range(config.contact_rounds):
        motion = _contact_fixes(motion, config, g1, log, round_index)
    if config.velocity_fraction or config.max_joint_speed:
        motion, info = fixes.clamp_joint_velocity(motion, config.velocity_fraction or 1.0, config.max_joint_speed)
        log("joint_speed", {k: v for k, v in info.items() if k != "limits_rad_s"})
    motion, info = fixes.clamp_joint_limits(motion, 0.0, g1)
    log("joint_limits_final", {k: v for k, v in info.items() if k != "per_joint_frames"})
    motion, info = fixes.remove_floor_penetration(motion)
    log("floor_penetration", info)

    end_hold_s = config.hold_s if config.end_hold_s is None else config.end_hold_s
    if config.hold_s > 0 or end_hold_s > 0:
        motion, info = hold_pose(motion, config.hold_s, end_hold_s, config.ease_s)
        log("hold", info)
    if config.plant_hover_m:
        motion, info = fixes.plant_still_feet(motion, config.plant_hover_m, g1=g1)
        log("plant_still_feet_final", info)
    motion, info = _canonical(motion, config.heading)
    log("canonicalize_final", info)

    motion.meta.update({"hand": "R" if mirrored else hand, "cleaned": True})
    report = quality_report(motion, config.thresholds, g1=g1)
    return CleanResult(motion, steps, report, raw_report, hand, hand_source, mirrored, segment)


def _contact_fixes(motion: Motion, config: CleanConfig, g1, log, round_index: int) -> Motion:
    """Vertical drift, ground alignment and foot slide from one round of contact detection.

    Repeated because each round makes the next detection better: a gliding root makes
    planted feet move faster than the contact speed threshold, so round 1 misses some
    contacts; once its slide fix has removed most of the glide, round 2 finds them.
    """
    suffix = "" if round_index == 0 else f"_round{round_index + 1}"
    fk = forward_kinematics(motion, g1)
    contacts, _ = contacts_from_fk(fk, motion.fps, config.contact, floor_known=round_index > 0)
    if config.drift_cutoff_hz:
        motion, info = fixes.remove_vertical_drift(motion, contacts, fk, config.drift_cutoff_hz)
        log("vertical_drift" + suffix, info)
        fk = forward_kinematics(motion, g1)
    if config.ground_smoothing_hz:
        motion, info = fixes.align_to_ground(motion, contacts, fk, config.ground_smoothing_hz)
        log("ground_align" + suffix, info)
    if config.plant_hover_m:
        motion, info = fixes.plant_still_feet(motion, config.plant_hover_m, g1=g1)
        log("plant_still_feet" + suffix, info)
    if config.fix_slide:
        fk = forward_kinematics(motion, g1)
        contacts, _ = contacts_from_fk(fk, motion.fps, config.contact, floor_known=True)
        motion, info = fixes.remove_foot_slide(motion, contacts, config.slide_blend_s, g1=g1)
        log("foot_slide" + suffix, info)
    return motion


def _canonical(motion: Motion, heading_source: str) -> tuple[Motion, dict]:
    fk = forward_kinematics(motion) if heading_source in ("feet", "line") else None
    heading = initial_heading(motion, fk, heading_source)
    motion, info = canonicalize(motion, heading)
    return motion, {**info, "heading_source": heading_source}


def provenance_for(result: CleanResult, source_clip: str, config: CleanConfig) -> Provenance:
    fps = result.motion.fps
    if result.segment:
        s, e = result.segment["start"], result.segment["end"]
        conf = result.segment.get("confidence")
        segment = f"frames {s}-{e} ({s / fps:.2f}-{e / fps:.2f} s)"
        if conf is not None:
            segment += f", auto {result.segment['label']} conf {conf:.2f}"
    else:
        segment = "whole clip"
    hand = f"{result.hand} ({result.hand_source})" + (" -> mirrored to R" if result.mirrored else "")
    names = []
    for step in result.steps:
        name = step["step"].split("_round")[0]
        if name not in ("canonicalize", "canonicalize_final", "canonicalize_straight", "segment") and name not in names:
            names.append(name)
    details = []
    for name in names:
        if name == "smooth":
            details.append(f"smooth {config.smooth_hz:g} Hz")
        elif name == "joint_speed":
            details.append(f"speed <= {config.velocity_fraction or 1:g}x motor" + (f", <= {config.max_joint_speed:g} rad/s" if config.max_joint_speed else ""))
        elif name == "time_scale":
            details.append(f"time x{config.time_scale:g}")
        elif name == "hold":
            end = config.hold_s if config.end_hold_s is None else config.end_hold_s
            details.append(f"hold {config.hold_s:g} s" + (f" (end {end:g} s)" if end != config.hold_s else ""))
        elif name == "foot_slide":
            details.append("foot slide (root + leg IK)")
        elif name == "straighten":
            step = next(st for st in result.steps if st["step"] == "straighten")
            if step.get("skipped"):
                details.append("straighten skipped (real turn)")
            else:
                details.append(f"straighten line ({step['line_source']}; {step['max_correction_deg']:.0f} deg heading drift removed, lateral x{config.lateral_keep:g})")
        else:
            details.append(name.replace("_", " "))
    details.append(f"heading {config.heading} -> +x")
    return Provenance(source_clip, segment, hand, ", ".join(details), summary_line(result.report))


def config_dict(config: CleanConfig) -> dict:
    return asdict(config)


def segments_table(segments: list[Segment], fps: float) -> list[dict]:
    return [
        {"start": s.start, "end": s.end, "start_s": s.start / fps, "end_s": s.end / fps, "label": s.label, "confidence": round(s.confidence, 3)}
        for s in segments
    ]
