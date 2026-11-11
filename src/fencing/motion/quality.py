"""Per-motion quality metrics and a pass / warn / fail verdict.

Metrics assume the published convention: floor at z = 0. Thresholds are first guesses to
be tightened once Phase 1.5 shows what BeyondMimic tolerates; they live in one dataclass
so the verdict can be re-run with other numbers.

Besides the physical checks (slip, penetration, limits, speed) a plausibility gate asks
whether this is a fencer at all: a tracker that jumps between people or a non-fencing video
gives a clean-looking but impossible motion (the robot folded backwards, the root
teleporting, sitting on the floor), and GMR's own fit statistics from the retarget sidecar
say when the robot could not follow the keypoints. Any of those fails the motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from fencing.motion.contacts import ContactParams, contact_pattern, contact_phases, contacts_from_fk
from fencing.motion.fixes import hardware_velocity_limits
from fencing.motion.io import JOINT_NAMES, Motion, validate_motion
from fencing.motion.kinematics import FKResult, G1Model, forward_kinematics, load_g1


@dataclass(frozen=True)
class QualityThresholds:
    slip_warn_m: float = 0.02
    slip_fail_m: float = 0.05
    penetration_warn_m: float = 0.005
    penetration_fail_m: float = 0.02
    float_warn_m: float = 0.02  # planted foot hovering (95th percentile)
    float_fail_m: float = 0.05
    limit_tolerance_rad: float = 1e-3
    speed_warn_fraction: float = 0.6  # of the motor no-load limit
    speed_fail_fraction: float = 0.9
    min_duration_s: float = 0.5
    # Plausibility, calibrated on the Goal 1 clips. Good fencing clips: torso tilt <= 24 deg,
    # root <= 4.0 m/s, pelvis >= 0.59 m, GMR IK error median <= 1.4, at-limit <= 5 %. A
    # tracker switching fencers / a stretching video: tilt > 50 deg for 2.5+ s, root 6.6-7.7
    # m/s, pelvis < 0.5 m for 0.7+ s, IK error 4.7-5.4, at-limit 22-47 %.
    torso_tilt_warn_deg: float = 30.0  # torso z-axis from vertical, any frame
    torso_tilt_fail_deg: float = 45.0  # ... held for sustain_s
    pelvis_low_warn_m: float = 0.55  # pelvis above the floor, any frame (deep lunge ~0.6)
    pelvis_low_fail_m: float = 0.50  # ... held for sustain_s
    pelvis_high_fail_m: float = 0.90  # ... held for sustain_s (G1 standing straight ~0.79)
    sustain_s: float = 0.2
    root_speed_warn_mps: float = 4.5  # root ground speed; a lunge peaks ~3 m/s
    root_speed_fail_mps: float = 6.0  # 0.2 m in one frame: a teleport
    ik_error_warn: float = 2.0  # sidecar retarget.ik_error_median (GMR units)
    ik_error_fail: float = 3.0
    limit_frac_warn: float = 0.10  # sidecar retarget.joint_limit_frac
    limit_frac_fail: float = 0.20


def slip_metrics(fk: FKResult, contacts: np.ndarray, fps: float) -> dict:
    xy = fk.sole_center[..., :2]
    max_dev, total, phase_count = 0.0, 0.0, 0
    for f in range(2):
        for s, e in contact_phases(contacts[:, f]):
            phase_count += 1
            anchor = np.median(xy[s:e, f], axis=0)
            max_dev = max(max_dev, float(np.linalg.norm(xy[s:e, f] - anchor, axis=1).max()))
            total += float(np.linalg.norm(np.diff(xy[s:e, f], axis=0), axis=1).sum())
    contact_frames = int(contacts.sum())
    return {
        "max_deviation_m": max_dev,
        "total_slide_m": total,
        "mean_slide_speed_mps": total * fps / contact_frames if contact_frames else 0.0,
        "contact_phases": phase_count,
    }


def _sustained(values: np.ndarray, frames: int, high: bool = True) -> float:
    """The most extreme level `values` stays beyond for at least `frames` frames in a row:
    the max over windows of the window minimum (high=True), or the min of window maxima."""
    frames = max(1, min(frames, len(values)))
    if high:
        return float(minimum_filter1d(values, frames, mode="nearest").max())
    return float(maximum_filter1d(values, frames, mode="nearest").min())


def plausibility_metrics(motion: Motion, fk: FKResult, floor: np.ndarray, th: QualityThresholds) -> dict:
    up = fk.body_rot[:, fk.body_names.index("torso_link")][:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))
    pelvis = fk.pelvis_pos[:, 2] - floor
    window = int(round(th.sustain_s * motion.fps))
    stats = (motion.meta.get("sidecar") or {}).get("retarget") or {}

    def stat(key: str) -> float | None:
        value = stats.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "torso_tilt_max_deg": float(tilt.max()),
        "torso_tilt_sustained_deg": _sustained(tilt, window),
        "pelvis_height_min_m": float(pelvis.min()),
        "pelvis_low_sustained_m": _sustained(pelvis, window, high=False),
        "pelvis_high_sustained_m": _sustained(pelvis, window),
        "retarget_ik_error_median": stat("ik_error_median"),
        "retarget_joint_limit_frac": stat("joint_limit_frac"),
    }


def quality_report(
    motion: Motion,
    thresholds: QualityThresholds = QualityThresholds(),
    fk: FKResult | None = None,
    g1: G1Model | None = None,
    floor_at_zero: bool = True,
) -> dict:
    problems = validate_motion(motion)
    if problems:
        return {"verdict": "fail", "reasons": problems, "format_ok": False}
    g1 = g1 or load_g1()
    fk = fk or forward_kinematics(motion, g1)
    contacts, floor = contacts_from_fk(fk, motion.fps, ContactParams(), floor_known=floor_at_zero)
    lowest = fk.foot_lowest_z - floor[:, None]
    lower_foot = lowest.min(axis=1)
    planted_heights = lowest[contacts]

    over = np.maximum(g1.joint_lower - motion.joint_pos, motion.joint_pos - g1.joint_upper)
    violations = over > thresholds.limit_tolerance_rad
    speeds = np.abs(np.diff(motion.joint_pos, axis=0)) * motion.fps
    speed_fraction = speeds / hardware_velocity_limits()
    j_fast = int(np.unravel_index(np.argmax(speed_fraction), speed_fraction.shape)[1]) if speeds.size else 0
    root_speed = np.linalg.norm(np.diff(motion.root_pos[:, :2], axis=0), axis=1) * motion.fps

    metrics = {
        "frames": motion.n_frames,
        "fps": motion.fps,
        "duration_s": motion.duration_s,
        "foot_slip": slip_metrics(fk, contacts, motion.fps),
        "penetration": {
            "max_depth_m": float(max(0.0, -lower_foot.min())),
            "frames_below_5mm": int((lower_foot < -0.005).sum()),
        },
        "planted_float_p95_m": float(np.percentile(planted_heights, 95)) if planted_heights.size else None,
        "joint_limits": {
            "violations": int(violations.sum()),
            "max_violation_rad": float(max(0.0, over.max())),
            "joints": sorted({JOINT_NAMES[j] for j in np.flatnonzero(violations.any(axis=0))}),
        },
        "joint_speed": {
            "max_rad_s": float(speeds.max(initial=0.0)),
            "fastest_joint": JOINT_NAMES[j_fast],
            "max_fraction_of_motor_limit": float(speed_fraction.max(initial=0.0)),
        },
        "root": {
            "height_min_m": float(motion.root_pos[:, 2].min()),
            "height_max_m": float(motion.root_pos[:, 2].max()),
            "max_ground_speed_mps": float(root_speed.max(initial=0.0)),
            "start_xy": motion.root_pos[0, :2].tolist(),
        },
        "contacts": {
            "fraction": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean())},
            "phases": {"left": len(contact_phases(contacts[:, 0])), "right": len(contact_phases(contacts[:, 1]))},
            "pattern": contact_pattern(contacts),
            "no_contact_frames": int((~contacts.any(axis=1)).sum()),
        },
        "plausibility": plausibility_metrics(motion, fk, floor, thresholds),
    }
    verdict, reasons = _verdict(metrics, thresholds)
    return {"verdict": verdict, "reasons": reasons, "format_ok": True, "metrics": metrics, "thresholds": asdict(thresholds)}


def _verdict(m: dict, th: QualityThresholds) -> tuple[str, list[str]]:
    fails, warns = [], []

    def grade(value: float | None, warn: float, fail: float, what: str) -> None:
        if value is None:
            return
        if value > fail:
            fails.append(f"{what} {value:.3f} > {fail}")
        elif value > warn:
            warns.append(f"{what} {value:.3f} > {warn}")

    grade(m["foot_slip"]["max_deviation_m"], th.slip_warn_m, th.slip_fail_m, "foot slip (m)")
    grade(m["penetration"]["max_depth_m"], th.penetration_warn_m, th.penetration_fail_m, "floor penetration (m)")
    grade(m["planted_float_p95_m"], th.float_warn_m, th.float_fail_m, "planted-foot float p95 (m)")
    grade(
        m["joint_speed"]["max_fraction_of_motor_limit"],
        th.speed_warn_fraction,
        th.speed_fail_fraction,
        f"joint speed / motor limit ({m['joint_speed']['fastest_joint']})",
    )
    if m["joint_limits"]["violations"]:
        fails.append(
            f"{m['joint_limits']['violations']} joint-limit violations, worst {m['joint_limits']['max_violation_rad']:.3f} rad "
            f"({', '.join(m['joint_limits']['joints'])})"
        )
    _plausibility_verdict(m["plausibility"], m["root"]["max_ground_speed_mps"], th, fails, warns)
    if m["duration_s"] < th.min_duration_s:
        warns.append(f"duration {m['duration_s']:.2f} s < {th.min_duration_s} s")
    if m["contacts"]["fraction"]["left"] == 0 and m["contacts"]["fraction"]["right"] == 0:
        warns.append("no foot contact detected at all")
    if fails:
        return "fail", fails + warns
    if warns:
        return "warn", warns
    return "pass", []


def _plausibility_verdict(p: dict, root_speed: float, th: QualityThresholds, fails: list, warns: list) -> None:
    held = f"for >= {th.sustain_s:g} s"
    if p["torso_tilt_sustained_deg"] > th.torso_tilt_fail_deg:
        fails.append(f"implausible: torso tilt {p['torso_tilt_sustained_deg']:.0f} deg {held} > {th.torso_tilt_fail_deg:g}")
    elif p["torso_tilt_max_deg"] > th.torso_tilt_warn_deg:
        warns.append(f"torso tilt {p['torso_tilt_max_deg']:.0f} deg > {th.torso_tilt_warn_deg:g}")
    if p["pelvis_low_sustained_m"] < th.pelvis_low_fail_m:
        fails.append(f"implausible: pelvis height {p['pelvis_low_sustained_m']:.2f} m {held} < {th.pelvis_low_fail_m:g}")
    elif p["pelvis_height_min_m"] < th.pelvis_low_warn_m:
        warns.append(f"pelvis height {p['pelvis_height_min_m']:.2f} m < {th.pelvis_low_warn_m:g}")
    if p["pelvis_high_sustained_m"] > th.pelvis_high_fail_m:
        fails.append(f"implausible: pelvis height {p['pelvis_high_sustained_m']:.2f} m {held} > {th.pelvis_high_fail_m:g}")
    if root_speed > th.root_speed_fail_mps:
        fails.append(f"implausible: root ground speed {root_speed:.1f} m/s > {th.root_speed_fail_mps:g} (teleport)")
    elif root_speed > th.root_speed_warn_mps:
        warns.append(f"root ground speed {root_speed:.1f} m/s > {th.root_speed_warn_mps:g}")
    for key, warn, fail, what in (
        ("retarget_ik_error_median", th.ik_error_warn, th.ik_error_fail, "retarget IK error median"),
        ("retarget_joint_limit_frac", th.limit_frac_warn, th.limit_frac_fail, "retarget frames at joint limits"),
    ):
        value = p[key]
        if value is None:
            continue
        if value > fail:
            fails.append(f"implausible: {what} {value:.2f} > {fail:g}")
        elif value > warn:
            warns.append(f"{what} {value:.2f} > {warn:g}")


def summary_line(report: dict) -> str:
    """One line for the provenance table."""
    if not report.get("format_ok"):
        return f"{report['verdict']}: {'; '.join(report['reasons'])}"
    m = report["metrics"]
    return (
        f"{report['verdict']}; slip {m['foot_slip']['max_deviation_m'] * 100:.1f} cm, "
        f"pen {m['penetration']['max_depth_m'] * 100:.1f} cm, "
        f"limits {m['joint_limits']['violations']}, "
        f"max {m['joint_speed']['max_rad_s']:.1f} rad/s, "
        f"{m['duration_s']:.2f} s"
    )
