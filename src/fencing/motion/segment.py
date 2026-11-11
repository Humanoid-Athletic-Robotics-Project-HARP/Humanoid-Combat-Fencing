"""Find single-motion segments in a long raw trajectory and label them.

Heuristics (all distances in the fencing-line frame, so camera heading does not matter):

1. Handedness picks the front foot (sword side). The fencing line u is the median
   direction back foot -> front foot over wide-stance frames.
2. STILL frames: both feet planted and pelvis ground speed < still_speed. A foot is planted
   when it is in contact or barely moving: monocular depth noise makes raw feet float in and
   out of contact, but a foot standing still is still standing. Still runs of
   at least min_still_s are en garde holds; the median stance width / pelvis height over
   them is the en garde reference.
3. ACTIVE intervals lie between still runs. An interval is split where both feet are
   planted at roughly en garde width (0.8-1.2x) -- the moment between two actions -- and at
   every valley of the stance width: each step widens the stance (leading foot) and
   narrows it again (trailing foot), so back-to-back steps that never stop are still cut
   one per step. A lunge widens once and recovers once, so it has no interior valley.
4. Each piece is classified:
     lunge    stance widens to >= lunge_stance_ratio x en garde and the pelvis drops
              >= lunge_drop_m, the widening is led by the front foot and the back foot
              stays planted apart from at most one advance step (a retreat or an
              advance widens the stance mid-step too); confidence is halved when the stance does not come back
              (no recovery seen).
     advance  pelvis moves +min_step_m .. max_step_m along u, both feet step, the front
              foot leaves the floor first.
     retreat  same backwards, back foot first.
   advance_lunge: adjacent advance -> lunge pieces with no still hold between them, or
              a lunge piece whose back foot already stepped >= min_step_m forward
              before full extension (the advance never paused at en garde width).
5. en_garde: still runs >= min_en_garde_s with a fencing-width stance.
6. Every segment is padded with up to pad_s of the neighbouring hold so it starts and ends
   in en garde where the data allows.
Confidence is in [0, 1] and is a margin score (how far past each threshold), not a probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks

from fencing.motion.contacts import ContactParams, contacts_from_fk
from fencing.motion.filters import lowpass, runs, smoothstep
from fencing.motion.io import Motion
from fencing.motion.kinematics import FKResult, forward_kinematics
from fencing.motion.normalize import detect_handedness

LABELS = ("en_garde", "advance", "retreat", "lunge", "advance_lunge")


@dataclass(frozen=True)
class SegmentParams:
    still_speed_mps: float = 0.12
    min_still_s: float = 0.2
    min_en_garde_s: float = 0.8
    min_en_garde_width_m: float = 0.25
    min_active_s: float = 0.15
    min_step_m: float = 0.08
    max_step_m: float = 0.8
    lunge_stance_ratio: float = 1.4
    lunge_drop_m: float = 0.025
    pad_s: float = 0.3
    planted_speed_mps: float = 0.15  # a foot slower than this counts as planted
    valley_prominence: float = 0.15  # x en garde width, for splitting at stance valleys
    valley_max_ratio: float = 1.2  # x en garde width: a valley must be near en garde


@dataclass(frozen=True)
class Segment:
    start: int  # first frame (inclusive)
    end: int  # last frame + 1
    label: str
    confidence: float
    features: dict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LineFeatures:
    """Per-frame quantities in the fencing-line frame."""

    front: int  # foot index of the front foot (1 = right for a right-hander)
    line: np.ndarray  # (2,) unit fencing direction
    front_s: np.ndarray  # (T,) front foot position along the line
    back_s: np.ndarray
    pelvis_s: np.ndarray
    pelvis_z: np.ndarray
    pelvis_speed: np.ndarray  # (T,) ground-plane speed, m/s
    stance: np.ndarray  # (T,) front_s - back_s
    contacts: np.ndarray  # (T, 2)
    foot_speed: np.ndarray  # (T, 2) ground-plane speed of each sole, m/s
    hand: str
    hand_confidence: float


def line_features(motion: Motion, fk: FKResult | None = None, hand: str | None = None) -> LineFeatures:
    fk = fk or forward_kinematics(motion)
    contacts, floor = contacts_from_fk(fk, motion.fps, ContactParams())
    if hand is None:
        detected = detect_handedness(fk)
        hand, hand_conf = detected.hand, detected.confidence
    else:
        hand_conf = 1.0
    front = 1 if hand == "R" else 0
    feet_xy = fk.sole_center[..., :2]
    sep = feet_xy[:, front] - feet_xy[:, 1 - front]
    width = np.linalg.norm(sep, axis=1)
    wide = width > np.percentile(width, 25)
    unit = sep[wide] / width[wide, None]
    line = np.median(unit, axis=0)
    line /= max(np.linalg.norm(line), 1e-9)
    pelvis_xy = lowpass(fk.pelvis_pos[:, :2], 3.0, motion.fps)
    speed = np.linalg.norm(np.gradient(pelvis_xy, 1.0 / motion.fps, axis=0), axis=1)
    front_s, back_s = feet_xy[:, front] @ line, feet_xy[:, 1 - front] @ line
    foot_speed = np.linalg.norm(np.gradient(lowpass(feet_xy, 4.0, motion.fps), 1.0 / motion.fps, axis=0), axis=-1)
    return LineFeatures(
        front=front,
        line=line,
        front_s=front_s,
        back_s=back_s,
        pelvis_s=fk.pelvis_pos[:, :2] @ line,
        pelvis_z=fk.pelvis_pos[:, 2] - floor,
        pelvis_speed=speed,
        stance=front_s - back_s,
        contacts=contacts,
        foot_speed=foot_speed,
        hand=hand,
        hand_confidence=hand_conf,
    )


def _planted(f: LineFeatures, p: SegmentParams) -> np.ndarray:
    return f.contacts | (f.foot_speed < p.planted_speed_mps)


def _margin(value: float, threshold: float, scale: float) -> float:
    """0 at the threshold, 1 at threshold + scale (clipped)."""
    return float(np.clip((value - threshold) / scale, 0.0, 1.0))


def _first_liftoff(contact: np.ndarray, start: int, end: int) -> int | None:
    off = np.flatnonzero(~contact[start:end])
    return int(start + off[0]) if off.size else None


def _classify(f: LineFeatures, start: int, end: int, ref_width: float, ref_height: float, p: SegmentParams) -> Segment | None:
    sl = slice(start, end)
    stance_ratio = float(f.stance[sl].max() / ref_width)
    drop = float(ref_height - f.pelvis_z[sl].min())
    displacement = float(f.pelvis_s[end - 1] - f.pelvis_s[start])
    end_ratio = float(f.stance[end - 1] / ref_width)
    front_moved = float(np.ptp(f.front_s[sl]))
    back_moved = float(np.ptp(f.back_s[sl]))
    front_off = _first_liftoff(f.contacts[:, f.front], start, end)
    back_off = _first_liftoff(f.contacts[:, 1 - f.front], start, end)
    peak = start + int(np.argmax(f.stance[sl]))
    widening = float(f.stance[peak] - f.stance[start])
    front_lead = float(f.front_s[peak] - f.front_s[start])
    back_retreat = float(f.back_s[sl].min() - f.back_s[start])
    # A back-foot step forward before full extension means the lunge was preceded by an
    # advance that never paused at en garde width (no split point was found).
    back_advance = float(f.back_s[peak] - f.back_s[start])
    feats = {
        "stance_ratio_max": stance_ratio,
        "pelvis_drop_m": drop,
        "displacement_m": displacement,
        "end_stance_ratio": end_ratio,
        "front_foot_travel_m": front_moved,
        "back_foot_travel_m": back_moved,
    }
    # Apart from that one advance step the back foot stays put: a piece where both feet
    # travel the same distance is a step, however wide the stance gets mid-step.
    back_planted = back_moved - max(back_advance, 0.0) <= 2 * p.min_step_m
    lunge_shape = front_lead >= 0.5 * widening and back_retreat > -p.min_step_m and back_moved <= p.max_step_m and back_planted
    if stance_ratio >= p.lunge_stance_ratio and drop >= p.lunge_drop_m and lunge_shape:
        recovered = end_ratio < 0.5 * (1.0 + p.lunge_stance_ratio)
        conf = 0.5 * _margin(stance_ratio, p.lunge_stance_ratio, 0.4) + 0.5 * _margin(drop, p.lunge_drop_m, 0.05)
        feats["recovered"] = bool(recovered)
        feats["back_foot_advance_before_peak_m"] = back_advance
        label = "advance_lunge" if back_advance >= p.min_step_m else "lunge"
        return Segment(start, end, label, (0.5 + 0.5 * conf) * (1.0 if recovered else 0.5), feats)
    if p.min_step_m <= abs(displacement) <= p.max_step_m and min(front_moved, back_moved) >= 0.5 * p.min_step_m:
        forward = displacement > 0
        label = "advance" if forward else "retreat"
        leader_ok = None
        if front_off is not None and back_off is not None:
            leader_ok = (front_off <= back_off) if forward else (back_off <= front_off)
        feats["lead_foot_order_ok"] = leader_ok
        conf = 0.5 + 0.5 * _margin(abs(displacement), p.min_step_m, 0.15)
        if leader_ok is False:
            conf *= 0.5
        return Segment(start, end, label, conf, feats)
    return None


def _split_points(f: LineFeatures, start: int, end: int, ref_width: float, fps: float, p: SegmentParams) -> list[int]:
    """Frames inside an active interval where one action ends and the next begins."""
    both = _planted(f, p)[start:end].all(axis=1)
    ratio = f.stance[start:end] / ref_width
    neutral = both & (ratio > 0.8) & (ratio < 1.2)
    cuts = []
    for s, e in runs(neutral):
        if s > 0 and e < end - start:  # touching the edge means it is part of the hold
            cuts.append(start + (s + e) // 2)
    stance = lowpass(f.stance, 3.0, fps)[start:end]
    valleys, _ = find_peaks(-stance, prominence=p.valley_prominence * ref_width)
    min_gap = p.min_active_s * fps
    for v in valleys:
        if stance[v] < p.valley_max_ratio * ref_width and all(abs(start + v - c) >= min_gap for c in cuts):
            cuts.append(start + int(v))
    return sorted(cuts)


def find_segments(
    motion: Motion,
    params: SegmentParams = SegmentParams(),
    hand: str | None = None,
    fk: FKResult | None = None,
) -> tuple[list[Segment], LineFeatures]:
    f = line_features(motion, fk, hand)
    fps = motion.fps
    still_mask = _planted(f, params).all(axis=1) & (f.pelvis_speed < params.still_speed_mps)
    still = [(s, e) for s, e in runs(still_mask) if e - s >= params.min_still_s * fps]
    if still:
        frames = np.concatenate([np.arange(s, e) for s, e in still])
        ref_width = float(np.median(f.stance[frames]))
        # A held lunge extension is also "still"; it is not en garde, and splitting there
        # would cut a lunge from its recovery. Holds are the majority of still frames, so
        # the first median is close to en garde width; refine it without the wide runs.
        still = [(s, e) for s, e in still if np.median(f.stance[s:e]) < 1.25 * ref_width]
    if still:
        frames = np.concatenate([np.arange(s, e) for s, e in still])
        ref_width = float(np.median(f.stance[frames]))
        ref_height = float(np.median(f.pelvis_z[frames]))
    else:
        ref_width = float(np.percentile(f.stance, 25))
        ref_height = float(np.percentile(f.pelvis_z, 75))
    ref_width = max(ref_width, 0.05)

    pad = int(round(params.pad_s * fps))
    boundaries = [0] + [x for span in still for x in span] + [motion.n_frames]
    active = [(boundaries[i], boundaries[i + 1]) for i in range(0, len(boundaries), 2)]
    pieces: list[Segment] = []
    for a, b in active:
        if b - a < params.min_active_s * fps:
            continue
        cuts = [a] + _split_points(f, a, b, ref_width, fps, params) + [b]
        for s, e in zip(cuts, cuts[1:]):
            if e - s >= params.min_active_s * fps:
                seg = _classify(f, s, e, ref_width, ref_height, params)
                if seg is not None:
                    pieces.append(seg)

    segments = _merge_advance_lunge(pieces, still)
    segments = [_pad(seg, still, pad, motion.n_frames) for seg in segments]
    for s, e in still:
        if e - s >= params.min_en_garde_s * fps:
            width = float(np.median(f.stance[s:e]))
            if width >= params.min_en_garde_width_m:
                conf = 0.5 + 0.5 * _margin(width, params.min_en_garde_width_m, 0.15)
                segments.append(Segment(s, e, "en_garde", conf * max(f.hand_confidence, 0.5), {"stance_width_m": width}))
    segments.sort(key=lambda seg: (seg.start, seg.end))
    return segments, f


def _merge_advance_lunge(pieces: list[Segment], still: list[tuple[int, int]]) -> list[Segment]:
    merged: list[Segment] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        hold_between = nxt is not None and any(cur.end <= s and e <= nxt.start for s, e in still)
        if cur.label == "advance" and nxt is not None and nxt.label == "lunge" and not hold_between:
            feats = {"advance": cur.features, "lunge": nxt.features}
            merged.append(Segment(cur.start, nxt.end, "advance_lunge", min(cur.confidence, nxt.confidence), feats))
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


def _pad(seg: Segment, still: list[tuple[int, int]], pad: int, n: int) -> Segment:
    """Extend into the neighbouring holds (only into still frames, never into other actions)."""
    start, end = seg.start, seg.end
    for s, e in still:
        if e == seg.start or (s < seg.start <= e):
            start = max(s, seg.start - pad)
        if s == seg.end or (s <= seg.end < e):
            end = min(e, seg.end + pad)
    return Segment(max(0, start), min(n, end), seg.label, seg.confidence, seg.features)


def best_segment(segments: list[Segment], label: str) -> Segment | None:
    candidates = [s for s in segments if s.label == label]
    return max(candidates, key=lambda s: (s.confidence, s.end - s.start)) if candidates else None


# ---------------------------------------------------------------- trimming


def hold_pose(motion: Motion, start_hold_s: float = 0.5, end_hold_s: float = 0.5, ease_s: float = 0.25) -> tuple[Motion, dict]:
    """Add a static hold of the first/last frame at both ends.

    BeyondMimic starts episodes at random phases but the first frames anchor the initial
    state; a clean static start and end makes the motion easy to enter and leave. The
    first/last `ease_s` of the motion are eased so velocity ramps from zero instead of
    jumping at the junction.
    """
    fps = motion.fps
    n_start, n_end = int(round(start_hold_s * fps)), int(round(end_hold_s * fps))
    n_ease = min(int(round(ease_s * fps)), motion.n_frames // 2)
    out = motion.copy()
    if n_ease > 1:
        ramp = smoothstep(np.linspace(0.0, 1.0, n_ease))[:, None]
        for arr in (out.root_pos, out.joint_pos):
            first, last = arr[0].copy(), arr[-1].copy()
            arr[:n_ease] = first + ramp * (arr[:n_ease] - first)
            arr[-n_ease:] = last + ramp[::-1] * (arr[-n_ease:] - last)
    pieces = [
        np.repeat(out.to_array()[:1], n_start, axis=0),
        out.to_array(),
        np.repeat(out.to_array()[-1:], n_end, axis=0),
    ]
    result = Motion.from_array(np.concatenate(pieces), fps, dict(motion.meta))
    return result, {"start_hold_s": start_hold_s, "end_hold_s": end_hold_s, "ease_s": ease_s}


def blend_to_pose(motion: Motion, pose_joints: np.ndarray, blend_s: float = 0.4) -> tuple[Motion, dict]:
    """Blend joints at both ends toward a reference en garde pose (e.g. a published
    en_garde_vN first frame), so every motion starts and ends in the same stance."""
    n = min(int(round(blend_s * motion.fps)), motion.n_frames // 2)
    out = motion.copy()
    if n < 2:
        return out, {"blend_frames": 0}
    w = smoothstep(np.linspace(1.0, 0.0, n))[:, None]
    out.joint_pos[:n] = w * pose_joints + (1 - w) * out.joint_pos[:n]
    out.joint_pos[-n:] = w[::-1] * pose_joints + (1 - w[::-1]) * out.joint_pos[-n:]
    return out, {"blend_frames": n}
