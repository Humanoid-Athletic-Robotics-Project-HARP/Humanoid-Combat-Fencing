"""Scored key frames of bladework motions: the instants the blade-tip error is judged at.

keyframes.py *builds* a bladework motion from blade keys (en garde -> parry -> en garde,
...). This module says which of those keys a policy is scored on, records them in the
motion's sidecar (`key_frames`, next to `blade.keys`) and finds them again from the NPZ a
policy was trained on.

Rule: every key except the rest guard (`en_garde`) and hand-backs to the base arm. For
the Goal 1 recipes that is exactly one key: the held parry, the cut or thrust landing, the
end of the extension, the cut at lunge touchdown. Its time is the frame the keyframer
*reached* the target (KeyReport.frame), so the number comes from the recipe's own schedule,
never from hand-typed seconds.

Pure python (json, hashlib) so mjlab eval, CPU sim2sim and MTC scoring can all use it.

    fencing motion key-frames [SIDECAR ...] [--write]   # backfill published sidecars
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fencing import paths

REST_TARGETS = ("en_garde",)
BASE_ACTION = "base"  # keyframes.BASE_ACTION; not imported to keep this module free of mujoco
SIDECAR_FIELD = "key_frames"


def key_frames_from_blade_keys(keys: list[dict], fps: float) -> list[dict]:
    """Scored keys from a sidecar's `blade.keys` (KeyReport dicts), in script order."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    scored = []
    for key in keys:
        if key["action"] == BASE_ACTION or key["target"] in REST_TARGETS:
            continue
        frame = int(key["frame"])
        scored.append({"index": int(key["index"]), "target": key["target"], "kind": key["kind"], "frame": frame,
                       "time_s": round(frame / fps, 6)})
    return scored


def with_key_frames(record: dict) -> dict:
    """Copy of a motion sidecar record with `key_frames` derived from its blade keys."""
    keys = (record.get("blade") or {}).get("keys")
    if not keys:
        raise ValueError(f"{record.get('file', 'sidecar')} has no blade keys: not a bladework motion")
    return {**record, SIDECAR_FIELD: key_frames_from_blade_keys(keys, float(record["fps"]))}


def key_steps(key_frames: list[dict], step_dt: float, num_frames: int) -> list[int]:
    """Reference frame at the policy rate for each key: mjlab's csv_to_npz samples the motion
    at t = k * step_dt from t = 0, so key time t lands on frame round(t / step_dt). Keys past
    the last frame are dropped."""
    steps = [int(round(float(k["time_s"]) / step_dt)) for k in key_frames]
    return [s for s in steps if 0 <= s < num_frames]


@dataclass(frozen=True)
class ResolvedKeyFrames:
    steps: tuple[int, ...]  # reference frames at the policy rate (= policy steps from the motion start)
    sidecar: str
    keys: tuple[dict, ...]

    def report(self) -> dict:
        return {"sidecar": self.sidecar, "steps": list(self.steps), "keys": list(self.keys)}


def _warn(message: str) -> None:
    print(f"[key-frames] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _warn(f"cannot read {path}: {error}; no key frames")
        return {}
    return data if isinstance(data, dict) else {}


def resolve_for_npz(npz: str | Path | None, step_dt: float, num_frames: int) -> ResolvedKeyFrames | None:
    """Key frames of the motion an NPZ was converted from, or None when it has none.

    NPZ -> its `fencing train npz` sidecar (source CSV + sha256) -> the CSV's motion sidecar
    -> `key_frames`. None, never an error, when any link is missing (a video motion, an NPZ
    converted by hand, an un-backfilled sidecar): scoring must not break on that. A CSV whose
    sha256 no longer matches the NPZ's record is refused, since its sidecar may describe a
    different motion; so is an NPZ whose own sha256 differs from the record's `npz_sha256`
    (when the record has one). None too when no key lies inside the motion.
    """
    if npz is None:
        return None
    npz = Path(npz)
    npz_record_path = npz.with_suffix(".json")
    if not npz_record_path.is_file():
        return None
    npz_record = _read_json(npz_record_path)
    source = npz_record.get("source_csv")
    if not source or not Path(source).is_file():
        return None
    csv = Path(source)
    expected = npz_record.get("source_csv_sha256")
    if expected and _sha256(csv) != expected:
        _warn(f"{csv} sha256 differs from the one recorded when {npz.name} was made; ignoring its key frames")
        return None
    npz_expected = npz_record.get("npz_sha256")
    if npz_expected and npz.is_file() and _sha256(npz) != npz_expected:
        _warn(f"{npz.name} npz_sha256 differs from its record (re-converted?); ignoring its key frames")
        return None
    sidecar = csv.with_suffix(".json")
    if not sidecar.is_file():
        return None
    key_frames = _read_json(sidecar).get(SIDECAR_FIELD)
    if not key_frames:
        return None
    steps = key_steps(key_frames, step_dt, num_frames)
    if not steps:  # every key past the motion end: report exactly as a run with no key frames
        return None
    return ResolvedKeyFrames(tuple(steps), str(sidecar), tuple(key_frames))


# ---------------------------------------------------------------- backfill


def backfill(sidecar: Path, write: bool = False) -> bool:
    """Derive `key_frames` for one published sidecar; True if it is missing or differs.
    Rewrites the file (same sorted, indented format as publishing) only with write=True."""
    from fencing.motion.io import write_sidecar

    record = json.loads(Path(sidecar).read_text())
    updated = with_key_frames(record)
    if record.get(SIDECAR_FIELD) == updated[SIDECAR_FIELD]:
        return False
    if write:
        write_sidecar(sidecar, updated)
    return True


def newest_bladework_sidecars(motions_dir: Path = paths.MOTIONS_DIR) -> list[Path]:
    """Sidecar of the newest published version of every bladework recipe."""
    from fencing.motion.publish import existing_versions
    from fencing.saber.recipes import RECIPES

    found = []
    for name in sorted(RECIPES):
        versions = existing_versions(name, motions_dir)
        if versions:
            sidecar = Path(motions_dir) / f"{name}_v{versions[-1]}.json"
            if sidecar.is_file():
                found.append(sidecar)
    return found
