"""Publish a cleaned motion as REPO/motions/<motion>_v<N>.csv with a provenance row.

Versions are never overwritten: N auto-increments and the CSV is created with O_EXCL, so
two concurrent publishes of the same motion get different versions instead of clobbering
each other. A `<motion>_v<N>.json` next to it keeps the full cleanup log and quality report;
the README row is the human summary.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import re
from dataclasses import dataclass
from pathlib import Path

from fencing import paths
from fencing.motion.io import Motion, write_csv, write_sidecar

MOTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
VERSIONED_FILE_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)_v(?P<version>\d+)\.csv$")
KNOWN_MOTIONS = ("en_garde", "advance", "retreat", "extension", "lunge", "advance_lunge", "advance_lunge_cut", "thrust")
KNOWN_PREFIXES = ("parry_", "cut_")
PROVENANCE_HEADING = "## Provenance"


class PublishError(RuntimeError):
    pass


@dataclass
class Provenance:
    source_clip: str  # clip_id (and video_log.md # when known)
    segment: str  # e.g. "frames 120-210 (4.00-7.00 s)"
    hand: str  # "R", "L -> mirrored", ...
    cleanup: str  # short list of steps
    notes: str  # metrics summary etc.


def check_motion_name(name: str, allow_unknown: bool = False) -> None:
    if not MOTION_NAME_RE.match(name):
        raise PublishError(f"motion name {name!r} must be lowercase letters, digits and underscores")
    if re.search(r"_v\d+$", name):
        raise PublishError(f"motion name {name!r} already ends in a version suffix; pass the bare name")
    known = name in KNOWN_MOTIONS or name.startswith(KNOWN_PREFIXES)
    if not known and not allow_unknown:
        raise PublishError(
            f"motion name {name!r} is not one of {', '.join(KNOWN_MOTIONS)} or parry_*/cut_* "
            "(motions/README.md naming); pass allow_unknown to override"
        )


def existing_versions(name: str, motions_dir: Path) -> list[int]:
    versions = []
    for path in motions_dir.glob(f"{name}_v*.csv"):
        match = VERSIONED_FILE_RE.match(path.name)
        if match and match.group("name") == name:
            versions.append(int(match.group("version")))
    return sorted(versions)


def next_version(name: str, motions_dir: Path) -> int:
    versions = existing_versions(name, motions_dir)
    return (versions[-1] + 1) if versions else 1


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def provenance_row(filename: str, prov: Provenance) -> str:
    cells = [f"`{filename}`", prov.source_clip, prov.segment, prov.hand, prov.cleanup, prov.notes]
    return "| " + " | ".join(_escape_cell(c) for c in cells) + " |"


def _is_placeholder_row(line: str) -> bool:
    return line.strip().startswith("|") and not line.replace("|", "").strip()


def append_provenance_row(readme: Path, row: str) -> None:
    """Insert `row` at the end of the table under '## Provenance' (drops the empty placeholder row)."""
    with open(readme, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        lines = fh.read().splitlines()
        try:
            heading = next(i for i, line in enumerate(lines) if line.strip() == PROVENANCE_HEADING)
        except StopIteration:
            raise PublishError(f"{readme} has no '{PROVENANCE_HEADING}' section to append to") from None
        table_start = next((i for i in range(heading + 1, len(lines)) if lines[i].lstrip().startswith("|")), None)
        if table_start is None:
            raise PublishError(f"{readme}: no table under '{PROVENANCE_HEADING}'")
        table_end = table_start
        while table_end < len(lines) and lines[table_end].lstrip().startswith("|"):
            table_end += 1
        body = [line for line in lines[table_start + 2 : table_end] if not _is_placeholder_row(line)]
        lines[table_start + 2 : table_end] = body + [row]
        fh.seek(0)
        fh.write("\n".join(lines) + "\n")
        fh.truncate()


def publish_motion(
    motion: Motion,
    name: str,
    provenance: Provenance,
    motions_dir: Path = paths.MOTIONS_DIR,
    record: dict | None = None,
    allow_unknown_name: bool = False,
) -> Path:
    """Write motions/<name>_v<N>.csv (+ .json record) and append the README provenance row."""
    check_motion_name(name, allow_unknown_name)
    motions_dir = Path(motions_dir)
    readme = motions_dir / "README.md"
    if not readme.exists():
        raise PublishError(f"{readme} not found; motions_dir must be the repo's motions/ folder")
    version = next_version(name, motions_dir)
    for _ in range(100):
        path = motions_dir / f"{name}_v{version}.csv"
        try:
            write_csv(motion, path, exclusive=True)
            break
        except FileExistsError:
            version += 1
    else:
        raise PublishError(f"could not find a free version for {name} in {motions_dir}")
    full_record = {
        "file": path.name,
        "version": version,
        "published_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fps": motion.fps,
        "frames": motion.n_frames,
        "provenance": provenance.__dict__,
        **(record or {}),
    }
    write_sidecar(path.with_suffix(".json"), full_record)
    append_provenance_row(readme, provenance_row(path.name, provenance))
    return path
