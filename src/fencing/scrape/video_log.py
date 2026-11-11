"""Render the scrape manifest into the markdown table in data/video_log.md.

Auto columns (Video, Channel, Dur, Guess) are refreshed from the manifest on every sync.
Human columns (Motion .. Notes) are written once, when a row is created, and never
overwritten, so hand-entered tags survive re-syncs. The one exception is filling a cell that
is still empty: a record may carry `_fill` ({column: value}, e.g. View/Start/End from the
screen report), applied only to blank cells. A video the scraper later excluded has its
Status moved from the default `new` to `rejected`; a blank Status or anything a human typed
stays. Row numbers are stable: a video keeps its # forever, new videos get max(#) + 1, and rows
are never dropped. `annotate_video_log` applies fencing.scrape.annotate's tags under the same
rules, plus one more Status move, `rejected` -> `retargeted`, which only direct evidence (a
retarget output or a published motion) asks for. Quality is never filled.

No cell is ever written with a `|` in it, not even the GFM-escaped form: plain-text readers and
renderers without the escape rule split on it and shift the rest of the row. Titles and
notes get `/` instead. A row split by a bare `|` someone typed is realigned on read only when
the shift can be proven (see `_fit`); otherwise parsing stops with the line and row number
rather than guessing, because the next sync would write the guess back for good.
"""

from __future__ import annotations

import re

COLUMNS = ["#", "Video", "Channel", "Dur", "Guess", "Motion", "Start", "End", "Hand", "View",
           "Fencers", "Quality", "Status", "Notes"]
AUTO_COLUMNS = ("Video", "Channel", "Dur", "Guess")
HUMAN_ONLY = ("#", "Quality")  # no data source may fill these: numbering, and an A/B/C judgement
HEADER_ALIASES = {"Link": "Video"}  # the table's original hand-made layout
TITLE_CHARS = 70
YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
PIPE = re.compile(r"\s*\\?\|\s*")
STATUSES = ("new", "gvhmr_ok", "retargeted", "rejected")
TIMESTAMP = r"\d+:\d{2}"
# Columns whose cells come from a closed vocabulary: a realigned row must fit all of them.
CLOSED_CELLS = {"Start": f"({TIMESTAMP})?", "End": f"({TIMESTAMP})?", "Hand": "[RL]?",
                "View": "(side|front|angled)?", "Fencers": r"\d*", "Quality": "[ABC]?",
                "Status": "|".join(STATUSES)}
# Text earlier syncs wrote themselves, with less detail than they can give now: replaceable as if
# empty. A person typing exactly this is not a real risk; anything else in the cell stays.
STALE_AUTO_TEXT = {"Notes": {"screen: no usable range"}}
STATUS_UPGRADES = {"new": {"gvhmr_ok", "retargeted", "rejected"}, "rejected": {"retargeted"}}


def _split_row(line: str) -> list[str]:
    cells = UNESCAPED_PIPE.split(line.strip())
    return [c.strip() for c in cells[1:-1]]


def _escape(text: str) -> str:
    return PIPE.sub(" / ", text).replace("\n", " ").strip()


def _lines_up(cells: list[str], header: list[str], title_split: bool) -> bool:
    patterns = dict(CLOSED_CELLS, Dur=TIMESTAMP) if title_split else CLOSED_CELLS
    return all(re.fullmatch(patterns[name], cells[i]) for i, name in enumerate(header) if name in patterns)


def _fit(cells: list[str], header: list[str]) -> list[str] | None:
    """Undo the shift a bare `|` inside a cell causes (one extra cell per pipe), or None when
    the shift cannot be proven.

    Status anchors it. Found where the header expects it, nothing before it moved, so the extra
    cells can only have come out of Notes, the column after it. Found exactly `extra` cells
    right of that, the split was before it; it is taken to be in the title only if Dur then
    reads as a duration, which rules out a pipe typed into Motion or Guess (it would land the
    motion in Dur). Either way every closed-vocabulary cell must fit afterwards.
    """
    extra = len(cells) - len(header)
    if extra <= 0:
        return cells
    if "Status" not in header or "Notes" != header[-1]:
        return None
    status = header.index("Status")
    if cells[status] in STATUSES:
        fitted = cells[:len(header) - 1] + [" / ".join(c for c in cells[len(header) - 1:] if c)]
        return fitted if _lines_up(fitted, header, title_split=False) else None
    if "Video" in header and "Dur" in header and cells[status + extra] in STATUSES:
        video = header.index("Video")
        fitted = cells[:video] + [" / ".join(cells[video:video + extra + 1])] + cells[video + extra + 1:]
        return fitted if _lines_up(fitted, header, title_split=True) else None
    return None


def _find_table(lines: list[str]) -> tuple[int, int] | None:
    """[start, end) line range of the first table whose header starts with '| #'."""
    for start, line in enumerate(lines):
        if line.lstrip().startswith("| #"):
            end = start
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            return start, end
    return None


def parse_table(markdown: str) -> tuple[list[str], list[dict]]:
    """Header and rows of the video table; each row also carries its YouTube id as '_id'."""
    lines = markdown.splitlines()
    span = _find_table(lines)
    if span is None:
        return [], []
    start, end = span
    header = [HEADER_ALIASES.get(name, name) for name in _split_row(lines[start])]
    rows = []
    for number, line in enumerate(lines[start + 2:end], start=start + 3):  # skip the |---| separator
        cells = _fit(_split_row(line), header)
        if cells is None:
            raw = _split_row(line)
            raise ValueError(
                f"video log line {number}, row #{raw[0] if raw else '?'}: {len(raw)} cells for {len(header)} "
                "columns and no provable realignment; a `|` typed inside a cell? Replace it with `/`.")
        row = {name: cells[i] if i < len(cells) else "" for i, name in enumerate(header)}
        match = YOUTUBE_ID.search(row.get("Video", ""))
        row["_id"] = match.group(1) if match else None
        rows.append(row)
    return header, rows


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _auto_cells(record: dict) -> dict[str, str]:
    title = record.get("title") or ""
    if len(title) > TITLE_CHARS:
        title = title[:TITLE_CHARS - 1] + "…"
    return {
        "Video": f"[{record['id']}]({record['url']}) {_escape(title)}",
        "Channel": _escape(record.get("channel") or ""),
        "Dur": _format_duration(record.get("duration")),
        "Guess": record.get("motion_guess") or "",
    }


def _row_number(row: dict) -> int:
    try:
        return int(row["#"])
    except (KeyError, ValueError):
        return 0  # rows without a usable number sort first and never block numbering


def _render(rows: list[dict]) -> list[str]:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(_escape(row.get(col, "")) for col in COLUMNS) + " |" for row in rows]
    return lines


def _fill_blank(row: dict, fill: dict[str, str]) -> list[str]:
    """Write `fill` into the row's empty hand-edited cells; the columns actually written."""
    written = []
    for col, value in fill.items():
        empty = not row.get(col) or row[col] in STALE_AUTO_TEXT.get(col, ())
        if col in COLUMNS and col not in AUTO_COLUMNS and col not in HUMAN_ONLY and empty and value \
                and _escape(value) != row.get(col):
            row[col] = _escape(value)
            written.append(col)
    return written


def _upgrade_status(row: dict, status: str | None, allowed: dict[str, set[str]] | None = None) -> bool:
    """Move Status off the default `new` (or along `allowed`); anything else a human typed,
    including a blank, stays."""
    if status in (allowed or {"new": set(STATUSES)}).get(row.get("Status"), set()) and status != row["Status"]:
        row["Status"] = status
        return True
    return False


def _replace_table(markdown: str, rows: list[dict]) -> str:
    lines = markdown.splitlines()
    span = _find_table(lines)
    if span is None:
        lines += [""] + _render(rows)
    else:
        lines[span[0]:span[1]] = _render(rows)
    return "\n".join(lines) + "\n"


def sync_video_log(markdown: str, records: list[dict]) -> str:
    """Return the markdown with its video table merged with `records` (manifest records)."""
    _, existing = parse_table(markdown)
    rows = [r for r in existing if any(v for k, v in r.items() if k not in {"#", "_id"})]
    by_id = {row["_id"]: row for row in rows if row["_id"]}
    next_number = max((_row_number(r) for r in rows), default=0) + 1

    for record in records:
        row = by_id.get(record["id"])
        if row is None:
            row = {col: "" for col in COLUMNS}
            row.update({"#": str(next_number), "_id": record["id"],
                        "Motion": record.get("motion_guess") or "", "Status": "new"})
            next_number += 1
            rows.append(row)
            by_id[record["id"]] = row
        row.update(_auto_cells(record))
        _fill_blank(row, record.get("_fill") or {})
        if record.get("status") == "excluded":
            _upgrade_status(row, "rejected")
            _fill_blank(row, {"Notes": f"excluded: {record.get('exclude_reason') or 'unusable'}"})

    rows.sort(key=_row_number)
    return _replace_table(markdown, rows)


def annotate_video_log(markdown: str, annotations: dict[str, dict]) -> tuple[str, dict[str, int]]:
    """Apply fencing.scrape.annotate's {yt_id: {"fill": {...}, "status": ...}} to existing rows
    under the same rules as a sync: empty cells only, Status only off the default `new`.
    Returns the markdown and how many cells each column gained."""
    _, rows = parse_table(markdown)
    rows = [r for r in rows if any(v for k, v in r.items() if k not in {"#", "_id"})]
    filled: dict[str, int] = {}
    for row in rows:
        annotation = annotations.get(row["_id"]) if row["_id"] else None
        if not annotation:
            continue
        written = _fill_blank(row, annotation.get("fill") or {})
        if _upgrade_status(row, annotation.get("status"), STATUS_UPGRADES):
            written.append("Status")
        for col in written:
            filled[col] = filled.get(col, 0) + 1
    rows.sort(key=_row_number)
    return _replace_table(markdown, rows), filled
