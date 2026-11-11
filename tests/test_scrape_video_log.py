"""Rendering the manifest into data/video_log.md (no network)."""

from __future__ import annotations

import pytest

from fencing.scrape.video_log import parse_table, sync_video_log

OLD_FORMAT = """# Video log

Intro text that must survive.

| # | Link | Motion | Start | End | Hand | View | Fencers | Quality | Status | Notes |
|---|------|--------|-------|-----|------|------|---------|---------|--------|-------|
| 1 | | | | | | | | | | |

Trailing text that must survive.
"""


def record(vid: str, title: str = "Sabre lunge drill", motion: str = "lunge", **extra) -> dict:
    return {
        "id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": title,
        "channel": "Club",
        "duration": 125,
        "motion_guess": motion,
        "score": 7.5,
        "status": "downloaded",
        **extra,
    }


def rows_by_id(markdown: str) -> dict[str, dict]:
    _, rows = parse_table(markdown)
    return {r["_id"]: r for r in rows if r.get("_id")}


def test_first_sync_replaces_placeholder_and_keeps_surrounding_text():
    out = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA"), record("BBBBBBBBBBB", motion="parry")])
    assert "Intro text that must survive." in out
    assert "Trailing text that must survive." in out
    rows = rows_by_id(out)
    assert rows["AAAAAAAAAAA"]["#"] == "1"
    assert rows["BBBBBBBBBBB"]["#"] == "2"
    assert rows["AAAAAAAAAAA"]["Motion"] == "lunge"
    assert rows["AAAAAAAAAAA"]["Status"] == "new"
    assert rows["AAAAAAAAAAA"]["Dur"] == "2:05"
    assert "https://www.youtube.com/watch?v=AAAAAAAAAAA" in rows["AAAAAAAAAAA"]["Video"]
    assert len(parse_table(out)[1]) == 2  # empty placeholder row dropped


def test_resync_keeps_numbers_and_human_fields():
    first = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA"), record("BBBBBBBBBBB")])
    edited = first.replace("| lunge | lunge |", "| lunge | en_garde |", 1)
    assert rows_by_id(edited)["AAAAAAAAAAA"]["Motion"] == "en_garde"
    edited = edited.replace("| new |", "| gvhmr_ok |", 1)

    # New video arrives first in manifest order; existing rows must keep their numbers.
    second = sync_video_log(edited, [record("CCCCCCCCCCC"), record("BBBBBBBBBBB"), record("AAAAAAAAAAA")])
    rows = rows_by_id(second)
    assert rows["AAAAAAAAAAA"]["#"] == "1"
    assert rows["BBBBBBBBBBB"]["#"] == "2"
    assert rows["CCCCCCCCCCC"]["#"] == "3"
    assert rows["AAAAAAAAAAA"]["Motion"] == "en_garde"
    assert rows["AAAAAAAAAAA"]["Status"] == "gvhmr_ok"


def test_rows_missing_from_manifest_are_kept():
    first = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA")])
    second = sync_video_log(first, [record("BBBBBBBBBBB")])
    assert set(rows_by_id(second)) == {"AAAAAAAAAAA", "BBBBBBBBBBB"}


def test_manual_rows_without_youtube_id_are_kept():
    manual = OLD_FORMAT.replace(
        "| 1 | | | | | | | | | | |",
        "| 4 | https://vimeo.com/123 | lunge | 0:10 | 0:20 | R | side | 1 | A | new | from a coach |",
    )
    out = sync_video_log(manual, [record("AAAAAAAAAAA")])
    _, rows = parse_table(out)
    vimeo = next(r for r in rows if "vimeo" in r["Video"])
    assert vimeo["#"] == "4" and vimeo["Notes"] == "from a coach" and vimeo["Hand"] == "R"
    assert rows_by_id(out)["AAAAAAAAAAA"]["#"] == "5"


def test_pipes_in_titles_do_not_break_the_table():
    out = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA", title="Sabre | lunge [drill]")])
    header, rows = parse_table(out)
    assert len(rows[0]) - 1 == len(header)  # every column parsed, plus the _id key
    assert rows[0]["Video"].count("](") == 1


def test_auto_columns_refresh_on_resync():
    first = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA", motion="lunge")])
    second = sync_video_log(first, [record("AAAAAAAAAAA", motion="parry")])
    row = rows_by_id(second)["AAAAAAAAAAA"]
    assert row["Guess"] == "parry"
    assert row["Motion"] == "lunge"  # human-owned once written


def test_fill_only_writes_blank_human_cells():
    first = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA")])
    filled = sync_video_log(first, [{**record("AAAAAAAAAAA"),
                                      "_fill": {"View": "side", "Start": "0:03", "Motion": "parry", "Video": "x"}}])
    row = rows_by_id(filled)["AAAAAAAAAAA"]
    assert row["View"] == "side" and row["Start"] == "0:03"
    assert row["Motion"] == "lunge"  # already set: the fill never overwrites
    assert "AAAAAAAAAAA" in row["Video"]  # auto columns are not fillable

    hand_edited = filled.replace("| side |", "| front |", 1)
    again = sync_video_log(hand_edited, [{**record("AAAAAAAAAAA"), "_fill": {"View": "side"}}])
    assert rows_by_id(again)["AAAAAAAAAAA"]["View"] == "front"


def test_excluded_video_is_marked_rejected_unless_a_human_set_the_status():
    first = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA"), record("BBBBBBBBBBB")])
    edited = first.replace("| new |", "| gvhmr_ok |", 1)  # row A
    excluded = [{**record(vid), "status": "excluded", "exclude_reason": "vertical video"}
                for vid in ("AAAAAAAAAAA", "BBBBBBBBBBB")]
    rows = rows_by_id(sync_video_log(edited, excluded))
    assert rows["AAAAAAAAAAA"]["Status"] == "gvhmr_ok"
    assert rows["BBBBBBBBBBB"]["Status"] == "rejected"
    assert rows["BBBBBBBBBBB"]["Notes"] == "excluded: vertical video"


def naive_cells(markdown: str) -> list[list[str]]:
    """Table rows split on every `|`, the way a plain text tool or a renderer without the GFM
    escape rule reads them."""
    return [line.split("|")[1:-1] for line in markdown.splitlines() if line.startswith("| ")]


def test_pipes_in_titles_and_channels_never_reach_the_file():
    out = sync_video_log(OLD_FORMAT, [record("AAAAAAAAAAA", title="Fencing Foil Epee Sabre | Footwork | Training",
                                             channel="Club | Academy")])
    assert all(len(cells) == 14 for cells in naive_cells(out))
    row = rows_by_id(out)["AAAAAAAAAAA"]
    assert row["Video"].endswith("Fencing Foil Epee Sabre / Footwork / Training")
    assert row["Channel"] == "Club / Academy"


def test_escaped_pipe_already_in_the_file_is_rewritten_on_resync():
    # The shape of rows 21/43/47/49 before the fix: valid GFM, but every other reader shifts
    # the cells after the title by one column per pipe.
    escaped = OLD_FORMAT.replace(
        "| 1 | | | | | | | | | | |",
        "| 1 | [AAAAAAAAAAA](https://www.youtube.com/watch?v=AAAAAAAAAAA) Sabre \\| Footwork | lunge | 0:09 | 0:26"
        " | R | angled | 13 | B | gvhmr_ok | coach in \\| frame |",
    )
    out = sync_video_log(escaped, [])
    assert all(len(cells) == 14 for cells in naive_cells(out))
    row = rows_by_id(out)["AAAAAAAAAAA"]
    assert (row["Hand"], row["View"], row["Quality"], row["Status"]) == ("R", "angled", "B", "gvhmr_ok")
    assert row["Notes"] == "coach in / frame"


CURRENT_FORMAT = """# Video log

| # | Video | Channel | Dur | Guess | Motion | Start | End | Hand | View | Fencers | Quality | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [AAAAAAAAAAA](https://www.youtube.com/watch?v=AAAAAAAAAAA) Sabre ROW
"""


def with_row(cells: str) -> str:
    return CURRENT_FORMAT.replace(" ROW", cells)


def test_row_split_by_a_stray_pipe_in_the_title_is_realigned():
    # An unescaped `|` in the title pushes every later cell one column right. It is undone only
    # when Status turns up exactly that far right and every closed-vocabulary cell then fits.
    stray = with_row(" | Footwork | Club | 2:05 | lunge | lunge | 0:09 | 0:26 | L | side | 1 | A | retargeted"
                     " | hand-typed |")
    row = rows_by_id(sync_video_log(stray, []))["AAAAAAAAAAA"]
    assert (row["Channel"], row["Dur"], row["Motion"], row["Start"], row["End"]) == ("Club", "2:05", "lunge", "0:09", "0:26")
    assert (row["Hand"], row["View"], row["Fencers"], row["Quality"]) == ("L", "side", "1", "A")
    assert (row["Status"], row["Notes"]) == ("retargeted", "hand-typed")
    assert row["Video"].endswith("Sabre / Footwork")


def test_stray_pipe_in_notes_stays_in_notes():
    # Status where the header expects it means nothing before it moved: the extra cell can only
    # have come out of Notes, the column after it.
    stray = OLD_FORMAT.replace(
        "| 1 | | | | | | | | | | |",
        "| 1 | https://vimeo.com/123 | lunge | 0:10 | 0:20 | R | side | 1 | A | new | slow | then fast |",
    )
    _, rows = parse_table(sync_video_log(stray, []))
    assert rows[0]["Status"] == "new" and rows[0]["Notes"] == "slow / then fast"


@pytest.mark.parametrize("cells", [
    # title split, Status blank: Status would read "A", Quality "1", Hand "0:26"
    " | Footwork | Club | 0:59 | lunge | lunge | 0:09 | 0:26 | R | side | 1 | A |  | note |",
    # title split, Status misspelled
    " | Footwork | Club | 0:59 | lunge | lunge | 0:09 | 0:26 | R | side | 1 | A | gvhmrok | note |",
    # title split and a Notes split: gvhmr_ok is two cells right, not one
    " | Footwork | Club | 0:59 | lunge | lunge | 0:09 | 0:26 | R | side | 1 | A | gvhmr_ok | coach | said |",
    # Status blank and a Notes split that happens to start with a status word
    " | Club | 0:59 | lunge | lunge | 0:09 | 0:26 | R | side | 1 | A |  | rejected | by coach |",
    # a pipe typed into Motion: realigning as a title split would push "lunge" into Guess
    " | Club | 0:59 | lunge | lunge | advance | 0:09 | 0:26 | R | side | 1 | A | gvhmr_ok |  |",
])
def test_row_with_a_stray_pipe_that_cannot_be_placed_stops_the_sync(cells):
    with pytest.raises(ValueError, match=r"line 5\b.*row #1\b"):
        sync_video_log(with_row(cells), [])


def test_blank_status_is_not_treated_as_the_default():
    blank = with_row(" | Club | 2:05 | lunge | lunge |  |  |  |  |  |  |  |  |")
    rows = rows_by_id(sync_video_log(blank, [{**record("AAAAAAAAAAA"), "status": "excluded"}]))
    assert rows["AAAAAAAAAAA"]["Status"] == ""
