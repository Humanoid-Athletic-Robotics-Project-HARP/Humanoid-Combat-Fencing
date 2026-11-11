"""`fencing scrape log --annotate`: tagging video-log rows from data the pipeline already wrote
(shortlist, screen reports, pose/retarget sidecars, published motion sidecars). No network."""

from __future__ import annotations

import json
from pathlib import Path

from fencing import paths
from fencing.scrape import cli
from fencing.scrape.annotate import clip_evidence, published_sources, video_annotations
from fencing.scrape.manifest import save_manifest
from fencing.scrape.video_log import annotate_video_log, parse_table, sync_video_log

PUB, CLIP, REJ, NONE = "pubxxxxxxxx", "clipxxxxxxx", "rejxxxxxxxx", "nonexxxxxxx"


def record(vid: str, motion: str = "lunge") -> dict:
    return {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": f"Sabre {motion} {vid}",
            "channel": "Club", "duration": 125, "motion_guess": motion, "score": 8.0, "status": "downloaded"}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def cleaned_motion(motions: Path, name: str, yt_id: str, hand: str | None, prov_hand: str,
                   hand_source: str = "sidecar") -> None:
    """Sidecar of a video-derived motion as `fencing motion publish` writes it."""
    sidecar = {"file": f"{name}.csv", "provenance": {"hand": prov_hand,
                                                      "source_clip": f"{yt_id}_0-5000 (youtube {yt_id}, 0.3-4.9 s)"}}
    if hand:
        sidecar |= {"hand": hand, "hand_source": hand_source}
    write_json(motions / f"{name}.json", sidecar)


def bladework_motion(motions: Path, name: str, yt_id: str) -> None:
    # The keyframed saber arm is always the robot's right arm: its provenance hand says nothing
    # about the fencer in the source video.
    source = f"{yt_id}_0-5000 (youtube {yt_id}, 0.3-4.9 s) via en_garde_v1.csv"
    write_json(motions / f"{name}.json", {"file": f"{name}.csv", "base": {"source": source, "synthetic": False},
                                          "provenance": {"hand": "R", "source_clip": source}})


def synthetic_motion(motions: Path, name: str) -> None:
    source = "SYNTHETIC (fencing.motion.synthetic: en_garde 1.00s; no video)"
    write_json(motions / f"{name}.json", {"file": f"{name}.csv", "base": {"source": source, "synthetic": True},
                                          "provenance": {"hand": "R", "source_clip": source}})


def processed_clip(data: Path, yt_id: str, start_ms: int, end_ms: int, hand: str | None, retargeted: bool = True):
    clip = f"{yt_id}_{start_ms}-{end_ms}"
    sidecar = {"clip_id": clip, "yt_id": yt_id, "source_start_s": start_ms / 1000, "source_end_s": end_ms / 1000,
               "handedness_guess": hand}
    write_json(data / "pose" / clip / "pose.json", sidecar)
    if retargeted:
        write_json(data / "retarget" / "csv" / f"{clip}.json", sidecar)


def shortlist_range(yt_id: str, start: float, end: float, rank: float, view: str = "oblique", people: int = 1) -> dict:
    return {"yt_id": yt_id, "start": start, "end": end, "rank": rank, "view": view, "people": people}


def screen(*ranges: tuple[float, float, bool], **video_level) -> dict:
    return video_level | {"ranges": [{"start_s": s, "end_s": e, "usable": ok, "view": "side", "people_median": 1,
                        "full_body_frac": 1.0, "camera": "static" if ok else "moving",
                        "reasons": [] if ok else ["camera moving (0.4)"]} for s, e, ok in ranges]}


# --- sources --------------------------------------------------------------------------------

def test_published_sources_take_the_hand_from_cleaned_motions_only(tmp_path):
    cleaned_motion(tmp_path, "advance_lunge_v1", PUB, hand="L", prov_hand="L (sidecar) -> mirrored to R")
    cleaned_motion(tmp_path, "advance_lunge_v2", PUB, hand=None, prov_hand="L (sidecar) -> mirrored to R")
    bladework_motion(tmp_path, "cut_head_v2", PUB)
    synthetic_motion(tmp_path, "cut_head_v1")
    sources = published_sources(tmp_path)
    assert set(sources) == {PUB}
    assert sources[PUB]["direct"] == ["advance_lunge_v1", "advance_lunge_v2"]
    assert sources[PUB]["base"] == ["cut_head_v2"]
    assert sources[PUB]["hands"] == {"L"}  # the bladework "R" is the robot's arm, not the fencer's


def test_detected_hand_is_a_guess_and_never_counts(tmp_path):
    # With no pose-stage handedness the cleaner runs detect_handedness, which answers R or L
    # even at confidence 0; only a hand the pose sidecar or a human config gave is evidence.
    cleaned_motion(tmp_path, "lunge_v1", PUB, hand="R", prov_hand="R (detected (confidence 0.00))",
                   hand_source="detected (confidence 0.00)")
    cleaned_motion(tmp_path, "lunge_v2", PUB, hand=None, prov_hand="L (detected (confidence 0.10))")
    assert published_sources(tmp_path)[PUB]["hands"] == set()
    cleaned_motion(tmp_path, "lunge_v3", PUB, hand="L", prov_hand="L (config)", hand_source="config")
    assert published_sources(tmp_path)[PUB]["hands"] == {"L"}


def test_clip_evidence_collects_decided_hands_and_retargeted_ranges(tmp_path):
    processed_clip(tmp_path, CLIP, 100900, 108700, "R")
    processed_clip(tmp_path, CLIP, 116600, 133600, None)  # undecided: neither for nor against R
    processed_clip(tmp_path, NONE, 0, 5000, "L", retargeted=False)
    clips = clip_evidence(tmp_path / "pose", tmp_path / "retarget")
    assert clips[CLIP]["hands"] == {"R"}
    assert clips[CLIP]["retargeted"] == [(100.9, 108.7), (116.6, 133.6)]
    assert clips[NONE] == {"hands": {"L"}, "retargeted": []}


# --- what each video gets --------------------------------------------------------------------

def test_best_ranked_range_gives_start_end_view_and_fencers():
    shortlist = {"ranges": [shortlist_range(CLIP, 10.2, 11.5, 0.4, view="front", people=2),
                            shortlist_range(CLIP, 30.0, 38.25, 0.9, view="oblique")]}
    notes = video_annotations({CLIP: record(CLIP)}, shortlist, {}, {}, {})[CLIP]
    assert notes["fill"] == {"Start": "0:30", "End": "0:39", "View": "angled", "Fencers": "1"}
    assert notes["status"] is None


def test_view_needs_the_same_agreement_between_ranges_as_the_sync():
    disagree = {"ranges": [shortlist_range(CLIP, 0, 10, 0.9, view="side"), shortlist_range(CLIP, 20, 30, 0.5, view="front")]}
    assert "View" not in video_annotations({CLIP: record(CLIP)}, disagree, {}, {}, {})[CLIP]["fill"]
    agree = {"ranges": [shortlist_range(CLIP, 0, 2, 0.9, view="front"), shortlist_range(CLIP, 20, 30, 0.5, view="oblique")]}
    assert video_annotations({CLIP: record(CLIP)}, agree, {}, {}, {})[CLIP]["fill"]["View"] == "angled"


def test_unknown_view_and_empty_detection_leave_those_cells_alone():
    shortlist = {"ranges": [shortlist_range(CLIP, 1, 5, 0.5, view="unknown", people=0)]}
    fill = video_annotations({CLIP: record(CLIP)}, shortlist, {}, {}, {})[CLIP]["fill"]
    assert "View" not in fill and "Fencers" not in fill


def test_published_source_is_retargeted_with_its_hand_and_motions_named():
    sources = {PUB: {"direct": ["advance_lunge_v1", "advance_lunge_v2", "advance_lunge_v3"],
                     "base": ["advance_lunge_cut_v2", "advance_lunge_cut_v3"], "hands": {"L"}}}
    clips = {PUB: {"hands": {"L"}, "retargeted": [(0.0, 20.0)]}}
    notes = video_annotations({PUB: record(PUB)}, {"ranges": []}, {}, sources, clips)[PUB]
    assert notes["status"] == "retargeted"
    assert notes["fill"]["Hand"] == "L"
    assert notes["fill"]["Notes"] == "published: advance_lunge v1-3; as base: advance_lunge_cut v2-3"


def test_conflicting_hand_evidence_leaves_hand_empty():
    sources = {PUB: {"direct": ["lunge_v1"], "base": [], "hands": {"R"}}}
    clips = {PUB: {"hands": {"L"}, "retargeted": [(0.0, 5.0)]}}
    fill = video_annotations({PUB: record(PUB)}, {"ranges": []}, {}, sources, clips)[PUB]["fill"]
    assert "Hand" not in fill


def test_retargeted_but_unpublished_clip_says_so():
    clips = {CLIP: {"hands": {"R"}, "retargeted": [(100.9, 108.7), (116.6, 133.6)]}}
    notes = video_annotations({CLIP: record(CLIP)}, {"ranges": []}, {}, {}, clips)[CLIP]
    assert notes["status"] == "retargeted" and notes["fill"]["Hand"] == "R"
    assert notes["fill"]["Notes"] == "retargeted, not published: 1:40-1:49, 1:56-2:14"


def test_screened_video_without_a_usable_range_is_rejected_with_the_reason():
    reports = {REJ: screen((0, 10, False))}
    notes = video_annotations({REJ: record(REJ)}, {"ranges": []}, reports, {}, {})[REJ]
    assert notes["status"] == "rejected"
    assert notes["fill"] == {"Notes": "screen: no usable range (camera moving)"}


def test_screen_without_any_range_still_gives_a_reason():
    reports = {REJ: screen(full_body_frac=0.035, camera="moving", chosen_track_id=2)}
    notes = video_annotations({REJ: record(REJ)}, {"ranges": []}, reports, {}, {})[REJ]
    assert notes["fill"]["Notes"] == "screen: no usable range (camera moving; full body visible in only 4% of frames)"
    nobody = video_annotations({REJ: record(REJ)}, {"ranges": []}, {REJ: screen(chosen_track_id=None)}, {}, {})
    assert nobody[REJ]["fill"]["Notes"] == "screen: no usable range (no fencer tracked)"


def test_unscreened_video_gets_nothing():
    assert video_annotations({NONE: record(NONE)}, {"ranges": []}, {}, {}, {}) == {}


# --- applying them to the log ----------------------------------------------------------------

def test_annotations_fill_only_empty_cells_and_upgrade_only_the_default_status():
    log = sync_video_log("# Video log\n", [record(PUB), record(CLIP), record(REJ)])
    rows = {r["_id"]: r for r in parse_table(log)[1]}
    assert rows[PUB]["Status"] == "new"
    # A human already tagged CLIP: its View, Notes and Status must survive.
    clip_line = next(line for line in log.splitlines() if CLIP in line)
    edited = clip_line.replace("|  |  |  |  |  |  | new |  |", "|  |  |  | front |  |  | gvhmr_ok | coach said B |")
    log = log.replace(clip_line, edited)
    annotations = {
        PUB: {"fill": {"Hand": "L", "Notes": "published: lunge v1", "Quality": "A"}, "status": "retargeted"},
        CLIP: {"fill": {"View": "angled", "Hand": "R", "Notes": "retargeted, not published: 0:01-0:05"},
               "status": "retargeted"},
        REJ: {"fill": {}, "status": "rejected"},
    }
    out, filled = annotate_video_log(log, annotations)
    rows = {r["_id"]: r for r in parse_table(out)[1]}
    assert (rows[PUB]["Hand"], rows[PUB]["Status"], rows[PUB]["Notes"]) == ("L", "retargeted", "published: lunge v1")
    assert rows[PUB]["Quality"] == ""  # A/B/C is a human judgement, never filled
    assert (rows[CLIP]["View"], rows[CLIP]["Status"], rows[CLIP]["Notes"]) == ("front", "gvhmr_ok", "coach said B")
    assert rows[CLIP]["Hand"] == "R"
    assert rows[REJ]["Status"] == "rejected"
    assert filled == {"Hand": 2, "Notes": 1, "Status": 2}


def test_reasonless_screen_note_from_older_syncs_gets_the_reason():
    log = sync_video_log("# Video log\n", [{**record(REJ), "_fill": {"Notes": "screen: no usable range"}}])
    out, _ = annotate_video_log(log, {REJ: {"fill": {"Notes": "screen: no usable range (no fencer tracked)"},
                                            "status": "rejected"}})
    assert {r["_id"]: r for r in parse_table(out)[1]}[REJ]["Notes"] == "screen: no usable range (no fencer tracked)"


def test_rejected_moves_to_retargeted_on_direct_evidence_but_never_back():
    log = sync_video_log("# Video log\n", [record(PUB), record(CLIP)])
    log = log.replace("| new |", "| rejected |")
    out, filled = annotate_video_log(log, {PUB: {"fill": {}, "status": "retargeted"}})
    assert {r["_id"]: r for r in parse_table(out)[1]}[PUB]["Status"] == "retargeted"
    out, filled = annotate_video_log(log.replace("| rejected |", "| gvhmr_ok |"),
                                     {PUB: {"fill": {}, "status": "retargeted"}, CLIP: {"fill": {}, "status": "rejected"}})
    rows = {r["_id"]: r for r in parse_table(out)[1]}
    assert (rows[PUB]["Status"], rows[CLIP]["Status"]) == ("gvhmr_ok", "gvhmr_ok") and filled == {}


def test_annotating_twice_changes_nothing_the_second_time():
    log = sync_video_log("# Video log\n", [record(PUB), record(REJ)])
    annotations = {PUB: {"fill": {"Hand": "R", "Start": "0:01", "Notes": "published: lunge v1"}, "status": "retargeted"},
                   REJ: {"fill": {"Notes": "screen: no usable range (no fencer tracked)"}, "status": "rejected"}}
    once, _ = annotate_video_log(log, annotations)
    twice, filled = annotate_video_log(once, annotations)
    assert twice == once and filled == {}


def test_log_stops_without_writing_when_a_row_cannot_be_read(tmp_path):
    manifest = tmp_path / "videos" / "manifest.jsonl"
    save_manifest(manifest, {PUB: record(PUB)})
    video_log = tmp_path / "video_log.md"
    broken = ("# Video log\n\n| # | Video | Channel | Dur | Guess | Motion | Start | End | Hand | View | Fencers"
              " | Quality | Status | Notes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
              f"| 7 | [{PUB}](https://www.youtube.com/watch?v={PUB}) a | b | Club | 0:59 | lunge | lunge | 0:09 | 0:26"
              " | R | side | 1 | A |  | x |\n")
    video_log.write_text(broken, encoding="utf-8")
    assert cli.main(["--manifest", str(manifest), "log", "--video-log", str(video_log)]) == 1
    assert video_log.read_text(encoding="utf-8") == broken


def test_log_annotate_flag_tags_rows_from_files_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MOTIONS_DIR", tmp_path / "motions")
    monkeypatch.setattr(paths, "POSE", tmp_path / "pose")
    monkeypatch.setattr(paths, "RETARGET", tmp_path / "retarget")
    manifest = tmp_path / "videos" / "manifest.jsonl"
    save_manifest(manifest, {PUB: record(PUB), REJ: record(REJ)})
    write_json(manifest.with_name("shortlist.json"), {"ranges": [shortlist_range(PUB, 0.33, 4.93, 0.8)]})
    write_json(tmp_path / "pose" / REJ / "screen.json", screen((0, 10, False)))
    cleaned_motion(tmp_path / "motions", "lunge_v1", PUB, hand="R", prov_hand="R (sidecar)")
    processed_clip(tmp_path, PUB, 0, 5000, "R")
    video_log = tmp_path / "video_log.md"
    video_log.write_text("# Video log\n", encoding="utf-8")

    base = ["--manifest", str(manifest), "log", "--video-log", str(video_log)]
    assert cli.main(base) == 0
    plain = {r["_id"]: r for r in parse_table(video_log.read_text(encoding="utf-8"))[1]}
    assert plain[PUB]["Status"] == "new" and plain[PUB]["Hand"] == ""  # annotation is opt-in

    assert cli.main([*base, "--annotate"]) == 0
    rows = {r["_id"]: r for r in parse_table(video_log.read_text(encoding="utf-8"))[1]}
    assert (rows[PUB]["Start"], rows[PUB]["End"], rows[PUB]["View"]) == ("0:00", "0:05", "angled")
    assert (rows[PUB]["Hand"], rows[PUB]["Status"], rows[PUB]["Notes"]) == ("R", "retargeted", "published: lunge v1")
    assert rows[REJ]["Status"] == "rejected"
