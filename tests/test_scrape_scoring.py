"""Metadata filter, scoring and motion guessing for `fencing scrape` (no network)."""

from __future__ import annotations

import pytest

from fencing.scrape.scoring import (
    ScrapeFilter,
    detect_weapons,
    evaluate,
    fold,
    guess_motion,
    select_for_download,
)


def entry(title: str, description: str = "", duration: float | None = 120, **extra) -> dict:
    return {
        "id": extra.pop("id", "abcdefghijk"),
        "url": extra.pop("url", "https://www.youtube.com/watch?v=abcdefghijk"),
        "title": title,
        "description": description,
        "duration": duration,
        "channel": extra.pop("channel", "Some Club"),
        **extra,
    }


FILTER = ScrapeFilter()


# --- text normalisation -----------------------------------------------------------------

def test_fold_strips_latin_accents_but_keeps_hangul_and_cyrillic():
    assert fold("Épée Leçon") == "epee lecon"
    assert fold("펜싱 사브르") == "펜싱 사브르"
    assert fold("Сабля") == "сабля"


@pytest.mark.parametrize(
    "text, weapons",
    [
        ("Sabre footwork drill", {"saber"}),
        ("saber lesson", {"saber"}),
        ("Épée lunge", {"epee"}),
        ("fleuret escrime", {"foil"}),
        ("kardvívás lábmunka", {"saber"}),
        ("sciabola scherma", {"saber"}),
        ("펜싱 사브르", {"saber"}),
        ("сабля фехтование", {"saber"}),
        ("foil vs sabre differences", {"foil", "saber"}),
        ("lightsaber duel", set()),
    ],
)
def test_detect_weapons(text, weapons):
    assert detect_weapons(fold(text)) == weapons


# --- motion guess -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, motion",
    [
        ("Sabre advance lunge drill", "advance_lunge"),
        ("Sabre step-lunge", "advance_lunge"),
        ("How to lunge in saber", "lunge"),
        ("sabre escrime fente", "lunge"),
        ("sciabola scherma affondo", "lunge"),
        ("Сабля фехтование выпад урок", "lunge"),
        ("펜싱 사브르 런지 훈련", "lunge"),
        ("Saber parry 5 and riposte", "parry"),
        ("Sabre head cut technique", "cut"),
        ("Saber advance and retreat footwork", "advance"),
        ("Sabre retreat drill", "retreat"),
        ("Saber en garde position for beginners", "en_garde"),
        ("kardvívás lábmunka", "footwork"),
    ],
)
def test_guess_motion_from_text(text, motion):
    assert guess_motion(fold(text), fallback="unknown")[0] == motion


def test_guess_motion_falls_back_to_query_motion():
    assert guess_motion(fold("Sabre session with coach Bob"), fallback="parry") == ("parry", [])


# --- evaluate: rejections ---------------------------------------------------------------

def test_foil_only_is_rejected():
    assert evaluate(entry("Foil fencing lunge drill"), "lunge", FILTER).reject_reason == "foil_or_epee_only"


def test_epee_with_accent_is_rejected():
    assert evaluate(entry("Épée footwork tutorial"), "footwork", FILTER).reject_reason == "foil_or_epee_only"


def test_weaponless_fencing_video_is_kept_but_ranked_below_saber():
    generic = evaluate(entry("Fencing lunge drill"), "lunge", FILTER)
    saber = evaluate(entry("Sabre lunge drill"), "lunge", FILTER)
    assert generic.reject_reason is None
    assert saber.score >= generic.score + 3


def test_weaponless_video_is_rejected_when_saber_required():
    strict = ScrapeFilter(require_saber=True)
    assert evaluate(entry("Fencing lunge drill"), "lunge", strict).reject_reason == "no_saber_mention"


def test_weaponless_non_fencing_video_is_rejected():
    assert evaluate(entry("Lunges for leg day"), "lunge", FILTER).reject_reason == "no_fencing_context"


@pytest.mark.parametrize(
    "title",
    [
        "Beat Saber expert+ full combo",
        "Sabre dance - Khachaturian",
        "Classical sabre fencing lesson",
        "Sciabola militare britannica - affondo",
        "Stage combat sabre fencing",
        "Fix Your Lunge - HEMA fencing",
    ],
)
def test_non_sport_saber_is_blocklisted(title):
    assert evaluate(entry(title), "cut", FILTER).reject_reason == "blocklisted"


@pytest.mark.parametrize(
    "title",
    ["Assalti sciabola scherma", "[펜싱] 남자 사브르 레전드 경기"],
)
def test_foreign_language_bouts_are_rejected(title):
    assert evaluate(entry(title), "lunge", FILTER).reject_reason == "bout_or_highlights"


@pytest.mark.parametrize("title, motion", [("펜싱 사브르 팡트", "lunge"), ("펜싱 사브르 마르세 팡트", "advance_lunge")])
def test_korean_transliterations(title, motion):
    assert evaluate(entry(title), "footwork", FILTER).motion_guess == motion


def test_saber_without_fencing_context_is_rejected():
    assert evaluate(entry("Sabre review unboxing"), "cut", FILTER).reject_reason == "no_fencing_context"


def test_bout_highlights_are_rejected():
    verdict = evaluate(entry("Men's Sabre Fencing Final Szilagyi vs Samele - Highlights"), "lunge", FILTER)
    assert verdict.reject_reason == "bout_or_highlights"


def test_bout_word_with_drill_signal_is_kept_but_penalised():
    drill = evaluate(entry("Sabre lunge drill"), "lunge", FILTER)
    mixed = evaluate(entry("Sabre lunge drill from the world cup"), "lunge", FILTER)
    assert mixed.reject_reason is None
    assert mixed.score < drill.score


@pytest.mark.parametrize("duration", [5, 1500])
def test_duration_bounds(duration):
    verdict = evaluate(entry("Sabre lunge drill", duration=duration), "lunge", FILTER)
    assert verdict.reject_reason == "duration"


def test_duration_bounds_are_configurable():
    verdict = evaluate(entry("Sabre lunge drill", duration=5), "lunge", ScrapeFilter(min_duration=1))
    assert verdict.reject_reason is None


def test_unknown_duration_is_not_rejected():
    assert evaluate(entry("Sabre lunge drill", duration=None), "lunge", FILTER).reject_reason is None


def test_shorts_are_skipped_unless_allowed():
    short = entry("Sabre lunge drill", duration=30, url="https://www.youtube.com/shorts/abcdefghijk")
    assert evaluate(short, "lunge", FILTER).reject_reason == "short"
    assert evaluate(short, "lunge", ScrapeFilter(skip_shorts=False)).reject_reason is None


def test_livestreams_are_skipped():
    live = entry("Sabre lesson", duration=None, live_status="is_live")
    assert evaluate(live, "lunge", FILTER).reject_reason == "live"


# --- evaluate: ranking ------------------------------------------------------------------

def test_drill_outranks_plain_mention():
    drill = evaluate(entry("Sabre lunge drill for beginners"), "lunge", FILTER)
    plain = evaluate(entry("Sabre fencing at our club"), "lunge", FILTER)
    assert drill.score > plain.score
    assert drill.motion_guess == "lunge"


def test_short_videos_outrank_long_ones():
    short = evaluate(entry("Sabre lunge drill", duration=90), "lunge", FILTER)
    long = evaluate(entry("Sabre lunge drill", duration=1000), "lunge", FILTER)
    assert short.score > long.score


def test_repeated_query_hits_raise_score():
    once = evaluate(entry("Sabre lunge drill"), "lunge", FILTER)
    many = evaluate(entry("Sabre lunge drill", hits=4), "lunge", FILTER)
    assert many.score > once.score


def test_channel_name_counts_as_saber_mention():
    verdict = evaluate(entry("Footwork drill 3", channel="Sabre Academy Fencing"), "footwork", FILTER)
    assert verdict.reject_reason is None


# --- selection --------------------------------------------------------------------------

def rec(vid: str, motion: str, score: float, status: str = "candidate") -> dict:
    return {"id": vid, "motion_guess": motion, "score": score, "status": status}


def test_select_round_robins_across_motions_by_score():
    records = [
        rec("l1", "lunge", 9), rec("l2", "lunge", 8), rec("l3", "lunge", 7),
        rec("p1", "parry", 5), rec("a1", "advance", 6),
    ]
    chosen = [r["id"] for r in select_for_download(records, max_n=4, min_score=0)]
    assert chosen == ["l1", "a1", "p1", "l2"]


def test_select_skips_non_candidates_and_low_scores():
    records = [rec("d", "lunge", 9, "downloaded"), rec("r", "lunge", 9, "rejected"),
               rec("low", "lunge", 1), rec("ok", "lunge", 5)]
    assert [r["id"] for r in select_for_download(records, max_n=10, min_score=2)] == ["ok"]


def test_select_can_retry_failed():
    records = [rec("f", "lunge", 9, "failed")]
    assert select_for_download(records, max_n=10, min_score=0) == []
    assert [r["id"] for r in select_for_download(records, 10, 0, retry_failed=True)] == ["f"]


def test_select_caps_videos_per_channel():
    records = [{**rec(f"b{i}", "lunge", 10 - i), "channel": "Big Channel"} for i in range(5)]
    records.append({**rec("o1", "lunge", 1), "channel": "Other"})
    chosen = [r["id"] for r in select_for_download(records, 10, 0, max_per_channel=2)]
    assert chosen == ["b0", "b1", "o1"]


@pytest.mark.parametrize("title", ["Mariel Zagunis en garde position", "Oh Sang-uk 오상욱 런지"])
def test_famous_saber_fencers_count_as_saber_mentions(title):
    assert "saber" in detect_weapons(fold(title))


@pytest.mark.parametrize("title", ["Russ Mitchell: Historical Fencer sabre footwork",
                                   "Казачья шашка и сабля, фехтование",
                                   "19th Century British Sabre Fundamentals - Basic Footwork",
                                   "Polish sabre long cuts drill 1",
                                   "#1 ДУЭЛЬНАЯ САБЛЯ фехтование",
                                   "Escrime en fauteuil roulant - Wheelchair Fencing - Sabre",
                                   "Seated Lunge Drill | Fencing Tutorial (Foil, Epee, Saber)",
                                   "L'escrime artistique à la rapière - la retraite sabre",
                                   "Fencing stance for one-handed sword, hand-and-a-half sword, sabre"])
def test_more_historical_saber_is_blocklisted(title):
    assert evaluate(entry(title), "cut", FILTER).reject_reason == "blocklisted"


def test_shorts_hashtag_in_title_counts_as_short():
    verdict = evaluate(entry("Sabre lunge drill #shorts", duration=40), "lunge", FILTER)
    assert verdict.reject_reason == "short"


def test_misspelled_szilagyi_still_counts_as_saber():
    assert "saber" in detect_weapons(fold("Aron Szliagyi Individual Footwork"))


def test_very_long_videos_rank_below_medium_ones():
    medium = evaluate(entry("Sabre lunge drill", duration=700), "lunge", FILTER)
    very_long = evaluate(entry("Sabre lunge drill", duration=1100), "lunge", FILTER)
    assert medium.score > very_long.score


def test_weighted_select_fills_the_motion_furthest_below_its_share():
    records = [rec(f"f{i}", "footwork", 10 - i) for i in range(6)] + [rec(f"p{i}", "parry", 20 - i) for i in range(6)]
    chosen = [r["id"] for r in select_for_download(records, 5, 0, weights={"footwork": 4, "parry": 1})]
    # Parry has better scores, but footwork's 4x weight earns it most of the slots.
    assert chosen.count("p0") == 1 and sum(c.startswith("f") for c in chosen) == 4


def test_weighted_select_counts_videos_already_downloaded():
    records = [rec(f"d{i}", "footwork", 9, "downloaded") for i in range(4)]
    records += [rec("f1", "footwork", 9), rec("l1", "lunge", 5)]
    chosen = [r["id"] for r in select_for_download(records, 1, 0, weights={"footwork": 1, "lunge": 1})]
    assert chosen == ["l1"]


def test_failed_downloads_retry_until_attempts_run_out():
    fresh = {**rec("f1", "lunge", 9, "failed"), "attempts": 1}
    spent = {**rec("f2", "lunge", 9, "failed"), "attempts": 3}
    gone = {**rec("u1", "lunge", 9, "unavailable"), "attempts": 1}
    chosen = [r["id"] for r in select_for_download([fresh, spent, gone], 10, 0, max_attempts=3)]
    assert chosen == ["f1"]
    forced = select_for_download([fresh, spent, gone], 10, 0, retry_failed=True, max_attempts=3)
    assert sorted(r["id"] for r in forced) == ["f1", "f2"]
