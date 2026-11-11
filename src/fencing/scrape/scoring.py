"""Pure metadata filtering, scoring and motion guessing for scraped videos.

Everything here works on yt-dlp's flat search entries (title, description snippet,
duration, channel, url, live_status) so a candidate can be judged before downloading.
Text is accent-folded and lower-cased first; all patterns below are written against
folded text ("épée" -> "epee", "vívás" -> "vivas").
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

MOTIONS = ("en_garde", "advance", "retreat", "lunge", "advance_lunge", "parry", "cut", "footwork")


@dataclass(frozen=True)
class ScrapeFilter:
    min_duration: float = 10
    max_duration: float = 20 * 60
    skip_shorts: bool = True
    skip_live: bool = True
    # Off by default: many good footwork tutorials name no weapon at all. They are kept
    # but ranked below saber-specific videos (see SABER_BONUS).
    require_saber: bool = False


@dataclass(frozen=True)
class Verdict:
    score: float
    motion_guess: str
    motion_tags: list[str]
    reject_reason: str | None


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


WEAPON_PATTERNS = {
    # \bsab... rather than sab... so "lightsaber" does not count as a sabre mention.
    "saber": _compile([r"\bsab(re|er|res|ers|reur|reuse|lista)\b", r"\bsciabol", r"\bkard",
                       r"сабл", r"사브르", r"\bszabl", r"\bsabel\b", r"\bsable\b", r"佩剑", r"サーブル",
                       # Well-known sabreurs: their names imply the weapon even when titles don't.
                       r"zagunis", r"sz(il|li)agyi", r"dershwitz", r"morehouse", r"\bsamele", r"\bapithy",
                       r"오상욱", r"구본길", r"поздняков"]),
    "foil": _compile([r"\bfoils?\b", r"\bfleuret", r"\bfioretto", r"\bflorett", r"рапир", r"플뢰레",
                      r"花剑"]),
    "epee": _compile([r"\bepees?\b", r"\bspada\b", r"шпаг", r"에페", r"\bdegen\b", r"parbajtor",
                      r"重剑", r"\bespada\b"]),
}

FENCING_WORDS = _compile([
    r"\bfenc", r"escrim", r"scherm", r"vivas", r"фехтов", r"펜싱", r"esgrim", r"\bfecht",
    r"szermier", r"击剑", r"フェンシング", r"\bfie\b", r"\bpiste\b",
])
# Enough context next to a sabre mention, but not alone: "lunge" is also a gym exercise.
TECHNIQUE_WORDS = _compile([
    r"\ben ?garde", r"\blunge", r"\bflunge", r"\bfente", r"\bparr(y|ies)", r"\briposte",
    r"\bfootwork",
])

# Not sport fencing at all, even though "sabre" appears.
BLOCKLIST = _compile([
    r"light ?sab", r"beat ?sab", r"star ?wars", r"\bhema\b", r"longsword", r"kendo", r"kumdo",
    r"sab(re|er)[ -]?(dance|tooth)", r"sabretooth", r"historical fenc", r"шашк", r"казач", r"польская сабл", r"тяжел", r"sabre laser",
    r"skallagrim", r"gladiatoria", r"\bdaab\b", r"\bcentury\b", r"polish sab", r"дуэльн",
    r"duel(l)?ing sab", r"\brapier", r"artisti", r"артистич", r"пластун", r"рукопаш",
    r"hand-and-a-half", r"one-handed sword",
    # Seated fencing/drills have no footwork to learn from.
    r"\bseated\b", r"wheelchair", r"fauteuil", r"колясоч", r"휠체어", r"\bcavalry",
    r"\bminecraft", r"\broblox", r"\bfortnite", r"\banime\b", r"\bfate\b",
    # Classical, military and stage sabre move differently from the modern sport.
    r"\bclassical (sab|fenc)", r"классическ", r"\bmilitar", r"\bstage (combat|fencing)", r"сценич",
    r"хема", r"battlefront",
])

# Bouts, highlights and talk: two fencers overlapping or nobody fencing at all.
BOUT_TERMS = _compile([
    r"\bvs\.?\b", r"\bversus\b", r"highlight", r"\bbouts?\b", r"\bfinals?\b", r"semi-?final",
    r"\bfinale\b", r"gold medal", r"world cup", r"grand prix", r"championship", r"olympic games",
    r"compilation", r"best of", r"\btop ?\d+", r"\bfails?\b", r"funny", r"reaction", r"podcast",
    r"interview", r"\bvlog", r"documentary", r"\bmatch(es)?\b", r"\btouches\b", r"\bt(8|16|32|64)\b",
    r"\bassaut", r"\bassalt", r"финал", r"결승", r"경기", r"득점", r"livestream", r"live stream", r"\bstreamed\b",
])

# Drills, lessons and tutorials: one fencer, slow and repetitive -- what pose estimation wants.
FORMAT_TERMS = [
    r"\bdrill", r"\blessons?\b", r"\btutorial", r"\bhow to\b", r"\btechni", r"\bfootwork",
    r"\bexercis", r"\btraining\b", r"slow[ -]?mo", r"\btips?\b", r"\bbeginner", r"\bcoach",
    r"\bdemonstrat", r"\bbasics?\b", r"\bfundamental", r"\blecon", r"\bexercice", r"entrainement",
    r"\blezione", r"allenamento", r"esercizi", r"edzes", r"gyakorl", r"урок", r"трениров",
    r"упражнен", r"техник", r"레슨", r"훈련", r"강좌", r"기초", r"\bclase", r"entrenamiento",
    r"ejercicio", r"ubung", r"\btrening",
]
FORMAT_PATTERNS = [re.compile(p) for p in FORMAT_TERMS]

# Ordered by specificity: the first motion that matches is the primary guess, so
# "advance lunge" beats "lunge" and a lunge drill that mentions "en garde" stays a lunge.
MOTION_PATTERNS: dict[str, re.Pattern] = {
    "advance_lunge": _compile([r"advance[ -]?(and |& )?lunge", r"step[ -]?lunge", r"marche[ -]?fente",
                               r"마르세 ?팡트",
                               r"passo[ -]?affondo", r"balestra", r"jump[ -]?lunge"]),
    "lunge": _compile([r"\blung(e|es|ing)\b", r"\bflunge", r"\bfente", r"affondo", r"kitores",
                       r"выпад", r"런지", r"팡트", r"\bfondo\b"]),
    "parry": _compile([r"\bparr(y|ies|ying)", r"\bparades?\b", r"\bparat(a|e)\b", r"haritas",
                       r"защит", r"패리", r"방어", r"\bripost"]),
    "cut": _compile([r"\bcuts?\b", r"head ?cut", r"\bcolpo", r"fendente", r"\btagli", r"vagas",
                     r"удар", r"베기", r"\battacks? to the (head|flank|chest|cheek)"]),
    "advance": _compile([r"\badvanc", r"\bmarche\b", r"passo avanti", r"шаг вперед", r"전진",
                         r"\bavance"]),
    "retreat": _compile([r"\bretreat", r"\bretraite", r"passo indietro", r"шаг назад", r"후진",
                         r"\bretroce"]),
    "en_garde": _compile([r"\ben[ -]?garde", r"\bon guard", r"\bguard position", r"\bstance\b",
                          r"in guardia", r"стойк", r"자세"]),
    "footwork": _compile([r"\bfootwork", r"jeu de jambes", r"travail de(s)? jambes", r"lavoro di gambe",
                          r"gioco di gambe", r"labmunka", r"работа ног", r"передвижен", r"풋워크",
                          r"발동작", r"juego de pies", r"beinarbeit"]),
}


# Large enough that a saber video outranks an otherwise identical weapon-less one.
SABER_BONUS = 3.0

# Shorts are vertical and seconds long; search results usually give them a /shorts/ url, but
# some come back as /watch?v= with only the hashtag in the title to give them away.
SHORTS_TAG = re.compile(r"#shorts?\b")

# Target mix of the downloaded set, as relative weights per motion guess. Footwork drills
# carry advance, retreat and en garde in one steady shot, so they get the biggest share;
# parries and cuts only matter later, once the saber is in the loop. Roughly the week plan's
# ~50-clip mix, scaled: for 80 videos ~25 footwork, ~15 lunge, ~9 each en garde and
# advance-lunge, ~6 each advance and retreat, ~5 each parry and cut.
DOWNLOAD_WEIGHTS: dict[str, float] = {
    "footwork": 4.0, "lunge": 2.5, "en_garde": 1.5, "advance_lunge": 1.5,
    "advance": 1.0, "retreat": 1.0, "parry": 0.75, "cut": 0.75,
}


def fold(text: str) -> str:
    """Lower-case and strip combining accents.

    Re-composing with NFC afterwards matters: NFKD splits Hangul syllables into jamo,
    and those are letters (not combining marks), so they survive and must be rejoined.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def detect_weapons(folded_text: str) -> set[str]:
    return {weapon for weapon, pattern in WEAPON_PATTERNS.items() if pattern.search(folded_text)}


def guess_motion(folded_text: str, fallback: str) -> tuple[str, list[str]]:
    """Primary motion and every motion mentioned; the fallback is the query's motion."""
    tags = [motion for motion, pattern in MOTION_PATTERNS.items() if pattern.search(folded_text)]
    return (tags[0] if tags else fallback), tags


def _reject_reason(entry: dict, text: str, flt: ScrapeFilter) -> str | None:
    if flt.skip_live and entry.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
        return "live"
    if flt.skip_shorts and ("/shorts/" in (entry.get("url") or "") or SHORTS_TAG.search(fold(entry.get("title") or ""))):
        return "short"
    duration = entry.get("duration")
    if duration is not None and not flt.min_duration <= duration <= flt.max_duration:
        return "duration"
    if BLOCKLIST.search(text):
        return "blocklisted"
    weapons = detect_weapons(text)
    if weapons and "saber" not in weapons:
        return "foil_or_epee_only"
    if flt.require_saber and "saber" not in weapons:
        return "no_saber_mention"
    has_technique_context = "saber" in weapons and TECHNIQUE_WORDS.search(text)
    if not (FENCING_WORDS.search(text) or has_technique_context):
        return "no_fencing_context"
    return None


def evaluate(entry: dict, query_motion: str, flt: ScrapeFilter) -> Verdict:
    """Judge one search entry: reject it, or give it a relevance score and motion guess."""
    title = fold(entry.get("title") or "")
    text = " ".join([title, fold(entry.get("description") or ""), fold(entry.get("channel") or "")])
    motion_guess, motion_tags = guess_motion(text, fallback=query_motion)

    reject_reason = _reject_reason(entry, text, flt)
    if reject_reason:
        return Verdict(0.0, motion_guess, motion_tags, reject_reason)

    format_hits = sum(1 for p in FORMAT_PATTERNS if p.search(text))
    bout_hits = len({m.group(0) for m in BOUT_TERMS.finditer(text)})
    if bout_hits and not format_hits:
        return Verdict(0.0, motion_guess, motion_tags, "bout_or_highlights")

    score = min(2.0 * format_hits, 6.0) - min(3.0 * bout_hits, 9.0)
    if motion_tags:
        score += 2.0
        if MOTION_PATTERNS[motion_tags[0]].search(title):
            score += 1.0
    if "saber" in detect_weapons(text):
        score += SABER_BONUS
        if "saber" in detect_weapons(title):
            score += 1.0
    if len(detect_weapons(text)) > 1:
        score -= 1.0  # weapon-comparison videos split their time across weapons
    duration = entry.get("duration")
    if duration is not None:
        # Short clips tend to be single-motion demonstrations and are cheap to screen.
        # Long ones are mostly talk or partner lessons, and cost the most to download and screen.
        score += (2.0 if duration <= 180 else 1.0 if duration <= 480 else 0.0 if duration <= 600
                  else -1.0 if duration <= 900 else -2.0)
    # Found by several independent queries: a more robust relevance signal than any one keyword.
    score += min(0.5 * (int(entry.get("hits") or 1) - 1), 2.0)
    return Verdict(round(score, 2), motion_guess, motion_tags, None)


def select_for_download(
    records: list[dict], max_n: int, min_score: float, retry_failed: bool = False,
    max_per_channel: int | None = None, max_attempts: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Pick up to max_n candidates, spread across motion guesses, best score first per motion.

    Plain top-N would fill up with whichever motion has the most tutorials (footwork). Without
    `weights` this is a plain round-robin over motions (strongest motion first). With weights,
    each next pick goes to the motion furthest below its target share of the whole set --
    counting videos already downloaded, so a re-run tops up the thin motions instead of
    starting the round-robin over.

    Failed downloads come back while they have fewer than `max_attempts` attempts (or always,
    with retry_failed). The per-channel cap (counting already-downloaded videos) spreads the
    set over more fencers and camera setups, which matters more for pose/retargeting than volume.
    """
    def eligible_status(r: dict) -> bool:
        if r.get("status") == "candidate":
            return True
        if r.get("status") != "failed":
            return False
        return retry_failed or (max_attempts is not None and int(r.get("attempts") or 0) < max_attempts)

    downloaded = [r for r in records if r.get("status") == "downloaded"]
    per_channel = Counter(r.get("channel") for r in downloaded)
    eligible = sorted((r for r in records if eligible_status(r) and r.get("score", 0) >= min_score),
                      key=lambda r: -r["score"])
    by_motion: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        if max_per_channel is not None and per_channel[record.get("channel")] >= max_per_channel:
            continue
        per_channel[record.get("channel")] += 1
        by_motion[record.get("motion_guess") or "unknown"].append(record)

    chosen: list[dict] = []
    if weights is None:
        queues = sorted(by_motion.values(), key=lambda queue: -queue[0]["score"])
        while len(chosen) < max_n and any(queues):
            for queue in queues:
                if queue and len(chosen) < max_n:
                    chosen.append(queue.pop(0))
        return chosen

    have = Counter(r.get("motion_guess") or "unknown" for r in downloaded)
    while len(chosen) < max_n:
        open_motions = [m for m, queue in by_motion.items() if queue]
        if not open_motions:
            break
        # Furthest below target share first; ties go to the motion with the better next video.
        motion = min(open_motions, key=lambda m: (have[m] / weights.get(m, 0.5), -by_motion[m][0]["score"]))
        chosen.append(by_motion[motion].pop(0))
        have[motion] += 1
    return chosen
