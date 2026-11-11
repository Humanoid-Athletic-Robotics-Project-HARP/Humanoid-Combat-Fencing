"""`fencing scrape`: find, filter and download saber fencing videos; sync data/video_log.md.

The whole collection stage, fully automatic, resumable and safe to re-run:

    fencing scrape run --max-downloads 80 --screen
        1. search   every query in queries.yaml (skips searches done in the last 7 days)
        2. rescore  re-apply the scoring rules to every record
        3. download the best candidates, spread over motions (<= 720p H.264 mp4), each one
                    checked with OpenCV; failures are retried by later runs up to
                    --max-attempts times, permanent errors (private, removed...) never
        4. log      sync data/video_log.md
        5. screen   (--screen) `fencing screen` on the new downloads, then write the shortlist

Single steps:

    fencing scrape search --max-per-query 20      # candidates -> HCF_DATA/videos/manifest.jsonl
    fencing scrape download --max-downloads 80 --dry-run
    fencing scrape verify                         # re-check every downloaded file with OpenCV
    fencing scrape rescore                        # re-apply the scoring rules
    fencing scrape log                            # manifest (+ screen reports) -> data/video_log.md
    fencing scrape log --annotate                 # ... + shortlist, pose/retarget, motions/ tags
    fencing scrape shortlist                      # screen reports -> HCF_DATA/videos/shortlist.json
    fencing scrape stats

Videos land in HCF_DATA/videos/raw/<youtube_id>.mp4 -- never inside the repo. Re-running any
command skips work already done: files on disk count as downloaded, screened videos are not
re-screened.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import yaml
from yt_dlp.utils import DownloadError

from fencing import paths
from fencing.scrape.annotate import clip_evidence, published_sources, video_annotations
from fencing.scrape.manifest import (
    MAX_ATTEMPTS,
    exclude_record,
    load_manifest,
    manifest_stats,
    record_failure,
    rescore_record,
    save_manifest,
    upsert_hit,
)
from fencing.scrape.scoring import DOWNLOAD_WEIGHTS, ScrapeFilter, select_for_download
from fencing.scrape.shortlist import build_shortlist, load_screen_reports, log_fill
from fencing.scrape.video_log import annotate_video_log, parse_table, sync_video_log
from fencing.scrape.youtube import (
    BrokenVideoError,
    ThrottledError,
    download_video,
    probe_video,
    search,
    verify_video,
)

DEFAULT_QUERIES = Path(__file__).with_name("queries.yaml")


def log(message: str) -> None:
    print(f"[scrape] {message}", file=sys.stderr, flush=True)


def load_search_targets(queries_path: Path, max_per_query: int, motions: list[str] | None
                        ) -> list[tuple[str, str, str]]:
    """(yt-dlp target, query text, motion) for every query and channel search in the bank."""
    bank = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    targets = []
    for motion, queries in (bank.get("motions") or {}).items():
        if motions and motion not in motions:
            continue
        targets += [(f"ytsearch{max_per_query}:{q}", q, motion) for q in queries]
    if not motions or "footwork" in motions:
        for channel in bank.get("channels") or []:
            for q in bank.get("channel_queries") or []:
                url = f"{channel['url'].rstrip('/')}/search?query={quote_plus(q)}"
                targets.append((url, f"{channel['name']}: {q}", "footwork"))
    return targets


def scrape_filter(args: argparse.Namespace) -> ScrapeFilter:
    return ScrapeFilter(min_duration=args.min_duration, max_duration=args.max_duration,
                        skip_shorts=not args.allow_shorts, require_saber=args.require_saber)


def search_state_path(manifest: Path) -> Path:
    return manifest.with_name("search_state.json")


def load_search_state(path: Path) -> dict[str, str]:
    """{yt-dlp search target: ISO time it last succeeded}; lets an interrupted search resume."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(f"ignoring unreadable {path}; every search will run again")
        return {}


def save_search_state(path: Path, state: dict[str, str]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_fresh(stamp: str | None, max_age_days: float) -> bool:
    if not stamp or max_age_days <= 0:
        return False
    return datetime.now(UTC) - datetime.fromisoformat(stamp) < timedelta(days=max_age_days)


def cmd_search(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest)
    targets = load_search_targets(args.queries, args.max_per_query, args.motions)
    state_path = search_state_path(args.manifest)
    state = load_search_state(state_path)
    todo = [t for t in targets if not is_fresh(state.get(t[0]), args.search_max_age_days)]
    flt, failed_queries = scrape_filter(args), []
    log(f"{len(targets)} searches ({len(targets) - len(todo)} done in the last "
        f"{args.search_max_age_days:g} days, skipped), manifest has {len(records)} records")
    for index, (target, query, motion) in enumerate(todo, start=1):
        try:
            entries = search(target)
        except ThrottledError as error:
            log(f"YouTube is throttling us, stopping search early: {error}")
            save_manifest(args.manifest, records)
            return 3
        except DownloadError as error:
            log(f"search failed for {query!r}: {error}")
            failed_queries.append(query)
            continue
        before = len(records)
        accepted = sum(upsert_hit(records, e, query, motion, flt)["status"] != "rejected" for e in entries)
        log(f"[{index}/{len(todo)}] {query!r}: {len(entries)} hits, {accepted} kept, "
            f"{len(records) - before} new")
        save_manifest(args.manifest, records)
        state[target] = datetime.now(UTC).isoformat(timespec="seconds")
        save_search_state(state_path, state)
        time.sleep(random.uniform(args.search_sleep, 2 * args.search_sleep))
    if failed_queries:
        log(f"{len(failed_queries)} searches failed: {failed_queries}")
    print_stats(records)
    return 1 if failed_queries and len(failed_queries) == len(todo) else 0


def cmd_rescore(args: argparse.Namespace) -> int:
    records, flt = load_manifest(args.manifest), scrape_filter(args)
    changed = sum(rescore_record(record, flt) for record in records.values())
    save_manifest(args.manifest, records)
    log(f"rescored; {changed} records changed status")
    print_stats(records)
    return 0


def mark_downloaded(record: dict, path: Path, info: dict) -> str:
    """Probe and verify a file on disk; returns the record's new status.

    Broken files are deleted and count as a failed attempt (the next run downloads them
    again); readable but vertical videos are kept and excluded.
    """
    record.update({k: v for k, v in info.items() if v is not None})
    try:
        record.update(probe_video(path))
        record.update(verify_video(path, record.get("duration")))
    except (BrokenVideoError, RuntimeError) as error:
        path.unlink(missing_ok=True)
        record.pop("path", None)
        return record_failure(record, f"broken file: {error}")
    record.update(status="downloaded", path=str(path), error=None,
                  downloaded_at=record.get("downloaded_at") or datetime.now(UTC).isoformat(timespec="seconds"))
    if record["cv_height"] > record["cv_width"]:
        exclude_record(record, f"vertical video ({record['cv_width']}x{record['cv_height']})")
    return record["status"]


def raw_dir_bytes(videos_dir: Path) -> int:
    return sum(p.stat().st_size for p in videos_dir.glob("*.mp4")) if videos_dir.exists() else 0


def adopt_files_on_disk(records: dict[str, dict], videos_dir: Path) -> None:
    """Resume: a file from an interrupted or earlier run counts as downloaded once it verifies."""
    for record in records.values():
        path = videos_dir / f"{record['id']}.mp4"
        if record["status"] not in {"downloaded", "excluded"} and path.exists():
            status = mark_downloaded(record, path, {})
            log(f"found existing {path.name}: {status}")


def download_queue(records: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    return select_for_download(list(records.values()), len(records), args.min_score, args.retry_failed,
                               max_per_channel=args.max_per_channel, max_attempts=args.max_attempts,
                               weights=DOWNLOAD_WEIGHTS)


def download(args: argparse.Namespace) -> tuple[int, list[str]]:
    """The download step; returns (exit status, ids downloaded by this call)."""
    records = load_manifest(args.manifest)
    adopt_files_on_disk(records, args.videos_dir)
    save_manifest(args.manifest, records)

    queue = download_queue(records, args)
    if args.dry_run:
        for record in queue[:args.max_downloads]:
            print(f"{record['score']:5.1f}  {record['motion_guess']:<13} {record.get('duration') or 0:>6.0f}s  "
                  f"{record['id']}  {record['title'][:70]}")
        log(f"dry run: would download {min(len(queue), args.max_downloads)} of {len(queue)} eligible")
        return 0, []

    budget_bytes = args.max_total_gb * 1e9
    new_ids: list[str] = []
    failed = throttle_pauses = 0
    queue.reverse()  # pop() from the end = best first
    while queue and len(new_ids) < args.max_downloads:
        if raw_dir_bytes(args.videos_dir) >= budget_bytes:
            log(f"stopping: {args.videos_dir} reached the {args.max_total_gb} GB cap")
            break
        record = queue.pop()
        log(f"downloading {record['id']} ({record['motion_guess']}, score {record['score']}): "
            f"{record['title'][:70]}")
        try:
            path, info = download_video(record["url"], args.videos_dir, args.max_height)
            status = mark_downloaded(record, path, info)
            if status == "downloaded":
                new_ids.append(record["id"])
            else:
                failed += status != "excluded"
                log(f"{record['id']}: {status} ({record.get('error') or record.get('exclude_reason')})")
        except ThrottledError as error:
            # YouTube's bot check is per-IP and usually lifts within minutes; a bounded
            # pause-and-retry keeps an unattended run going without hammering it.
            if throttle_pauses >= args.max_throttle_pauses:
                log(f"YouTube is still throttling after {throttle_pauses} pauses, stopping "
                    f"(re-run later to resume): {error}")
                save_manifest(args.manifest, records)
                return 3, new_ids
            throttle_pauses += 1
            log(f"throttled ({error.args[0][:80]}...); pausing {args.throttle_pause:.0f} s "
                f"[{throttle_pauses}/{args.max_throttle_pauses}]")
            queue.append(record)
            time.sleep(args.throttle_pause)
            continue
        except (DownloadError, FileNotFoundError, RuntimeError) as error:
            status = record_failure(record, str(error))
            failed += 1
            log(f"{status} {record['id']} (attempt {record['attempts']}): {error}")
        save_manifest(args.manifest, records)
        if queue and len(new_ids) < args.max_downloads:
            time.sleep(random.uniform(args.download_sleep, 2 * args.download_sleep))
    save_manifest(args.manifest, records)
    log(f"downloaded {len(new_ids)}, failed {failed}")
    print_stats(records)
    return 0, new_ids


def cmd_download(args: argparse.Namespace) -> int:
    status, new_ids = download(args)
    if getattr(args, "screen", False) and not args.dry_run:
        status = max(status, screen_and_shortlist(args, new_ids))
    return status


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-check every downloaded file; broken ones are deleted and queued for another attempt."""
    records = load_manifest(args.manifest)
    adopt_files_on_disk(records, args.videos_dir)
    counts: Counter = Counter()
    for record in records.values():
        if record["status"] != "downloaded":
            continue
        path = args.videos_dir / f"{record['id']}.mp4"
        if not path.exists():
            record.update(status="candidate", path=None, error="file missing from videos dir")
            counts["missing"] += 1
            log(f"{record['id']}: file missing, back to candidate")
            continue
        status = mark_downloaded(record, path, {})
        counts[status] += 1
        if status != "downloaded":
            log(f"{record['id']}: {status} ({record.get('error') or record.get('exclude_reason')})")
    save_manifest(args.manifest, records)
    log(f"verified: {dict(counts)}")
    return 1 if counts["failed"] or counts["unavailable"] else 0


def unscreened_downloads(records: dict[str, dict], first: list[str]) -> list[str]:
    """Downloaded ids without a screen.json: `first`, then the rest by motion priority and score."""
    rest = sorted((r for r in records.values() if r["status"] == "downloaded" and r["id"] not in first),
                  key=lambda r: (-DOWNLOAD_WEIGHTS.get(r.get("motion_guess") or "", 0.5), -(r.get("score") or 0)))
    ordered = [i for i in first if records.get(i, {}).get("status") == "downloaded"] + [r["id"] for r in rest]
    return [i for i in ordered if not (paths.POSE / i / "screen.json").exists()]


def screen_and_shortlist(args: argparse.Namespace, new_ids: list[str]) -> int:
    """Run `fencing screen` (sequentially, GPU) on every downloaded video not screened yet --
    this run's downloads first -- then rebuild the shortlist."""
    status = 0
    yt_ids = unscreened_downloads(load_manifest(args.manifest), new_ids)
    if yt_ids:
        from fencing.pose import screen  # deferred: pulls in the pose stage only when asked

        log(f"screening {len(yt_ids)} downloads ({len(new_ids)} new this run)")
        status = screen.main(yt_ids)
    write_shortlist(args)
    return status


def write_shortlist(args: argparse.Namespace) -> dict:
    records = load_manifest(args.manifest)
    reports = load_screen_reports(paths.POSE, sorted(records))
    shortlist = build_shortlist(records, reports)
    out = args.manifest.with_name("shortlist.json")
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(shortlist, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    log(f"shortlist: {shortlist['usable_ranges']} usable ranges from {shortlist['screened_videos']} "
        f"screened videos -> {out}")
    return shortlist


def cmd_shortlist(args: argparse.Namespace) -> int:
    shortlist = write_shortlist(args)
    for motion, entries in shortlist["by_motion"].items():
        print(f"\n{motion} ({len(entries)} ranges)")
        for e in entries[:args.top]:
            print(f"  {e['rank']:.3f}  {e['yt_id']}  {e['start']:7.1f}-{e['end']:7.1f}s  people {e['people']}  "
                  f"{e['view']:<7} full {e['full_body_frac']:.2f}  {e['title'][:50]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    status = 0
    if not args.skip_search:
        status = cmd_search(args)
        if status == 3:
            log("search was throttled; downloading from the candidates we already have")
    cmd_rescore(args)
    download_status, new_ids = download(args)
    status = max(status, download_status)
    if args.dry_run:
        return status
    cmd_log(args)
    if args.screen:
        status = max(status, screen_and_shortlist(args, new_ids))
        cmd_log(args)  # again, now with the screen's View/Start/End
    return status


def cmd_log(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest)
    markdown = args.video_log.read_text(encoding="utf-8") if args.video_log.exists() else "# Video log\n"
    try:
        in_table = {row["_id"] for row in parse_table(markdown)[1] if row["_id"]}
    except ValueError as error:  # a row we cannot place: writing anything would bake in a guess
        log(f"{args.video_log}: {error}; nothing written")
        return 1
    wanted = {"downloaded", "candidate"} if args.include_candidates else {"downloaded"}
    # Excluded videos only update rows they already have (to mark them rejected).
    chosen = sorted((r for r in records.values()
                     if r["status"] in wanted or (r["status"] == "excluded" and r["id"] in in_table)),
                    key=lambda r: (r.get("downloaded_at") or "~", -r.get("score", 0)))
    reports = load_screen_reports(paths.POSE, [r["id"] for r in chosen])
    chosen = [{**r, "_fill": log_fill(r, reports[r["id"]])} if r["id"] in reports and r["status"] == "downloaded"
              else r for r in chosen]
    markdown = sync_video_log(markdown, chosen)
    log(f"synced {len(chosen)} videos into {args.video_log} ({len(reports)} with screen reports)")
    if args.annotate:
        markdown = annotate_log(markdown, records, args.manifest.with_name("shortlist.json"))
    args.video_log.write_text(markdown, encoding="utf-8")
    return 0


def annotate_log(markdown: str, records: dict[str, dict], shortlist_path: Path) -> str:
    """Fill empty hand-edited cells from the shortlist, screen reports, pose/retarget sidecars
    and published motions (see fencing.scrape.annotate)."""
    if shortlist_path.exists():
        shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    else:
        log(f"no {shortlist_path}: Start/End/View/Fencers come from screen reports only")
        shortlist = {}
    annotations = video_annotations(records, shortlist, load_screen_reports(paths.POSE, sorted(records)),
                                    published_sources(paths.MOTIONS_DIR), clip_evidence(paths.POSE, paths.RETARGET))
    markdown, filled = annotate_video_log(markdown, annotations)
    log(f"annotated: {dict(filled) or 'nothing new'}")
    return markdown


def print_stats(records: dict[str, dict]) -> None:
    stats = manifest_stats(records)
    print(f"records: {len(records)}  by status: {stats['by_status']}")
    print(f"reject reasons: {stats['reject_reasons']}")
    if stats["exclude_reasons"]:
        print(f"excluded after download: {stats['exclude_reasons']}")
    candidates = Counter(r.get("motion_guess") for r in records.values() if r["status"] == "candidate")
    print(f"candidates by motion: {dict(candidates.most_common())}")
    print(f"downloaded by motion: {dict(Counter(stats['downloaded_by_motion']).most_common())}")
    print(f"downloaded: {stats['downloaded_bytes'] / 1e9:.2f} GB, "
          f"{stats['downloaded_seconds'] / 3600:.2f} h of video")


def cmd_stats(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest)
    print_stats(records)
    on_disk = raw_dir_bytes(args.videos_dir)
    print(f"on disk in {args.videos_dir}: {on_disk / 1e9:.2f} GB")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing scrape", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=paths.VIDEO_MANIFEST)
    parser.add_argument("--videos-dir", type=Path, default=paths.VIDEOS_RAW)
    sub = parser.add_subparsers(dest="command", required=True)

    filter_args = argparse.ArgumentParser(add_help=False)
    filter_args.add_argument("--min-duration", type=float, default=10, help="seconds")
    filter_args.add_argument("--max-duration", type=float, default=20 * 60, help="seconds")
    filter_args.add_argument("--allow-shorts", action="store_true", help="keep YouTube Shorts")
    filter_args.add_argument("--require-saber", action="store_true",
                             help="reject videos that name no weapon (default: keep, rank lower)")

    search_args = argparse.ArgumentParser(add_help=False)
    search_args.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="query bank YAML")
    search_args.add_argument("--max-per-query", type=int, default=20)
    search_args.add_argument("--motions", nargs="+", help="only these query groups")
    search_args.add_argument("--search-sleep", type=float, default=2.0,
                             help="seconds between requests (randomised up to 2x)")
    search_args.add_argument("--search-max-age-days", type=float, default=7.0,
                             help="skip searches that already ran within this many days (0 = rerun all)")

    download_args = argparse.ArgumentParser(add_help=False)
    download_args.add_argument("--max-downloads", type=int, default=80)
    download_args.add_argument("--min-score", type=float, default=6.0,
                               help="candidates below this score are never downloaded")
    download_args.add_argument("--max-height", type=int, default=720)
    download_args.add_argument("--max-total-gb", type=float, default=30.0,
                               help="stop once the raw video folder reaches this size")
    download_args.add_argument("--max-per-channel", type=int, default=6,
                               help="cap per YouTube channel, for fencer/camera variety")
    download_args.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                               help="failed downloads are retried by later runs until this many attempts")
    download_args.add_argument("--retry-failed", action="store_true",
                               help="also retry failures that used up their attempts")
    download_args.add_argument("--screen", action="store_true",
                               help="afterwards run `fencing screen` on new downloads and write shortlist.json")
    download_args.add_argument("--throttle-pause", type=float, default=600.0,
                               help="seconds to wait when YouTube bot-checks us")
    download_args.add_argument("--max-throttle-pauses", type=int, default=4)
    download_args.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    download_args.add_argument("--download-sleep", type=float, default=15.0,
                               help="seconds between downloads (randomised up to 2x)")

    log_args = argparse.ArgumentParser(add_help=False)
    log_args.add_argument("--video-log", type=Path, default=paths.VIDEO_LOG)
    log_args.add_argument("--include-candidates", action="store_true")
    log_args.add_argument("--annotate", action="store_true",
                          help="also fill empty hand-edited cells (Start/End/View/Fencers/Hand/Notes) and "
                               "upgrade Status `new` from the shortlist, pose/retarget sidecars and motions/*.json")

    sub.add_parser("search", parents=[search_args, filter_args], help="collect candidates").set_defaults(
        func=cmd_search)
    sub.add_parser("download", parents=[download_args], help="download top candidates").set_defaults(
        func=cmd_download)
    run_parser = sub.add_parser("run", parents=[search_args, filter_args, download_args, log_args],
                                help="search, rescore, download, log (and screen with --screen)")
    run_parser.add_argument("--skip-search", action="store_true", help="use the candidates already found")
    run_parser.set_defaults(func=cmd_run)
    sub.add_parser("rescore", parents=[filter_args], help="re-apply scoring rules").set_defaults(
        func=cmd_rescore)
    sub.add_parser("verify", help="re-check downloaded files with OpenCV").set_defaults(func=cmd_verify)
    sub.add_parser("log", parents=[log_args], help="sync data/video_log.md from the manifest").set_defaults(
        func=cmd_log)
    shortlist_parser = sub.add_parser("shortlist", help="rank screened ranges into videos/shortlist.json")
    shortlist_parser.add_argument("--top", type=int, default=10, help="ranges printed per motion")
    shortlist_parser.set_defaults(func=cmd_shortlist)
    sub.add_parser("stats", help="counts per status and motion, size on disk").set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
