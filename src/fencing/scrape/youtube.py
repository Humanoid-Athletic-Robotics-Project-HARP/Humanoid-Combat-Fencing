"""yt-dlp, ffprobe and OpenCV wrappers: the only network/IO side of `fencing scrape`."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

# yt-dlp needs a JS runtime for YouTube's player challenges; node is what this box has
# (the API default is deno). Equivalent to `--js-runtimes node` on the command line.
BASE_OPTIONS = {"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}}}

# Phrases in yt-dlp errors that mean YouTube is pushing back on us rather than this
# one video being broken; carrying on would only dig the hole deeper.
THROTTLE_MARKERS = ("HTTP Error 429", "Too Many Requests", "not a bot", "rate-limit")

# Info fields worth keeping in the manifest after a download.
INFO_FIELDS = ("title", "channel", "channel_url", "duration", "upload_date", "license", "fps",
               "width", "height", "vcodec", "format_id", "view_count", "webpage_url")


# A download shorter than this fraction of YouTube's duration is treated as truncated.
VERIFY_MIN_LENGTH_FRACTION = 0.9


class ThrottledError(RuntimeError):
    """YouTube is rate-limiting or bot-checking this IP; stop and try again later."""


def _raise_if_throttled(error: DownloadError) -> None:
    if any(marker in str(error) for marker in THROTTLE_MARKERS):
        raise ThrottledError(str(error)) from error


def search(target: str, retries: int = 3, backoff_seconds: float = 10.0) -> list[dict]:
    """Flat search entries (videos only) for `ytsearchN:<q>` or a channel search URL."""
    options = {**BASE_OPTIONS, "extract_flat": "in_playlist", "skip_download": True}
    for attempt in range(1, retries + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(target, download=False)
            break
        except DownloadError as error:
            _raise_if_throttled(error)
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * attempt)
    entries = result.get("entries") or []
    # Channel searches also return playlists and channels; keep only single videos.
    return [e for e in entries
            if e and e.get("id") and ("/watch?" in (e.get("url") or "") or "/shorts/" in (e.get("url") or ""))]


def download_video(url: str, out_dir: Path, max_height: int) -> tuple[Path, dict]:
    """Download one video (video stream only, <= max_height, mp4) and return (path, info)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        **BASE_OPTIONS,
        # Prefer H.264 in mp4: OpenCV and the pose tools read it without surprises.
        # Audio is useless for pose estimation, so video-only streams halve the download.
        "format": (f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}][ext=mp4]"
                   f"/bv*[height<={max_height}]/b[height<={max_height}]"),
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "sleep_interval_requests": 1,
        "overwrites": False,
        "continuedl": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as error:
        _raise_if_throttled(error)
        raise
    path = out_dir / f"{info['id']}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp finished but {path} is missing (format {info.get('ext')})")
    return path, {key: info.get(key) for key in INFO_FIELDS}


def probe_video(path: Path) -> dict:
    """Width, height, fps, codec, size and duration of the first video stream."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
               "-show_entries", "format=duration,size", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {completed.stderr.strip()}")
    probe = json.loads(completed.stdout)
    if not probe.get("streams"):
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream, container = probe["streams"][0], probe.get("format", {})
    numerator, _, denominator = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": round(fps, 3),
        "codec": stream.get("codec_name"),
        "filesize": int(container["size"]) if "size" in container else path.stat().st_size,
        "probed_duration": float(container["duration"]) if "duration" in container else None,
    }


class BrokenVideoError(RuntimeError):
    """The file on disk is not a video the pipeline can read (truncated, no frames, ...)."""


def verify_video(path: Path, expected_duration: float | None = None) -> dict:
    """Open the file with OpenCV, the reader every later stage uses, and decode real frames.

    Checks the frame count, fps and size, decodes the first frame and one near the end, and
    compares the length with YouTube's duration to catch truncated downloads. Returns the
    cv_* fields for the manifest; raises BrokenVideoError if any check fails.
    """
    import cv2  # deferred: only needed after a download, keeps `scrape search` light

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise BrokenVideoError(f"OpenCV cannot open {path}")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames <= 0 or not 1.0 <= fps <= 240.0 or width <= 0 or height <= 0:
            raise BrokenVideoError(f"{path.name}: bad stream info (frames={frames}, fps={fps}, {width}x{height})")
        for index in (0, max(frames - int(fps) - 1, 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape[:2] != (height, width):
                raise BrokenVideoError(f"{path.name}: cannot decode frame {index} of {frames}")
    finally:
        cap.release()
    seconds = frames / fps
    if expected_duration and seconds < VERIFY_MIN_LENGTH_FRACTION * expected_duration - 2.0:
        raise BrokenVideoError(f"{path.name}: {seconds:.0f} s on disk but YouTube says {expected_duration:.0f} s "
                               "(truncated download?)")
    return {"cv_frames": frames, "cv_fps": round(fps, 3), "cv_width": width, "cv_height": height,
            "cv_seconds": round(seconds, 2)}
