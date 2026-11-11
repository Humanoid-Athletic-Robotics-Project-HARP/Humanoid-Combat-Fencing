"""MediaPipe Pose Landmarker on one fencer track of one video segment.

Runs inside the `pose` venv, launched by `fencing pose`:

    python -m fencing.pose.mediapipe_worker --video V.mp4 --start 12.0 --end 19.5 \
        --track-npz pose/<yt_id>/track.npz --track-id 3 --out pose/<clip_id>/mediapipe_raw.npz

Each frame is cropped to a square around the chosen fencer's box before it reaches MediaPipe,
so the second fencer or the coach can never be the one that gets landmarked.
Model: pose_landmarker_heavy.task (Apache-2.0), CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from fencing.pose.landmarks import smooth_time
from fencing.pose.screening import track_from_arrays
from fencing.pose.tracking import interpolate_track_boxes, square_crop_box

MAX_PROCESS_FPS = 32.0
CROP_ENLARGE = 1.3


def _crop_padded(frame: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Pixels inside `crop` (xyxy, may exceed the image), black where it does."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = np.round(crop).astype(int)
    out = np.zeros((y1 - y0, x1 - x0, 3), dtype=frame.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def _track_boxes_at_times(track_npz: Path, track_id: int, times_s: np.ndarray) -> np.ndarray:
    with np.load(track_npz) as data:
        track = track_from_arrays(data, track_id)
        frame_times = np.asarray(data["frame_times"])
    # Map clip times onto (fractional) screen-frame indices, then interpolate the track there.
    fractional_frames = np.interp(times_s, frame_times, np.arange(len(frame_times)))
    return interpolate_track_boxes(track, fractional_frames)


def run(args: argparse.Namespace) -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not args.model.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {args.model} (run scripts/setup_linux_gpu.sh pose)")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / MAX_PROCESS_FPS + 0.49)))

    cap.set(cv2.CAP_PROP_POS_MSEC, max(args.start - 0.5, 0.0) * 1000.0)
    frames, times = [], []
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > args.end + 1e-3:
            break
        if t >= args.start - 1e-3 and index % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
                times.append(t)
        if t >= args.start - 1e-3:
            index += 1
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"fewer than 2 frames decoded from {args.video} in [{args.start}, {args.end}] s")
    times_s = np.asarray(times)

    boxes = _track_boxes_at_times(args.track_npz, args.track_id, times_s)
    # A jittery crop makes MediaPipe's landmarks jitter with it; smooth the box over ~0.3 s.
    boxes = smooth_time(boxes, fps / stride, window_s=0.3)
    crops = np.stack([square_crop_box(b, CROP_ENLARGE, width, height) for b in boxes])

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    n = len(frames)
    image_xy = np.full((n, 33, 2), np.nan)
    visibility = np.zeros((n, 33))
    world = np.full((n, 33, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    t_start = time.time()
    for i, (frame, crop) in enumerate(zip(frames, crops, strict=True)):
        patch = cv2.cvtColor(_crop_padded(frame, crop), cv2.COLOR_BGR2RGB)
        timestamp_ms = int(round((times_s[i] - times_s[0]) * 1000.0)) + i  # strictly increasing
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=patch), timestamp_ms)
        if not result.pose_landmarks:
            continue
        side = crop[2] - crop[0]
        image_xy[i] = [(crop[0] + p.x * side, crop[1] + p.y * side) for p in result.pose_landmarks[0]]
        visibility[i] = [min(p.visibility, p.presence) for p in result.pose_landmarks[0]]
        world[i] = [(p.x, p.y, p.z) for p in result.pose_world_landmarks[0]]
        valid[i] = True
    landmarker.close()
    elapsed = time.time() - t_start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        times_s=times_s,
        image_xy=image_xy,
        visibility=visibility,
        world=world,
        valid=valid,
        crops=crops,
        width=width,
        height=height,
        source_fps=fps,
        stride=stride,
        seconds=elapsed,
    )
    print(
        f"[mediapipe] {n} frames, {valid.mean():.0%} with a pose, {1000 * elapsed / n:.0f} ms/frame -> {args.out}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--track-npz", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
