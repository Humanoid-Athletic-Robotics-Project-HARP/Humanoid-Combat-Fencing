"""Person detection + COCO keypoints + shot-cut and camera-motion signals for one video.

Runs inside the `pose` venv (torch + torchvision), launched by `fencing screen`:

    python -m fencing.pose.detect_worker --video V.mp4 --out detections.npz

Detector: torchvision Keypoint R-CNN (BSD-3 code, COCO weights). It gives person boxes and
17 COCO keypoints in one pass, which the screen stage needs for "full body visible" and the
side/front view guess. Frames are strided down to <= 10 fps (--max-fps) and batched.
Everything downstream (tracking, scoring) is pure numpy in fencing.pose.tracking/screening.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Screening only needs boxes, keypoint visibility, cuts and camera motion, none of which change
# much within 1/10 s (a lunge moves the fencer's box ~20% of its width per frame at 10 fps, well
# inside the tracker's IoU gate), so frames are strided down to <= 10 fps (30 -> 10, 60 -> 10,
# 25 -> 8.3). That cuts the per-video cost 3x; consumers (fencing pose) interpolate track boxes
# by time. Batching amortises the per-call overhead, which dominates when the CPU is contended.
MAX_PROCESS_FPS = 10.0
BATCH_SIZE = 4  # bigger batches gain little and cost GPU memory shared with training jobs
DETECTOR_MIN_SIZE = 576  # shorter image side fed to the detector; 720p in, ~2x faster than 800
MIN_PERSON_SCORE = 0.5
MOTION_WIDTH = 640  # camera-motion estimate runs on a downscaled grey frame


def _histogram(gray_small: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray_small], [0], None, [64], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def _camera_motion(prev_gray: np.ndarray, gray: np.ndarray, person_boxes_small: np.ndarray) -> tuple[float, float, int]:
    """Median background displacement (dx, dy) in MOTION_WIDTH pixels, and the inlier count.

    People are masked out so a fencer crossing a static frame does not read as a pan.
    """
    mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
    for x0, y0, x1, y1 in person_boxes_small.astype(int):
        pad = 8
        mask[max(y0 - pad, 0) : y1 + pad, max(x0 - pad, 0) : x1 + pad] = 0
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
    if corners is None or len(corners) < 10:
        return float("nan"), float("nan"), 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, winSize=(21, 21), maxLevel=3)
    good = status.reshape(-1) == 1
    if good.sum() < 10:
        return float("nan"), float("nan"), int(good.sum())
    flow = (moved - corners).reshape(-1, 2)[good]
    dx, dy = np.median(flow, axis=0)
    return float(dx), float(dy), int(good.sum())


def run(video: Path, out: Path, device: str, max_fps: float = MAX_PROCESS_FPS) -> None:
    import torch
    import torchvision

    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / max_fps + 0.49)))  # max_fps 10: 60 -> 6, 30 -> 3, 24 -> 3

    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights, min_size=DETECTOR_MIN_SIZE)
    model.eval().to(device)

    motion_scale = MOTION_WIDTH / width
    frame_times, hist_diff, cam_motion, cam_inliers = [], [], [], []
    det_frame, det_box, det_score, det_kps = [], [], [], []
    state = {"prev_gray": None, "prev_hist": None, "processed": 0}
    t_start = time.time()

    def flush(batch: list[tuple[float, np.ndarray]]) -> None:
        """Detect people in a batch of frames, then add the per-frame cut/motion signals in order."""
        if not batch:
            return
        # Upload uint8 BGR and do the colour flip and float conversion on the GPU: the CPU is the
        # scarce resource here (screening runs niced next to training jobs).
        tensors = [
            torch.from_numpy(frame).to(device, non_blocking=True).flip(-1).permute(2, 0, 1).float().div_(255.0)
            for _, frame in batch
        ]
        results = model(tensors)
        for (time_s, frame), result in zip(batch, results, strict=True):
            keep = result["scores"] >= MIN_PERSON_SCORE
            boxes = result["boxes"][keep].cpu().numpy()
            scores = result["scores"][keep].cpu().numpy()
            # Keypoint R-CNN's 3rd keypoint column is a constant 1; the real confidence is
            # keypoints_scores (a heatmap logit), so store that instead.
            kp_xy = result["keypoints"][keep][..., :2].cpu().numpy()
            kp_conf = result["keypoints_scores"][keep].cpu().numpy()[..., None]
            processed = state["processed"]
            for box, score, kps in zip(boxes, scores, np.concatenate([kp_xy, kp_conf], axis=-1), strict=True):
                det_frame.append(processed)
                det_box.append(box)
                det_score.append(score)
                det_kps.append(kps)

            gray = cv2.cvtColor(cv2.resize(frame, None, fx=motion_scale, fy=motion_scale), cv2.COLOR_BGR2GRAY)
            hist = _histogram(gray)
            if state["prev_gray"] is None:
                hist_diff.append(0.0)
                cam_motion.append((0.0, 0.0))
                cam_inliers.append(0)
            else:
                hist_diff.append(float(cv2.compareHist(state["prev_hist"], hist, cv2.HISTCMP_BHATTACHARYYA)))
                dx, dy, n = _camera_motion(state["prev_gray"], gray, boxes * motion_scale)
                cam_motion.append((dx, dy))
                cam_inliers.append(n)
            state["prev_gray"], state["prev_hist"] = gray, hist
            frame_times.append(time_s)
            state["processed"] = processed + 1
            if state["processed"] % 300 == 0:
                rate = state["processed"] / (time.time() - t_start)
                print(f"[detect] {video.name}: {state['processed']} frames ({rate:.1f} fps)", flush=True)

    batch: list[tuple[float, np.ndarray]] = []
    frame_index = 0
    with torch.inference_mode():
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            batch.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame))
            frame_index += 1
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch = []
        flush(batch)
    processed = state["processed"]
    cap.release()
    if processed == 0:
        raise RuntimeError(f"no frames decoded from {video}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name and rename, so a killed run never leaves a truncated detections.npz
    # that a later `fencing screen` would trust.
    partial = out.with_name(f".{out.stem}.partial.npz")
    np.savez_compressed(
        partial,
        video=str(video),
        source_fps=fps,
        stride=stride,
        width=width,
        height=height,
        frame_times=np.asarray(frame_times),
        hist_diff=np.asarray(hist_diff),
        cam_motion=np.asarray(cam_motion, dtype=np.float64) / motion_scale,  # full-res pixels
        cam_inliers=np.asarray(cam_inliers),
        det_frame=np.asarray(det_frame, dtype=np.int64),
        det_box=np.asarray(det_box, dtype=np.float32).reshape(-1, 4),
        det_score=np.asarray(det_score, dtype=np.float32),
        det_kps=np.asarray(det_kps, dtype=np.float32).reshape(-1, 17, 3),
    )
    os.replace(partial, out)
    elapsed = time.time() - t_start
    print(f"[detect] {video.name}: {processed} frames in {elapsed:.1f}s -> {out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-fps", type=float, default=MAX_PROCESS_FPS, help="stride frames down to this rate")
    args = parser.parse_args(argv)
    run(args.video, args.out, args.device, args.max_fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
