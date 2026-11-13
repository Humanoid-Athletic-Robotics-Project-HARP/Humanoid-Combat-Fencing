"""GVHMR on one clip, on the pipeline's fencer. Runs inside the `gvhmr` venv, launched by
`fencing pose` with the working directory HCF_DATA/gvhmr (GVHMR finds inputs/checkpoints
relative to it):

    python -m fencing.pose.gvhmr_worker --video clips/<clip_id>.mp4 --output-root pose/<clip_id>/gvhmr \
        --start 12.0 --track-npz pose/<yt_id>/track.npz --track-id 3 \
        [--mediapipe-npz pose/<clip_id>/mediapipe_raw.npz] [--predict] [--render]

It drives GVHMR's own demo code (third_party/GVHMR/tools/demo/demo.py, imported, never
edited) with one change: demo.py tracks the largest person, so this writes the fencer's boxes
to GVHMR's preprocess/bbx.pt first, and demo.py's run_preprocess then skips its tracker.

    1. the clip (already cut at 30 fps) -> <output-root>/<clip_id>/0_input_video.mp4
    2. YOLOv8x person detections (GVHMR's Tracker) -> preprocess/yolo_tracks.npz; the fencer's
       box per frame (fencing.pose.gvhmr_boxes) -> preprocess/bbx.pt
    3. demo.run_preprocess: ViTPose-H 2D keypoints and HMR2 image features on those boxes
    4. fencer check: ViTPose vs MediaPipe skeleton agreement -> <output-root>/fencer_check.json
       and fencer_check.png (boxes and keypoints on 6 frames)
    5. --predict (needs body_models/smplx/SMPLX_NEUTRAL.npz): the GVHMR network -> hmr4d_results.pt
    6. --render (also needs body_models/smpl/SMPL_NEUTRAL.pkl): demo.py's mesh videos. Rendering
       loads hmr4d/utils/body_model/*.pt relative to the working directory, so it runs from
       third_party/GVHMR. A render failure is printed as a warning and never fails the clip.

Steps 1-4 need no body model; every output is cached and reused on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from fencing import paths
from fencing.pose import gvhmr_boxes as gb
from fencing.pose.screening import track_from_arrays
from fencing.pose.tracking import interpolate_track_boxes

GVHMR_SRC = paths.THIRD_PARTY / "GVHMR"
BOX_ENLARGE = 1.2  # demo.py's get_bbx_xys_from_xyxy(base_enlarge=1.2)
COCO17_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
OVERLAY_FRAMES = 6
OVERLAY_WIDTH = 640


def load_demo_module():
    """third_party/GVHMR/tools/demo/demo.py as a module (its main() is guarded)."""
    spec = importlib.util.spec_from_file_location("gvhmr_demo", GVHMR_SRC / "tools" / "demo" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose_cfg(video_name: str, output_root: Path):
    """demo.py's config with -s (static camera), without its argparse."""
    from hmr4d.configs import register_store_gvhmr
    from hydra import compose, initialize_config_module

    overrides = [f"video_name={video_name}", "static_cam=True", "verbose=False", "use_dpvo=False", f"output_root={output_root}"]
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        return compose(config_name="demo", overrides=overrides)


def pipeline_boxes(track_npz: Path, track_id: int, times_s: np.ndarray) -> np.ndarray:
    with np.load(track_npz) as data:
        track = track_from_arrays(data, track_id)
        frame_times = np.asarray(data["frame_times"])
    fractional_frames = np.interp(times_s, frame_times, np.arange(len(frame_times)))
    return interpolate_track_boxes(track, fractional_frames)


def yolo_detections(video_path: str, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame, track id, xyxy box) of every YOLOv8x person detection, cached in `cache`."""
    if cache.exists():
        with np.load(cache) as data:
            return data["frame"], data["track_id"], data["box"]
    from hmr4d.utils.preproc import Tracker

    history = Tracker().track(video_path)
    rows = [(f, det["id"], det["bbx_xyxy"]) for f, dets in enumerate(history) for det in dets]
    frame = np.array([r[0] for r in rows], dtype=np.int64)
    track_id = np.array([r[1] for r in rows], dtype=np.int64)
    box = np.array([r[2] for r in rows], dtype=np.float64).reshape(-1, 4)
    np.savez_compressed(cache, frame=frame, track_id=track_id, box=box)
    return frame, track_id, box


def guide_digest(guide: np.ndarray) -> str:
    """sha256 of the screen-track boxes used as the guide: shape, then little-endian float64 bytes."""
    array = np.ascontiguousarray(guide, dtype="<f8")
    return hashlib.sha256(repr(array.shape).encode() + array.tobytes()).hexdigest()


def fencer_key(args, n_frames: int, guide: np.ndarray) -> dict:
    """What the preprocessing cache depends on. The guide digest covers the track.npz contents
    (a re-run of screening can rewrite it under the same path and id)."""
    return {
        "track_npz": str(args.track_npz), "track_id": args.track_id, "start_s": round(args.start, 3),
        "frames": n_frames, "guide_sha256": guide_digest(guide),
    }  # fmt: skip


def cached_boxes_meta(meta_path: Path, bbx_path: Path, key: dict, boxes_npz: Path) -> dict | None:
    """The cached box summary when it was made for `key`, else None. A cache written before the
    key had a guide digest is checked on the guide it stored (fencer_boxes.npz `pipeline`), and
    when it matches its key is rewritten in place with the digest, so it is not recomputed."""
    if not (meta_path.exists() and bbx_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    stored = meta.get("key") or {}
    if stored == key:
        return meta
    if "guide_sha256" in stored:
        return None
    if stored != {k: v for k, v in key.items() if k != "guide_sha256"}:
        return None
    if boxes_npz.exists():
        with np.load(boxes_npz) as data:
            if guide_digest(data["pipeline"]) != key["guide_sha256"]:
                return None
    meta["key"] = key
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_fencer_boxes(cfg, args, n_frames: int) -> dict:
    """preprocess/bbx.pt from the pipeline's fencer track; returns the box summary."""
    import torch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    preprocess = Path(cfg.preprocess_dir)
    meta_path = preprocess / "fencer_boxes.json"
    times = gb.clip_frame_times(args.start, n_frames)
    guide = pipeline_boxes(args.track_npz, args.track_id, times)
    key = fencer_key(args, n_frames, guide)
    meta = cached_boxes_meta(meta_path, Path(cfg.paths.bbx), key, preprocess / "fencer_boxes.npz")
    if meta is not None:
        return meta
    # New or changed fencer: everything computed on the old boxes is stale.
    for name in ("bbx", "vitpose", "vit_features"):
        Path(cfg.paths[name]).unlink(missing_ok=True)
    Path(cfg.paths.hmr4d_results).unlink(missing_ok=True)

    frame, _, box = yolo_detections(cfg.video_path, preprocess / "yolo_tracks.npz")
    boxes, source, matched_iou = gb.select_fencer_boxes(frame, box, guide)
    bbx_xyxy = torch.from_numpy(boxes).float()
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=BOX_ENLARGE).float()
    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, cfg.paths.bbx)
    np.savez_compressed(preprocess / "fencer_boxes.npz", boxes=boxes, pipeline=guide, source=source, matched_iou=matched_iou)
    meta = {"key": key, **gb.box_summary(source, matched_iou)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def fencer_check(cfg, args, box_meta: dict) -> dict:
    import torch

    preprocess = Path(cfg.preprocess_dir)
    with np.load(preprocess / "fencer_boxes.npz") as data:
        boxes, guide = data["boxes"], data["pipeline"]
    vitpose = torch.load(cfg.paths.vitpose).cpu().numpy()
    times = gb.clip_frame_times(args.start, len(boxes))
    check = {"clip": Path(args.video).stem, "track_id": args.track_id, "fps": gb.GVHMR_FPS, "boxes": box_meta}
    mp = None
    if args.mediapipe_npz and args.mediapipe_npz.exists():
        with np.load(args.mediapipe_npz) as data:
            mp = {k: data[k] for k in ("times_s", "image_xy", "valid")}
        check["keypoints"] = gb.keypoint_agreement(vitpose, times, mp["times_s"], mp["image_xy"], mp["valid"], boxes)
    else:
        check["keypoints"] = {"frames_compared": 0, "same_person": None, "note": "no mediapipe_raw.npz"}
    out_root = Path(args.output_root)
    (out_root / "fencer_check.json").write_text(json.dumps(check, indent=2) + "\n")
    draw_overlay(cfg.video_path, boxes, guide, vitpose, times, mp, out_root / "fencer_check.png", check)
    return check


def draw_overlay(video_path, boxes, guide, vitpose, times, mp, out_png: Path, check: dict) -> None:
    """Contact sheet: chosen box (green), pipeline box (yellow), other YOLO people (red, thin),
    ViTPose limbs (cyan), MediaPipe landmarks (magenta)."""
    import cv2

    with np.load(Path(video_path).parent / "preprocess" / "yolo_tracks.npz") as data:
        det_frame, det_box = data["frame"], data["box"]
    picks = np.linspace(0, len(boxes) - 1, OVERLAY_FRAMES).round().astype(int)
    cap = cv2.VideoCapture(str(video_path))
    tiles = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        for b in det_box[det_frame == f]:
            cv2.rectangle(img, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 0, 255), 1)
        cv2.rectangle(img, tuple(int(v) for v in guide[f, :2]), tuple(int(v) for v in guide[f, 2:]), (0, 255, 255), 2)
        cv2.rectangle(img, tuple(int(v) for v in boxes[f, :2]), tuple(int(v) for v in boxes[f, 2:]), (0, 255, 0), 3)
        for a, b in COCO17_EDGES:
            if vitpose[f, a, 2] > gb.VITPOSE_MIN_CONF and vitpose[f, b, 2] > gb.VITPOSE_MIN_CONF:
                cv2.line(img, tuple(int(v) for v in vitpose[f, a, :2]), tuple(int(v) for v in vitpose[f, b, :2]), (255, 255, 0), 3)
        if mp is not None:
            j = int(np.argmin(np.abs(mp["times_s"] - times[f])))
            if mp["valid"][j]:
                for x, y in mp["image_xy"][j, gb.MEDIAPIPE_LIMBS]:
                    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)
        cv2.rectangle(img, (0, 0), (330, 42), (0, 0, 0), -1)
        cv2.putText(img, f"frame {f}  t={times[f]:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        scale = OVERLAY_WIDTH / img.shape[1]
        tiles.append(cv2.resize(img, (OVERLAY_WIDTH, int(round(img.shape[0] * scale)))))
    cap.release()
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    kp, bx = check["keypoints"], check["boxes"]
    title = (
        f"{check['clip']}  track {check['track_id']}  yolo {bx['yolo_frac']:.0%} bridged {bx['bridged_frac']:.0%} "
        f"pipeline {bx['pipeline_frac']:.0%}  same_person={kp.get('same_person')} "
        f"dist_med={kp.get('centroid_dist_median')} within={kp.get('frac_within')}"
    )
    band = np.zeros((40, sheet.shape[1], 3), dtype=sheet.dtype)
    cv2.putText(band, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack([band, sheet]))


def copy_input_video(src: Path, cfg) -> int:
    from hmr4d.utils.video_io_utils import get_video_lwh

    dst = Path(cfg.video_path)
    n = get_video_lwh(src)[0]
    if not dst.exists() or get_video_lwh(dst)[0] != n:
        shutil.copy2(src, dst)
    return n


def predict(cfg, demo) -> None:
    import hydra
    import torch
    from hmr4d.utils.net_utils import detach_to_cpu

    if Path(cfg.paths.hmr4d_results).exists():
        print(f"[gvhmr] reusing {cfg.paths.hmr4d_results}", flush=True)
        return
    data = demo.load_data_dict(cfg)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    tic = time.time()
    with torch.no_grad():
        pred = detach_to_cpu(model.predict(data, static_cam=True))
    print(f"[gvhmr] network {time.time() - tic:.1f}s for {int(data['length'])} frames", flush=True)
    torch.save(pred, cfg.paths.hmr4d_results)


def render(cfg, demo) -> None:
    here = os.getcwd()
    try:
        os.chdir(GVHMR_SRC)  # demo.py loads hmr4d/utils/body_model/*.pt relative to the cwd
        demo.render_incam(cfg)
        demo.render_global(cfg)
        if not Path(cfg.paths.incam_global_horiz_video).exists():
            demo.merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)
    except Exception as exc:  # noqa: BLE001  rendering is QA only; hmr4d_results.pt is already saved
        print(f"[gvhmr] WARNING: render failed, results are unaffected: {type(exc).__name__}: {exc}", flush=True)
    finally:
        os.chdir(here)


def run(args: argparse.Namespace) -> None:
    import torch

    demo = load_demo_module()
    cfg = compose_cfg(args.video.stem, args.output_root.resolve())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    n_frames = copy_input_video(args.video, cfg)
    tic = time.time()
    box_meta = write_fencer_boxes(cfg, args, n_frames)
    with torch.no_grad():
        demo.run_preprocess(cfg)
    check = fencer_check(cfg, args, box_meta)
    print(f"[gvhmr] preprocess {time.time() - tic:.1f}s; fencer check {json.dumps(check['keypoints'])}", flush=True)
    if args.predict:
        predict(cfg, demo)
        if args.render:
            render(cfg, demo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True, help="clip cut at 30 fps; its stem names GVHMR's output folder")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True, help="source time of the clip's first frame, s")
    parser.add_argument("--track-npz", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--mediapipe-npz", type=Path, help="mediapipe_raw.npz of the clip, for the fencer check")
    parser.add_argument("--predict", action="store_true", help="also run the GVHMR network (needs SMPLX_NEUTRAL.npz)")
    parser.add_argument("--render", action="store_true", help="also render demo.py's mesh videos (needs SMPL_NEUTRAL.pkl)")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
