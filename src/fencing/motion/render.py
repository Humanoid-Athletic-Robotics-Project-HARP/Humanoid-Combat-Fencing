"""Kinematic replay to MP4 / contact-sheet PNG with MuJoCo offscreen rendering.

Independent of mjlab and W&B: a seconds-long visual check of a CSV. Each motion gets its
own camera over a 0.25 m checker floor: fixed for short footwork (slides and drift show
as motion against the tiles), following the smoothed pelvis for long walks. Planted feet get a green marker; any sole point below
the floor gets a red one.

MUJOCO_GL must be chosen before mujoco is imported; fencing.motion.cli sets it to egl. On
this machine EGL resolves to Mesa llvmpipe, i.e. CPU rendering, so it never touches the GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fencing.motion.contacts import ContactParams, contacts_from_fk  # noqa: E402
from fencing.motion.filters import lowpass  # noqa: E402
from fencing.motion.io import Motion  # noqa: E402
from fencing.motion.kinematics import G1Model, forward_kinematics, load_g1  # noqa: E402

CONTACT_RGBA = np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32)
PENETRATION_RGBA = np.array([0.95, 0.1, 0.1, 1.0], dtype=np.float32)
FIXED_CAMERA_MAX_EXTENT_M = 1.2


def build_scene_model(g1: G1Model, max_width: int = 1280, max_height: int = 960, saber: bool = False) -> mujoco.MjModel:
    """The training MJCF plus a checker floor and lights (the bare robot file has neither).

    saber=True welds the nominal training saber to the right wrist (visual only, no joint,
    so the qpos layout is unchanged).
    """
    spec = mujoco.MjSpec.from_file(str(g1.mjcf_path))
    if saber:
        from fencing.saber.spec import attach_saber

        attach_saber(spec)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, max_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, max_height)
    # Shadow maps cost ~1 s per frame on llvmpipe (CPU GL); flat lighting is enough to judge feet.
    spec.visual.quality.shadowsize = 0
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.add_texture(
        name="render_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.85, 0.9, 0.95],
        rgb2=[0.55, 0.65, 0.75],
        width=64,
        height=64,
    )
    spec.add_texture(
        name="floor_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.82, 0.80],
        rgb2=[0.55, 0.57, 0.60],
        width=256,
        height=256,
    )
    material = spec.add_material(name="floor_mat")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_checker"
    # texrepeat 2 per metre on a 2-square checker texture -> 0.25 m tiles.
    material.texrepeat = [2.0, 2.0]
    material.texuniform = True
    spec.worldbody.add_geom(
        name="render_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1],
        material="floor_mat",
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(pos=[0, 0, 4.0], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=False)
    spec.worldbody.add_light(pos=[2.0, -3.0, 3.0], dir=[-0.4, 0.6, -0.7], diffuse=[0.4, 0.4, 0.4], castshadow=False)
    model = spec.compile()
    if model.nq != g1.model.nq:
        raise RuntimeError(f"render scene changed nq ({model.nq} vs {g1.model.nq}); qpos layout would not match")
    return model


class MotionRenderer:
    def __init__(self, width: int = 480, height: int = 360, g1: G1Model | None = None, saber: bool = False):
        self.g1 = g1 or load_g1()
        self.model = build_scene_model(self.g1, saber=saber)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.width, self.height = width, height
        # The blade reaches ~0.9 m past the wrist (above the head in quinte): aim higher, back off.
        self.lookat_z, self.extra_distance = (0.75, 0.6) if saber else (0.45, 0.0)

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> MotionRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def prepare(self, motion: Motion, azimuth: float = 90.0, elevation: float = -12.0) -> dict:
        """FK, contacts and camera for one motion (computed once, not per frame).

        The camera is fixed when the whole motion fits in ~1.2 m (fencing footwork): slides
        are then plain motion against the floor tiles. Longer motions (walks) would make
        the robot tiny, so the camera follows the low-passed pelvis instead; slides still
        show as a planted foot drifting over the tiles. azimuth 90 puts +x (forward after
        canonicalisation) left-to-right in the image.
        """
        fk = forward_kinematics(motion, self.g1)
        contacts, _ = contacts_from_fk(fk, motion.fps, ContactParams(), floor_known=True)
        xy = motion.root_pos[:, :2]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation = azimuth, elevation
        follow = None
        if float(np.max(hi - lo)) <= FIXED_CAMERA_MAX_EXTENT_M:
            camera.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, self.lookat_z]
            camera.distance = 2.2 + 1.4 * float(np.max(hi - lo)) + self.extra_distance
        else:
            follow = lowpass(xy, 0.5, motion.fps)
            camera.distance = 2.6 + self.extra_distance
        return {"motion": motion, "qpos": self.g1.qpos_from_motion(motion), "fk": fk, "contacts": contacts, "camera": camera, "follow": follow}

    def frame(self, prepared: dict, t: int, label: str = "") -> np.ndarray:
        motion, fk, contacts, camera = prepared["motion"], prepared["fk"], prepared["contacts"], prepared["camera"]
        if prepared["follow"] is not None:
            camera.lookat[:] = [*prepared["follow"][t], self.lookat_z]
        self.data.qpos[:] = prepared["qpos"][t]
        mujoco.mj_kinematics(self.model, self.data)
        self.renderer.update_scene(self.data, camera)
        for f in range(2):
            if contacts[t, f]:
                # Flat disc on the floor around the planted foot: visible from the side.
                self._marker(fk.sole_center[t, f] * [1, 1, 0] + [0, 0, 0.002], CONTACT_RGBA, 0.09, flat=True)
            if fk.foot_lowest_z[t, f] < -0.005:
                self._marker(fk.sole_center[t, f] + [0, 0, 0.06], PENETRATION_RGBA, 0.03)
        image = np.array(self.renderer.render(), copy=True)
        _put_text(image, f"{label} f{t} {t / motion.fps:.2f}s".strip())
        return image

    def _marker(self, pos: np.ndarray, rgba: np.ndarray, radius: float, flat: bool = False) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER if flat else mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.002, 0]) if flat else np.array([radius, 0, 0]),
            np.asarray(pos, dtype=np.float64),
            np.eye(3).flatten(),
            rgba,
        )
        scene.ngeom += 1


def _put_text(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def contact_sheet(
    motions: list[tuple[str, Motion]],
    out_path: str | Path,
    columns: int = 8,
    width: int = 320,
    height: int = 260,
    azimuth: float = 90.0,
    saber: bool = False,
    frames: list[int] | None = None,
) -> Path:
    """One row per motion, `columns` frames at the same relative times in each row.

    frames: explicit frame indices instead (e.g. blade keyframes), clipped to each motion.
    """
    with MotionRenderer(width, height, saber=saber) as renderer:
        rows = []
        for label, motion in motions:
            prepared = renderer.prepare(motion, azimuth)
            if frames is not None:
                idx = np.clip(np.asarray(frames, dtype=int), 0, motion.n_frames - 1)
            else:
                idx = np.linspace(0, motion.n_frames - 1, columns).round().astype(int)
            rows.append(np.concatenate([renderer.frame(prepared, int(t), label) for t in idx], axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write contact sheet to {out_path}")
    return out_path


def render_video(
    motions: list[tuple[str, Motion]],
    out_path: str | Path,
    width: int = 480,
    height: int = 360,
    azimuth: float = 90.0,
    saber: bool = False,
) -> Path:
    """MP4 at the motion fps; several motions are placed side by side (before | after)."""
    fps = motions[0][1].fps
    n = max(m.n_frames for _, m in motions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(motions), height))
    if not writer.isOpened():
        raise OSError(f"could not open video writer for {out_path}")
    try:
        with MotionRenderer(width, height, saber=saber) as renderer:
            prepared = [(label, renderer.prepare(m, azimuth)) for label, m in motions]
            for t in range(n):
                tiles = [renderer.frame(p, min(t, p["motion"].n_frames - 1), label) for label, p in prepared]
                writer.write(cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out_path
