"""MP4 writing and a reference "ghost" overlay for MuJoCo offscreen renders.

Numpy + mujoco only, so both the fencing venv (sim2sim) and the mjlab venv (train play)
can use it. Frames go to ffmpeg over a pipe (H.264, yuv420p, plays everywhere); without
ffmpeg we fall back to OpenCV's mp4v writer.
"""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FALLBACK_FFMPEG = Path.home() / ".local" / "bin" / "ffmpeg"
GHOST_RGBA = (0.45, 0.75, 0.45, 0.35)


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return str(FALLBACK_FFMPEG) if FALLBACK_FFMPEG.is_file() else None


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.frames = 0
        self._process = None
        self._cv_writer = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            self._process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(self.path)],
                stdin=subprocess.PIPE,
            )
            self.backend = "ffmpeg"
        else:
            import cv2

            self._cv_writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
            if not self._cv_writer.isOpened():
                raise RuntimeError(f"neither ffmpeg nor OpenCV could open {self.path} for writing")
            self.backend = "opencv"


    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError(f"frame {frame.shape} does not match video size {self.size}")
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if self._process is not None:
            self._process.stdin.write(frame.tobytes())
        else:
            import cv2

            self._cv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> Path:
        if self._process is not None:
            self._process.stdin.close()
            if self._process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path} (exit {self._process.returncode})")
            self._process = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        return self.path


def ghost_model(model: mujoco.MjModel, rgba=GHOST_RGBA) -> mujoco.MjModel:
    """Copy of `model` whose visual geoms are tinted and collision geoms hidden (as mjlab)."""
    ghost = copy.deepcopy(model)
    for geom in range(ghost.ngeom):
        if ghost.geom_contype[geom] != 0 or ghost.geom_conaffinity[geom] != 0 or ghost.geom_bodyid[geom] == 0:
            ghost.geom_rgba[geom, 3] = 0.0
        else:
            ghost.geom_rgba[geom] = rgba
    return ghost


class SceneRenderer:
    """Offscreen renderer with a tracking camera and an optional reference ghost."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480,
                 track_body: int | None = None, with_ghost: bool = True):
        render_model = copy.deepcopy(model)
        render_model.vis.global_.offwidth = max(width, render_model.vis.global_.offwidth)
        render_model.vis.global_.offheight = max(height, render_model.vis.global_.offheight)
        render_model.light_castshadow[:] = False  # shadows cost ~4x on llvmpipe (EGL on WSL2)
        render_model.mat_reflectance[:] = 0.0
        self.model = render_model
        self.renderer = mujoco.Renderer(render_model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        if track_body is not None:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = track_body
        self.camera.distance = 3.0
        self.camera.azimuth = 120.0
        self.camera.elevation = -10.0
        self.ghost = ghost_model(render_model) if with_ghost else None
        self.ghost_data = mujoco.MjData(self.ghost) if with_ghost else None
        self._option = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()

    def render(self, data: mujoco.MjData, ghost_qpos: np.ndarray | None = None) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera)
        if self.ghost is not None and ghost_qpos is not None:
            self.ghost_data.qpos[:] = ghost_qpos
            mujoco.mj_kinematics(self.ghost, self.ghost_data)
            mujoco.mjv_addGeoms(self.ghost, self.ghost_data, self._option, self._perturb,
                                mujoco.mjtCatBit.mjCAT_DYNAMIC.value, self.renderer.scene)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
