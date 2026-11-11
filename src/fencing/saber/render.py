"""Offscreen renders of the saber G1 in en garde and the parry/cut poses from reach.json.

Uses the arm joint angles the reach analysis solved, so the pictures show exactly what the
table claims. Needs an OpenGL backend: MUJOCO_GL=egl (default here) or osmesa.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (MUJOCO_GL must be set before the import)

from fencing.saber.reach import PELVIS_POS, stance_qpos  # noqa: E402
from fencing.saber.spec import NOMINAL_SABER, SaberSpec, attach_saber, g1_spec  # noqa: E402

WIDTH, HEIGHT = 480, 480


def _scene_spec(saber: SaberSpec) -> mujoco.MjSpec:
    """Saber G1 plus a floor and lights; render-only additions, never used for physics."""
    spec = attach_saber(g1_spec(), saber)
    grid = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                            rgb1=[0.82, 0.82, 0.85], rgb2=[0.68, 0.68, 0.72],
                            width=256, height=256)
    floor_material = spec.add_material(name="floor")
    floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = grid.name
    floor_material.texrepeat = [8, 8]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[4, 4, 0.05], material="floor")
    floor.contype = 0
    floor.conaffinity = 0
    spec.worldbody.add_light(pos=[1.5, -1.5, 3.0], dir=[-0.4, 0.4, -1.0], diffuse=[0.8] * 3)
    spec.worldbody.add_light(pos=[-1.5, 1.0, 2.5], dir=[0.4, -0.3, -1.0], diffuse=[0.4] * 3)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    return spec


def _camera(azimuth: float, elevation: float = -12.0, distance: float = 2.4) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = PELVIS_POS + np.array([0.15, 0.0, 0.15])
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera


def _label(image: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
    except ImportError:  # labels are a convenience; the picture is still correct without
        return image
    image = image.copy()
    cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2,
                cv2.LINE_AA)
    return image


def _save_png(path: Path, image: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def render_poses(reach_json: Path, out_dir: Path, saber: SaberSpec = NOMINAL_SABER,
                 joint_set: str = "arm") -> list[Path]:
    """One PNG per target (front-right and side views side by side) plus a contact sheet."""
    report = json.loads(Path(reach_json).read_text())
    model = _scene_spec(saber).compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    options = mujoco.MjvOption()
    options.sitegroup[3] = 1  # show blade_tip (red) and guard (blue) sites

    written, tiles = [], []
    for result in report["results"]:
        if result["joint_set"] != joint_set:
            continue
        data.qpos[:] = stance_qpos(model)
        for name, value in result["joint_positions"].items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        views = []
        for azimuth in (150.0, 90.0):  # front-right, then from the robot's right side
            renderer.update_scene(data, camera=_camera(azimuth), scene_option=options)
            views.append(renderer.render())
        verdict = "reachable" if result["reachable"] else "NOT reachable"
        image = _label(np.concatenate(views, axis=1), f"{result['target']} ({verdict})")
        path = Path(out_dir) / f"{result['target']}.png"
        _save_png(path, image)
        written.append(path)
        tiles.append(image)

    if tiles:
        columns = 2
        while len(tiles) % columns:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.concatenate(tiles[i:i + columns], axis=1) for i in range(0, len(tiles), columns)]
        sheet = np.concatenate(rows, axis=0)
        sheet_path = Path(out_dir) / "contact_sheet.png"
        _save_png(sheet_path, sheet[::2, ::2])  # half size keeps the sheet light to open
        written.append(sheet_path)
    renderer.close()
    return written
