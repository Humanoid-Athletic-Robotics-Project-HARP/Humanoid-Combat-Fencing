"""Which robot MTC's simulator must run a policy on, and the saber plant's models.

A saber policy (task `Fencing-Tracking-Flat-G1-Saber`) was trained with the saber welded to
the right wrist (fencing.saber.spec.attach_saber). unitree_description's G1, which MTC loads by
default, has no saber, so such a policy needs a plant that does:

- MJCF for mujoco_sim_ros2: the rootfs's own g1.xml with the saber body inserted under
  right_wrist_yaw_link. The body comes from attach_saber on that same file (so mass, inertia,
  mount, geoms and sites are exactly the training ones); only that element is copied into the
  original XML, which keeps the rest byte-for-byte what MuJoCo 3.2.7 in the rootfs already loads
  (the venv's newer MuJoCo would otherwise rewrite the whole file).
- URDF for the controller's pinocchio model and robot_state_publisher: the xacro output of
  unitree_description's robot.xacro (simulation:=mujoco) plus saber.spec.urdf_snippet().

`python -m fencing.sim2sim.mtc_plant <export dir> --models-dir DIR` prints `g1`, or writes the
saber models into DIR and prints `g1_saber <urdf> <mjcf>` (for scripts/mtc_sim2sim.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fencing import paths

PLAIN = "g1"
SABER = "g1_saber"

MTC_DIR = paths.DATA / "mtc"
ROOTFS_G1_MJCF = MTC_DIR / "noble/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
BASE_URDF = MTC_DIR / "g1_mujoco.urdf"  # xacro robot.xacro robot_type:=g1 simulation:=mujoco
MODELS_DIR = MTC_DIR / "models"
SABER_MJCF_NAME = "g1_saber.xml"
SABER_URDF_NAME = "g1_saber.urdf"


class PlantError(ValueError):
    """The policy's plant cannot be determined, or does not match the model given."""


def task_has_saber(task: str) -> bool:
    return "saber" in (part.lower() for part in task.split("-"))


def _has_saber_body(xml: Path) -> bool:
    from fencing.saber.spec import SABER_BODY

    # mjlab prefixes scene names with the entity ("robot/saber").
    return any(b.get("name", "").split("/")[-1] == SABER_BODY for b in ET.parse(xml).getroot().iter("body"))


def policy_plant(export_dir: Path) -> str:
    """PLAIN or SABER, from the export's task id and from its mjlab scene; both must agree.

    The scene is what the policy was trained in; the task id is an independent record of the
    same choice. One alone decides when the other is missing; neither is refused."""
    export_dir = Path(export_dir)
    info_path = export_dir / "export.json"
    task = json.loads(info_path.read_text()).get("task") if info_path.is_file() else None
    scene = export_dir / "scene" / "scene.xml"
    by_task = task_has_saber(task) if task else None
    by_scene = _has_saber_body(scene) if scene.is_file() else None
    if by_task is None and by_scene is None:
        raise PlantError(f"cannot tell whether {export_dir} is a saber policy: no task id in {info_path} "
                         f"and no {scene}")
    if by_task is not None and by_scene is not None and by_task != by_scene:
        raise PlantError(f"task id {task!r} and {scene} disagree about the saber "
                         f"(task says {'saber' if by_task else 'plain'}, scene {'has' if by_scene else 'has no'} "
                         f"saber body): refusing to guess the plant")
    return SABER if (by_scene if by_scene is not None else by_task) else PLAIN


def require_plant(plant: str, mtc_mjcf: Path) -> None:
    """Refuse to score a policy against a recording made on the other robot."""
    has_saber = _has_saber_body(Path(mtc_mjcf))
    if plant == SABER and not has_saber:
        raise PlantError(f"saber policy, but MTC's MJCF {mtc_mjcf} has no saber body: it would be judged on "
                         f"the wrong robot (scripts/mtc_sim2sim.sh picks the saber plant)")
    if plant == PLAIN and has_saber:
        raise PlantError(f"plain-G1 policy, but MTC's MJCF {mtc_mjcf} has a saber body")


def write_saber_mjcf(g1_xml: Path, out: Path) -> Path:
    import mujoco

    from fencing.saber.spec import PARENT_BODY, SABER_BODY, attach_saber

    g1_xml, out = Path(g1_xml).resolve(), Path(out).resolve()
    spec = attach_saber(mujoco.MjSpec.from_file(str(g1_xml)))  # raises if it already has a saber
    spec.compile()  # fail here, with MuJoCo's message, rather than in the simulator
    generated = ET.fromstring(spec.to_xml())
    saber = next(b for b in generated.iter("body") if b.get("name") == SABER_BODY)

    tree = ET.parse(g1_xml)
    root = tree.getroot()
    parent = next(b for b in root.iter("body") if b.get("name") == PARENT_BODY)
    parent.append(saber)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    # Relative, not absolute: the file is loaded both on the host (scoring, start qpos) and in
    # the rootfs, which sees $HCF_DATA as /hostdata.
    mesh_dir = (g1_xml.parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", os.path.relpath(mesh_dir, out.parent))
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="unicode")
    return out


def saber_urdf(base_urdf: str) -> str:
    from fencing.saber.spec import PARENT_BODY, SABER_BODY, urdf_snippet

    links = {link.get("name") for link in ET.fromstring(base_urdf).iter("link")}
    if PARENT_BODY not in links:
        raise ValueError(f"URDF has no {PARENT_BODY} link to mount the saber on")
    if SABER_BODY in links:
        raise ValueError(f"URDF already has a {SABER_BODY!r} link")
    head, sep, tail = base_urdf.rpartition("</robot>")
    return head + urdf_snippet() + sep + tail


def write_saber_plant(out_dir: Path, g1_xml: Path = ROOTFS_G1_MJCF, base_urdf: Path = BASE_URDF) -> tuple[Path, Path]:
    """(urdf, mjcf) of the G1 with the training saber, regenerated from the current spec."""
    out_dir = Path(out_dir)
    mjcf = write_saber_mjcf(g1_xml, out_dir / SABER_MJCF_NAME)
    urdf = out_dir / SABER_URDF_NAME
    urdf.write_text(saber_urdf(Path(base_urdf).read_text()))
    return urdf, mjcf


def main(argv: list[str] | None = None) -> int:
    from fencing.sim2sim.cli import resolve_export_dir

    parser = argparse.ArgumentParser(prog="python -m fencing.sim2sim.mtc_plant", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="export dir, run dir, run dir name or run name (as for sim2sim)")
    parser.add_argument("--checkpoint")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)
    try:
        plant = policy_plant(resolve_export_dir(args.target, args.checkpoint))
    except PlantError as error:
        print(f"[mtc] FAIL: {error}", file=sys.stderr)
        return 2
    if plant == PLAIN:
        print(PLAIN)
        return 0
    for need in (ROOTFS_G1_MJCF, BASE_URDF):
        if not need.is_file():
            print(f"[mtc] FAIL: {need} missing (docs/mtc.md, 'Saber plant')", file=sys.stderr)
            return 2
    urdf, mjcf = write_saber_plant(args.models_dir)
    print(SABER, urdf, mjcf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
