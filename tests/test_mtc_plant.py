"""MTC's plant for saber policies: detection, generated models, and equivalence with mjlab.

The unit tests build tiny synthetic models. The equivalence tests at the bottom compare the
saber plant generated from the rootless ROS install ($HCF_DATA/mtc, docs/mtc.md) with the
exact mjlab scene of the newest saber export, and skip when either is absent.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from fencing import paths
from fencing.saber import spec as saber_spec
from fencing.sim2sim import mtc, mtc_plant

SABER_TASK = "Fencing-Tracking-Flat-G1-Saber"
PLAIN_TASK = "Mjlab-Tracking-Flat-Unitree-G1"

# Shaped like unitree_description's g1.xml where it matters: relative meshdir, a childclass on
# the root body, and the saber's parent link at the end of an arm.
WRIST_MJCF = """<mujoco model="g1">
  <compiler angle="radian" meshdir="../meshes/g1"/>
  <default><default class="g1"><joint damping="0.1"/><geom contype="1" conaffinity="1"/></default></default>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8" childclass="g1">
      <freejoint/>
      <geom size="0.05" mass="5"/>
      <body name="right_elbow_link" pos="0 -0.2 0">
        <joint name="right_elbow_joint" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
        <body name="right_wrist_yaw_link" pos="0.2 0 0" quat="0.9238795 0 0 0.3826834">
          <joint name="right_wrist_yaw_joint" axis="0 0 1"/>
          <geom size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

WRIST_URDF = """<?xml version="1.0"?>
<robot name="robot">
  <link name="right_wrist_yaw_link"><inertial><mass value="0.5"/></inertial></link>
  <ros2_control name="sim" type="system"/>
</robot>
"""


def _export_dir(tmp_path: Path, task: str | None, scene_body: str | None) -> Path:
    d = tmp_path / "model_2999"
    (d / "scene").mkdir(parents=True)
    info = {"motion_frames": 10} if task is None else {"task": task, "motion_frames": 10}
    (d / "export.json").write_text(json.dumps(info))
    if scene_body is not None:
        (d / "scene" / "scene.xml").write_text(
            f'<mujoco><worldbody><body name="robot/pelvis"><body name="{scene_body}"/></body></worldbody></mujoco>')
    return d


def test_saber_task_with_saber_in_its_scene_runs_on_the_saber_plant(tmp_path):
    assert mtc_plant.policy_plant(_export_dir(tmp_path, SABER_TASK, "robot/saber")) == mtc_plant.SABER


def test_plain_task_without_saber_in_its_scene_runs_on_unitree_descriptions_g1(tmp_path):
    assert mtc_plant.policy_plant(_export_dir(tmp_path, PLAIN_TASK, "robot/torso_link")) == mtc_plant.PLAIN


def test_task_id_and_scene_that_disagree_about_the_saber_are_refused(tmp_path):
    # Either could be wrong (a renamed task, a hand-edited scene); guessing would score the
    # policy on the wrong robot without anyone noticing.
    with pytest.raises(mtc_plant.PlantError, match="disagree"):
        mtc_plant.policy_plant(_export_dir(tmp_path, PLAIN_TASK, "robot/saber"))


def test_export_with_neither_task_id_nor_scene_is_refused(tmp_path):
    with pytest.raises(mtc_plant.PlantError, match="cannot tell"):
        mtc_plant.policy_plant(_export_dir(tmp_path, None, None))


def test_one_source_is_enough_when_the_other_is_missing(tmp_path):
    assert mtc_plant.policy_plant(_export_dir(tmp_path / "a", None, "robot/saber")) == mtc_plant.SABER
    assert mtc_plant.policy_plant(_export_dir(tmp_path / "b", SABER_TASK, None)) == mtc_plant.SABER


def test_task_id_detection_is_by_dash_separated_word_not_substring():
    assert mtc_plant.task_has_saber("Fencing-Tracking-Flat-G1-Saber")
    assert not mtc_plant.task_has_saber("Mjlab-Tracking-Flat-Unitree-G1")
    assert not mtc_plant.task_has_saber("Fencing-Tracking-Flat-G1-Sabermetrics")


def _write_wrist_mjcf(tmp_path: Path) -> Path:
    src = tmp_path / "share" / "mjcf" / "g1.xml"
    src.parent.mkdir(parents=True)
    src.write_text(WRIST_MJCF)
    return src


def _tip_in_wrist(model) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    wrist = next(b for b in range(model.nbody) if model.body(b).name.split("/")[-1] == saber_spec.PARENT_BODY)
    tip = next(s for s in range(model.nsite) if model.site(s).name.split("/")[-1] == saber_spec.TIP_SITE)
    return data.xmat[wrist].reshape(3, 3).T @ (data.site_xpos[tip] - data.xpos[wrist])


def test_saber_mjcf_welds_the_spec_saber_to_the_right_wrist(tmp_path):
    import mujoco

    out = mtc_plant.write_saber_mjcf(_write_wrist_mjcf(tmp_path), tmp_path / "models" / "g1_saber.xml")
    model = mujoco.MjModel.from_xml_path(str(out))
    plain = mujoco.MjModel.from_xml_string(WRIST_MJCF)
    saber = model.body(saber_spec.SABER_BODY)
    assert model.body(int(saber.parentid[0])).name == saber_spec.PARENT_BODY
    # A weld: no new qpos, so the hold plugin's start vector keeps its length.
    assert (model.nq, model.nv, model.nbody) == (plain.nq, plain.nv, plain.nbody + 1)
    props = saber_spec.mass_properties()
    assert saber.mass[0] == pytest.approx(props.mass)
    np.testing.assert_allclose(saber.ipos, props.com, atol=1e-9)
    # MJCF stores the mount quaternion to 6 digits (mjlab's scene.xml does too): ~1e-7 m at the tip.
    np.testing.assert_allclose(_tip_in_wrist(model), saber_spec.tip_in_hand(saber_spec.NOMINAL_SABER), atol=1e-6)
    # The root's childclass enables collisions; the saber must not inherit that.
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == saber.id]
    assert geoms and all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in geoms)


def test_saber_mjcf_points_its_meshdir_back_at_the_original_meshes(tmp_path):
    # Written outside the package share, so a relative meshdir must be rebased; relative (not
    # absolute) so the same file loads on the host and inside the rootfs (/hostdata).
    src = _write_wrist_mjcf(tmp_path)
    out = mtc_plant.write_saber_mjcf(src, tmp_path / "models" / "g1_saber.xml")
    meshdir = ET.parse(out).getroot().find("compiler").get("meshdir")
    assert not Path(meshdir).is_absolute()
    assert (out.parent / meshdir).resolve() == (src.parent / "../meshes/g1").resolve()


def test_saber_mjcf_refuses_a_model_that_already_has_a_saber(tmp_path):
    src = _write_wrist_mjcf(tmp_path)
    once = mtc_plant.write_saber_mjcf(src, tmp_path / "once.xml")
    with pytest.raises(ValueError, match="saber"):
        mtc_plant.write_saber_mjcf(once, tmp_path / "twice.xml")


def test_saber_urdf_adds_the_fixed_saber_link_under_the_right_wrist():
    root = ET.fromstring(mtc_plant.saber_urdf(WRIST_URDF))
    joint = next(j for j in root.iter("joint") if j.find("child").get("link") == saber_spec.SABER_BODY)
    assert joint.get("type") == "fixed" and joint.find("parent").get("link") == saber_spec.PARENT_BODY
    assert root.find("ros2_control") is not None  # the simulator's hardware interface survives
    with pytest.raises(ValueError, match="saber"):
        mtc_plant.saber_urdf(mtc_plant.saber_urdf(WRIST_URDF))


def test_saber_urdf_refuses_a_urdf_without_the_parent_link():
    with pytest.raises(ValueError, match=saber_spec.PARENT_BODY):
        mtc_plant.saber_urdf('<robot name="robot"><link name="pelvis"/></robot>')


def test_scoring_a_saber_policy_on_a_plant_without_the_saber_is_refused(tmp_path):
    plain = _write_wrist_mjcf(tmp_path)
    saber = mtc_plant.write_saber_mjcf(plain, tmp_path / "g1_saber.xml")
    mtc_plant.require_plant(mtc_plant.SABER, saber)
    mtc_plant.require_plant(mtc_plant.PLAIN, plain)
    with pytest.raises(mtc_plant.PlantError, match="no saber"):
        mtc_plant.require_plant(mtc_plant.SABER, plain)
    with pytest.raises(mtc_plant.PlantError, match="has a saber"):
        mtc_plant.require_plant(mtc_plant.PLAIN, saber)


def test_settle_window_defaults_to_zero_like_python_sim2sim():
    # With the start hold, a settle only drops the start transient and flatters MTC.
    args = mtc.build_parser().parse_args(["some_export"])
    assert args.settle_seconds == 0.0
    assert mtc.build_parser().parse_args(["x", "--settle-seconds", "1"]).settle_seconds == 1.0


# --- Equivalence with the mjlab scene (real models; skipped without them) -------------------------


def _newest_saber_scene() -> Path | None:
    scenes = sorted(paths.EVAL.glob("*_tip/model_*/scene/scene.xml"))
    return scenes[-1] if scenes else None


@pytest.fixture(scope="module")
def plants(tmp_path_factory):
    import mujoco

    g1_xml, base_urdf, scene = mtc_plant.ROOTFS_G1_MJCF, mtc_plant.BASE_URDF, _newest_saber_scene()
    for need in (g1_xml, base_urdf, scene):
        if need is None or not need.is_file():
            pytest.skip(f"needs {g1_xml}, {base_urdf} (docs/mtc.md) and a saber export under {paths.EVAL}")
    urdf, mjcf = mtc_plant.write_saber_plant(tmp_path_factory.mktemp("models"), g1_xml, base_urdf)
    return {"mtc": mujoco.MjModel.from_xml_path(str(mjcf)), "mjlab": mujoco.MjModel.from_xml_path(str(scene)),
            "urdf": ET.parse(urdf).getroot(), "scene": scene}


def _body(model, name: str) -> int:
    return next(b for b in range(model.nbody) if model.body(b).name.split("/")[-1] == name)


def test_mtc_saber_plant_has_the_mjlab_scenes_total_robot_mass(plants):
    mtc_model, mjlab = plants["mtc"], plants["mjlab"]
    assert mtc_model.body_subtreemass[_body(mtc_model, "pelvis")] == pytest.approx(
        mjlab.body_subtreemass[_body(mjlab, "pelvis")], abs=1e-6)


def test_mtc_saber_body_has_the_mjlab_scenes_mount_and_inertial(plants):
    mtc_model, mjlab = plants["mtc"], plants["mjlab"]
    a, b = _body(mtc_model, saber_spec.SABER_BODY), _body(mjlab, saber_spec.SABER_BODY)
    assert mtc_model.body(int(mtc_model.body_parentid[a])).name.split("/")[-1] == saber_spec.PARENT_BODY
    assert mjlab.body(int(mjlab.body_parentid[b])).name.split("/")[-1] == saber_spec.PARENT_BODY
    for field in ("body_pos", "body_quat", "body_mass", "body_ipos", "body_iquat", "body_inertia"):
        np.testing.assert_allclose(getattr(mtc_model, field)[a], getattr(mjlab, field)[b], atol=1e-6, err_msg=field)


def test_mtc_saber_blade_tip_is_where_the_mjlab_scene_puts_it_in_the_wrist_frame(plants):
    np.testing.assert_allclose(_tip_in_wrist(plants["mtc"]), _tip_in_wrist(plants["mjlab"]), atol=1e-6)


def test_mtc_saber_has_the_mjlab_scenes_collision_geometry(plants):
    def saber_collision(model):
        body = _body(model, saber_spec.SABER_BODY)
        return sorted((int(model.geom_type[g]), tuple(np.round(model.geom_size[g], 6)))
                      for g in range(model.ngeom)
                      if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g]))
    assert saber_collision(plants["mtc"]) == saber_collision(plants["mjlab"])


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # URDF: fixed-axis roll about x, then pitch about y, then yaw about z.
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_mtc_saber_urdf_matches_the_mjlab_scene_to_its_printed_precision(plants):
    # The controller's pinocchio model and robot_state_publisher read this, not the MJCF.
    root, mjlab = plants["urdf"], plants["mjlab"]
    joints = {j.get("name"): j for j in root.iter("joint")}
    mount, tip = joints["saber_mount_joint"].find("origin"), joints["blade_tip_joint"].find("origin")
    rotation = _rpy_matrix(*map(float, mount.get("rpy").split()))
    tip_in_wrist = (np.array(mount.get("xyz").split(), float)
                    + rotation @ np.array(tip.get("xyz").split(), float))
    np.testing.assert_allclose(tip_in_wrist, _tip_in_wrist(mjlab), atol=1e-4)
    saber = next(link for link in root.iter("link") if link.get("name") == saber_spec.SABER_BODY)
    inertial = saber.find("inertial")
    b = _body(mjlab, saber_spec.SABER_BODY)
    assert float(inertial.find("mass").get("value")) == pytest.approx(mjlab.body_mass[b], abs=1e-4)
    inertia = inertial.find("inertia")
    np.testing.assert_allclose([float(inertia.get(k)) for k in ("ixx", "iyy", "izz")], mjlab.body_inertia[b],
                               rtol=1e-5)
    np.testing.assert_allclose(np.array(inertial.find("origin").get("xyz").split(), float), mjlab.body_ipos[b],
                               atol=1e-5)
