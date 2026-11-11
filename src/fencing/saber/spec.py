"""Saber geometry, mass properties and how it is bolted to the G1 (ADR 0002).

The saber is one rigid body ("saber") welded to `right_wrist_yaw_link`, the end link of
the 29-DoF G1's right arm. No joint: the fixed mount has no compliance worth modelling.

Saber frame (origin at the centre of the grip):
    +x  along the blade, towards the tip
    +z  the back (false) edge, towards the thumb of the rubber hand
    -z  the cutting edge
Along +x: grip [-grip/2, +grip/2], guard plate, then the blade up to the tip.

Only numpy and mujoco are imported here, so the fencing venv (no torch) can build the
model for evaluation, and the mjlab venv can reuse the same function for training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from fencing.paths import G1_MJCF, REPO_ROOT

PARENT_BODY = "right_wrist_yaw_link"
SABER_BODY = "saber"
TIP_SITE = "blade_tip"
GUARD_SITE = "guard"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MJCF_OUT = ASSETS_DIR / "g1_29dof_saber.xml"
DEFAULT_URDF_SNIPPET_OUT = ASSETS_DIR / "saber_link_snippet.urdf"

# Total-mass range used for domain randomization during training (kg). Covers bare foam
# trainers (~0.12 kg) up to a padded trainer plus a heavier printed mount (~0.35 kg).
MASS_RANDOMIZATION_KG = (0.12, 0.35)


@dataclass(frozen=True)
class SaberSpec:
    """Dimensions (m), masses (kg) and mount pose of the training saber.

    Defaults are the nominal weapon chosen in ADR 0002: a foam trainer with a 0.75 m
    blade (85 % of the FIE 0.88 m maximum, scaled towards a 1.32 m robot), 0.88 m overall
    and 0.20 kg including the printed grip socket.
    """

    blade_length: float = 0.75
    blade_radius: float = 0.012  # foam sleeve over a fibreglass/plastic core
    blade_mass: float = 0.10
    guard_thickness: float = 0.010
    guard_width: float = 0.070  # along y (across the hand)
    guard_height: float = 0.100  # along z (knuckle bow)
    guard_mass: float = 0.030
    grip_length: float = 0.120
    grip_radius: float = 0.015
    grip_mass: float = 0.070  # handle plus the printed socket that replaces the fingers
    # Grip centre in the right_wrist_yaw_link frame: inside the rubber hand's fist, which
    # spans x = 0.04..0.17 m there (measured from the hand mesh).
    mount_pos: tuple[float, float, float] = (0.10, 0.0, 0.0)
    # Blade raised towards the thumb (+z of the hand) relative to the forearm axis, like a
    # fencer's handshake grip. 30 deg won the mount sweep in docs/saber.md (15 deg, the
    # human-like guess, reached fewer parries). Positive yaw swings the tip towards the
    # palm (+y, inwards).
    mount_pitch_deg: float = 30.0
    mount_yaw_deg: float = 0.0
    # Off by default for Goal 1: see ADR 0002 ("Collision").
    collision: bool = False
    rgba_blade: tuple[float, float, float, float] = (0.95, 0.85, 0.15, 1.0)
    rgba_guard: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)
    rgba_grip: tuple[float, float, float, float] = (0.45, 0.25, 0.10, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "blade_length", "blade_radius", "blade_mass", "guard_thickness", "guard_width",
            "guard_height", "guard_mass", "grip_length", "grip_radius", "grip_mass",
        ):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"SaberSpec.{name} must be a positive finite number, got {value!r}")
        if not -90.0 < self.mount_pitch_deg < 90.0:
            raise ValueError(f"mount_pitch_deg must be in (-90, 90), got {self.mount_pitch_deg}")
        if not -90.0 < self.mount_yaw_deg < 90.0:
            raise ValueError(f"mount_yaw_deg must be in (-90, 90), got {self.mount_yaw_deg}")
        if len(self.mount_pos) != 3:
            raise ValueError(f"mount_pos must have 3 components, got {self.mount_pos!r}")

    # Positions along the saber x axis.
    @property
    def guard_center_x(self) -> float:
        return self.grip_length / 2 + self.guard_thickness / 2

    @property
    def blade_base_x(self) -> float:
        """The guard site: where the blade leaves the guard (the forte)."""
        return self.grip_length / 2 + self.guard_thickness

    @property
    def tip_x(self) -> float:
        return self.blade_base_x + self.blade_length

    @property
    def blade_center_x(self) -> float:
        return self.blade_base_x + self.blade_length / 2

    @property
    def total_length(self) -> float:
        return self.grip_length + self.guard_thickness + self.blade_length

    @property
    def total_mass(self) -> float:
        return self.blade_mass + self.guard_mass + self.grip_mass

    def with_total_mass(self, total_mass: float) -> SaberSpec:
        """Same weapon with every part scaled to a new total mass (uniform density change)."""
        if not (math.isfinite(total_mass) and total_mass > 0):
            raise ValueError(f"total_mass must be a positive finite number, got {total_mass!r}")
        k = total_mass / self.total_mass
        return replace(
            self, blade_mass=self.blade_mass * k, guard_mass=self.guard_mass * k,
            grip_mass=self.grip_mass * k,
        )


NOMINAL_SABER = SaberSpec()


# ---------------------------------------------------------------------------
# Mass properties from geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassProperties:
    mass: float
    com: np.ndarray = field(repr=False)  # (3,) in the saber frame
    inertia: np.ndarray = field(repr=False)  # (3, 3) about the COM, saber-frame axes


def _cylinder_x_inertia(mass: float, radius: float, length: float) -> np.ndarray:
    """Solid cylinder whose axis is x, about its own centre."""
    axial = mass * radius**2 / 2
    transverse = mass * (3 * radius**2 + length**2) / 12
    return np.diag([axial, transverse, transverse])


def _box_inertia(mass: float, size_x: float, size_y: float, size_z: float) -> np.ndarray:
    """Solid box with full side lengths size_*, about its own centre."""
    return mass / 12 * np.diag(
        [size_y**2 + size_z**2, size_x**2 + size_z**2, size_x**2 + size_y**2]
    )


def _shift_inertia(inertia: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    """Parallel-axis theorem: inertia about a point `offset` away from the part's COM."""
    return inertia + mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))


def saber_parts(saber: SaberSpec) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    """(name, mass, centre in saber frame, inertia about own centre) for grip, guard, blade."""
    return [
        ("grip", saber.grip_mass, np.zeros(3),
         _cylinder_x_inertia(saber.grip_mass, saber.grip_radius, saber.grip_length)),
        ("guard", saber.guard_mass, np.array([saber.guard_center_x, 0.0, 0.0]),
         _box_inertia(saber.guard_mass, saber.guard_thickness, saber.guard_width,
                      saber.guard_height)),
        ("blade", saber.blade_mass, np.array([saber.blade_center_x, 0.0, 0.0]),
         _cylinder_x_inertia(saber.blade_mass, saber.blade_radius, saber.blade_length)),
    ]


def mass_properties(saber: SaberSpec = NOMINAL_SABER) -> MassProperties:
    """Composite mass, COM and inertia of the saber, from its three solid parts."""
    parts = saber_parts(saber)
    total = sum(m for _, m, _, _ in parts)
    com = sum(m * c for _, m, c, _ in parts) / total
    inertia = sum(_shift_inertia(i, m, c - com) for _, m, c, i in parts)
    return MassProperties(mass=total, com=com, inertia=inertia)


# ---------------------------------------------------------------------------
# Mount pose
# ---------------------------------------------------------------------------


def _quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    return q


def mount_quat(saber: SaberSpec) -> np.ndarray:
    """wxyz rotation from the wrist-link frame to the saber frame.

    Yaw about the hand z axis, then pitch: rotating about -y lifts +x towards +z.
    """
    yaw = _quat_axis_angle([0, 0, 1], math.radians(saber.mount_yaw_deg))
    pitch = _quat_axis_angle([0, -1, 0], math.radians(saber.mount_pitch_deg))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, yaw, pitch)
    return q


def blade_direction_in_hand(saber: SaberSpec) -> np.ndarray:
    """Unit blade direction expressed in the right_wrist_yaw_link frame."""
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.array([1.0, 0.0, 0.0]), mount_quat(saber))
    return out


def tip_in_hand(saber: SaberSpec) -> np.ndarray:
    """Blade-tip position in the right_wrist_yaw_link frame (what tests check against)."""
    return np.asarray(saber.mount_pos, float) + saber.tip_x * blade_direction_in_hand(saber)


# ---------------------------------------------------------------------------
# MjSpec construction
# ---------------------------------------------------------------------------

_X_TO_CYLINDER_AXIS = _quat_axis_angle([0, 1, 0], math.pi / 2)  # geom z axis -> saber x


def _add_visual_geom(body: mujoco.MjsBody, name: str, **kwargs) -> mujoco.MjsGeom:
    # Every attribute set explicitly: geoms added through MjSpec do not reliably inherit the
    # G1 "visual" class, and a default contype=1 geom would silently start colliding.
    geom = body.add_geom(name=name, **kwargs)
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2
    geom.mass = 0.0  # inertia comes from the body's explicit <inertial>, not from geoms
    return geom


def attach_saber(spec: mujoco.MjSpec, saber: SaberSpec = NOMINAL_SABER,
                 parent: str = PARENT_BODY) -> mujoco.MjSpec:
    """Add the saber body (visual geoms, sites, explicit inertia) under `parent`. Mutates spec.

    `right_wrist_yaw_link` is the last body of the G1 kinematic tree, so the saber becomes
    the last body and every existing body keeps its index: NPZ motions built for the plain
    robot stay valid (see docs/saber.md, "Motion NPZ compatibility").
    """
    parent_body = spec.body(parent)
    if parent_body is None:
        raise ValueError(f"parent body {parent!r} not found in spec")
    if spec.body(SABER_BODY) is not None:
        raise ValueError(f"spec already has a body named {SABER_BODY!r}; attach_saber called twice?")

    props = mass_properties(saber)
    body = parent_body.add_body(name=SABER_BODY, pos=list(saber.mount_pos),
                                quat=list(mount_quat(saber)))
    body.explicitinertial = True
    body.mass = props.mass
    body.ipos = props.com
    # Parts are symmetric about the saber x axis, so the inertia is already diagonal.
    off_diagonal = props.inertia - np.diag(np.diag(props.inertia))
    if np.abs(off_diagonal).max() > 1e-12:
        raise AssertionError(f"saber inertia not diagonal in its frame: {props.inertia}")
    body.inertia = np.diag(props.inertia)

    half_grip = saber.grip_length / 2
    _add_visual_geom(body, "saber_grip", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[saber.grip_radius, half_grip, 0], pos=[0, 0, 0],
                     quat=list(_X_TO_CYLINDER_AXIS), rgba=list(saber.rgba_grip))
    _add_visual_geom(body, "saber_guard", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[saber.guard_thickness / 2, saber.guard_width / 2,
                           saber.guard_height / 2],
                     pos=[saber.guard_center_x, 0, 0], rgba=list(saber.rgba_guard))
    _add_visual_geom(body, "saber_blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[saber.blade_radius, saber.blade_length / 2, 0],
                     pos=[saber.blade_center_x, 0, 0], quat=list(_X_TO_CYLINDER_AXIS),
                     rgba=list(saber.rgba_blade))

    if saber.collision:
        # Name ends in "_collision" on purpose: mjlab's G1 CollisionCfg regex then gives it
        # the same contype/conaffinity/condim as the robot's other collision capsules.
        geom = body.add_geom(name="saber_blade_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                             size=[saber.blade_radius, saber.blade_length / 2, 0],
                             pos=[saber.blade_center_x, 0, 0], quat=list(_X_TO_CYLINDER_AXIS),
                             rgba=[0.2, 0.6, 0.2, 0.3])
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 3
        geom.mass = 0.0

    for name, x, rgba in ((GUARD_SITE, saber.blade_base_x, [0, 0, 1, 1]),
                          (TIP_SITE, saber.tip_x, [1, 0, 0, 1])):
        site = body.add_site(name=name, pos=[x, 0, 0], size=[0.012, 0, 0], rgba=rgba)
        site.group = 3
    return spec


def g1_spec(mjcf_path: Path = G1_MJCF) -> mujoco.MjSpec:
    if not Path(mjcf_path).exists():
        raise FileNotFoundError(f"G1 MJCF not found at {mjcf_path} (is third_party/mjlab checked out?)")
    return mujoco.MjSpec.from_file(str(mjcf_path))


def g1_saber_spec(saber: SaberSpec = NOMINAL_SABER) -> mujoco.MjSpec:
    """The upstream G1 MJCF with the saber attached. Fresh spec on every call."""
    return attach_saber(g1_spec(), saber)


def g1_saber_model(saber: SaberSpec = NOMINAL_SABER) -> mujoco.MjModel:
    return g1_saber_spec(saber).compile()


def write_mjcf(out_path: Path = DEFAULT_MJCF_OUT, saber: SaberSpec = NOMINAL_SABER) -> Path:
    """Write a standalone MJCF of the G1 + saber that plain MuJoCo can load.

    Mesh paths are rewritten relative to the output file so the file works on every
    machine that has the repo (with third_party/mjlab) at any location.
    """
    out_path = Path(out_path).resolve()
    spec = g1_saber_spec(saber)
    spec.compile()  # fail here, with MuJoCo's message, rather than in the written file
    xml = spec.to_xml()
    mesh_dir = (Path(G1_MJCF).parent / "assets").resolve()
    rel = Path(_relpath(mesh_dir, out_path.parent)).as_posix()
    xml, n = re.subn(r'meshdir="[^"]*"', f'meshdir="{rel}"', xml)
    if n != 1:
        raise RuntimeError(f"expected exactly one meshdir attribute in generated XML, found {n}")
    header = (
        "<!-- Generated by `fencing saber build-mjcf` from the upstream mjlab G1 MJCF plus "
        "fencing.saber.spec. Do not edit by hand; regenerate. -->\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + xml)
    return out_path


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


def urdf_snippet(saber: SaberSpec = NOMINAL_SABER) -> str:
    """Fixed-joint link for the Unitree G1 URDF (Isaac Sim / Isaac Lab, future work).

    Kept as a snippet: Isaac Sim does not run on this WSL2 machine, so it is untested there.
    """
    props = mass_properties(saber)
    q = mount_quat(saber)
    # URDF wants fixed-axis roll-pitch-yaw; convert from the wxyz quaternion.
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    ixx, iyy, izz = np.diag(props.inertia)
    mx, my, mz = saber.mount_pos
    blade_len = saber.blade_length
    return f"""<!-- Saber for the G1 29-DoF URDF: paste inside <robot>. Generated by fencing.saber.spec.
     Isaac Lab's URDF importer merges fixed joints by default: set merge_fixed_joints=False
     (or keep the blade_tip link) or the blade_tip frame disappears. -->
<joint name="saber_mount_joint" type="fixed">
  <parent link="{PARENT_BODY}"/>
  <child link="{SABER_BODY}"/>
  <origin xyz="{mx:.4f} {my:.4f} {mz:.4f}" rpy="{roll:.6f} {pitch:.6f} {yaw:.6f}"/>
</joint>
<link name="{SABER_BODY}">
  <inertial>
    <origin xyz="{props.com[0]:.5f} 0 0" rpy="0 0 0"/>
    <mass value="{props.mass:.4f}"/>
    <inertia ixx="{ixx:.6e}" ixy="0" ixz="0" iyy="{iyy:.6e}" iyz="0" izz="{izz:.6e}"/>
  </inertial>
  <visual>
    <origin xyz="{saber.blade_center_x:.4f} 0 0" rpy="0 1.570796 0"/>
    <geometry><cylinder radius="{saber.blade_radius}" length="{blade_len}"/></geometry>
  </visual>
  <visual>
    <origin xyz="{saber.guard_center_x:.4f} 0 0" rpy="0 0 0"/>
    <geometry><box size="{saber.guard_thickness} {saber.guard_width} {saber.guard_height}"/></geometry>
  </visual>
  <visual>
    <origin xyz="0 0 0" rpy="0 1.570796 0"/>
    <geometry><cylinder radius="{saber.grip_radius}" length="{saber.grip_length}"/></geometry>
  </visual>
</link>
<joint name="blade_tip_joint" type="fixed">
  <parent link="{SABER_BODY}"/>
  <child link="{TIP_SITE}"/>
  <origin xyz="{saber.tip_x:.4f} 0 0" rpy="0 0 0"/>
</joint>
<link name="{TIP_SITE}"/>
<joint name="guard_joint" type="fixed">
  <parent link="{SABER_BODY}"/>
  <child link="{GUARD_SITE}"/>
  <origin xyz="{saber.blade_base_x:.4f} 0 0" rpy="0 0 0"/>
</joint>
<link name="{GUARD_SITE}"/>
"""


def write_urdf_snippet(out_path: Path = DEFAULT_URDF_SNIPPET_OUT,
                       saber: SaberSpec = NOMINAL_SABER) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(urdf_snippet(saber))
    return out_path


__all__ = [
    "ASSETS_DIR", "DEFAULT_MJCF_OUT", "GUARD_SITE", "MASS_RANDOMIZATION_KG", "MassProperties",
    "NOMINAL_SABER", "PARENT_BODY", "REPO_ROOT", "SABER_BODY", "SaberSpec", "TIP_SITE",
    "attach_saber", "blade_direction_in_hand", "g1_saber_model", "g1_saber_spec", "g1_spec",
    "mass_properties", "mount_quat", "saber_parts", "tip_in_hand", "urdf_snippet", "write_mjcf",
    "write_urdf_snippet",
]
