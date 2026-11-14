"""Bending saber blade: a chain of rigid segments joined by passive spring hinges.

Opt-in. spec.attach_saber builds it when SaberSpec.flex_segments > 0 (the default, 0, keeps the
rigid blade), and flex_eval.py swaps it into exported policy scenes. Training refuses it for now
(mjlab_task.saber_g1_robot_cfg). The keyframer and the blade-path tools refuse a model with extra
joints, so they keep planning with the straight blade.

Model (pseudo-rigid-body chain):
  * grip + guard stay welded to right_wrist_yaw_link, exactly where spec.attach_saber puts them;
  * the blade (spec.blade_length, spec.blade_mass, spec.blade_radius) is split into N equal
    segments; each segment hangs from the previous one by two passive hinges with no actuator, so
    neither the policy nor the motors drive them (no torsion):
      - the *_y hinges bend the blade in the saber's x-z plane, towards an edge ("edgewise": the
        stiff direction of a flat blade; spec.py puts the cutting edge on -z);
      - the *_z hinges bend it in the x-y plane, towards a flat side ("flatwise": the floppy one);
  * all hinges of one direction share one stiffness, chosen so the clamped blade's first bending
    frequency in that direction equals its target (linear eigen-analysis, exact for the chain);
    damping is stiffness-proportional, so each direction's first mode has damping ratio zeta;
  * total mass, COM and inertia of the straight saber equal spec.mass_properties (checked).

Physics limit: MuJoCo integrates joint springs explicitly (Euler and implicitfast alike), so a
stiff chain of light segments needs a small timestep. `stable_timestep` picks one for evaluation
from the chain's highest mode; `check_stable` simulates the chain to test a given timestep.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.linalg

from fencing import paths
from fencing.saber.spec import NOMINAL_SABER, SaberSpec, mass_properties, saber_parts
from fencing.saber.spec import TIP_SITE as SPEC_TIP_SITE

MJLAB_G1_ASSETS = paths.THIRD_PARTY / "mjlab" / "src" / "mjlab" / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "assets"
SABER_BODY = "robot/saber"
TIP_SITE = "robot/blade_tip"
CYLINDER_X_QUAT = "0.707107 0 0.707107 0"  # geom z axis -> saber x, as spec._X_TO_CYLINDER_AXIS


@dataclass(frozen=True)
class FlexBlade:
    segments: int
    # Target first bending frequency (Hz) towards the flat side, blade clamped at the guard, no
    # gravity. Also used towards the edge when f1_edge_hz is None.
    f1_hz: float
    zeta: float = 0.05  # damping ratio of the first mode in each direction
    f1_edge_hz: float | None = None  # target first bending frequency towards the edge

    def __post_init__(self) -> None:
        if self.segments < 1:
            raise ValueError(f"segments must be >= 1, got {self.segments}")
        if self.f1_hz <= 0 or not 0 <= self.zeta < 1:
            raise ValueError(f"need f1_hz > 0 and 0 <= zeta < 1, got {self.f1_hz}, {self.zeta}")
        if self.f1_edge_hz is not None and self.f1_edge_hz <= 0:
            raise ValueError(f"f1_edge_hz must be > 0 or None, got {self.f1_edge_hz}")

    @property
    def edge_hz(self) -> float:
        return self.f1_hz if self.f1_edge_hz is None else self.f1_edge_hz

    @classmethod
    def from_saber(cls, saber: SaberSpec) -> FlexBlade:
        """The bending blade a SaberSpec asks for (its flex_* fields)."""
        if saber.flex_segments < 1:
            raise ValueError("this SaberSpec has a rigid blade (flex_segments=0)")
        return cls(saber.flex_segments, saber.flex_f1_flatwise_hz, saber.flex_damping_ratio,
                   saber.flex_f1_edgewise_hz or None)


@dataclass(frozen=True)
class ChainModes:
    stiffness: float  # N m / rad, per flatwise (*_z) hinge
    damping: float  # N m s / rad, per flatwise hinge
    frequencies_hz: np.ndarray  # flatwise bending modes of the clamped chain, ascending
    static_tip_droop_m: float  # under gravity, blade horizontal with a flat side down
    stiffness_edge: float  # N m / rad, per edgewise (*_y) hinge
    damping_edge: float  # N m s / rad, per edgewise hinge
    frequencies_edge_hz: np.ndarray  # edgewise bending modes, ascending


def _segment(saber: SaberSpec, n: int) -> tuple[float, float, np.ndarray]:
    length, mass = saber.blade_length / n, saber.blade_mass / n
    r = saber.blade_radius
    inertia = np.array([mass * r**2 / 2, mass * (3 * r**2 + length**2) / 12, mass * (3 * r**2 + length**2) / 12])
    return length, mass, inertia


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.atleast_1d(v))


def _hinges(modes: ChainModes):
    """(axis name, axis, stiffness, damping) of the two hinges at every blade joint."""
    return (("y", (0, 1, 0), modes.stiffness_edge, modes.damping_edge),  # towards the edge
            ("z", (0, 0, 1), modes.stiffness, modes.damping))  # towards the flat side


def _chain_xml(saber: SaberSpec, n: int, modes: ChainModes, name_prefix: str, rgba: str) -> str:
    """Nested <body> chain for the blade, rooted at the guard face (saber.blade_base_x)."""
    length, mass, inertia = _segment(saber, n)
    xml = f'<site name="{name_prefix}blade_tip" pos="{length:.9g} 0 0" size="0.012" group="3"/>'
    for i in reversed(range(n)):
        pos = f"{saber.blade_base_x:.9g} 0 0" if i == 0 else f"{length:.9g} 0 0"
        joints = "".join(
            f'<joint name="{name_prefix}saber_flex_{i}_{axis_name}" type="hinge" axis="{_fmt(axis)}" pos="0 0 0" '
            f'stiffness="{k:.9g}" damping="{c:.9g}" springref="0" armature="0" '
            f'frictionloss="0" limited="false"/>'
            for axis_name, axis, k, c in _hinges(modes))
        xml = (f'<body name="{name_prefix}saber_blade_{i}" pos="{pos}">{joints}'
               f'<inertial pos="{length / 2:.9g} 0 0" mass="{mass:.9g}" diaginertia="{_fmt(inertia)}"/>'
               f'<geom name="{name_prefix}saber_blade_{i}_geom" type="cylinder" size="{saber.blade_radius:.9g} '
               f'{length / 2:.9g}" pos="{length / 2:.9g} 0 0" quat="{CYLINDER_X_QUAT}" contype="0" '
               f'conaffinity="0" group="2" mass="0" rgba="{rgba}"/>{xml}</body>')
    return xml


def handle_mass_properties(saber: SaberSpec = NOMINAL_SABER) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, COM (saber frame) and principal inertia of grip + guard (the part that stays welded)."""
    parts = [p for p in saber_parts(saber) if p[0] != "blade"]
    mass = sum(m for _, m, _, _ in parts)
    com = sum(m * c for _, m, c, _ in parts) / mass
    inertia = sum(i + m * ((c - com) @ (c - com) * np.eye(3) - np.outer(c - com, c - com)) for _, m, c, i in parts)
    return mass, com, np.diag(inertia)


def _clamped_chain(saber: SaberSpec, n: int, modes: ChainModes, gravity: str = "0 0 0",
                   dt: float = 0.0001) -> mujoco.MjModel:
    """The blade alone, its first segment hinged to a fixed base at the guard."""
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option gravity="{gravity}" timestep="{dt:.9g}"/><worldbody><body name="base">'
        f'{_chain_xml(saber, n, modes, "", "1 1 0 1")}</body></worldbody></mujoco>')


def chain_modes(blade: FlexBlade, saber: SaberSpec = NOMINAL_SABER) -> ChainModes:
    """Hinge stiffness/damping that give the target frequencies, plus all mode frequencies and droop."""
    n = blade.segments
    # Unit stiffness everywhere. The segments are round, so the y and z chains are identical and
    # decoupled when straight: one plane's eigenvalues serve both directions.
    unit = ChainModes(1.0, 0.0, np.zeros(1), 0.0, 1.0, 0.0, np.zeros(1))
    model = _clamped_chain(saber, n, unit)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)  # MuJoCo 3.11 signature: (model, data, dst)
    plane = np.arange(0, model.nv, 2)  # the *_y hinges
    mass_matrix = full[np.ix_(plane, plane)]
    omega_sq_unit = np.sort(scipy.linalg.eigh(np.eye(len(plane)), mass_matrix, eigvals_only=True))

    def direction(f1: float) -> tuple[float, float, np.ndarray]:
        k = (2 * math.pi * f1) ** 2 / omega_sq_unit[0]
        c = 2 * blade.zeta * k / (2 * math.pi * f1)  # stiffness-proportional: zeta on mode 1
        return k, c, np.sqrt(omega_sq_unit * k) / (2 * math.pi)

    k, c, frequencies = direction(blade.f1_hz)
    k_edge, c_edge, frequencies_edge = direction(blade.edge_hz)

    # Static droop: settle under gravity with heavy damping, blade horizontal, flat side down.
    dt = min(1e-4, 0.3 / (2 * math.pi * float(max(frequencies.max(), frequencies_edge.max()))))
    heavy = _clamped_chain(saber, n, ChainModes(k, 4 * k / (2 * math.pi * blade.f1_hz), frequencies, 0.0, k_edge,
                                                4 * k_edge / (2 * math.pi * blade.edge_hz), frequencies_edge),
                           gravity="0 -9.81 0", dt=dt)
    hd = mujoco.MjData(heavy)
    for _ in range(int(3.0 / heavy.opt.timestep)):
        mujoco.mj_step(heavy, hd)
    droop = -float(hd.site("blade_tip").xpos[1])
    return ChainModes(k, c, frequencies, droop, k_edge, c_edge, frequencies_edge)


def check_stable(blade: FlexBlade, timestep: float, saber: SaberSpec = NOMINAL_SABER,
                 integrator: int = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST), seconds: float = 2.0,
                 margin: float = 1.05) -> None:
    """Raise ValueError if the blade blows up at `timestep`.

    Simulates the blade alone (guard held fixed), released from a small bend in both directions,
    and watches its energy: a stable chain's energy only falls (damping), an unstable one's grows
    every step. MuJoCo would otherwise reset an unstable simulation without saying so.

    The test runs at timestep * margin, because on the robot the blade is slightly less stable
    than alone. Checked at 5 ms with the cut_head_v4, parry_prime_v3 and cut_chest_v3 policies:
    2 pieces ran at 12.5 Hz and blew up at 12.9 Hz (this test alone: 12.9 Hz); 3 pieces ran at
    8 Hz and blew up at 9 Hz (alone: 8.0 Hz); 5 pieces ran at 6 Hz and blew up at 7 Hz (alone:
    6.6 Hz). With margin 1.05 no blade passes here that blew up there."""
    model = _clamped_chain(saber, blade.segments, chain_modes(blade, saber), dt=timestep * margin)
    model.opt.integrator = int(integrator)
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.05
    mujoco.mj_forward(model, data)
    start = float(data.energy.sum())
    for _ in range(max(1, round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        energy = float(data.energy.sum())
        if (data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or not math.isfinite(energy)
                or energy > 1.5 * start):
            raise ValueError(
                f"a {blade.segments}-piece blade at {blade.f1_hz:g} Hz flatwise / {blade.edge_hz:g} Hz "
                f"edgewise blows up at a {timestep * 1000:g} ms timestep: use fewer pieces or a "
                f"smaller timestep")


def stable_timestep(modes: ChainModes, step_dt: float = 0.02, safety: float = 0.5,
                    candidates=(0.005, 0.0025, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002,
                                0.00001)) -> float:
    """Largest candidate timestep (dividing step_dt) with omega_max * dt <= safety.

    Semi-implicit Euler is stable for an undamped spring while omega * dt < 2; 0.5 leaves margin
    for the coupling to the moving wrist (a free 10-segment chain at 5 ms went unstable well below 2).
    This is an accuracy margin for evaluation; check_stable tests plain stability."""
    omega_max = 2 * math.pi * float(max(modes.frequencies_hz.max(), modes.frequencies_edge_hz.max()))
    for dt in candidates:
        if omega_max * dt <= safety and abs(round(step_dt / dt) * dt - step_dt) < 1e-12:
            return dt
    raise ValueError(f"no candidate timestep is stable for omega_max = {omega_max:.0f} rad/s")


def scene_with_meshes(scene_xml: str) -> str:
    """Exported scenes reference meshdir="assets", which the repo does not commit; point it at
    mjlab's copy of the same G1 meshes (all 35 files present at the pinned v1.6.0)."""
    return scene_xml.replace('meshdir="assets"', f'meshdir="{MJLAB_G1_ASSETS}"')


def flexible_scene(scene_xml: str, blade: FlexBlade, saber: SaberSpec = NOMINAL_SABER) -> tuple[str, ChainModes]:
    """An exported scene.xml with its welded saber blade replaced by the hinge chain."""
    match = re.search(r'(<body name="robot/saber"[^>]*>)(.*?)(</body>)', scene_xml, flags=re.DOTALL)
    if match is None:
        raise ValueError("no <body name=\"robot/saber\"> in the scene")
    head, inner, tail = match.groups()
    if "<body" in inner:
        raise ValueError("the saber body already has child bodies; is it already flexible?")
    keep = [line for line in re.findall(r"<[^>]+/>", inner)
            if not re.search(r'name="robot/(saber_blade|blade_tip)"|<inertial', line)]
    if not any('name="robot/saber_grip"' in k for k in keep) or not any('name="robot/guard"' in k for k in keep):
        raise ValueError("unexpected saber body: grip geom or guard site missing")
    blade_rgba = re.search(r'name="robot/saber_blade"[^>]*rgba="([^"]+)"', inner)
    mass, com, inertia = handle_mass_properties(saber)
    modes = chain_modes(blade, saber)
    new_inner = (f'<inertial pos="{_fmt(com)}" mass="{mass:.9g}" diaginertia="{_fmt(inertia)}"/>'
                 + "".join(keep)
                 + _chain_xml(saber, blade.segments, modes, "robot/",
                              blade_rgba.group(1) if blade_rgba else "0.95 0.85 0.15 1"))
    return scene_xml[:match.start()] + head + new_inner + tail + scene_xml[match.end():], modes


def _set_spring(joint: mujoco.MjsJoint, name: str, value: float) -> None:
    # MuJoCo 3.11 stores joint stiffness/damping as polynomial coefficients; the first is linear.
    current = getattr(joint, name)
    if np.ndim(current):
        setattr(joint, name, np.r_[value, np.zeros(np.size(current) - 1)])
    else:
        setattr(joint, name, value)


def add_bending_blade(body: mujoco.MjsBody, saber: SaberSpec) -> ChainModes:
    """Hang the blade chain from `body` (the welded grip + guard). Used by spec.attach_saber.

    Same chain as flexible_scene, built through MjSpec: bodies saber_blade_{i}, hinges
    saber_flex_{i}_y / _z, and the blade_tip site on the last segment."""
    blade = FlexBlade.from_saber(saber)
    modes = chain_modes(blade, saber)
    length, mass, inertia = _segment(saber, blade.segments)
    half = math.sqrt(0.5)
    parent = body
    for i in range(blade.segments):
        child = parent.add_body(name=f"saber_blade_{i}", pos=[saber.blade_base_x if i == 0 else length, 0, 0])
        child.explicitinertial = True
        child.mass = mass
        child.ipos = [length / 2, 0, 0]
        child.inertia = inertia
        for axis_name, axis, k, c in _hinges(modes):
            joint = child.add_joint(name=f"saber_flex_{i}_{axis_name}", type=mujoco.mjtJoint.mjJNT_HINGE,
                                    axis=list(axis), pos=[0, 0, 0])
            _set_spring(joint, "stiffness", k)
            _set_spring(joint, "damping", c)
            joint.springref = 0.0
            joint.armature = 0.0
            joint.frictionloss = 0.0
            joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE
        geom = child.add_geom(name=f"saber_blade_{i}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                              size=[saber.blade_radius, length / 2, 0], pos=[length / 2, 0, 0],
                              quat=[half, 0, half, 0], rgba=list(saber.rgba_blade))
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.mass = 0.0
        parent = child
    site = parent.add_site(name=SPEC_TIP_SITE, pos=[length, 0, 0], size=[0.012, 0, 0], rgba=[1, 0, 0, 1])
    site.group = 3
    return modes


def rigid_scene_for(scene_xml: str, saber: SaberSpec) -> str:
    """The exported scene with its rigid saber resized to `saber` (mass, inertia, blade, tip site).

    For a measured saber the rigid baseline must carry the same mass and blade length as the
    bending one, so the comparison isolates the bending. With NOMINAL_SABER the scene keeps its
    values (only the number formatting may change)."""
    match = re.search(r'(<body name="robot/saber"[^>]*>)(.*?)(</body>)', scene_xml, flags=re.DOTALL)
    if match is None or "<body" in match.group(2):
        raise ValueError("expected a rigid robot/saber body without child bodies")
    props = mass_properties(saber)
    inertial = (f'<inertial pos="{_fmt(props.com)}" mass="{props.mass:.9g}" '
                f'diaginertia="{_fmt(np.diag(props.inertia))}"/>')
    inner = re.sub(r"<inertial[^>]*/>", inertial, match.group(2))
    blade_size = f'size="{saber.blade_radius:.9g} {saber.blade_length / 2:.9g}"'
    blade_pos = f'pos="{saber.blade_center_x:.9g} 0 0"'
    inner = re.sub(r'(<geom name="robot/saber_blade"[^>]*?)size="[^"]*"([^>]*?)pos="[^"]*"',
                   lambda m: f"{m.group(1)}{blade_size}{m.group(2)}{blade_pos}", inner)
    tip_pos = f'pos="{saber.tip_x:.9g} 0 0"'
    inner = re.sub(r'(<site name="robot/blade_tip"[^>]*?)pos="[^"]*"', lambda m: f"{m.group(1)}{tip_pos}", inner)
    return scene_xml[:match.start()] + match.group(1) + inner + match.group(3) + scene_xml[match.end():]


def saber_mass_check(model: mujoco.MjModel, saber: SaberSpec = NOMINAL_SABER) -> float:
    """Total mass of the saber subtree (handle + blade segments); must equal spec.total_mass."""
    root = model.body(SABER_BODY).id
    ids = [b for b in range(model.nbody) if b == root or _is_descendant(model, b, root)]
    return float(model.body_mass[ids].sum())


def _is_descendant(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        body = int(model.body_parentid[body])
        if body == root:
            return True
    return False


def flex_joint_ids(model: mujoco.MjModel) -> list[int]:
    return [j for j in range(model.njnt) if "saber_flex_" in model.joint(j).name]
