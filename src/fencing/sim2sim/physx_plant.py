"""The exported mjlab robot as a PhysX articulation (ovphysx: CPU PhysX without Omniverse Kit).

`fencing sim2sim --engine physx` cross-checks a policy in a different physics engine. The
plant is authored as USD from the *compiled MuJoCo model* of the export's scene.xml, so
PhysX gets the bodies, masses, COMs, inertias, joint axes and limits, armature, collision
primitives, per-geom friction and excluded pairs that mjlab simulated; the PD gains come
from the ONNX contract and the effort limits from the scene's actuator forceranges.

Conventions the adapter hides:
- USD angles are degrees: joint limits, and drive stiffness/damping per degree. Tensor
  bindings read and write SI units and radians.
- PhysX quaternions are (x, y, z, w); everything returned here is (w, x, y, z).
- PhysX link linear velocity is the velocity of the centre of mass; mjlab's metrics and
  the IMU use the link frame origin (`com_to_origin_velocity`).

Needs `ovphysx`, `physx-usd-schemas`, `newton-usd-schemas` and `usd-core`, which are not
dependencies of this package (NVIDIA licence, ~1.7 GB): see docs/evaluation.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from fencing.sim2sim.quat import quat_apply, quat_mul

ROBOT_PATH = "/World/robot"
ENTITY_PREFIX = "robot/"
RADIANS_PER_DEGREE = np.pi / 180.0
UNLIMITED_FORCE = 3.4e38
INSTALL_HINT = ("the PhysX engine needs ovphysx, physx-usd-schemas, newton-usd-schemas and usd-core, which are not "
                "fencing dependencies; see docs/evaluation.md 'Sim-to-sim in PhysX' for the separate venv")


def require_ovphysx():
    """Import ovphysx (CPU only) or explain where it comes from."""
    os.environ.setdefault("OVPHYSX_DISABLE_GPU", "1")  # hard CPU-only mode, process-wide
    try:
        import ovphysx
    except ImportError as error:
        raise RuntimeError(INSTALL_HINT) from error
    return ovphysx


def strip_prefix(name: str) -> str:
    return name.removeprefix(ENTITY_PREFIX)


def quat_from_x_to(axis) -> np.ndarray:
    """(w, x, y, z) rotating +X onto `axis` (USD revolute joints turn about their frame's X)."""
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    x_axis = np.array([1.0, 0.0, 0.0])
    cos = float(np.dot(x_axis, axis))
    if cos < -1 + 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])  # half turn about z
    q = np.array([1.0 + cos, *np.cross(x_axis, axis)])
    return q / np.linalg.norm(q)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def com_to_origin_velocity(com_lin_vel, ang_vel, link_quat, com_local) -> np.ndarray:
    """World velocity of the link frame origin from PhysX's COM velocity (leading axes batch)."""
    return np.asarray(com_lin_vel) - np.cross(ang_vel, quat_apply(link_quat, com_local))


def origin_to_com_velocity(origin_lin_vel, ang_vel, link_quat, com_local) -> np.ndarray:
    return np.asarray(origin_lin_vel) + np.cross(ang_vel, quat_apply(link_quat, com_local))


@dataclass
class UsdOptions:
    # MuJoCo takes a pair's friction from the higher-priority geom (the G1 feet: 0.6 against the
    # terrain's 1.0), else the max. With "min" on every material PhysX gives the same foot-ground
    # and body-ground values; only self-contacts differ (MuJoCo: condim 1, frictionless).
    friction_combine_mode: str = "min"
    solver_position_iterations: int = 8  # Isaac Lab's G1 articulation settings
    solver_velocity_iterations: int = 4
    self_collisions: bool = True


def robot_bodies(model: mujoco.MjModel) -> list[int]:
    """The floating-base tree: its root (child of world with a free joint) first, then descendants."""
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and model.body_jntnum[b] == 1
             and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
    if len(roots) != 1:
        raise ValueError(f"expected one floating-base body in the scene, found {len(roots)}")
    return [b for b in range(model.nbody) if model.body_rootid[b] == roots[0]]


def check_ground_identity(model: mujoco.MjModel, geom: int) -> None:
    """The USD ground plane is authored at the origin facing +z, so the MuJoCo plane must be too."""
    name = model.geom(geom).name or f"geom{geom}"
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)  # the plane may sit in a static body, so check its world pose
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[geom])
    pose = np.concatenate([data.geom_xpos[geom], quat if quat[0] >= 0 else -quat])
    if not np.allclose(pose, [0, 0, 0, 1, 0, 0, 0], atol=1e-9):
        raise ValueError(f"ground plane {name!r} has world pose pos+quat {pose.tolist()}, expected identity "
                         "[0, 0, 0, 1, 0, 0, 0]")


def build_usd(model: mujoco.MjModel, joint_names, kp, kd, out: Path, options: UsdOptions | None = None) -> dict:
    """Write the robot and the ground plane as USD; returns a summary of what was authored."""
    require_ovphysx()
    from ovphysx.samples.python_samples.schema_support import physx_schema_paths
    from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    Plug.Registry().RegisterPlugins([str(p) for p in physx_schema_paths()])
    options = options or UsdOptions()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)  # qpos0: every hinge at zero, so PhysX DOF 0 == MuJoCo q 0
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    def attr(prim, name, type_name, value):
        prim.CreateAttribute(name, type_name).Set(value)

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*(gravity / np.linalg.norm(gravity))))
    scene.CreateGravityMagnitudeAttr(float(np.linalg.norm(gravity)))
    scene_prim = scene.GetPrim()
    scene_prim.ApplyAPI("PhysxSceneAPI")
    attr(scene_prim, "physxScene:solverType", Sdf.ValueTypeNames.Token, "TGS")
    attr(scene_prim, "physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt, round(1 / model.opt.timestep))
    attr(scene_prim, "physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, False)
    attr(scene_prim, "physxScene:broadphaseType", Sdf.ValueTypeNames.Token, "MBP")

    materials = {}

    def bind_friction(prim, friction: float):
        key = round(float(friction), 6)
        if key not in materials:
            material = UsdShade.Material.Define(stage, f"/World/materials/friction_{str(key).replace('.', 'p')}")
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(key)
            api.CreateDynamicFrictionAttr(key)
            api.CreateRestitutionAttr(0.0)
            material.GetPrim().ApplyAPI("PhysxMaterialAPI")
            attr(material.GetPrim(), "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token,
                 options.friction_combine_mode)
            attr(material.GetPrim(), "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, "min")
            materials[key] = material
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[key], UsdShade.Tokens.weakerThanDescendants, "physics")

    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise ValueError(f"expected one ground plane in the scene, found {len(planes)}")
    check_ground_identity(model, planes[0])
    ground = UsdGeom.Plane.Define(stage, "/World/ground")
    ground.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    bind_friction(ground.GetPrim(), float(model.geom_friction[planes[0], 0]))

    bodies = robot_bodies(model)
    UsdGeom.Xform.Define(stage, ROBOT_PATH)
    body_path = {}
    shapes = []
    for body in bodies:
        name = strip_prefix(model.body(body).name)
        body_path[body] = f"{ROBOT_PATH}/{name}"
        xform = UsdGeom.Xform.Define(stage, body_path[body])
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*data.xpos[body]))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, data.xquat[body])))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(model.body_mass[body]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*model.body_ipos[body]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*model.body_inertia[body]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(*map(float, model.body_iquat[body])))
        prim.ApplyAPI("PhysxRigidBodyAPI")
        for key in ("linearDamping", "angularDamping", "sleepThreshold", "stabilizationThreshold"):
            attr(prim, f"physxRigidBody:{key}", Sdf.ValueTypeNames.Float, 0.0)  # PhysX defaults are not zero
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            shapes.append(_author_shape(stage, model, geom, body_path[body]))
            bind_friction(stage.GetPrimAtPath(shapes[-1]), float(model.geom_friction[geom, 0]))

    root = stage.GetPrimAtPath(body_path[bodies[0]])
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.ApplyAPI("PhysxArticulationAPI")
    attr(root, "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int,
         options.solver_position_iterations)
    attr(root, "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int,
         options.solver_velocity_iterations)
    attr(root, "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, options.self_collisions)
    attr(root, "physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float, 0.0)
    attr(root, "physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float, 0.0)

    gains = {name: (float(p), float(d)) for name, p, d in zip(joint_names, kp, kd)}
    joint_actuator = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)
                      if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    UsdGeom.Scope.Define(stage, f"{ROBOT_PATH}/joints")
    revolute, welds = [], []
    for body in bodies[1:]:
        parent, child = body_path[int(model.body_parentid[body])], body_path[body]
        offset, rotation = model.body_pos[body], model.body_quat[body]  # child frame in parent at q = 0
        if model.body_jntnum[body] == 0:
            joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PATH}/joints/{strip_prefix(model.body(body).name)}")
            _connect(joint, parent, child, offset, rotation, np.zeros(3), np.array([1.0, 0, 0, 0]), Gf)
            welds.append(strip_prefix(model.body(body).name))
            continue
        index = int(model.body_jntadr[body])
        if model.body_jntnum[body] != 1 or model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
            raise NotImplementedError(f"body {model.body(body).name}: only one hinge or a weld per body")
        name = strip_prefix(model.joint(index).name)
        if name not in gains:
            raise ValueError(f"joint {name} is not in the policy's joint_names")
        frame = quat_from_x_to(model.jnt_axis[index])
        anchor = model.jnt_pos[index]
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROBOT_PATH}/joints/{name}")
        joint.CreateAxisAttr("X")
        _connect(joint, parent, child, offset + quat_apply(rotation, anchor), quat_mul(rotation, frame), anchor, frame,
                 Gf)
        if model.jnt_limited[index]:
            joint.CreateLowerLimitAttr(float(model.jnt_range[index, 0] / RADIANS_PER_DEGREE))
            joint.CreateUpperLimitAttr(float(model.jnt_range[index, 1] / RADIANS_PER_DEGREE))
        stiffness, damping = gains[name]
        actuator = joint_actuator.get(index)
        limited = actuator is not None and model.actuator_forcelimited[actuator]
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness * RADIANS_PER_DEGREE)
        drive.CreateDampingAttr(damping * RADIANS_PER_DEGREE)
        drive.CreateMaxForceAttr(float(model.actuator_forcerange[actuator, 1]) if limited else UNLIMITED_FORCE)
        drive.CreateTargetPositionAttr(0.0)
        prim = joint.GetPrim()
        prim.ApplyAPI("PhysxJointAxisAPI", "angular")
        dof = model.jnt_dofadr[index]
        attr(prim, "physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float, float(model.dof_armature[dof]))
        for key in ("staticFrictionEffort", "dynamicFrictionEffort"):
            attr(prim, f"physxJointAxis:angular:{key}", Sdf.ValueTypeNames.Float, float(model.dof_frictionloss[dof]))
        attr(prim, "physxJointAxis:angular:viscousFrictionCoefficient", Sdf.ValueTypeNames.Float,
             float(model.dof_damping[dof]) * RADIANS_PER_DEGREE)
        revolute.append(name)

    filtered = []
    for signature in model.exclude_signature:
        first, second = int(signature) >> 16, int(signature) & 0xFFFF
        if first in body_path and second in body_path:
            pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body_path[first]))
            pairs.CreateFilteredPairsRel().AddTarget(body_path[second])
            filtered.append([strip_prefix(model.body(first).name), strip_prefix(model.body(second).name)])
    stage.GetRootLayer().Save()
    return {"usd": str(out), "bodies": [strip_prefix(model.body(b).name) for b in bodies], "revolute_joints": revolute,
            "welds": welds, "collision_shapes": len(shapes), "filtered_pairs": filtered,
            "frictions": sorted(materials), "options": dict(options.__dict__)}


def _connect(joint, parent, child, pos0, rot0, pos1, rot1, Gf) -> None:
    joint.CreateBody0Rel().AddTarget(parent)
    joint.CreateBody1Rel().AddTarget(child)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    joint.CreateLocalRot0Attr(Gf.Quatf(*map(float, rot0)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    joint.CreateLocalRot1Attr(Gf.Quatf(*map(float, rot1)))


def _author_shape(stage, model: mujoco.MjModel, geom: int, body_path: str) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

    name = strip_prefix(model.geom(geom).name) or f"geom{geom}"
    path = f"{body_path}/{name}"
    kind, size = model.geom_type[geom], model.geom_size[geom]
    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        shape = (UsdGeom.Capsule if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else UsdGeom.Cylinder).Define(stage, path)
        shape.CreateAxisAttr("Z")
        shape.CreateRadiusAttr(float(size[0]))
        shape.CreateHeightAttr(float(2 * size[1]))  # MuJoCo size[1] is the half-length
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(float(size[0]))
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
    else:
        raise NotImplementedError(f"collision geom {name}: MuJoCo type {int(kind)} has no PhysX primitive here")
    shape.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*model.geom_pos[geom]))
    shape.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, model.geom_quat[geom])))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, 2 * size)))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return path


@dataclass
class Snapshot:
    """Robot state: joints in policy order, links in PhysX order, quaternions (w, x, y, z)."""

    q: np.ndarray
    qd: np.ndarray
    link_pos: np.ndarray  # (L, 3) link frame origins, world
    link_quat: np.ndarray  # (L, 4)
    link_lin_vel: np.ndarray  # (L, 3) world velocity of the link frame origin
    link_ang_vel: np.ndarray  # (L, 3) world


STATE_TENSORS = ("ARTICULATION_ROOT_POSE", "ARTICULATION_ROOT_VELOCITY", "ARTICULATION_LINK_POSE",
                 "ARTICULATION_LINK_VELOCITY", "ARTICULATION_DOF_POSITION", "ARTICULATION_DOF_VELOCITY",
                 "ARTICULATION_DOF_POSITION_TARGET", "ARTICULATION_DOF_VELOCITY_TARGET")


@dataclass
class PhysxPlant:
    """One floating-base articulation in a CPU PhysX scene, driven by joint position targets.

    Uses ovphysx tensor bindings (deprecated since ovphysx 0.6 in favour of PhysX.read/write,
    still present in 0.7) because they map straight onto caller-owned numpy arrays.
    """

    usd: Path
    joint_names: tuple[str, ...]
    _bindings: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        ovphysx = require_ovphysx()
        import ovstage
        from ovphysx.samples.python_samples.schema_support import physx_schema_paths
        from ovphysx.types import TensorType

        ovphysx.PhysX.set_cpu_mode(True)
        self._tensor_type = TensorType
        self.physx = ovphysx.PhysX()
        ovstage.population.register_usd_schemas(
            [*[str(p) for p in physx_schema_paths()], str(ovphysx.newton_schema_root())])
        self.stage = ovstage.Stage("fencing-sim2sim")
        ovstage.population.open_usd(self.stage, str(self.usd), ordinal=1, domains=ovstage.PopulationDomain.ALL)
        self.stage.advance_write_floor(ordinal=1).wait()
        self.physx.wait_task(self.physx.attach_ovstage(self.stage, read_ordinal=1))
        self._root_pattern = None
        for name in STATE_TENSORS:
            self.binding(name)
        self.link_names = [p.rsplit("/", 1)[-1] for p in self._bindings["ARTICULATION_LINK_POSE"].body_names]
        self.dof_names = [p.rsplit("/", 1)[-1] for p in self._bindings["ARTICULATION_DOF_POSITION"].dof_names]
        missing = [n for n in self.joint_names if n not in self.dof_names]
        if missing:
            raise KeyError(f"policy joints {missing} are not PhysX DOFs ({self.dof_names})")
        self.dof_index = np.array([self.dof_names.index(n) for n in self.joint_names])
        self.root_link = self.link_names.index(self._root_pattern.rsplit("/", 1)[-1])
        self.com_local = self.read("ARTICULATION_BODY_COM_POSE")[0, :, :3].astype(np.float64)
        self._targets = np.zeros(self._bindings["ARTICULATION_DOF_POSITION_TARGET"].shape, np.float32)

    def binding(self, name: str):
        if name not in self._bindings:
            if self._root_pattern is None:
                from pxr import Usd, UsdPhysics

                stage = Usd.Stage.Open(str(self.usd))
                roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
                if len(roots) != 1:
                    raise ValueError(f"{self.usd} has {len(roots)} articulation roots, expected 1")
                self._root_pattern = str(roots[0].GetPath())
            self._bindings[name] = self.physx.create_tensor_binding(
                pattern=self._root_pattern, tensor_type=getattr(self._tensor_type, name), raise_if_empty=True)
        return self._bindings[name]

    def read(self, name: str) -> np.ndarray:
        binding = self.binding(name)
        out = np.zeros(binding.shape, np.float32)
        binding.read(out)
        return out

    def write(self, name: str, value) -> None:
        binding = self.binding(name)
        binding.write(np.ascontiguousarray(np.asarray(value, np.float32).reshape(binding.shape)))

    def link_index(self, names) -> np.ndarray:
        return np.array([self.link_names.index(n) for n in names])

    def _dofs(self, values) -> np.ndarray:
        out = np.zeros(len(self.dof_names))
        out[self.dof_index] = values
        return out

    def reset(self, root_pos, root_quat, root_lin_vel_w, root_ang_vel_w, q, qd) -> None:
        """Place the robot. Root velocities are of the root link origin, world frame (mjlab's)."""
        root_quat = np.asarray(root_quat, dtype=np.float64)
        com_velocity = origin_to_com_velocity(root_lin_vel_w, root_ang_vel_w, root_quat, self.com_local[self.root_link])
        self.write("ARTICULATION_ROOT_POSE", np.concatenate([root_pos, wxyz_to_xyzw(root_quat)]))
        self.write("ARTICULATION_ROOT_VELOCITY", np.concatenate([com_velocity, root_ang_vel_w]))
        self.write("ARTICULATION_DOF_POSITION", self._dofs(q))
        self.write("ARTICULATION_DOF_VELOCITY", self._dofs(qd))
        self.write("ARTICULATION_DOF_VELOCITY_TARGET", np.zeros(len(self.dof_names)))
        self.set_targets(q)

    def set_targets(self, q_target) -> None:
        self._targets[0, self.dof_index] = q_target
        self._bindings["ARTICULATION_DOF_POSITION_TARGET"].write(self._targets)

    def step(self, dt: float) -> None:
        self.physx.step_sync(dt)

    def snapshot(self) -> Snapshot:
        pose = self.read("ARTICULATION_LINK_POSE")[0].astype(np.float64)
        velocity = self.read("ARTICULATION_LINK_VELOCITY")[0].astype(np.float64)
        quat = xyzw_to_wxyz(pose[:, 3:])
        angular = velocity[:, 3:]
        return Snapshot(
            q=self.read("ARTICULATION_DOF_POSITION")[0].astype(np.float64)[self.dof_index],
            qd=self.read("ARTICULATION_DOF_VELOCITY")[0].astype(np.float64)[self.dof_index],
            link_pos=pose[:, :3], link_quat=quat,
            link_lin_vel=com_to_origin_velocity(velocity[:, :3], angular, quat, self.com_local),
            link_ang_vel=angular,
        )

    def close(self) -> None:
        for binding in self._bindings.values():
            binding.destroy()
        self._bindings.clear()
        self.physx.wait_task(self.physx.detach_ovstage())
        self.stage.destroy()
        self.physx.destroy()


def property_report(plant: PhysxPlant, model: mujoco.MjModel, joint_names, kp, kd) -> dict:
    """Model properties read back from PhysX (SI units) against the MuJoCo model and the gains."""
    bodies = {strip_prefix(model.body(b).name): b for b in robot_bodies(model)}
    links = [bodies[n] for n in plant.link_names]
    mass = plant.read("ARTICULATION_BODY_MASS")[0]
    com = plant.read("ARTICULATION_BODY_COM_POSE")[0, :, :3]
    inertia = plant.read("ARTICULATION_BODY_INERTIA")[0].reshape(-1, 3, 3).astype(np.float64)
    inertia_diff = 0.0
    for k, body in enumerate(links):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
        rotation = rotation.reshape(3, 3)
        expected = rotation @ np.diag(model.body_inertia[body]) @ rotation.T  # link frame, about the COM
        inertia_diff = max(inertia_diff, float(np.abs(inertia[k] - expected).max()))
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    joints = [j if j >= 0 else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ENTITY_PREFIX + n)
              for j, n in zip(joints, joint_names)]
    actuator_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    force_limit = np.array([model.actuator_forcerange[actuator_of[j], 1] if j in actuator_of
                            and model.actuator_forcelimited[actuator_of[j]] else UNLIMITED_FORCE for j in joints])
    index = plant.dof_index
    stiffness = plant.read("ARTICULATION_DOF_STIFFNESS")[0][index]
    damping = plant.read("ARTICULATION_DOF_DAMPING")[0][index]
    return {
        "mass_total_kg": float(mass.sum()),
        "mass_max_abs_diff_kg": float(np.abs(mass - model.body_mass[links]).max()),
        "com_max_abs_diff_m": float(np.abs(com - model.body_ipos[links]).max()),
        "inertia_max_abs_diff": inertia_diff,
        "kp_max_rel_diff": float(np.abs(stiffness / np.asarray(kp) - 1).max()),
        "kd_max_rel_diff": float(np.abs(damping / np.asarray(kd) - 1).max()),
        "armature_max_abs_diff": float(np.abs(plant.read("ARTICULATION_DOF_ARMATURE")[0][index]
                                              - model.dof_armature[model.jnt_dofadr[joints]]).max()),
        "max_force_max_abs_diff": float(np.abs(plant.read("ARTICULATION_DOF_MAX_FORCE")[0][index]
                                               - force_limit).max()),
        "limit_max_abs_diff_rad": float(np.abs(plant.read("ARTICULATION_DOF_LIMIT")[0][index]
                                               - model.jnt_range[joints]).max()),
        "joint_velocity_limit_min": float(plant.read("ARTICULATION_DOF_MAX_VELOCITY")[0].min()),
    }
