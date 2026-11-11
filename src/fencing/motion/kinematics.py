"""Forward kinematics of the G1 with MuJoCo, batched over frames.

The model is the exact MJCF mjlab trains on (paths.G1_MJCF). CSV joint columns are mapped
to qpos addresses by joint NAME. Foot geometry comes from the model's seven sole capsules
per foot (`<side>_foot<k>_collision`), so "lowest foot point" means the same thing here
as in the physics the policy is trained in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from fencing import paths
from fencing.motion.io import JOINT_NAMES, Motion

FOOT_SIDES = ("left", "right")  # index 0 = left, 1 = right everywhere in this package
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
HAND_SITES = ("left_palm", "right_palm")
LEG_JOINTS = {
    side: tuple(f"{side}_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
    for side in FOOT_SIDES
}
N_SOLE_CAPSULES = 7


@dataclass(frozen=True)
class FootGeometry:
    """Sole points of one foot in its ankle_roll_link frame."""

    capsule_endpoints: np.ndarray  # (2 * N_SOLE_CAPSULES, 3) capsule segment endpoints
    radius: float  # capsule radius; sole surface is this far below the segments
    sole_center: np.ndarray  # (3,) mean of endpoints, lowered by radius
    toe: np.ndarray  # (3,) most forward endpoint, lowered by radius
    heel: np.ndarray  # (3,) most rearward endpoint, lowered by radius


class G1Model:
    """MuJoCo model plus the index maps the cleanup code needs."""

    def __init__(self, mjcf_path: Path):
        if not mjcf_path.exists():
            raise FileNotFoundError(
                f"G1 MJCF not found at {mjcf_path}; run scripts/setup_third_party.sh to fetch mjlab"
            )
        self.mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1 or self.model.jnt_qposadr[free[0]] != 0:
            raise ValueError(f"{mjcf_path}: expected one free joint at qpos[0:7], found {len(free)}")
        self.joint_ids = np.array([self._joint_id(name) for name in JOINT_NAMES])
        self.qpos_index = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_index = self.model.jnt_dofadr[self.joint_ids].astype(int)
        self.joint_lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.joint_axes = self.model.jnt_axis[self.joint_ids].copy()
        self.body_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)
        )
        self.foot_body_ids = np.array([self.body_id(name) for name in FOOT_BODIES])
        self.pelvis_body_id = self.body_id("pelvis")
        self.hand_site_ids = np.array([self._site_id(name) for name in HAND_SITES])
        self.feet = tuple(self._foot_geometry(side) for side in FOOT_SIDES)
        self.leg_columns = {side: np.array([JOINT_NAMES.index(n) for n in LEG_JOINTS[side]]) for side in FOOT_SIDES}

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} (CSV column) not found in {self.mjcf_path}")
        return jid

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site {name!r} not found in {self.mjcf_path}")
        return sid

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body {name!r} not found in {self.mjcf_path}")
        return bid

    def _foot_geometry(self, side: str) -> FootGeometry:
        m = self.model
        endpoints, radii = [], []
        for k in range(1, N_SOLE_CAPSULES + 1):
            name = f"{side}_foot{k}_collision"
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"sole capsule {name!r} not found in {self.mjcf_path}")
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError(f"{name} is geom type {m.geom_type[gid]}, expected a capsule")
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[gid])
            half = rot.reshape(3, 3)[:, 2] * m.geom_size[gid, 1]
            endpoints += [m.geom_pos[gid] + half, m.geom_pos[gid] - half]
            radii.append(m.geom_size[gid, 0])
        pts = np.array(endpoints)
        radius = float(np.max(radii))
        down = np.array([0.0, 0.0, radius])
        return FootGeometry(
            capsule_endpoints=pts,
            radius=radius,
            sole_center=pts.mean(axis=0) - down,
            toe=pts[np.argmax(pts[:, 0])] - down,
            heel=pts[np.argmin(pts[:, 0])] - down,
        )

    def qpos_from_motion(self, motion: Motion) -> np.ndarray:
        """(T, nq) qpos: root pos, root quat converted xyzw -> MuJoCo wxyz, joints by name."""
        qpos = np.zeros((motion.n_frames, self.model.nq))
        qpos[:, :3] = motion.root_pos
        qpos[:, 3] = motion.root_quat[:, 3]
        qpos[:, 4:7] = motion.root_quat[:, :3]
        qpos[:, self.qpos_index] = motion.joint_pos
        return qpos


@lru_cache(maxsize=4)
def load_g1(mjcf_path: str | None = None) -> G1Model:
    return G1Model(Path(mjcf_path) if mjcf_path else paths.G1_MJCF)


@dataclass
class FKResult:
    """World-frame kinematics for every frame. Foot arrays are (T, 2, ...) left, right."""

    body_pos: np.ndarray  # (T, nbody, 3)
    body_rot: np.ndarray  # (T, nbody, 3, 3)
    hand_pos: np.ndarray  # (T, 2, 3) palm sites
    foot_lowest_z: np.ndarray  # (T, 2) lowest sole point (exact for capsules)
    sole_center: np.ndarray  # (T, 2, 3)
    toe: np.ndarray  # (T, 2, 3)
    heel: np.ndarray  # (T, 2, 3)
    pelvis_pos: np.ndarray  # (T, 3)
    body_names: tuple[str, ...]

    def body(self, name: str) -> np.ndarray:
        return self.body_pos[:, self.body_names.index(name)]

    @property
    def foot_rot(self) -> np.ndarray:
        """(T, 2, 3, 3) ankle_roll_link orientation of each foot."""
        idx = [self.body_names.index(b) for b in FOOT_BODIES]
        return self.body_rot[:, idx]

    @property
    def foot_yaw(self) -> np.ndarray:
        """(T, 2) heading of each foot's toe direction in the ground plane (rad)."""
        forward = self.foot_rot[..., :, 0]
        return np.arctan2(forward[..., 1], forward[..., 0])


def forward_kinematics(motion: Motion, g1: G1Model | None = None) -> FKResult:
    g1 = g1 or load_g1()
    m = g1.model
    data = mujoco.MjData(m)
    qpos = g1.qpos_from_motion(motion)
    n = motion.n_frames
    body_pos = np.empty((n, m.nbody, 3))
    body_rot = np.empty((n, m.nbody, 3, 3))
    hand_pos = np.empty((n, 2, 3))
    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, data)
        body_pos[t] = data.xpos
        body_rot[t] = data.xmat.reshape(-1, 3, 3)
        hand_pos[t] = data.site_xpos[g1.hand_site_ids]
    return _assemble_fk(g1, body_pos, body_rot, hand_pos)


def _assemble_fk(g1: G1Model, body_pos, body_rot, hand_pos) -> FKResult:
    n = body_pos.shape[0]
    lowest = np.empty((n, 2))
    centers, toes, heels = (np.empty((n, 2, 3)) for _ in range(3))
    for f, (bid, foot) in enumerate(zip(g1.foot_body_ids, g1.feet)):
        rot, pos = body_rot[:, bid], body_pos[:, bid]
        endpoints_world = np.einsum("tij,kj->tki", rot, foot.capsule_endpoints) + pos[:, None, :]
        lowest[:, f] = endpoints_world[..., 2].min(axis=1) - foot.radius
        centers[:, f] = np.einsum("tij,j->ti", rot, foot.sole_center) + pos
        toes[:, f] = np.einsum("tij,j->ti", rot, foot.toe) + pos
        heels[:, f] = np.einsum("tij,j->ti", rot, foot.heel) + pos
    return FKResult(
        body_pos=body_pos,
        body_rot=body_rot,
        hand_pos=hand_pos,
        foot_lowest_z=lowest,
        sole_center=centers,
        toe=toes,
        heel=heels,
        pelvis_pos=body_pos[:, g1.pelvis_body_id].copy(),
        body_names=g1.body_names,
    )


def foot_point_world(g1: G1Model, data: mujoco.MjData, foot_index: int) -> np.ndarray:
    """Sole centre of one foot in the world, for a `data` that already ran kinematics."""
    bid = g1.foot_body_ids[foot_index]
    return data.xmat[bid].reshape(3, 3) @ g1.feet[foot_index].sole_center + data.xpos[bid]
