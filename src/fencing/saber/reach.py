"""Can the G1's right arm put the blade where saber parries and cuts need it? (Phase 1.3)

Each canonical position is a target for the blade: a guard (hand) position relative to the
right shoulder, scaled by the robot's reach, plus a blade direction. From those we get the
blade-tip position + blade direction relative to the pelvis, and solve damped-least-squares
IK on the right arm's 7 joints with the pelvis fixed in a knees-bent stance. A second pass
also frees the 3 waist joints, because fencers turn the torso.

The target poses are estimates written from fencing references and coaching descriptions,
not measured from video; docs/saber.md says so next to the table.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import mujoco
import numpy as np

from fencing.saber.spec import (
    GUARD_SITE, NOMINAL_SABER, PARENT_BODY, SABER_BODY, TIP_SITE, SaberSpec, g1_saber_model,
    g1_spec,
)

ARM_JOINTS = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
# The 5 Nm wrist motors whose excursion the bladework summary reports: mjlab's default
# action scale for them is small (see fencing.saber.mjlab_task.SABER_WRIST_ACTION_SCALE).
WRIST_PITCH_YAW_JOINTS = ("right_wrist_pitch_joint", "right_wrist_yaw_joint")

# Copied from mjlab's g1_constants.py (importing it would pull torch into the fencing venv).
EFFORT_LIMIT_NM = {
    "right_shoulder_pitch_joint": 25.0, "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0, "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0, "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}
SOFT_LIMIT_FACTOR = 0.9  # mjlab G1_ARTICULATION.soft_joint_pos_limit_factor

# Stance used for every solve: mjlab's KNEES_BENT_KEYFRAME (pelvis 0.76 m, facing +x).
PELVIS_POS = np.array([0.0, 0.0, 0.76])
STANCE_JOINTS = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
}

POSITION_TOLERANCE_M = 0.02
ANGLE_TOLERANCE_DEG = 5.0
LIMIT_MARGIN_RAD = 0.01
COLLISION_DEPTH_M = 0.005  # ignore grazing contacts shallower than this


@dataclass(frozen=True)
class BladeTarget:
    """A canonical saber position, from a right-handed fencer facing +x (y left, z up).

    guard_offset: guard position minus right-shoulder position, in units of the arm+grip
    reach (1.0 = fully extended). blade_direction: need not be normalised.
    """

    name: str
    kind: str  # guard | parry | cut | thrust
    guard_offset: tuple[float, float, float]
    blade_direction: tuple[float, float, float]
    description: str


CANONICAL_TARGETS: tuple[BladeTarget, ...] = (
    BladeTarget("en_garde", "guard", (0.55, -0.10, -0.45), (0.80, 0.10, 0.59),
                "Hand at waist height in front of the right hip, blade forward and ~35 deg up at the opponent's face (saber en garde in tierce/sixte)."),
    BladeTarget("parry_prime", "parry", (0.45, 0.45, 0.00), (0.25, 0.20, -0.95),
                "Hand at shoulder height in front of the left shoulder, pronated; blade hangs down and slightly forward, covering the inside low line."),
    BladeTarget("parry_seconde", "parry", (0.65, -0.10, -0.25), (0.50, -0.20, -0.85),
                "Hand at chest height, outside; blade points down-forward to the outside, covering the flank."),
    BladeTarget("parry_tierce", "parry", (0.50, -0.25, -0.45), (0.20, -0.20, 0.96),
                "Hand at waist height outside the right hip; blade near vertical, tip slightly forward and out."),
    BladeTarget("parry_quarte", "parry", (0.50, 0.35, -0.45), (0.20, 0.15, 0.97),
                "Hand at waist height across the body in front of the left hip; blade near vertical, covering the chest."),
    BladeTarget("parry_quinte", "parry", (0.35, -0.15, 0.55), (0.15, 1.00, 0.15),
                "Head parry: hand above and right of the head; blade horizontal across the top of the head, tip to the left."),
    BladeTarget("parry_sixte", "parry", (0.60, -0.20, -0.30), (0.50, -0.10, 0.86),
                "Hand at chest height outside, supinated; blade up and forward, tip at face height."),
    BladeTarget("cut_head", "cut", (0.90, 0.05, 0.15), (0.90, 0.05, -0.40),
                "Arm extended at shoulder height; blade forward and angled down onto the top of the opponent's mask."),
    BladeTarget("cut_flank", "cut", (0.85, 0.15, -0.20), (0.80, 0.50, -0.30),
                "Arm extended at chest height; blade sweeps to the left and slightly down onto the opponent's sword-arm side."),
    BladeTarget("cut_chest", "cut", (0.85, -0.05, -0.05), (0.85, -0.45, -0.25),
                "Arm extended; blade angled right and slightly down across the opponent's chest (inside line)."),
    BladeTarget("cut_cheek", "cut", (0.88, -0.05, 0.12), (0.85, -0.50, 0.05),
                "Arm extended at shoulder height; blade horizontal, angled right onto the side of the mask."),
    BladeTarget("point_thrust", "thrust", (0.95, 0.05, 0.00), (1.00, 0.02, -0.12),
                "Arm fully extended at shoulder height; blade continues the arm line, tip slightly down at the chest."),
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


@dataclass
class ArmModel:
    """Compiled saber G1 plus the indices the solver needs."""

    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    tip_site: int
    guard_site: int
    collision_model: mujoco.MjModel | None = None
    right_arm_bodies: frozenset[int] = field(default_factory=frozenset)


def build_arm_model(saber: SaberSpec = NOMINAL_SABER,
                    joint_names: tuple[str, ...] = ARM_JOINTS) -> ArmModel:
    model = g1_saber_model(saber)
    # A twin with the blade's collision capsule enabled, used only to flag poses where the
    # arm or blade would pass through the robot's own body.
    collision_model = g1_saber_model(_with_collision(saber))
    joint_ids = [model.joint(name).id for name in joint_names]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids):
        raise ValueError(f"all IK joints must be hinges: {joint_names}")
    right_arm = frozenset(
        b for b in range(model.nbody)
        if model.body(b).name.startswith("right_shoulder") or model.body(b).name in (
            "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
            PARENT_BODY, SABER_BODY)
    )
    return ArmModel(
        model=model, joint_names=tuple(joint_names),
        qpos_adr=np.array([model.jnt_qposadr[j] for j in joint_ids]),
        dof_adr=np.array([model.jnt_dofadr[j] for j in joint_ids]),
        lower=np.array([model.jnt_range[j][0] for j in joint_ids]),
        upper=np.array([model.jnt_range[j][1] for j in joint_ids]),
        tip_site=model.site(TIP_SITE).id, guard_site=model.site(GUARD_SITE).id,
        collision_model=collision_model, right_arm_bodies=right_arm,
    )


def _with_collision(saber: SaberSpec) -> SaberSpec:
    return replace(saber, collision=True)


def stance_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = model.qpos0.copy()
    qpos[0:3] = PELVIS_POS
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in STANCE_JOINTS.items():
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    return qpos


def right_shoulder_position(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return data.xanchor[model.joint("right_shoulder_pitch_joint").id].copy()


def arm_reach(arm: ArmModel, qpos: np.ndarray, samples: int = 20000, seed: int = 0) -> float:
    """Largest shoulder-to-guard distance over random arm configurations (metres)."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(arm.model)
    shoulder = right_shoulder_position(arm.model, qpos)
    best = 0.0
    for _ in range(samples):
        data.qpos[:] = qpos
        data.qpos[arm.qpos_adr] = rng.uniform(arm.lower, arm.upper)
        mujoco.mj_kinematics(arm.model, data)
        best = max(best, float(np.linalg.norm(data.site_xpos[arm.guard_site] - shoulder)))
    return best


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------


@dataclass
class IkResult:
    q: np.ndarray
    position_error_m: float
    angle_error_deg: float
    tip: np.ndarray
    direction: np.ndarray


def _blade_state(arm: ArmModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    tip = data.site_xpos[arm.tip_site].copy()
    direction = data.site_xmat[arm.tip_site].reshape(3, 3)[:, 0].copy()  # saber +x
    return tip, direction


def solve_ik(arm: ArmModel, base_qpos: np.ndarray, target_tip: np.ndarray,
             target_direction: np.ndarray, q_init: np.ndarray, iterations: int = 300,
             damping: float = 0.03, direction_weight: float = 0.5,
             max_step_rad: float = 0.2) -> IkResult:
    """Damped least squares on [tip position; blade direction], joints clamped to range.

    Roll about the blade axis is left free (edge orientation is not part of the target).
    direction_weight converts direction error to metres: 0.5 means 1 rad ~ 0.5 m.
    """
    model = arm.model
    data = mujoco.MjData(model)
    target_direction = target_direction / np.linalg.norm(target_direction)
    q = np.clip(q_init, arm.lower, arm.upper)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterations):
        data.qpos[:] = base_qpos
        data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        tip, direction = _blade_state(arm, data)
        error = np.concatenate([target_tip - tip,
                                direction_weight * (target_direction - direction)])
        if np.linalg.norm(error) < 1e-5:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.tip_site)
        # d(direction)/dt = omega x direction = -[direction]x omega
        jac = np.vstack([jacp[:, arm.dof_adr],
                         direction_weight * (-_skew(direction) @ jacr[:, arm.dof_adr])])
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
        largest = np.abs(step).max()
        if largest > max_step_rad:
            step *= max_step_rad / largest
        q = np.clip(q + step, arm.lower, arm.upper)

    data.qpos[:] = base_qpos
    data.qpos[arm.qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    tip, direction = _blade_state(arm, data)
    angle = math.degrees(math.acos(float(np.clip(direction @ target_direction, -1.0, 1.0))))
    return IkResult(q=q, position_error_m=float(np.linalg.norm(target_tip - tip)),
                    angle_error_deg=angle, tip=tip, direction=direction)


def self_collisions(arm: ArmModel, qpos: np.ndarray) -> list[str]:
    """Body pairs (involving the right arm or saber) that interpenetrate at this pose."""
    model = arm.collision_model
    if model is None:
        return []
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    pairs = set()
    depths = data.contact.dist[: data.ncon]
    geoms = data.contact.geom[: data.ncon]
    for depth, (geom1, geom2) in zip(depths, geoms):
        if depth > -COLLISION_DEPTH_M:
            continue
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 in arm.right_arm_bodies or body2 in arm.right_arm_bodies:
            pairs.add(" / ".join(sorted((model.body(body1).name, model.body(body2).name))))
    return sorted(pairs)


@dataclass
class ReachResult:
    target: str
    kind: str
    joint_set: str
    reachable: bool
    position_error_m: float
    angle_error_deg: float
    target_tip_rel_pelvis: list[float]
    target_direction: list[float]
    achieved_tip_rel_pelvis: list[float]
    joint_positions: dict[str, float]
    joints_at_limit: list[str]
    joints_beyond_soft_limit: list[str]
    self_collisions: list[str]


def _limit_report(arm: ArmModel, q: np.ndarray) -> tuple[list[str], list[str]]:
    at_limit, beyond_soft = [], []
    middle = (arm.lower + arm.upper) / 2
    half_range = (arm.upper - arm.lower) / 2
    for i, name in enumerate(arm.joint_names):
        if q[i] - arm.lower[i] < LIMIT_MARGIN_RAD or arm.upper[i] - q[i] < LIMIT_MARGIN_RAD:
            at_limit.append(name)
        if abs(q[i] - middle[i]) > SOFT_LIMIT_FACTOR * half_range[i]:
            beyond_soft.append(name)
    return at_limit, beyond_soft


def target_tip_and_direction(target: BladeTarget, shoulder: np.ndarray, reach: float,
                             saber: SaberSpec) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(target.blade_direction, float)
    direction /= np.linalg.norm(direction)
    guard = shoulder + reach * np.asarray(target.guard_offset, float)
    return guard + saber.blade_length * direction, direction


def solve_target(arm: ArmModel, target: BladeTarget, base_qpos: np.ndarray,
                 shoulder: np.ndarray, reach: float, saber: SaberSpec, restarts: int = 32,
                 seed: int = 0, joint_set: str = "arm") -> ReachResult:
    """Best of several random restarts; prefers solutions within tolerance and collision-free."""
    tip_target, direction_target = target_tip_and_direction(target, shoulder, reach, saber)
    rng = np.random.default_rng(seed)
    initial_guesses = [np.clip(np.zeros(len(arm.joint_names)), arm.lower, arm.upper)]
    initial_guesses += [rng.uniform(arm.lower, arm.upper) for _ in range(restarts - 1)]

    best_key, best = None, None
    for q0 in initial_guesses:
        result = solve_ik(arm, base_qpos, tip_target, direction_target, q0)
        within = (result.position_error_m <= POSITION_TOLERANCE_M
                  and result.angle_error_deg <= ANGLE_TOLERANCE_DEG)
        qpos = base_qpos.copy()
        qpos[arm.qpos_adr] = result.q
        collisions = self_collisions(arm, qpos) if within else []
        key = (not within, len(collisions) > 0,
               result.position_error_m + 0.01 * result.angle_error_deg)
        if best_key is None or key < best_key:
            best_key, best = key, (result, collisions, within)

    result, collisions, within = best
    if not within:  # report what the closest pose would hit too
        qpos = base_qpos.copy()
        qpos[arm.qpos_adr] = result.q
        collisions = self_collisions(arm, qpos)
    at_limit, beyond_soft = _limit_report(arm, result.q)
    pelvis = base_qpos[0:3]
    return ReachResult(
        target=target.name, kind=target.kind, joint_set=joint_set,
        reachable=bool(within and not collisions),
        position_error_m=round(result.position_error_m, 4),
        angle_error_deg=round(result.angle_error_deg, 2),
        target_tip_rel_pelvis=np.round(tip_target - pelvis, 3).tolist(),
        target_direction=np.round(direction_target, 3).tolist(),
        achieved_tip_rel_pelvis=np.round(result.tip - pelvis, 3).tolist(),
        joint_positions={n: round(float(v), 4) for n, v in zip(arm.joint_names, result.q)},
        joints_at_limit=at_limit, joints_beyond_soft_limit=beyond_soft,
        self_collisions=collisions,
    )


# ---------------------------------------------------------------------------
# Payload: what the saber costs the arm and the balance
# ---------------------------------------------------------------------------


def payload_report(arm_q: dict[str, float], saber: SaberSpec = NOMINAL_SABER,
                   heavy_mass_kg: float = 0.35,
                   wrist_angular_accel: float = 100.0) -> dict:
    """Static joint torques, COM shift and a wrist-cut torque estimate at one arm pose.

    wrist_angular_accel (rad/s^2) is a guess for a fast wrist-driven cut (0 -> 10 rad/s in
    0.1 s); the result scales linearly, so other guesses are easy to read off.
    """
    plain = g1_spec().compile()
    variants = {"no_saber": plain, "nominal": g1_saber_model(saber),
                "heavy": g1_saber_model(saber.with_total_mass(heavy_mass_kg))}
    report: dict = {"pose": arm_q, "heavy_mass_kg": heavy_mass_kg,
                    "wrist_angular_accel_rad_s2": wrist_angular_accel, "static_torque_nm": {},
                    "com_shift_mm": {}, "effort_limit_nm": EFFORT_LIMIT_NM}
    com_plain = None
    for label, model in variants.items():
        data = mujoco.MjData(model)
        data.qpos[:] = stance_qpos(model)
        for name, value in arm_q.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        report["static_torque_nm"][label] = {
            name: round(float(data.qfrc_inverse[model.jnt_dofadr[model.joint(name).id]]), 3)
            for name in ARM_JOINTS
        }
        com = data.subtree_com[1].copy()  # pelvis subtree = whole robot
        if com_plain is None:
            com_plain = com
        report["com_shift_mm"][label] = np.round((com - com_plain) * 1000, 1).tolist()
        if label != "no_saber":
            report.setdefault("wrist_cut_torque_nm", {})[label] = _wrist_cut_torque(
                model, data, wrist_angular_accel)
    return report


def _wrist_cut_torque(model: mujoco.MjModel, data: mujoco.MjData,
                      angular_accel: float) -> dict[str, float]:
    """I_axis * alpha for the hand + saber about the wrist pitch and yaw axes (gravity excluded)."""
    out = {}
    for joint_name in ("right_wrist_pitch_joint", "right_wrist_yaw_joint"):
        joint = model.joint(joint_name).id
        axis = data.xaxis[joint]
        anchor = data.xanchor[joint]
        inertia_about_axis = 0.0
        for body_name in (PARENT_BODY, SABER_BODY):
            body = model.body(body_name).id
            rotation = data.ximat[body].reshape(3, 3)
            inertia_world = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
            offset = data.xipos[body] - anchor
            perpendicular = offset - (offset @ axis) * axis
            inertia_about_axis += float(axis @ inertia_world @ axis
                                        + model.body_mass[body] * perpendicular @ perpendicular)
        out[joint_name] = round(inertia_about_axis * angular_accel, 3)
    return out


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------


MOUNT_SWEEP = tuple((pitch, mount_x) for mount_x in (0.10, 0.06)
                    for pitch in (0.0, 15.0, 30.0, 45.0, 60.0))


def run_reach_analysis(saber: SaberSpec = NOMINAL_SABER, restarts: int = 32,
                       include_waist: bool = True,
                       mount_sweep: tuple[tuple[float, float], ...] = MOUNT_SWEEP) -> dict:
    """Solve every canonical target; returns a JSON-ready dict.

    mount_sweep: (mount pitch deg, grip-centre x in the wrist link) pairs to compare. The
    targets stay those of the nominal saber so only the mount changes between rows.
    """
    arm = build_arm_model(saber, ARM_JOINTS)
    base = stance_qpos(arm.model)
    shoulder = right_shoulder_position(arm.model, base)
    reach = arm_reach(arm, base)

    results = [asdict(solve_target(arm, t, base, shoulder, reach, saber, restarts, seed=i))
               for i, t in enumerate(CANONICAL_TARGETS)]
    if include_waist:
        arm_waist = build_arm_model(saber, ARM_JOINTS + WAIST_JOINTS)
        results += [asdict(solve_target(arm_waist, t, base, shoulder, reach, saber, restarts,
                                        seed=i, joint_set="arm+waist"))
                    for i, t in enumerate(CANONICAL_TARGETS)]

    sweep = {}
    for pitch, mount_x in mount_sweep:
        variant = replace(saber, mount_pitch_deg=pitch, mount_pos=(mount_x, 0.0, 0.0))
        variant_arm = build_arm_model(variant, ARM_JOINTS)
        rows = [solve_target(variant_arm, t, base, shoulder, reach, variant, restarts, seed=i)
                for i, t in enumerate(CANONICAL_TARGETS)]
        sweep[f"pitch={pitch:g}deg,mount_x={mount_x:g}m"] = {
            "mount_pitch_deg": pitch, "mount_x_m": mount_x,
            "num_reachable": sum(r.reachable for r in rows),
            "reachable": [r.target for r in rows if r.reachable],
            "unreachable": {r.target: [r.position_error_m, r.angle_error_deg, r.self_collisions]
                            for r in rows if not r.reachable},
        }

    en_garde = next(r for r in results if r["target"] == "en_garde" and r["joint_set"] == "arm")
    joint_ranges = {name: [round(float(lo), 4), round(float(hi), 4)]
                    for name, lo, hi in zip(arm.joint_names, arm.lower, arm.upper)}
    return {
        "saber": {k: v for k, v in asdict(saber).items() if not k.startswith("rgba")},
        "stance": {"pelvis_pos": PELVIS_POS.tolist(), "joints": STANCE_JOINTS},
        "right_shoulder_rel_pelvis": np.round(shoulder - PELVIS_POS, 4).tolist(),
        "arm_reach_to_guard_m": round(reach, 4),
        "tolerances": {"position_m": POSITION_TOLERANCE_M, "angle_deg": ANGLE_TOLERANCE_DEG,
                       "collision_depth_m": COLLISION_DEPTH_M},
        "arm_joint_ranges_rad": joint_ranges,
        "targets": [asdict(t) for t in CANONICAL_TARGETS],
        "results": results,
        "mount_pitch_sweep": sweep,
        "payload": payload_report(en_garde["joint_positions"], saber),
        "notes": (
            "Target poses are estimates (fencing references, scaled to the G1's reach), not "
            "measurements. Pelvis fixed at mjlab's KNEES_BENT stance facing +x; legs and "
            "left arm static. 'reachable' = within tolerance and no self-collision deeper "
            "than collision_depth_m involving the right arm or blade."
        ),
    }


def markdown_table(report: dict) -> str:
    """The table pasted into docs/saber.md."""
    lines = [
        "| Target | Joints | Reachable | Tip err (cm) | Dir err (deg) | Joints at limit | Beyond 90 % soft limit | Self-collision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        short = lambda names: ", ".join(n.replace("right_", "").replace("_joint", "") for n in names) or "-"
        lines.append(
            f"| {r['target']} | {r['joint_set']} | {'yes' if r['reachable'] else '**no**'} | "
            f"{100 * r['position_error_m']:.1f} | {r['angle_error_deg']:.1f} | "
            f"{short(r['joints_at_limit'])} | {short(r['joints_beyond_soft_limit'])} | "
            f"{'; '.join(r['self_collisions']) or '-'} |"
        )
    return "\n".join(lines)
