"""Hand-edited blade keyframes on top of a G1 motion (roadmap Phase 1.4d).

Pose estimation from video cannot see wrist rotation or blade angle, so the saber's line
is keyframed: a *blade script* lists named blade targets (en garde, parries, cuts, ...)
with transition and hold times. For each key the right arm (and, only if the arm alone
cannot, waist yaw/pitch within small bounds) is solved with damped least squares so the
blade tip and blade direction land on the target. Between keys the controlled joints
follow minimum-jerk profiles, stretched where needed so no joint exceeds the speed cap.
Legs, root and the left arm always come from the base motion.

Target frame: a BladeTarget is a guard offset from the right shoulder (in units of arm
reach) plus a blade direction (fencing/saber/reach.py). Here the shoulder is the *base*
motion's right shoulder at the key's frame, and the axes are the fencing line's (world
+x by default, the published-motion convention), so targets follow the footwork.

    from fencing.saber.keyframes import BladeKey, apply_blade_script
    result = apply_blade_script(base_motion, [
        BladeKey("en_garde", move_s=0.0, hold_s=0.6),
        BladeKey("parry_quinte", move_s=0.3, hold_s=0.6),
        BladeKey("en_garde", move_s=0.45, hold_s=0.6),
    ])
    result.motion, result.keys  # Motion + per-key blade-tip errors

The action "base" hands the arm back to the base motion (blend out); a script that
starts with a key whose move_s > 0 blends in from the base arm.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import mujoco
import numpy as np

from fencing.motion.fixes import hardware_velocity_limits, rate_limit
from fencing.motion.io import JOINT_NAMES, Motion
from fencing.motion.kinematics import load_g1
from fencing.saber.reach import (
    ARM_JOINTS, CANONICAL_TARGETS, EFFORT_LIMIT_NM, SOFT_LIMIT_FACTOR, WRIST_PITCH_YAW_JOINTS, ArmModel, BladeTarget,
    _skew, arm_reach, build_arm_model, self_collisions, solve_ik, stance_qpos,
)
from fencing.saber.spec import NOMINAL_SABER, SaberSpec

BASE_ACTION = "base"
WAIST_IK_JOINTS = ("waist_yaw_joint", "waist_pitch_joint")
ARM_COLUMNS = [JOINT_NAMES.index(name) for name in ARM_JOINTS]
WRIST_PITCH_YAW_COLUMNS = [JOINT_NAMES.index(name) for name in WRIST_PITCH_YAW_JOINTS]
WAIST_COLUMNS = [JOINT_NAMES.index(name) for name in WAIST_IK_JOINTS]
CONTROLLED_COLUMNS = ARM_COLUMNS + WAIST_COLUMNS
MIN_JERK_PEAK_SPEED = 1.875  # max ds/dtau of the quintic 10t^3 - 15t^4 + 6t^5
SPEED_MARGIN = 1.05  # sampled min-jerk can peak slightly above the continuous bound
TORQUE_SAMPLES = 101  # points along a move for its inverse-dynamics torque demand
# reach.arm_reach(nominal saber, KNEES_BENT stance, 20000 samples, seed 0): the value in
# HCF_DATA/eval/saber/reach.json. Sampling it again costs seconds, so the nominal saber
# reuses it; any other saber is re-sampled the same way.
NOMINAL_ARM_REACH_M = 0.5717
EXTRA_RESTARTS_AFTER_VALID = 3
BLADE_PATHS = ("joint", "line")
# blade_path="line": samples along a move's blade line, IK iterations per sample, and the gain of the
# nullspace pull toward the joint-space interpolation (prototype on thrust/en_garde_v4: 60 and 180
# samples give the same path; gain 1.0 needed the shortest move, 0.52 s, at 1.3 cm tracking error).
LINE_SAMPLES = 60
LINE_IK_ITERATIONS = 30
LINE_NULLSPACE_GAIN = 1.0
LINE_DAMPING = 0.03
LINE_DIRECTION_WEIGHT = 0.5  # as reach.solve_ik: 1 rad of direction error ~ 0.5 m
LINE_MAX_STEP_RAD = 0.2
# A tracked line whose end is farther than this from the key's pose has reached another solution
# family; the move then stays a joint-space one (reported in the summary) rather than jump.
LINE_END_TOLERANCE_RAD = 0.1
# ... and so has a line the tip cannot follow (parry_quinte from the guard: 1.5 m off on the synthetic
# base, the blade has to go over the head).
LINE_TRACKING_TOLERANCE_M = 0.05
# ... and so does a move over which the base body alone carries the tip farther than this: the line is
# solved on the body at the reach frame, so a moving base (a lunge) would put the tip off it. The GVHMR
# guard's sway moves it 0.7-1.7 cm over thrust_v5's and cut_flank_v5's moves.
LINE_BODY_SHIFT_TOLERANCE_M = 0.03

# Targets the reach table lacks. Same conventions as reach.CANONICAL_TARGETS; estimates
# from fencing references, not measurements.
EXTRA_TARGETS: tuple[BladeTarget, ...] = (
    BladeTarget("extension", "guard", (0.88, -0.02, -0.05), (1.00, 0.00, 0.05),
                "Sword arm extended at shoulder height, blade continuing the arm, point threatening the opponent's chest/face."),
    BladeTarget("parry_tierce_high", "parry", (0.50, -0.25, -0.20), (0.15, -0.15, 0.97),
                "Tierce with the hand at chest height outside the right shoulder line; blade near vertical. "
                "Waist-height tierce is out of reach in this stance (ADR 0002)."),
    BladeTarget("parry_quarte_high", "parry", (0.50, 0.25, -0.20), (0.15, 0.15, 0.97),
                "Quarte with the hand at chest height across the body; blade near vertical, covering the chest."),
)
TARGET_ALIASES = {"en_garde_guard": "en_garde", "thrust": "point_thrust"}


class SchedulingError(ValueError):
    """The blade script's timing cannot be satisfied (e.g. a timed key with no room)."""


@dataclass(frozen=True)
class BladeKey:
    """One step of a blade script.

    action: a target name (blade_targets(), aliases allowed) or "base".
    move_s: requested transition time into this key; stretched if a joint would exceed
            the speed cap. 0 on the first key = start already in that pose.
    hold_s: minimum time the pose is held after it is reached (a later timed key may cut
            it short; the last key is held to the end of the motion).
    at_s:   if set, the key is reached exactly at this time instead of right after the
            previous hold (e.g. a cut landing at lunge touchdown).
    """

    action: str
    move_s: float = 0.4
    hold_s: float = 0.0
    at_s: float | None = None

    def __post_init__(self) -> None:
        if self.move_s < 0 or self.hold_s < 0:
            raise ValueError(f"{self.action}: move_s and hold_s must be >= 0, got {self.move_s}, {self.hold_s}")
        if self.at_s is not None and self.at_s < 0:
            raise ValueError(f"{self.action}: at_s must be >= 0, got {self.at_s}")


@dataclass(frozen=True)
class BladeConfig:
    velocity_fraction: float = 0.5  # of the motor no-load limit, as the motion cleanup
    soft_limit_factor: float = SOFT_LIMIT_FACTOR
    waist_yaw_bound_rad: float = 0.35  # waist may turn this far from the base motion's
    waist_pitch_bound_rad: float = 0.20
    allow_waist: bool = True
    position_tolerance_m: float = 0.02
    angle_tolerance_deg: float = 5.0
    restarts: int = 16
    heading: str = "x"  # "x": fencing line = world +x; "pelvis": per-frame pelvis yaw
    seed: int = 0
    saber: SaberSpec = NOMINAL_SABER
    # Whole-body still hold at the end: counted from the later of the last key's reach and the
    # base motion coming to rest, so the output ends with at least this long of stillness (a
    # base that already ends that still is not lengthened). mjlab resamples an env when its
    # motion ends, so a policy only learns to stand still for as long as the reference does,
    # and the eval holds the last frame for 2 s.
    end_hold_s: float = 0.0
    # How the arm moves between two keyed poses. "joint": minimum-jerk in joint space (every motion
    # published before thrust_v5). "line": the blade tip follows the straight line between the two
    # keys' tips, the blade direction turns at a constant rate (slerp), and the arm's redundant DoF
    # are pulled toward the joint-space interpolation; a new target is solved from the end of that
    # line, so the key lands in the solution family the move reaches. On the side-on GVHMR guard
    # (en_garde_v4) every point_thrust solution is >= 3 rad from the guard in joint space and a
    # joint-space move swept the tip to 7 cm above the floor (thrust_v4); see docs/saber.md.
    blade_path: str = "joint"

    def __post_init__(self) -> None:
        if self.end_hold_s < 0:
            raise ValueError(f"end_hold_s must be >= 0, got {self.end_hold_s}")
        if self.blade_path not in BLADE_PATHS:
            raise ValueError(f"blade_path must be one of {BLADE_PATHS}, got {self.blade_path!r}")


@dataclass
class KeyReport:
    index: int
    action: str
    target: str
    kind: str
    frame: int
    time_s: float
    move_s_requested: float
    move_s_used: float
    hold_s: float
    joint_set: str
    tip_error_m: float  # measured on the final motion at the key frame
    angle_error_deg: float
    hold_drift_m: float  # worst tip error over the hold (body moving under a fixed arm)
    collisions: list[str]
    waist_delta_rad: list[float]


@dataclass
class BladeResult:
    motion: Motion
    keys: list[KeyReport]
    summary: dict
    script: list[BladeKey] = field(default_factory=list)

    def record(self) -> dict:
        return {"script": [asdict(k) for k in self.script], "keys": [asdict(k) for k in self.keys],
                "summary": self.summary}


# ---------------------------------------------------------------- targets


def blade_targets() -> dict[str, BladeTarget]:
    return {t.name: t for t in CANONICAL_TARGETS + EXTRA_TARGETS}


def resolve_target(name: str) -> BladeTarget:
    targets = blade_targets()
    canonical = TARGET_ALIASES.get(name, name)
    if canonical not in targets:
        raise KeyError(f"unknown blade target {name!r}; known: {', '.join(sorted(targets) + sorted(TARGET_ALIASES))}")
    return targets[canonical]


# ---------------------------------------------------------------- timing (pure)


def minimum_jerk(tau: np.ndarray) -> np.ndarray:
    """Quintic 0 -> 1 with zero velocity and acceleration at both ends."""
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def minimum_jerk_derivatives(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ds/dtau and d2s/dtau2 of minimum_jerk."""
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return 30.0 * tau**2 * (1.0 - tau) ** 2, 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3


def min_move_frames(delta: np.ndarray, max_speed: np.ndarray, fps: float) -> int:
    """Fewest frames for a min-jerk move of `delta` whose peak speed stays under max_speed."""
    ratio = float(np.max(np.abs(delta) / max_speed)) if np.size(delta) else 0.0
    if ratio <= 0:
        return 0
    return int(math.ceil(MIN_JERK_PEAK_SPEED * ratio * fps * SPEED_MARGIN))


def schedule_keys(keys: list[BladeKey], fps: float, min_frames: list[int] | None = None) -> list[tuple[int, int, int]]:
    """(move start, reach, hold end) frame per key.

    Sequential keys start where the previous hold ends; timed keys (at_s) are reached at
    their time and may cut the previous hold short, but never start before the previous
    key has been reached.
    """
    if not keys:
        raise SchedulingError("blade script is empty")
    min_frames = min_frames or [0] * len(keys)
    spans: list[tuple[int, int, int]] = []
    prev_reach, prev_end = 0, 0
    for i, key in enumerate(keys):
        move = max(int(round(key.move_s * fps)), min_frames[i])
        if key.at_s is None:
            start = prev_end
            reach = start + move
        else:
            reach = int(round(key.at_s * fps))
            start = reach - move
            if start < (prev_reach if i else 0):
                raise SchedulingError(
                    f"key {i} {key.action!r} must be reached at {key.at_s:.2f} s but its {move}-frame move "
                    f"would start before the previous key is reached (frame {prev_reach}); "
                    "move it later, shorten move_s or lower the speed demand"
                )
        end = reach + int(round(key.hold_s * fps))
        spans.append((start, reach, end))
        prev_reach, prev_end = reach, end
    return spans


# ---------------------------------------------------------------- IK context


@dataclass
class _Context:
    arm: ArmModel  # 7 right-arm joints, soft limits
    arm_waist: ArmModel  # + waist yaw/pitch (bounds set per key)
    reach: float
    data: mujoco.MjData
    tip_site: int
    shoulder_joint: int


@lru_cache(maxsize=2)
def _context(saber: SaberSpec, soft_limit_factor: float) -> _Context:
    full = build_arm_model(saber, ARM_JOINTS + WAIST_IK_JOINTS)
    arm_waist = _soft_limited(full, soft_limit_factor)
    n = len(ARM_JOINTS)
    # Same compiled models, first 7 joints only (compiling again costs seconds: meshes).
    arm = replace(arm_waist, joint_names=ARM_JOINTS, qpos_adr=arm_waist.qpos_adr[:n],
                  dof_adr=arm_waist.dof_adr[:n], lower=arm_waist.lower[:n], upper=arm_waist.upper[:n])
    if arm.model.nq != load_g1().model.nq:
        raise RuntimeError(f"saber model nq={arm.model.nq} differs from the G1's; qpos layouts would not match")
    reach = NOMINAL_ARM_REACH_M if saber == NOMINAL_SABER else arm_reach(arm, stance_qpos(arm.model))
    return _Context(arm, arm_waist, reach, mujoco.MjData(arm.model), arm.tip_site,
                    arm.model.joint("right_shoulder_pitch_joint").id)


def _soft_limited(arm: ArmModel, factor: float) -> ArmModel:
    middle, half = (arm.lower + arm.upper) / 2, (arm.upper - arm.lower) / 2
    return replace(arm, lower=middle - factor * half, upper=middle + factor * half)


def _heading_rotation(qpos: np.ndarray, heading: str) -> np.ndarray:
    if heading == "x":
        return np.eye(3)
    if heading == "pelvis":
        w, x, y, z = qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"heading must be 'x' or 'pelvis', got {heading!r}")


def target_in_world(ctx: _Context, base_qpos: np.ndarray, target: BladeTarget, config: BladeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Blade tip + unit direction in the world for the base pose at one frame."""
    ctx.data.qpos[:] = base_qpos
    mujoco.mj_kinematics(ctx.arm.model, ctx.data)
    shoulder = ctx.data.xanchor[ctx.shoulder_joint].copy()
    rotation = _heading_rotation(base_qpos, config.heading)
    direction = rotation @ (np.asarray(target.blade_direction, float) / np.linalg.norm(target.blade_direction))
    guard = shoulder + rotation @ (ctx.reach * np.asarray(target.guard_offset, float))
    return guard + config.saber.blade_length * direction, direction


def _blade_tip(ctx: _Context, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ctx.data.qpos[:] = qpos
    mujoco.mj_kinematics(ctx.arm.model, ctx.data)
    tip = ctx.data.site_xpos[ctx.tip_site].copy()
    return tip, ctx.data.site_xmat[ctx.tip_site].reshape(3, 3)[:, 0].copy()


def _errors(ctx, qpos, tip_target, dir_target) -> tuple[float, float]:
    tip, direction = _blade_tip(ctx, qpos)
    angle = math.degrees(math.acos(float(np.clip(direction @ dir_target, -1.0, 1.0))))
    return float(np.linalg.norm(tip - tip_target)), angle


def move_torque_ratio(qpos: np.ndarray, arm_from: np.ndarray, arm_to: np.ndarray, move_s: float,
                      saber: SaberSpec = NOMINAL_SABER) -> dict[str, float]:
    """Peak |torque| / effort limit per right-arm joint for a minimum-jerk move of the 7 arm
    joints from arm_from to arm_to in move_s, the rest of the body pinned at qpos (inverse
    dynamics with the saber). Velocity-product and inertial torques scale with 1/move_s^2.

    The joint speed cap cannot see this: a straight-arm swing moves few joints, none fast, but
    swings the whole arm and blade as one long lever (cut_head_v3: 0.37x motor speed, 1.9x the
    shoulder pitch and 5.2x the wrist pitch effort limit)."""
    if move_s <= 0:
        raise ValueError(f"move_s must be > 0, got {move_s}")
    ctx = _context(saber, SOFT_LIMIT_FACTOR)
    model, data = ctx.arm.model, ctx.data
    dofs = ctx.arm.dof_adr
    delta = np.asarray(arm_to, float) - np.asarray(arm_from, float)
    limits = np.array([EFFORT_LIMIT_NM[name] for name in ARM_JOINTS])
    peak = np.zeros(len(ARM_JOINTS))
    tau = np.linspace(0.0, 1.0, TORQUE_SAMPLES)
    position = minimum_jerk(tau)
    speed, acceleration = minimum_jerk_derivatives(tau)
    for s, ds, dds in zip(position, speed, acceleration):
        data.qpos[:] = qpos
        data.qpos[ctx.arm.qpos_adr] = arm_from + s * delta
        data.qvel[:] = 0.0
        data.qvel[dofs] = ds / move_s * delta
        data.qacc[:] = 0.0
        data.qacc[dofs] = dds / move_s**2 * delta
        mujoco.mj_inverse(model, data)
        peak = np.maximum(peak, np.abs(data.qfrc_inverse[dofs]))
    return {name: float(value) for name, value in zip(ARM_JOINTS, peak / limits)}


# ---------------------------------------------------------------- blade_path="line"


def _slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    omega = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    if omega < 1e-9:
        return a
    return (math.sin((1 - s) * omega) * a + math.sin(s * omega) * b) / math.sin(omega)


def track_blade_line(ctx: _Context, body_qpos: np.ndarray, arm_from: np.ndarray, tip_to: np.ndarray,
                     dir_to: np.ndarray, arm_toward: np.ndarray | None = None,
                     samples: int = LINE_SAMPLES) -> tuple[np.ndarray, float]:
    """(arm path (samples + 1, 7) at s = k / samples, worst tip distance from the line).

    IK continuation with the body pinned at body_qpos: the tip on the straight line from arm_from's
    tip to tip_to, the blade direction slerped to dir_to. With arm_toward, the arm's redundant DoF
    are pulled toward the joint-space line arm_from -> arm_toward (nullspace of the blade task), so
    the arm does not wander through its 2 free DoF on the way."""
    arm, model = ctx.arm, ctx.arm.model
    data = mujoco.MjData(model)
    dofs = arm.dof_adr
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    tip_from, dir_from = _blade_tip(ctx, _with_arm(body_qpos, arm, arm_from))
    q = np.asarray(arm_from, float).copy()
    outside = (q < arm.lower - 1e-9) | (q > arm.upper + 1e-9)
    if outside.any():
        # Clipping would move the arm in one frame at the move's first sample. Keyed poses come from
        # IK inside these limits, so this only happens to an arm pose from elsewhere.
        names = [ARM_JOINTS[j] for j in np.flatnonzero(outside)]
        raise ValueError(f"blade line start pose is outside the soft joint limits on {names}: "
                         f"{q[outside].round(4).tolist()}")
    path, worst = [q.copy()], 0.0
    for k in range(1, samples + 1):
        s = k / samples
        tip_goal, dir_goal = tip_from + s * (tip_to - tip_from), _slerp(dir_from, dir_to, s)
        reference = None if arm_toward is None else arm_from + s * (arm_toward - arm_from)
        for _ in range(LINE_IK_ITERATIONS):
            data.qpos[:] = body_qpos
            data.qpos[arm.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            tip = data.site_xpos[arm.tip_site]
            direction = data.site_xmat[arm.tip_site].reshape(3, 3)[:, 0]
            error = np.concatenate([tip_goal - tip, LINE_DIRECTION_WEIGHT * (dir_goal - direction)])
            mujoco.mj_jacSite(model, data, jacp, jacr, arm.tip_site)
            jac = np.vstack([jacp[:, dofs], LINE_DIRECTION_WEIGHT * (-_skew(direction) @ jacr[:, dofs])])
            pinv = jac.T @ np.linalg.inv(jac @ jac.T + LINE_DAMPING**2 * np.eye(6))
            step = pinv @ error
            if reference is not None:
                step += (np.eye(len(q)) - pinv @ jac) @ (LINE_NULLSPACE_GAIN * (reference - q))
            largest = np.abs(step).max()
            if largest > LINE_MAX_STEP_RAD:
                step *= LINE_MAX_STEP_RAD / largest
            q = np.clip(q + step, arm.lower, arm.upper)
        tip, _ = _blade_tip(ctx, _with_arm(body_qpos, arm, q))
        worst = max(worst, float(np.linalg.norm(tip - tip_goal)))
        path.append(q.copy())
    return np.array(path), worst


def _with_arm(body_qpos: np.ndarray, arm: ArmModel, arm_q: np.ndarray) -> np.ndarray:
    qpos = body_qpos.copy()
    qpos[arm.qpos_adr[:len(ARM_JOINTS)]] = arm_q
    return qpos


def line_move(ctx: _Context, body_qpos: np.ndarray, arm_from: np.ndarray, arm_to: np.ndarray) -> dict:
    """The blade-line path between two arm poses on one body. The tracked end is corrected onto
    arm_to linearly in s (zero at the start), so the path starts and ends exactly on the keys."""
    tip_to, dir_to = _blade_tip(ctx, _with_arm(body_qpos, ctx.arm, arm_to))
    path, worst = track_blade_line(ctx, body_qpos, arm_from, tip_to, dir_to, arm_toward=arm_to)
    mismatch = np.asarray(arm_to, float) - path[-1]
    s = np.linspace(0.0, 1.0, len(path))[:, None]
    return {"path": path + s * mismatch, "tracking_max_m": worst,
            "end_mismatch_rad": float(np.abs(mismatch).max())}


def path_at(path: np.ndarray, s: float) -> np.ndarray:
    """Arm pose at s in [0, 1] on a path sampled at s = k / (len - 1)."""
    grid = np.linspace(0.0, 1.0, len(path))
    return np.array([np.interp(s, grid, path[:, j]) for j in range(path.shape[1])])


def _min_jerk_path(path: np.ndarray, samples: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tau, q, dq/dtau and d2q/dtau2 of a path run on the minimum-jerk time profile. The path's
    s-derivatives are finite differences on its own samples (differentiating the piecewise-linear
    interpolation would put acceleration spikes at every sample)."""
    tau = np.linspace(0.0, 1.0, samples)
    s = minimum_jerk(tau)
    ds, dds = minimum_jerk_derivatives(tau)
    grid = np.linspace(0.0, 1.0, len(path))
    d1 = np.gradient(path, grid, axis=0)
    d2 = np.gradient(d1, grid, axis=0)

    def at(values: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(s, grid, values[:, j]) for j in range(values.shape[1])], axis=1)

    p1 = at(d1)
    return tau, at(path), p1 * ds[:, None], at(d2) * (ds**2)[:, None] + p1 * dds[:, None]


def path_move_frames(path: np.ndarray, max_speed: np.ndarray, fps: float) -> int:
    """Fewest frames for the min-jerk run along `path` whose joint speeds stay under max_speed."""
    _, q, _, _ = _min_jerk_path(path)
    # Speeds of the pose sequence itself (what the assembled frames sample), not of the smoothed derivative.
    ratio = float(np.max(np.abs(np.diff(q, axis=0)) * (len(q) - 1) / max_speed))
    return int(math.ceil(ratio * fps * SPEED_MARGIN)) if ratio > 0 else 0


def path_torque_ratio(qpos: np.ndarray, path: np.ndarray, move_s: float,
                      saber: SaberSpec = NOMINAL_SABER) -> dict[str, float]:
    """move_torque_ratio for a move along a sampled arm path instead of a straight joint line."""
    if move_s <= 0:
        raise ValueError(f"move_s must be > 0, got {move_s}")
    ctx = _context(saber, SOFT_LIMIT_FACTOR)
    model, data, dofs = ctx.arm.model, ctx.data, ctx.arm.dof_adr
    tau, q, dq, ddq = _min_jerk_path(path, TORQUE_SAMPLES * 10 + 1)
    limits = np.array([EFFORT_LIMIT_NM[name] for name in ARM_JOINTS])
    peak = np.zeros(len(ARM_JOINTS))
    for k in range(0, len(tau), 10):
        data.qpos[:] = qpos
        data.qpos[ctx.arm.qpos_adr] = q[k]
        data.qvel[:] = 0.0
        data.qvel[dofs] = dq[k] / move_s
        data.qacc[:] = 0.0
        data.qacc[dofs] = ddq[k] / move_s**2
        mujoco.mj_inverse(model, data)
        peak = np.maximum(peak, np.abs(data.qfrc_inverse[dofs]))
    return {name: float(value) for name, value in zip(ARM_JOINTS, peak / limits)}


@dataclass
class _Solution:
    arm_q: np.ndarray  # (7,) absolute
    waist_delta: np.ndarray  # (2,) relative to the base motion's waist
    joint_set: str
    position_error_m: float
    angle_error_deg: float
    collisions: list[str]
    line_seeded: bool = False  # solved from the end of a blade line (blade_path "line")

    def ok(self, config: BladeConfig) -> bool:
        return (self.position_error_m <= config.position_tolerance_m
                and self.angle_error_deg <= config.angle_tolerance_deg and not self.collisions)


def _solve_key(ctx: _Context, base_qpos: np.ndarray, target: BladeTarget, warm_arm: np.ndarray,
               config: BladeConfig, rng: np.random.Generator) -> _Solution:
    """Arm only first; arm + waist (bounded) only when the arm alone cannot do it."""
    tip_target, dir_target = target_in_world(ctx, base_qpos, target, config)
    best = _best_of(ctx.arm, ctx, base_qpos, tip_target, dir_target, warm_arm, config, rng, "arm")
    if best.ok(config) or not config.allow_waist:
        return best
    waist_adr = ctx.arm_waist.qpos_adr[len(ARM_JOINTS):]
    waist_base = base_qpos[waist_adr]
    bounds = np.array([config.waist_yaw_bound_rad, config.waist_pitch_bound_rad])
    lower, upper = ctx.arm_waist.lower.copy(), ctx.arm_waist.upper.copy()
    lower[len(ARM_JOINTS):] = np.maximum(lower[len(ARM_JOINTS):], waist_base - bounds)
    upper[len(ARM_JOINTS):] = np.minimum(upper[len(ARM_JOINTS):], waist_base + bounds)
    if np.any(lower > upper):
        return best  # base waist already outside the soft range: leave the waist alone
    bounded = replace(ctx.arm_waist, lower=lower, upper=upper)
    warm = np.concatenate([warm_arm, waist_base])
    with_waist = _best_of(bounded, ctx, base_qpos, tip_target, dir_target, warm, config, rng, "arm+waist")
    return with_waist if _rank(with_waist, config) < _rank(best, config) else best


def _rank(solution: _Solution, config: BladeConfig) -> tuple:
    within = (solution.position_error_m <= config.position_tolerance_m
              and solution.angle_error_deg <= config.angle_tolerance_deg)
    return (not within, bool(solution.collisions),
            0.0 if within else solution.position_error_m + 0.01 * solution.angle_error_deg)


def _best_of(arm: ArmModel, ctx: _Context, base_qpos, tip_target, dir_target, warm, config, rng, joint_set) -> _Solution:
    n_arm = len(ARM_JOINTS)
    guesses = [np.clip(warm, arm.lower, arm.upper)] + [rng.uniform(arm.lower, arm.upper) for _ in range(config.restarts)]
    best_key, best, valid_seen = None, None, -1
    for k, guess in enumerate(guesses):
        if valid_seen >= 0 and k - valid_seen > EXTRA_RESTARTS_AFTER_VALID:
            break  # a few more tries for a closer valid pose, then stop: IK is the cost here
        result = solve_ik(arm, base_qpos, tip_target, dir_target, guess)
        qpos = base_qpos.copy()
        qpos[arm.qpos_adr] = result.q
        within = (result.position_error_m <= config.position_tolerance_m
                  and result.angle_error_deg <= config.angle_tolerance_deg)
        collisions = self_collisions(ctx.arm, qpos) if within else []
        if within and not collisions and valid_seen < 0:
            valid_seen = k
        # Among valid solutions prefer the one closest to the warm start: smooth, short moves.
        closeness = float(np.linalg.norm(result.q - warm))
        key = (not within, bool(collisions),
               closeness if within else result.position_error_m + 0.01 * result.angle_error_deg)
        if best_key is None or key < best_key:
            waist_delta = (result.q[n_arm:] - base_qpos[arm.qpos_adr[n_arm:]]) if len(result.q) > n_arm else np.zeros(2)
            best_key = key
            best = _Solution(result.q[:n_arm].copy(), waist_delta, joint_set, result.position_error_m,
                             result.angle_error_deg, collisions)
        if k == 0 and valid_seen == 0:
            break  # the warm start itself is valid, and closest to itself by construction
    return best


# ---------------------------------------------------------------- assembly


def _rest_frame(base: Motion, tol: float = 1e-6) -> int:
    """First frame from which root position, orientation and joints all equal the final frame."""
    moving = np.zeros(base.n_frames, dtype=bool)
    for values in (base.root_pos, base.root_quat, base.joint_pos):
        moving |= np.abs(values - values[-1]).max(axis=1) > tol
    indices = np.flatnonzero(moving)
    return int(indices[-1]) + 1 if indices.size else 0


def _pad_base(base: Motion, n_frames: int) -> Motion:
    if n_frames <= base.n_frames:
        return base.copy()
    extra = np.repeat(base.to_array()[-1:], n_frames - base.n_frames, axis=0)
    return Motion.from_array(np.concatenate([base.to_array(), extra]), base.fps, dict(base.meta))


def _key_values(i: int, keys: list[BladeKey], solutions: dict[int, _Solution], base_arm: np.ndarray, base_waist: np.ndarray):
    """Callables t -> (arm (7,), waist (2,)) for key i (-1 = base before the first key)."""
    if i < 0 or keys[i].action == BASE_ACTION:
        return lambda t: (base_arm[t], base_waist[t])
    sol = solutions[i]
    return lambda t: (sol.arm_q, base_waist[t] + sol.waist_delta)


def _assemble(base: Motion, keys, spans, solutions, paths: dict[int, np.ndarray] | None = None) -> Motion:
    out = base.copy()
    base_arm = base.joint_pos[:, ARM_COLUMNS]
    base_waist = base.joint_pos[:, WAIST_COLUMNS]
    values = [_key_values(i, keys, solutions, base_arm, base_waist) for i in range(-1, len(keys))]
    for t in range(base.n_frames):
        # Last key whose move has started at t (or -1 = still on the base motion).
        i = max((k for k, (start, _, _) in enumerate(spans) if start <= t), default=-1)
        if i < 0:
            continue
        start, reach, _ = spans[i]
        arm_new, waist_new = values[i + 1](t)
        if t < reach:
            s = float(minimum_jerk((t - start) / (reach - start)))
            arm_old, waist_old = values[i](t)
            arm_new = path_at(paths[i], s) if paths and i in paths else (1 - s) * arm_old + s * arm_new
            waist_new = (1 - s) * waist_old + s * waist_new
        out.joint_pos[t, ARM_COLUMNS] = arm_new
        out.joint_pos[t, WAIST_COLUMNS] = waist_new
    return out


def _move_delta(i: int, keys, spans, solutions, base: Motion) -> np.ndarray:
    """Joint change of the move into key i (controlled joints), for the speed stretch."""
    start, reach, _ = spans[i]
    start = min(start, base.n_frames - 1)
    reach = min(reach, base.n_frames - 1)

    def pose(k: int, t: int) -> np.ndarray:
        if k < 0 or keys[k].action == BASE_ACTION:
            return base.joint_pos[t, CONTROLLED_COLUMNS]
        sol = solutions[k]
        return np.concatenate([sol.arm_q, base.joint_pos[t, WAIST_COLUMNS] + sol.waist_delta])

    return pose(i, reach) - pose(i - 1, start)


def apply_blade_script(base: Motion, keys: list[BladeKey], config: BladeConfig = BladeConfig(),
                       tail_s: float = 0.0) -> BladeResult:
    """Keyframe the saber arm over `base`. Output length: max(base, last key's end, the later of
    the last key's reach and the base coming to rest + tail_s), so a tail_s > 0 motion ends with
    at least tail_s of whole-body stillness."""
    for key in keys:
        if key.action != BASE_ACTION:
            resolve_target(key.action)
    ctx = _context(config.saber, config.soft_limit_factor)
    g1 = load_g1()
    fps = base.fps
    speed_cap = config.velocity_fraction * hardware_velocity_limits()[CONTROLLED_COLUMNS]
    min_frames = [0] * len(keys)
    solutions: dict[int, _Solution] = {}
    tail = int(round(tail_s * fps))
    rest = _rest_frame(base) if tail else 0
    for _ in range(5):  # stretch -> re-solve at the new frames -> converge (usually 2 rounds)
        spans = schedule_keys(keys, fps, min_frames)
        n_frames = max(base.n_frames, spans[-1][2] + 1, max(spans[-1][1], rest) + 1 + tail)
        padded = _pad_base(base, n_frames)
        qpos = g1.qpos_from_motion(padded)
        solutions = _solve_all(ctx, keys, spans, qpos, padded, config, solutions)
        lines = _line_moves(ctx, keys, spans, solutions, qpos) if config.blade_path == "line" else {}
        needed = [min_move_frames(_move_delta(i, keys, spans, solutions, padded), speed_cap, fps)
                  if i or spans[0][1] > 0 else 0 for i in range(len(keys))]
        for i, line in lines.items():
            if line["used"]:
                waist = min_move_frames(_move_delta(i, keys, spans, solutions, padded)[len(ARM_JOINTS):],
                                        speed_cap[len(ARM_JOINTS):], fps)
                needed[i] = max(path_move_frames(line["path"], speed_cap[:len(ARM_JOINTS)], fps), waist)
        current = [reach - start for start, reach, _ in spans]
        if all(n <= c for n, c in zip(needed, current)):
            break
        min_frames = [max(m, n) for m, n in zip(min_frames, needed)]
    else:
        raise SchedulingError(f"speed stretch did not converge for script {[k.action for k in keys]}")

    paths = {i: line["path"] for i, line in lines.items() if line["used"]}
    motion = _assemble(padded, keys, spans, solutions, paths)
    motion, clamp = _clamp_speed(motion, speed_cap)
    motion.meta = {**base.meta, "bladework": True}
    reports = _key_reports(ctx, keys, spans, solutions, motion, padded, config)
    summary = _summary(ctx, motion, padded, reports, clamp, config, paths)
    if config.blade_path == "line":
        summary["line_moves"] = [{"key": i, "action": keys[i].action, "used": line["used"],
                                  "tracking_max_m": round(line["tracking_max_m"], 4),
                                  "end_mismatch_rad": round(line["end_mismatch_rad"], 4),
                                  "body_shift_m": round(line["body_shift_m"], 4)}
                                 for i, line in sorted(lines.items())]
    return BladeResult(motion, reports, summary, list(keys))


def _line_moves(ctx, keys, spans, solutions, qpos) -> dict[int, dict]:
    """blade_path="line": the path of every move between two keyed arm-only poses, solved on the body
    at the key's reach frame. A move whose tracked end lands in another family, or whose tip cannot
    follow the line, stays joint-space."""
    lines = {}
    for i in range(1, len(keys)):
        if BASE_ACTION in (keys[i].action, keys[i - 1].action):
            continue
        line = _line_check(ctx, qpos, spans[i], solutions[i - 1], solutions[i])
        if line is not None:
            lines[i] = line
    return lines


def _line_check(ctx, qpos, span, before: _Solution, after: _Solution) -> dict | None:
    """line_move between two keyed poses plus whether the move can use it; None where no line is
    solved (no move, or a waist turn: the line is solved for the arm alone)."""
    start, reach, _ = span
    if reach <= start or np.any(before.waist_delta != 0) or np.any(after.waist_delta != 0):
        return None
    line = line_move(ctx, qpos[min(reach, len(qpos) - 1)], before.arm_q, after.arm_q)
    line["body_shift_m"] = _body_tip_shift(ctx, qpos, start, reach, before.arm_q)
    line["used"] = (line["end_mismatch_rad"] <= LINE_END_TOLERANCE_RAD
                    and line["tracking_max_m"] <= LINE_TRACKING_TOLERANCE_M
                    and line["body_shift_m"] <= LINE_BODY_SHIFT_TOLERANCE_M)
    return line


def _line_used(ctx, qpos, span, before: _Solution, after: _Solution) -> bool:
    line = _line_check(ctx, qpos, span, before, after)
    return line is not None and line["used"]


def _body_tip_shift(ctx, qpos, start: int, reach: int, arm_q: np.ndarray) -> float:
    """How far the base body alone moves the tip of a fixed arm pose over a move. The line is solved
    on the body at the reach frame; a body that moves during the move carries the tip off it."""
    last = len(qpos) - 1
    tip_start, _ = _blade_tip(ctx, _with_arm(qpos[min(start, last)], ctx.arm, arm_q))
    tip_reach, _ = _blade_tip(ctx, _with_arm(qpos[min(reach, last)], ctx.arm, arm_q))
    return float(np.linalg.norm(tip_reach - tip_start))


def _solve_all(ctx, keys, spans, qpos, padded, config, previous_round: dict[int, _Solution]) -> dict[int, _Solution]:
    rng = np.random.default_rng(config.seed)
    solutions: dict[int, _Solution] = {}
    by_action: dict[str, _Solution] = {}
    warm = padded.joint_pos[0, ARM_COLUMNS]
    for i, key in enumerate(keys):
        if key.action == BASE_ACTION:
            warm = padded.joint_pos[spans[i][1], ARM_COLUMNS]
            continue
        target = resolve_target(key.action)
        frame_qpos = qpos[spans[i][1]].copy()
        previous = by_action.get(target.name)
        if previous is not None and _reuse_ok(ctx, previous, frame_qpos, target, config):
            solutions[i] = previous  # same target again (e.g. back to guard): identical pose
        elif (i in previous_round and _reuse_ok(ctx, previous_round[i], frame_qpos, target, config)
              and not (previous_round[i].line_seeded
                       and not _line_used(ctx, qpos, spans[i], solutions[i - 1], previous_round[i]))):
            # Only the timing moved; the pose still fits. A pose solved from a blade line is kept only
            # while its move can still use that line: otherwise it is solved again below, and falls
            # back to the joint-space solution.
            solutions[i] = previous_round[i]
        else:
            # A repeated target that no longer fits (the body moved under it) is re-solved from
            # its earlier pose, not from the previous key: the arm has 2 redundant DoF, and a warm
            # start from the cut lands in another solution family (advance_lunge_cut_v4: wrist
            # yaw -1.12 in the first guard, +1.45 in the last), which a policy cannot tell apart.
            seed = previous.arm_q if previous is not None else warm
            solution = None
            if config.blade_path == "line" and previous is None and i > 0 and keys[i - 1].action != BASE_ACTION:
                # Solve from where the blade line from the previous key ends: the family the move reaches.
                # Kept only if the move can then use the line; otherwise the key is solved as in joint
                # space, from the same random state, so a fallen-back move is the joint-space one.
                state = rng.bit_generator.state
                tip_target, dir_target = target_in_world(ctx, frame_qpos, target, config)
                line_end = track_blade_line(ctx, frame_qpos, warm, tip_target, dir_target)[0][-1]
                solved = _solve_key(ctx, frame_qpos, target, line_end, config, rng)
                if _line_used(ctx, qpos, spans[i], solutions[i - 1], solved):
                    solution = replace(solved, line_seeded=True)
                else:
                    rng.bit_generator.state = state
            solutions[i] = solution if solution is not None else _solve_key(ctx, frame_qpos, target, seed, config, rng)
        by_action.setdefault(target.name, solutions[i])
        warm = solutions[i].arm_q
    return solutions


def _reuse_ok(ctx, sol: _Solution, frame_qpos, target, config) -> bool:
    qpos = _apply_solution(ctx, frame_qpos, sol)
    tip_target, dir_target = target_in_world(ctx, frame_qpos, target, config)
    pos, angle = _errors(ctx, qpos, tip_target, dir_target)
    return (pos <= config.position_tolerance_m and angle <= config.angle_tolerance_deg
            and not self_collisions(ctx.arm, qpos))


def _apply_solution(ctx: _Context, frame_qpos: np.ndarray, sol: _Solution) -> np.ndarray:
    qpos = frame_qpos.copy()
    qpos[ctx.arm.qpos_adr] = sol.arm_q
    waist_adr = ctx.arm_waist.qpos_adr[len(ARM_JOINTS):]
    qpos[waist_adr] = frame_qpos[waist_adr] + sol.waist_delta
    return qpos


def _clamp_speed(motion: Motion, speed_cap: np.ndarray) -> tuple[Motion, dict]:
    """Safety net on the controlled joints only (moves are already timed to the cap)."""
    joints = motion.joint_pos[:, CONTROLLED_COLUMNS]
    speeds = np.abs(np.diff(joints, axis=0)) * motion.fps
    over = speeds > speed_cap * (1 + 1e-6)
    if not over.any():
        return motion, {"frames_over": 0, "max_change_rad": 0.0}
    limited = rate_limit(joints, speed_cap / motion.fps)
    out = motion.copy()
    out.joint_pos[:, CONTROLLED_COLUMNS] = limited
    return out, {"frames_over": int(over.any(axis=1).sum()), "max_change_rad": float(np.abs(limited - joints).max())}


def _key_reports(ctx, keys, spans, solutions, motion: Motion, padded: Motion, config) -> list[KeyReport]:
    g1 = load_g1()
    qpos_out = g1.qpos_from_motion(motion)
    qpos_base = g1.qpos_from_motion(padded)
    reports = []
    for i, (key, (start, reach, end)) in enumerate(zip(keys, spans)):
        hold_end = spans[i + 1][0] if i + 1 < len(keys) else motion.n_frames
        hold_end = max(hold_end, reach + 1)
        if key.action == BASE_ACTION:
            reports.append(KeyReport(i, key.action, BASE_ACTION, BASE_ACTION, reach, reach / motion.fps, key.move_s,
                                     (reach - start) / motion.fps, key.hold_s, "-", 0.0, 0.0, 0.0, [], [0.0, 0.0]))
            continue
        target = resolve_target(key.action)
        tip_target, dir_target = target_in_world(ctx, qpos_base[reach], target, config)
        pos, angle = _errors(ctx, qpos_out[reach], tip_target, dir_target)
        drift = 0.0
        for t in range(reach, min(hold_end, motion.n_frames)):
            tip_t, dir_t = target_in_world(ctx, qpos_base[t], target, config)
            drift = max(drift, _errors(ctx, qpos_out[t], tip_t, dir_t)[0])
        sol = solutions[i]
        reports.append(KeyReport(
            i, key.action, target.name, target.kind, reach, round(reach / motion.fps, 4), key.move_s,
            round((reach - start) / motion.fps, 4), key.hold_s, sol.joint_set, round(pos, 4), round(angle, 2),
            round(drift, 4), self_collisions(ctx.arm, qpos_out[reach]),
            [round(float(v), 4) for v in sol.waist_delta],
        ))
    return reports


def _summary(ctx, motion: Motion, padded: Motion, reports, clamp: dict, config: BladeConfig,
             paths: dict[int, np.ndarray] | None = None) -> dict:
    g1 = load_g1()
    qpos = g1.qpos_from_motion(motion)
    collision_frames, pairs = 0, set()
    for t in range(motion.n_frames):
        hits = self_collisions(ctx.arm, qpos[t])
        if hits:
            collision_frames += 1
            pairs.update(hits)
    joints = motion.joint_pos[:, CONTROLLED_COLUMNS]
    lower = g1.joint_lower[CONTROLLED_COLUMNS]
    upper = g1.joint_upper[CONTROLLED_COLUMNS]
    middle, half = (lower + upper) / 2, (upper - lower) / 2
    changed = np.abs(motion.joint_pos - padded.joint_pos).max(axis=0) > 1e-9
    beyond = np.abs(joints - middle) > config.soft_limit_factor * half + 1e-9
    # Only joints the script moved count: an untouched base waist is the base motion's business.
    beyond_names = sorted({JOINT_NAMES[CONTROLLED_COLUMNS[j]] for j in np.flatnonzero(beyond.any(axis=0))
                           if changed[CONTROLLED_COLUMNS[j]]})
    speeds = np.abs(np.diff(joints, axis=0)) * motion.fps
    fraction = speeds / hardware_velocity_limits()[CONTROLLED_COLUMNS]
    targets = [r for r in reports if r.action != BASE_ACTION]
    torque = _peak_torque_demand(motion, qpos, targets, config, paths)
    return {
        "frames": motion.n_frames,
        "max_tip_error_m": max((r.tip_error_m for r in targets), default=0.0),
        "max_angle_error_deg": max((r.angle_error_deg for r in targets), default=0.0),
        "max_hold_drift_m": max((r.hold_drift_m for r in targets), default=0.0),
        "keys_off_target": [r.action for r in targets
                            if r.tip_error_m > config.position_tolerance_m or r.angle_error_deg > config.angle_tolerance_deg],
        "collision_frames": collision_frames,
        "collision_pairs": sorted(pairs),
        "controlled_joints_beyond_soft_limit": beyond_names,
        "controlled_max_speed_fraction_of_motor": float(fraction.max(initial=0.0)),
        "speed_clamp": clamp,
        "waist_used": any(r.joint_set == "arm+waist" for r in targets),
        "wrist_pitch_yaw_max_abs_rad": float(np.abs(motion.joint_pos[:, WRIST_PITCH_YAW_COLUMNS]).max(initial=0.0)),
        **torque,
        "config": {k: v for k, v in asdict(config).items() if k != "saber"},
        "target_frame": f"base right shoulder, axes: {config.heading}",
        "arm_reach_m": round(ctx.reach, 4),
    }


def _peak_torque_demand(motion: Motion, qpos: np.ndarray, targets: list[KeyReport], config: BladeConfig,
                        paths: dict[int, np.ndarray] | None = None) -> dict:
    """Worst move_torque_ratio over the keyed moves of the final motion (body pinned at each reach);
    a blade-line move is measured along its path (path_torque_ratio)."""
    worst = {"max_torque_ratio": 0.0, "max_torque_ratio_joint": None, "max_torque_ratio_key": None}
    for report in targets:
        frames = int(round(report.move_s_used * motion.fps))
        if frames <= 0:
            continue
        reach, start = report.frame, report.frame - frames
        if paths and report.index in paths:
            ratios = path_torque_ratio(qpos[reach], paths[report.index], frames / motion.fps, config.saber)
        else:
            ratios = move_torque_ratio(qpos[reach], motion.joint_pos[start, ARM_COLUMNS],
                                       motion.joint_pos[reach, ARM_COLUMNS], frames / motion.fps, config.saber)
        joint = max(ratios, key=ratios.get)
        if ratios[joint] > worst["max_torque_ratio"]:
            worst = {"max_torque_ratio": round(ratios[joint], 3), "max_torque_ratio_joint": joint,
                     "max_torque_ratio_key": report.action}
    return worst


def script_summary(keys: list[BladeKey]) -> str:
    """Compact human-readable script for provenance rows: 'en_garde > parry_quinte 0.3s/0.6s > ...'."""
    parts = []
    for k in keys:
        timing = f"{k.move_s:g}s" + (f"/{k.hold_s:g}s" if k.hold_s else "") + (f"@{k.at_s:.2f}s" if k.at_s is not None else "")
        parts.append(f"{k.action} {timing}")
    return " > ".join(parts)
