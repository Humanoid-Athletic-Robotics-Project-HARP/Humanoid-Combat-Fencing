"""MediaPipe keypoints (world, z-up) -> the per-joint frames GMR consumes for `src_human="smplx"`.

What GMR expects (third_party/GMR/general_motion_retargeting/utils/smpl.py,
`get_gvhmr_data_offline_fast`): one dict per frame, SMPL-X joint name -> (position (3,),
quaternion wxyz (4,)), in a z-up world. GMR builds it from SMPL-X global joint rotations and
then left-multiplies everything by the y-up -> z-up rotation

    M = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]   (a +90 deg turn about x)

SMPL-X global joint rotations are the identity in the rest (T-) pose, whose canonical axes are
x = the body's left, y = up, z = forward (palms down, arms along +-x). So in GMR's world a
joint quaternion is simply the rotation whose columns are where the segment's canonical
(left, up, forward) axes point now; a rest-pose human facing -y has every joint at M.

We have no SMPL-X model, so we rebuild those frames from bone vectors: one axis from the bone
direction, the twist from a neighbouring point (knee hinge, elbow hinge, heel->ankle, index vs
pinky). Only the 14 joints in GMR's smplx_to_g1.json scale table are produced; GMR drops
anything else before the IK.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from fencing.pose import landmarks as lm

# Same rotation GMR applies to GVHMR output (smpl.py, get_gvhmr_data_offline_fast).
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

GMR_SMPLX_JOINTS = [
    "pelvis",
    "spine3",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# MediaPipe hip landmarks sit wider than true hip joint centres (they track the outside of
# the pelvis); pull them towards the midpoint so the robot's hip roll is not biased outwards.
HIP_WIDTH_SCALE = 0.8
# SMPL spine3 is the upper-chest joint, roughly three quarters of the way up the torso.
SPINE3_FRACTION = 0.75
# SMPL "foot" is the ball of the foot (toe base), not the toe tip MediaPipe reports.
BALL_OF_FOOT_FRACTION = 0.7
BALL_OF_FOOT_HEIGHT_M = 0.03
# A foot whose heel and toe are both this close to the floor is planted: its frame is made
# level (yaw only). MediaPipe's heel sits ~3 cm higher than its toe tip on a flat foot, which
# GMR's strong foot-orientation task would otherwise turn into a 20-30 deg toe-down robot foot.
FLAT_FOOT_FULL_M = 0.05
FLAT_FOOT_NONE_M = 0.10
# Below this bend the knee/elbow hinge axis is too noisy to set the limb twist.
HINGE_RELIABLE_BEND_RAD = np.deg2rad(25.0)
WORLD_UP = np.array([0.0, 0.0, 1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def frame_from_axes(primary_axis: int, primary: np.ndarray, secondary_axis: int, secondary: np.ndarray) -> np.ndarray:
    """Right-handed rotation matrices (..., 3, 3) with column `primary_axis` along `primary`
    and column `secondary_axis` along the part of `secondary` orthogonal to it."""
    if primary_axis == secondary_axis:
        raise ValueError(f"primary and secondary axis must differ, both are {primary_axis}")
    a = _unit(primary)
    b = _unit(secondary - np.sum(secondary * a, axis=-1, keepdims=True) * a)
    third_axis = 3 - primary_axis - secondary_axis
    is_cyclic = (primary_axis, secondary_axis) in {(0, 1), (1, 2), (2, 0)}
    c = np.cross(a, b) if is_cyclic else np.cross(b, a)
    cols = [None, None, None]
    cols[primary_axis], cols[secondary_axis], cols[third_axis] = a, b, c
    return np.stack(cols, axis=-1)


def _bend_weight(first_bone: np.ndarray, second_bone: np.ndarray) -> np.ndarray:
    """0 for a straight limb, 1 once it is bent past HINGE_RELIABLE_BEND_RAD."""
    sin_bend = np.linalg.norm(np.cross(_unit(first_bone), _unit(second_bone)), axis=-1)
    return np.clip(sin_bend / np.sin(HINGE_RELIABLE_BEND_RAD), 0.0, 1.0)[..., None]


def _hinge_hint(first_bone: np.ndarray, second_bone: np.ndarray, sign: float, fallback: np.ndarray) -> np.ndarray:
    """Hinge axis of a two-bone limb, blended into `fallback` as the limb straightens."""
    weight = _bend_weight(first_bone, second_bone)
    hinge = sign * _unit(np.cross(first_bone, second_bone))
    return _unit(weight * hinge + (1.0 - weight) * _unit(fallback))


def _foot_frame(heel: np.ndarray, toe: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    # Rest foot points forward (+z) and its up (+y) is roughly heel -> ankle.
    return frame_from_axes(2, toe - heel, 1, ankle - heel)


def flat_foot_weight(heel: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """(T,) 1 when heel and toe are both near the floor (z = 0), 0 once either is lifted."""
    highest = np.maximum(heel[..., 2], toe[..., 2])
    return np.clip((FLAT_FOOT_NONE_M - highest) / (FLAT_FOOT_NONE_M - FLAT_FOOT_FULL_M), 0.0, 1.0)


def _level_foot(foot_rot: np.ndarray, heel: np.ndarray, toe: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Rotate each foot frame towards level (forward horizontal, up = world up) by `weight`."""
    forward = toe - heel
    forward[..., 2] = 0.0
    level = frame_from_axes(2, forward, 1, np.broadcast_to(WORLD_UP, forward.shape))
    orig = Rotation.from_matrix(foot_rot)
    partial = Rotation.from_rotvec((orig.inv() * Rotation.from_matrix(level)).as_rotvec() * weight[:, None])
    return (orig * partial).as_matrix()


def _hand_frame(wrist: np.ndarray, index: np.ndarray, pinky: np.ndarray, is_left: bool) -> np.ndarray:
    # Rest pose: hand along +x (left) / -x (right), palm down, so index is forward of pinky.
    along = (index + pinky) / 2.0 - wrist
    return frame_from_axes(0, along if is_left else -along, 2, index - pinky)


def compute_joint_frames(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build GMR smplx joint positions and orientations from MediaPipe keypoints.

    keypoints: (T, 33, 3) MediaPipe landmarks in metres, z-up world.
    Returns positions (T, 14, 3) and quaternions wxyz (T, 14, 4), ordered as GMR_SMPLX_JOINTS.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (lm.NUM_LANDMARKS, 3):
        raise ValueError(f"expected keypoints of shape (T, 33, 3), got {keypoints.shape}")
    k = keypoints.astype(np.float64)
    p = {
        name: k[:, idx]
        for name, idx in vars(lm).items()
        if name.isupper() and isinstance(idx, int) and idx < lm.NUM_LANDMARKS
    }

    mid_hip = (p["LEFT_HIP"] + p["RIGHT_HIP"]) / 2.0
    mid_shoulder = (p["LEFT_SHOULDER"] + p["RIGHT_SHOULDER"]) / 2.0
    torso_up = mid_shoulder - mid_hip

    pelvis_rot = frame_from_axes(0, p["LEFT_HIP"] - p["RIGHT_HIP"], 1, _unit(torso_up) + WORLD_UP)
    spine_rot = frame_from_axes(0, p["LEFT_SHOULDER"] - p["RIGHT_SHOULDER"], 1, torso_up)
    pelvis_left = pelvis_rot[..., 0]

    rotations: dict[str, np.ndarray] = {"pelvis": pelvis_rot, "spine3": spine_rot}
    positions: dict[str, np.ndarray] = {
        "pelvis": mid_hip,
        "spine3": mid_hip + SPINE3_FRACTION * torso_up,
    }

    for side, is_left in (("left", True), ("right", False)):
        s = side.upper()
        hip, knee, ankle = p[f"{s}_HIP"], p[f"{s}_KNEE"], p[f"{s}_ANKLE"]
        heel, toe = p[f"{s}_HEEL"], p[f"{s}_FOOT_INDEX"]
        thigh, shin = knee - hip, ankle - knee

        foot_rot = _foot_frame(heel, toe, ankle)
        flat = flat_foot_weight(heel, toe)
        foot_rot = _level_foot(foot_rot, heel, toe, flat)
        # Straight leg: the knee hinge is undefined; split the twist between the pelvis and
        # the foot (a fencer's back foot is turned out ~90 deg, and the knee follows it).
        leg_fallback = _unit(pelvis_left) + _unit(foot_rot[..., 0])
        # Normal knee flexion gives thigh x shin along the body's left for both legs.
        knee_axis = _hinge_hint(thigh, shin, 1.0, leg_fallback)
        rotations[f"{side}_hip"] = frame_from_axes(1, -thigh, 0, knee_axis)
        rotations[f"{side}_knee"] = frame_from_axes(1, -shin, 0, knee_axis)
        rotations[f"{side}_foot"] = foot_rot

        positions[f"{side}_hip"] = mid_hip + HIP_WIDTH_SCALE * (hip - mid_hip)
        positions[f"{side}_knee"] = knee
        ball = heel + BALL_OF_FOOT_FRACTION * (toe - heel)
        ball_on_floor = ball.copy()
        ball_on_floor[:, 2] = np.minimum(heel[:, 2], toe[:, 2])
        ball = flat[:, None] * ball_on_floor + (1.0 - flat[:, None]) * ball
        positions[f"{side}_foot"] = ball + BALL_OF_FOOT_HEIGHT_M * foot_rot[..., 1]

        shoulder, elbow, wrist = p[f"{s}_SHOULDER"], p[f"{s}_ELBOW"], p[f"{s}_WRIST"]
        upper, fore = elbow - shoulder, wrist - elbow
        hand_rot = _hand_frame(wrist, p[f"{s}_INDEX"], p[f"{s}_PINKY"], is_left)
        # Elbow flexion: upper x fore points down (-y) for the left arm, up for the right.
        elbow_up = _hinge_hint(upper, fore, -1.0 if is_left else 1.0, hand_rot[..., 1])
        sign = 1.0 if is_left else -1.0
        rotations[f"{side}_shoulder"] = frame_from_axes(0, sign * upper, 1, elbow_up)
        rotations[f"{side}_elbow"] = frame_from_axes(0, sign * fore, 1, elbow_up)
        rotations[f"{side}_wrist"] = hand_rot

        positions[f"{side}_shoulder"] = shoulder
        positions[f"{side}_elbow"] = elbow
        positions[f"{side}_wrist"] = wrist

    pos = np.stack([positions[name] for name in GMR_SMPLX_JOINTS], axis=1)
    rot = np.stack([rotations[name] for name in GMR_SMPLX_JOINTS], axis=1)
    quat_xyzw = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].reshape(rot.shape[:2] + (4,))
    return pos, quat_wxyz


def frames_to_gmr_dicts(positions: np.ndarray, quats_wxyz: np.ndarray) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per-frame dicts in exactly the shape GeneralMotionRetargeting.retarget() takes."""
    return [
        {name: (positions[t, j].copy(), quats_wxyz[t, j].copy()) for j, name in enumerate(GMR_SMPLX_JOINTS)}
        for t in range(positions.shape[0])
    ]
