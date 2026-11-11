from functools import lru_cache

import numpy as np
import pytest

from fencing.motion.io import JOINT_NAMES
from fencing.motion.kinematics import forward_kinematics, load_g1
from fencing.motion.normalize import canonicalize, detect_handedness, mirror_map, mirror_motion, yaw_of
from fencing.motion.synthetic import add_jitter, synthesize


@lru_cache(maxsize=1)
def footwork():
    motion, _ = synthesize([("en_garde", 0.6), ("advance", 0.6), ("lunge", 1.2)])
    # Arbitrary joint noise so the mirror test exercises every joint, not just the legs.
    return add_jitter(motion, joint_sigma_rad=0.2, root_sigma_m=0.0, seed=3)


def _mirrored_body_index(names):
    swap = {n: n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_") for n in names}
    return [names.index(swap[n]) for n in names]


def test_csv_joints_map_to_mjcf_joints_by_name():
    import mujoco

    g1 = load_g1()
    for column, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(g1.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert g1.qpos_index[column] == g1.model.jnt_qposadr[jid]


def test_sole_geometry_comes_from_capsules():
    foot = load_g1().feet[0]
    assert foot.toe[0] > foot.sole_center[0] > foot.heel[0]
    assert foot.capsule_endpoints.shape == (14, 3)


def test_mirror_sign_rules_derived_from_axes():
    mmap = mirror_map()
    for j, name in enumerate(JOINT_NAMES):
        expected = 1.0 if "pitch" in name or "knee" in name or "elbow" in name else -1.0
        assert mmap.sign[j] == expected, name


def test_mirror_twice_is_identity():
    motion = footwork()
    twice = mirror_motion(mirror_motion(motion))
    np.testing.assert_allclose(twice.to_array(), motion.to_array(), atol=1e-12)


def test_mirrored_bodies_are_reflected_original_bodies():
    motion = footwork()
    fk = forward_kinematics(motion)
    fk_m = forward_kinematics(mirror_motion(motion))
    perm = _mirrored_body_index(list(fk.body_names))
    reflected = fk.body_pos[:, perm] * np.array([1.0, -1.0, 1.0])
    # 1e-4 m: the MJCF's left/right body offsets are symmetric to its printed precision.
    np.testing.assert_allclose(fk_m.body_pos, reflected, atol=1e-4)
    np.testing.assert_allclose(fk_m.hand_pos[:, ::-1] * [1, -1, 1], fk.hand_pos, atol=1e-4)


def test_handedness_detected_and_flipped_by_mirror():
    motion, _ = synthesize([("en_garde", 0.5)])
    right = detect_handedness(forward_kinematics(motion))
    left = detect_handedness(forward_kinematics(mirror_motion(motion)))
    assert right.hand == "R" and right.confidence > 0.5
    assert left.hand == "L" and left.confidence > 0.5


def test_square_stance_has_low_handedness_confidence():
    from fencing.motion.io import Motion

    standing = Motion(np.tile([0, 0, 0.79], (5, 1)), np.tile([0, 0, 0, 1.0], (5, 1)), np.zeros((5, 29)))
    assert detect_handedness(forward_kinematics(standing)).confidence < 0.2


def test_canonicalize_starts_at_origin_facing_x():
    from fencing.motion.normalize import rotate_about_z

    motion = rotate_about_z(footwork(), 2.0, np.array([-3.0, 1.0]))
    out, _ = canonicalize(motion)
    np.testing.assert_allclose(out.root_pos[0, :2], 0.0, atol=1e-9)
    assert abs(np.mean(yaw_of(out.root_quat[:9]))) < 1e-6
    np.testing.assert_allclose(out.root_pos[:, 2], motion.root_pos[:, 2])


@pytest.mark.parametrize("frames", [1, 3])
def test_fk_shapes(frames):
    motion = footwork().slice(0, frames + 1)
    fk = forward_kinematics(motion)
    assert fk.foot_lowest_z.shape == (frames + 1, 2)
    assert fk.sole_center.shape == (frames + 1, 2, 3)
