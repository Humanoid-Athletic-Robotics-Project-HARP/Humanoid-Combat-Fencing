"""Pure geometry of the license-free pose path: keypoints -> GMR joint frames, root trajectory,
left/right fixes, handedness."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fencing.pose import landmarks as lm
from fencing.pose.handedness import guess_handedness
from fencing.pose.joint_frames import (
    GMR_SMPLX_JOINTS,
    Y_UP_TO_Z_UP,
    compute_joint_frames,
    frame_from_axes,
    frames_to_gmr_dicts,
)
from fencing.pose.root_trajectory import (
    PinholeCamera,
    body_scale_factor,
    estimate_root_trajectory,
    fit_floor_plane,
    leg_torso_length,
    planted_weights,
    anchor_planted_feet,
    ray_rotations,
    snap_planted_feet,
    solve_root_translation,
    world_from_camera,
    world_from_floor_normal,
)


def t_pose_facing_minus_y() -> np.ndarray:
    """MediaPipe-layout T-pose in GMR's z-up world, facing -y (left = +x): SMPL-X rest pose."""
    k = np.zeros((lm.NUM_LANDMARKS, 3))
    for sign, side in ((1.0, "LEFT"), (-1.0, "RIGHT")):
        idx = lambda name: getattr(lm, f"{side}_{name}")  # noqa: E731
        k[idx("HIP")] = [sign * 0.10, 0.0, 0.95]
        k[idx("KNEE")] = [sign * 0.10, 0.0, 0.52]
        k[idx("ANKLE")] = [sign * 0.10, 0.0, 0.08]
        k[idx("HEEL")] = [sign * 0.10, 0.05, 0.02]
        k[idx("FOOT_INDEX")] = [sign * 0.10, -0.15, 0.02]
        k[idx("SHOULDER")] = [sign * 0.18, 0.0, 1.45]
        k[idx("ELBOW")] = [sign * 0.45, 0.0, 1.45]
        k[idx("WRIST")] = [sign * 0.70, 0.0, 1.45]
        k[idx("INDEX")] = [sign * 0.85, -0.03, 1.45]
        k[idx("PINKY")] = [sign * 0.83, 0.03, 1.45]
        k[idx("THUMB")] = [sign * 0.75, -0.06, 1.45]
    k[lm.NOSE] = [0.0, -0.10, 1.65]
    return k


def quat_angle_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    r1 = Rotation.from_quat(q1_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4))
    r2 = Rotation.from_quat(q2_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4))
    return np.rad2deg((r1.inv() * r2).magnitude())


def rest_quat_wxyz() -> np.ndarray:
    xyzw = Rotation.from_matrix(Y_UP_TO_Z_UP).as_quat()
    return xyzw[[3, 0, 1, 2]]


def test_frame_from_axes_is_right_handed_for_every_axis_pair():
    rng = np.random.default_rng(0)
    for primary in range(3):
        for secondary in range(3):
            if primary == secondary:
                continue
            r = frame_from_axes(primary, rng.normal(size=3), secondary, rng.normal(size=3))
            assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)
            assert np.isclose(np.linalg.det(r), 1.0)


def test_rest_pose_gives_gmr_rest_orientation_for_every_joint():
    pos, quat = compute_joint_frames(t_pose_facing_minus_y()[None])
    assert pos.shape == (1, len(GMR_SMPLX_JOINTS), 3)
    angles = quat_angle_deg(quat[0], np.tile(rest_quat_wxyz(), (len(GMR_SMPLX_JOINTS), 1)))
    worst = GMR_SMPLX_JOINTS[int(np.argmax(angles))]
    assert angles.max() < 1.0, f"{worst} is {angles.max():.1f} deg off the SMPL-X rest frame"


def test_yawed_body_rotates_every_joint_frame_by_the_same_yaw():
    yaw = Rotation.from_euler("z", 90, degrees=True)
    keypoints = yaw.apply(t_pose_facing_minus_y())
    _, quat = compute_joint_frames(keypoints[None])
    expected = (yaw * Rotation.from_matrix(Y_UP_TO_Z_UP)).as_quat()[[3, 0, 1, 2]]
    assert quat_angle_deg(quat[0], np.tile(expected, (len(GMR_SMPLX_JOINTS), 1))).max() < 1.0


def test_bent_knee_keeps_hinge_on_body_left_and_thigh_along_bone():
    k = t_pose_facing_minus_y()
    k[lm.LEFT_KNEE] = [0.10, -0.25, 0.55]  # knee forward (-y)
    _, quat = compute_joint_frames(k[None])
    thigh = Rotation.from_quat(quat[0, GMR_SMPLX_JOINTS.index("left_hip"), [1, 2, 3, 0]]).as_matrix()
    assert np.allclose(thigh[:, 0], [1, 0, 0], atol=1e-6)
    expected_up = (k[lm.LEFT_HIP] - k[lm.LEFT_KNEE]) / np.linalg.norm(k[lm.LEFT_HIP] - k[lm.LEFT_KNEE])
    assert np.allclose(thigh[:, 1], expected_up, atol=1e-6)


def test_elbow_bent_forward_keeps_upper_arm_up_axis_vertical():
    k = t_pose_facing_minus_y()
    k[lm.LEFT_WRIST] = [0.45, -0.25, 1.45]
    k[lm.LEFT_INDEX] = [0.45, -0.40, 1.45]
    k[lm.LEFT_PINKY] = [0.47, -0.38, 1.45]
    _, quat = compute_joint_frames(k[None])
    upper = Rotation.from_quat(quat[0, GMR_SMPLX_JOINTS.index("left_shoulder"), [1, 2, 3, 0]]).as_matrix()
    assert np.allclose(upper[:, 1], [0, 0, 1], atol=1e-6)
    assert np.allclose(upper[:, 0], [1, 0, 0], atol=1e-6)


def test_ball_of_foot_sits_between_heel_and_toe_and_above_floor():
    pos, _ = compute_joint_frames(t_pose_facing_minus_y()[None])
    foot = pos[0, GMR_SMPLX_JOINTS.index("left_foot")]
    assert -0.15 < foot[1] < 0.05
    assert foot[2] > 0.02


def test_planted_foot_frame_is_level_and_lifted_foot_keeps_its_pitch():
    planted = t_pose_facing_minus_y()
    planted[lm.LEFT_HEEL, 2] = 0.05  # MediaPipe heel above toe tip on a flat foot
    planted[lm.LEFT_FOOT_INDEX, 2] = 0.0
    lifted = planted.copy()
    lifted[[lm.LEFT_HEEL, lm.LEFT_FOOT_INDEX, lm.LEFT_ANKLE], 2] += 0.25
    _, quats = compute_joint_frames(np.stack([planted, lifted]))
    j = GMR_SMPLX_JOINTS.index("left_foot")
    forward = Rotation.from_quat(quats[:, j][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    assert abs(forward[0, 2]) < 1e-6, "planted foot should point level"
    assert forward[1, 2] < -0.2, "lifted foot keeps its toe-down pitch"


def test_gmr_dicts_have_exactly_the_ik_joint_names_and_wxyz_quats():
    pos, quat = compute_joint_frames(np.stack([t_pose_facing_minus_y()] * 3))
    frames = frames_to_gmr_dicts(pos, quat)
    assert len(frames) == 3
    assert set(frames[0]) == set(GMR_SMPLX_JOINTS)
    p, q = frames[0]["pelvis"]
    assert p.shape == (3,) and q.shape == (4,)
    assert np.allclose(np.abs(q), np.abs(rest_quat_wxyz()), atol=1e-6)


def test_compute_joint_frames_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(T, 33, 3\)"):
        compute_joint_frames(np.zeros((2, 17, 3)))


# ---- root trajectory --------------------------------------------------------------------


def camera_oriented_skeleton() -> np.ndarray:
    """T-pose turned side-on to the camera, in MediaPipe world axes (x right, y down, z away),
    hip-centred."""
    z_up = Rotation.from_euler("z", 90, degrees=True).apply(t_pose_facing_minus_y())  # faces +x
    cam_from_world = world_from_camera(0.0).T
    k = z_up @ cam_from_world.T
    mid_hip = (k[lm.LEFT_HIP] + k[lm.RIGHT_HIP]) / 2.0
    return k - mid_hip


def as_mediapipe_world(points_cam_rel: np.ndarray, hip_pixel: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    """MediaPipe orients world landmarks as if the person were straight ahead of the camera
    (measured on real clips: body yaw follows the viewing-ray angle with slope ~1)."""
    return points_cam_rel @ ray_rotations(hip_pixel[None], camera)[0]


def synthetic_clip(num_frames: int = 60, depth: float = 5.0, pitch_down_deg: float = 0.0):
    camera = PinholeCamera.from_image(1280, 720)
    skeleton = camera_oriented_skeleton()
    xs = np.linspace(-1.0, 1.0, num_frames)  # walk 2 m along the image x axis
    hip_height_below_camera = 0.3
    image_xy, world = [], []
    # A camera pitched down by theta sees level points rotated by +theta about its x axis.
    tilt = Rotation.from_euler("x", pitch_down_deg, degrees=True)
    for x in xs:
        root = np.array([x, hip_height_below_camera, depth])
        points_cam = tilt.apply(skeleton + root)
        image = camera.project(points_cam)
        image_xy.append(image)
        hip_pixel = (image[lm.LEFT_HIP] + image[lm.RIGHT_HIP]) / 2.0
        world.append(as_mediapipe_world(tilt.apply(skeleton), hip_pixel, camera))
    return camera, np.array(image_xy), np.array(world), xs


def test_solve_root_translation_recovers_exact_camera_position():
    camera = PinholeCamera.from_image(1280, 720)
    skeleton = camera_oriented_skeleton()
    root = np.array([0.4, 0.2, 6.0])
    image = camera.project(skeleton + root)
    solved = solve_root_translation(image, skeleton, np.ones(len(skeleton)), camera)
    assert np.allclose(solved, root, atol=1e-6)


def test_body_scale_factor_maps_skeleton_to_requested_height():
    world = camera_oriented_skeleton()[None].repeat(5, axis=0)
    s = body_scale_factor(world, 1.75)
    assert np.isclose(leg_torso_length(world * s)[0], 0.779 * 1.75, rtol=1e-3)


def test_root_trajectory_tracks_lateral_walk_with_feet_on_floor():
    camera, image_xy, world, xs = synthetic_clip()
    true_scale = body_scale_factor(world, 1.75)
    result = estimate_root_trajectory(
        image_xy, world / true_scale, np.ones(world.shape[:2]), camera, fps=30.0, human_height_m=1.75
    )
    k = result.keypoints_world
    mid_hip = (k[:, lm.LEFT_HIP] + k[:, lm.RIGHT_HIP]) / 2.0
    assert np.allclose(mid_hip[:, 0], xs - xs[0], atol=0.02), "root should advance 2 m along +x"
    assert np.abs(mid_hip[:, 1]).max() < 0.02, "no drift towards/away from the camera"
    assert np.allclose(k[:, lm.FEET, 2].min(axis=1), 0.0, atol=1e-6)
    assert abs(result.pitch_down_rad) < np.deg2rad(1.0)


def test_ray_rotations_map_the_optical_axis_onto_each_pixel_ray():
    camera = PinholeCamera.from_image(1280, 720)
    pixels = np.array([[640.0, 360.0], [1200.0, 360.0], [100.0, 650.0]])
    rotations = ray_rotations(pixels, camera)
    rays = np.c_[(pixels - [camera.cx, camera.cy]) / camera.focal_px, np.ones(3)]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    assert np.allclose(rotations @ [0.0, 0.0, 1.0], rays, atol=1e-12)
    assert np.allclose(rotations[0], np.eye(3))
    assert np.allclose(np.linalg.det(rotations), 1.0)


def test_root_trajectory_keeps_heading_when_walking_across_the_image():
    """Uncorrected, the ray-relative landmarks would turn the body by the ray angle (~38 deg
    over this 2 m walk at 3 m); corrected, the hip line stays along the walk."""
    camera, image_xy, world, xs = synthetic_clip(depth=3.0)
    result = estimate_root_trajectory(image_xy, world, np.ones(world.shape[:2]), camera, fps=30.0)
    k = result.keypoints_world
    hips = k[:, lm.RIGHT_HIP, :2] - k[:, lm.LEFT_HIP, :2]
    hip_yaw = np.rad2deg(np.arctan2(hips[:, 1], hips[:, 0]))
    assert np.ptp(hip_yaw) < 2.0
    raw = estimate_root_trajectory(image_xy, world, np.ones(world.shape[:2]), camera, fps=30.0, ray_correct=False)
    hips_raw = raw.keypoints_world[:, lm.RIGHT_HIP, :2] - raw.keypoints_world[:, lm.LEFT_HIP, :2]
    assert np.ptp(np.rad2deg(np.arctan2(hips_raw[:, 1], hips_raw[:, 0]))) > 25.0


def test_root_trajectory_removes_camera_pitch():
    camera, image_xy, world, xs = synthetic_clip(pitch_down_deg=15.0)
    result = estimate_root_trajectory(image_xy, world, np.ones(world.shape[:2]), camera, fps=30.0)
    assert np.isclose(np.rad2deg(result.pitch_down_rad), 15.0, atol=1.0)
    k = result.keypoints_world
    torso = (k[:, lm.LEFT_SHOULDER] + k[:, lm.RIGHT_SHOULDER] - k[:, lm.LEFT_HIP] - k[:, lm.RIGHT_HIP]) / 2
    assert np.allclose(torso[:, :2], 0.0, atol=0.01), "torso should be vertical in the world"


def test_camera_from_fov_matches_definition():
    cam = PinholeCamera.from_image(1000, 500, hfov_deg=90.0)
    assert np.isclose(cam.focal_px, 500.0)
    with pytest.raises(ValueError, match="hfov_deg"):
        PinholeCamera.from_image(1000, 500, hfov_deg=0.0)


def standing_clip(num_frames: int = 45, depth: float = 5.0, raise_right_foot_m: float = 0.0):
    """Fencer standing still side-on, camera 1.2 m above the floor. MediaPipe-style world
    landmarks can carry a wrong right-foot height (what oblique views produce); the image
    always shows the true pose with both feet on the floor."""
    camera = PinholeCamera.from_image(1280, 720)
    skeleton = camera_oriented_skeleton()
    root = np.array([0.0, 0.25, depth])  # hip 0.25 m below a 1.2 m high camera
    image = camera.project(skeleton + root)
    world = as_mediapipe_world(skeleton, (image[lm.LEFT_HIP] + image[lm.RIGHT_HIP]) / 2.0, camera)
    right_foot = [lm.RIGHT_ANKLE, lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX]
    world[right_foot, 1] -= raise_right_foot_m  # camera y points down
    return camera, np.stack([image] * num_frames), np.stack([world] * num_frames)


def test_world_from_floor_normal_matches_pitch_rotation():
    pitch = np.deg2rad(12.0)
    rotation = world_from_camera(pitch)
    assert np.allclose(world_from_floor_normal(rotation[2]), rotation, atol=1e-9)


def test_fit_floor_plane_recovers_tilted_floor_and_camera_height():
    rng = np.random.default_rng(0)
    rotation = world_from_camera(np.deg2rad(10.0))
    num_frames = 80
    world_pts = np.zeros((num_frames, lm.NUM_LANDMARKS, 3))
    world_pts[:, :, 2] = 0.5  # everything else well above the floor
    world_pts[:, lm.FEET, 0] = rng.uniform(-1.5, 1.5, (num_frames, 1))
    world_pts[:, lm.FEET, 1] = rng.uniform(3.0, 6.0, (num_frames, 1))
    world_pts[:, lm.FEET, 2] = rng.uniform(0.0, 0.15, (num_frames, len(lm.FEET)))
    world_pts[:, lm.FEET[0], 2] = 0.0  # one foot point on the floor in every frame
    camera_height = 1.3
    points_cam = (world_pts - [0.0, 0.0, camera_height]) @ rotation  # world -> camera
    normal, height = fit_floor_plane(points_cam, world_from_camera(0.0)[2])
    assert np.allclose(normal, rotation[2], atol=1e-3)
    assert np.isclose(height, camera_height, atol=1e-3)


def test_planted_weights_separate_still_and_moving_feet():
    image = np.zeros((40, lm.NUM_LANDMARKS, 2))
    image[:, lm.LEFT_SHOULDER, 1] = image[:, lm.RIGHT_SHOULDER, 1] = 100.0
    image[:, lm.LEFT_HIP, 1] = image[:, lm.RIGHT_HIP, 1] = 200.0  # torso = 100 px
    for i in (lm.RIGHT_ANKLE, lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX):
        image[:, i, 0] = np.arange(40) * 10.0  # 300 px/s = 3 torso lengths per second
    weights = planted_weights(image, fps=30.0)
    assert np.all(weights[5:-5, 0] > 0.99) and np.all(weights[5:-5, 1] < 0.01)


def test_snap_planted_feet_lowers_only_planted_feet():
    k = np.stack([t_pose_facing_minus_y()] * 10)
    k[:, [lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX], 2] = 0.12
    planted = np.zeros((10, 2))
    planted[:, 1] = 1.0
    out = snap_planted_feet(k, planted, fps=30.0)
    assert np.allclose(out[:, lm.RIGHT_FOOT_INDEX, 2], 0.0, atol=1e-9)
    assert np.allclose(out[:, lm.RIGHT_KNEE, 2], k[:, lm.RIGHT_KNEE, 2] - 0.06, atol=1e-9)
    assert np.allclose(out[:, lm.LEFT_FOOT_INDEX], k[:, lm.LEFT_FOOT_INDEX])


def test_snap_planted_feet_makes_a_planted_foot_flat():
    k = np.stack([t_pose_facing_minus_y()] * 10)
    k[:, lm.LEFT_HEEL, 2] = 0.04  # MediaPipe heel above the toe tip
    k[:, lm.LEFT_FOOT_INDEX, 2] = 0.01
    out = snap_planted_feet(k, np.ones((10, 2)), fps=30.0)
    assert np.allclose(out[:, [lm.LEFT_HEEL, lm.LEFT_FOOT_INDEX], 2], 0.0, atol=1e-9)


def test_snap_planted_feet_removes_a_constant_foot_offset_from_swing_frames_too():
    k = np.stack([t_pose_facing_minus_y()] * 60)
    k[:, [lm.RIGHT_ANKLE, lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX], 2] += 0.08  # far foot too high
    k[40:, [lm.RIGHT_ANKLE, lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX], 2] += 0.10  # then a real step
    planted = np.ones((60, 2))
    planted[40:, 1] = 0.0
    out = snap_planted_feet(k, planted, fps=30.0)
    assert np.allclose(out[45:, lm.RIGHT_HEEL, 2], 0.10, atol=1e-6), "the step keeps its real lift"


def test_anchor_planted_feet_removes_root_jitter_but_keeps_slow_travel():
    num_frames, fps = 120, 30.0
    t = np.arange(num_frames) / fps
    true = np.stack([t_pose_facing_minus_y()] * num_frames)
    true[..., 0] += 0.25 * t[:, None]  # slow travel, 0.25 m/s, both feet "planted" (a slide)
    jitter = 0.03 * np.sin(2 * np.pi * 4.0 * t)  # 4 Hz, 3 cm depth noise
    noisy = true.copy()
    noisy[..., 1] += jitter[:, None]
    out = anchor_planted_feet(noisy, np.ones((num_frames, 2)), fps)
    hip = (out[:, lm.LEFT_HIP] + out[:, lm.RIGHT_HIP]) / 2.0
    true_hip = (true[:, lm.LEFT_HIP] + true[:, lm.RIGHT_HIP]) / 2.0
    inner = slice(15, -15)
    assert np.abs(hip[inner, 1] - true_hip[inner, 1]).max() < 0.008, "jitter removed"
    assert np.abs(hip[inner, 0] - true_hip[inner, 0]).max() < 0.03, "slow travel kept"


def test_anchor_planted_feet_leaves_swinging_feet_alone():
    k = np.stack([t_pose_facing_minus_y()] * 30)
    k[..., 1] += np.linspace(0.0, 0.3, 30)[:, None]
    out = anchor_planted_feet(k, np.zeros((30, 2)), fps=30.0)
    assert np.allclose(out, k)


def test_floor_constraint_puts_a_floating_far_foot_back_on_the_floor():
    camera, image_xy, world = standing_clip(raise_right_foot_m=0.10)
    plain = estimate_root_trajectory(image_xy, world, np.ones(world.shape[:2]), camera, 30.0, use_floor=False)
    floor = estimate_root_trajectory(image_xy, world, np.ones(world.shape[:2]), camera, 30.0)
    right = [lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX]
    assert plain.keypoints_world[:, right, 2].min() > 0.07, "without the floor the right foot hovers"
    assert np.abs(floor.keypoints_world[:, right, 2].min(axis=1)).max() < 0.01
    assert np.isclose(floor.camera_height_m, 1.2, atol=0.1)


# ---- landmarks clean-up ----------------------------------------------------------------


def test_left_right_leg_swap_is_undone():
    base = t_pose_facing_minus_y()[:, :2] * 100 + 500
    frames = np.stack([base + np.array([t, 0.0]) for t in range(6)])
    frames[:, lm.LEFT_ANKLE, 0] -= 40  # make the legs clearly distinct in the image
    frames[:, lm.RIGHT_ANKLE, 0] += 40
    corrupted = frames.copy()
    for a, b in lm.LEG_PAIRS:
        corrupted[3, [a, b]] = frames[3, [b, a]]
    (fixed,), swapped = lm.fix_left_right_swaps(corrupted, [corrupted], np.full(6, 150.0))
    assert np.allclose(fixed, frames)
    assert swapped[3, 0] and not swapped[:, 1].any()


def test_resample_uniform_handles_irregular_timestamps():
    times = np.array([0.0, 0.04, 0.07, 0.11])
    values = times[:, None] * 10.0
    out = lm.resample_uniform(times, values, np.array([0.0, 1 / 30, 2 / 30]))
    assert np.allclose(out[:, 0], [0.0, 10 / 30, 20 / 30])


def test_fill_gaps_interpolates_missing_frames():
    values = np.array([[0.0], [np.nan], [2.0]])
    out = lm.fill_gaps(values, np.array([True, False, True]))
    assert np.allclose(out[:, 0], [0.0, 1.0, 2.0])


# ---- handedness --------------------------------------------------------------------------


def en_garde(right_handed: bool) -> np.ndarray:
    """Side-on en garde on the +x piste: front foot along +x, back foot turned out (+y)."""
    k = t_pose_facing_minus_y()
    front, back = ("RIGHT", "LEFT") if right_handed else ("LEFT", "RIGHT")
    g = lambda side, name: getattr(lm, f"{side}_{name}")  # noqa: E731
    k[g(front, "ANKLE")] = [0.35, 0.0, 0.08]
    k[g(front, "HEEL")] = [0.30, 0.0, 0.02]
    k[g(front, "FOOT_INDEX")] = [0.50, 0.0, 0.02]
    k[g(back, "ANKLE")] = [-0.35, 0.0, 0.08]
    k[g(back, "HEEL")] = [-0.35, -0.05, 0.02]
    k[g(back, "FOOT_INDEX")] = [-0.35, 0.15, 0.02]
    # Sword arm extended forward, back arm bent at the hip.
    k[g(front, "WRIST")] = k[g(front, "SHOULDER")] + [0.55, 0.0, -0.05]
    k[g(front, "ELBOW")] = k[g(front, "SHOULDER")] + [0.28, 0.0, -0.05]
    k[g(back, "ELBOW")] = k[g(back, "SHOULDER")] + [-0.1, 0.0, -0.25]
    k[g(back, "WRIST")] = k[g(back, "SHOULDER")] + [0.0, 0.0, -0.35]
    return np.stack([k] * 30)


def test_handedness_from_en_garde_feet_and_arms():
    assert guess_handedness(en_garde(right_handed=True)) == "R"
    assert guess_handedness(en_garde(right_handed=False)) == "L"


def test_handedness_is_none_for_a_neutral_standing_pose():
    assert guess_handedness(np.stack([t_pose_facing_minus_y()] * 30)) is None
