"""Global root trajectory from a monocular, static camera, without any body model.

MediaPipe gives two things per frame: 2D landmarks in the image and "world" landmarks in
metres, centred on the hip midpoint and rotated like the camera (x right, y down, z away).
If the world landmarks are scaled to a known body height, the camera-frame position of the
hip centre is the translation that projects them onto the 2D landmarks through a pinhole
camera. That translation solve is linear (see `solve_root_translation`), so depth comes out
of the apparent body size for free.

Depth from body size is noisy (a 2% size error at 5 m is 10 cm) and MediaPipe's relative
depth between the two feet is often wrong in oblique views, which leaves one foot floating
10+ cm above the floor. So the size-based solve is only the first pass. The floor is then
fitted as a plane through the lowest foot point of every frame (static camera: one plane per
clip), and every foot that is planted -- still in the image -- is slid along its own camera
ray onto that plane (`apply_floor_constraint`). This keeps the 2D projection exact, gives the
planted foot a depth good to ~1-2 cm per pixel, and the root follows the planted feet.

MediaPipe also orients the world landmarks as if the fencer stood straight ahead of the
camera: on real side-view clips the body's yaw follows the viewing-ray angle with slope ~1
(correlation 0.95), so a fencer crossing the frame turned by 20-35 deg that never happened.
`ray_rotations` undoes that before anything else (`ray_correct`).

Assumptions (all configurable at the CLI): the fencer's height (default 1.75 m), the camera's
horizontal field of view (default: GVHMR's own guess, focal = image diagonal), a static camera,
a flat floor, and that at least one foot is on the floor in every frame (true for fencing
footwork; a balestra's short flight is flattened).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from fencing.pose import landmarks as lm
from fencing.pose.landmarks import smooth_time

DEFAULT_HUMAN_HEIGHT_M = 1.75
# Shin + thigh + hip-to-shoulder as a fraction of standing height (Drillis & Contini segment
# ratios: knee 0.285 H, hip 0.530 H, shoulder 0.818 H, ankle 0.039 H).
LEG_TORSO_FRACTION_OF_HEIGHT = (0.285 - 0.039) + (0.530 - 0.285) + (0.818 - 0.530)
# Pitch outside this range means the torso estimate is garbage, not a real camera angle.
MAX_CAMERA_PITCH_DOWN_RAD = np.deg2rad(35.0)
MAX_CAMERA_PITCH_UP_RAD = np.deg2rad(15.0)


@dataclass(frozen=True)
class PinholeCamera:
    focal_px: float
    cx: float
    cy: float

    @classmethod
    def from_image(cls, width: int, height: int, hfov_deg: float | None = None) -> PinholeCamera:
        if width <= 0 or height <= 0:
            raise ValueError(f"image size must be positive, got {width}x{height}")
        if hfov_deg is None:
            # Same default as GVHMR's estimate_K (hmr4d/utils/geo/hmr_cam.py): focal = diagonal.
            focal = float(np.hypot(width, height))
        else:
            if not 5.0 < hfov_deg < 170.0:
                raise ValueError(f"hfov_deg must be in (5, 170), got {hfov_deg}")
            focal = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(focal_px=focal, cx=width / 2.0, cy=height / 2.0)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        z = points_cam[..., 2:3]
        return np.concatenate(
            [self.focal_px * points_cam[..., 0:1] / z + self.cx, self.focal_px * points_cam[..., 1:2] / z + self.cy],
            axis=-1,
        )


def ray_rotations(pixels: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    """(T, 3, 3) rotations taking a camera that looks straight at each pixel to the real one.

    The smallest rotation that maps the optical axis (0, 0, 1) onto the pixel's viewing ray.
    MediaPipe's world landmarks live in that "looking straight at the person" frame, so
    `R @ landmark` puts them in camera axes.
    """
    rays = np.stack(
        [(pixels[:, 0] - camera.cx) / camera.focal_px, (pixels[:, 1] - camera.cy) / camera.focal_px, np.ones(len(pixels))],
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    axis = np.stack([-rays[:, 1], rays[:, 0], np.zeros(len(rays))], axis=-1)  # z x ray
    sin = np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.arctan2(sin, rays[:, 2:3])
    return Rotation.from_rotvec(axis / np.maximum(sin, 1e-12) * angle).as_matrix()


def leg_torso_length(world: np.ndarray) -> np.ndarray:
    """Per-frame shin + thigh (mean of both legs) + hip-mid to shoulder-mid, (T,)."""

    def seg(a: int, b: int) -> np.ndarray:
        return np.linalg.norm(world[:, a] - world[:, b], axis=-1)

    shin = (seg(lm.LEFT_KNEE, lm.LEFT_ANKLE) + seg(lm.RIGHT_KNEE, lm.RIGHT_ANKLE)) / 2.0
    thigh = (seg(lm.LEFT_HIP, lm.LEFT_KNEE) + seg(lm.RIGHT_HIP, lm.RIGHT_KNEE)) / 2.0
    mid_hip = (world[:, lm.LEFT_HIP] + world[:, lm.RIGHT_HIP]) / 2.0
    mid_shoulder = (world[:, lm.LEFT_SHOULDER] + world[:, lm.RIGHT_SHOULDER]) / 2.0
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)
    return shin + thigh + torso


def body_scale_factor(world: np.ndarray, human_height_m: float) -> float:
    """One scale for the whole clip that makes the skeleton `human_height_m` tall.

    One factor per clip (median over frames), not per frame: bone lengths do not change, and
    per-frame scaling would turn MediaPipe's depth noise into limb-length noise.
    """
    if human_height_m <= 0:
        raise ValueError(f"human height must be positive, got {human_height_m}")
    measured = float(np.median(leg_torso_length(world)))
    if measured <= 1e-6:
        raise ValueError("degenerate skeleton: leg + torso length is zero")
    return LEG_TORSO_FRACTION_OF_HEIGHT * human_height_m / measured


def solve_root_translation(
    image_xy: np.ndarray,
    world_rel: np.ndarray,
    weights: np.ndarray,
    camera: PinholeCamera,
    huber_px: float = 8.0,
) -> np.ndarray:
    """Camera-frame position of the world-landmark origin for one frame, shape (3,).

    For point i with camera position T + x_i and pixel u_i: (u_i - cx)(Z + z_i) = f(X + x_i),
    which is linear in T = (X, Y, Z). Same for v. Weighted least squares, then one Huber
    reweighting pass so a single bad landmark (a hand behind the body) cannot drag depth.
    """
    du = image_xy[:, 0] - camera.cx
    dv = image_xy[:, 1] - camera.cy
    f = camera.focal_px
    x, y, z = world_rel[:, 0], world_rel[:, 1], world_rel[:, 2]
    n = len(du)
    a = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    a[:n, 0], a[:n, 2], b[:n] = f, -du, du * z - f * x
    a[n:, 1], a[n:, 2], b[n:] = f, -dv, dv * z - f * y
    w = np.concatenate([weights, weights]).astype(np.float64)
    if np.count_nonzero(w > 0) < 6:
        raise ValueError("need at least 3 weighted landmarks to solve the root translation")
    solution = np.zeros(3)
    for _ in range(2):
        sw = np.sqrt(w)
        solution, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
        # Residuals are in units of f * metres; divide by Z to get pixels for the Huber cut.
        residual_px = np.abs(a @ solution - b) / max(solution[2], 1e-3)
        w = w * np.minimum(1.0, huber_px / np.maximum(residual_px, 1e-9))
    return solution


def estimate_camera_pitch(world_cam_oriented: np.ndarray) -> float:
    """Downward camera pitch (rad) from the median torso direction in camera axes.

    Fencers' torsos are close to vertical over a clip. Only the y-z (pitch) component is used:
    a side-on fencer's forward lean shows up in the image plane (x-y) and must not tilt the
    world. Roll is assumed zero.
    """
    mid_hip = (world_cam_oriented[:, lm.LEFT_HIP] + world_cam_oriented[:, lm.RIGHT_HIP]) / 2.0
    mid_sh = (world_cam_oriented[:, lm.LEFT_SHOULDER] + world_cam_oriented[:, lm.RIGHT_SHOULDER]) / 2.0
    up = mid_sh - mid_hip
    up_y, up_z = float(np.median(up[:, 1])), float(np.median(up[:, 2]))
    # Camera pitched down by theta sees world-up as (0, -cos theta, -sin theta).
    pitch = float(np.arctan2(-up_z, -up_y))
    return float(np.clip(pitch, -MAX_CAMERA_PITCH_UP_RAD, MAX_CAMERA_PITCH_DOWN_RAD))


def world_from_camera(pitch_down_rad: float) -> np.ndarray:
    """Rotation taking camera axes (x right, y down, z forward) to a z-up world whose x is the
    image's right and whose y points away from the camera along the floor."""
    c, s = np.cos(pitch_down_rad), np.sin(pitch_down_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, -s, c], [0.0, -c, -s]])


@dataclass
class RootTrajectoryResult:
    keypoints_world: np.ndarray  # (T, 33, 3) z-up metres, feet on z = 0, start at x = y = 0
    root_cam: np.ndarray  # (T, 3) hip centre in camera coordinates, smoothed
    scale: float
    pitch_down_rad: float
    camera: PinholeCamera
    floor_normal_cam: np.ndarray | None = None  # (3,) unit "up" in camera axes, None = torso pitch only
    camera_height_m: float | None = None
    planted: np.ndarray | None = None  # (T, 2) left/right planted weight in [0, 1]


def _smooth_root(root_cam: np.ndarray, fps: float) -> np.ndarray:
    # Depth from body size is the noisiest axis (a 2% size error at 5 m is 10 cm), so it gets
    # an outlier-rejecting median and a longer window than the image-plane axes.
    out = root_cam.copy()
    med = max(3, int(round(0.2 * fps)) | 1)
    out[:, 2] = median_filter(out[:, 2], size=med, mode="nearest")
    out[:, :2] = smooth_time(out[:, :2], fps, window_s=0.2)
    out[:, 2] = smooth_time(out[:, 2], fps, window_s=0.6)
    return out


def ground_and_center(keypoints_world: np.ndarray, fps: float) -> np.ndarray:
    """Put the lowest foot point on z = 0 in every frame and start the root at x = y = 0."""
    out = keypoints_world.copy()
    lowest = out[:, lm.FEET, 2].min(axis=1)
    lowest = smooth_time(lowest, fps, window_s=0.15)
    out[..., 2] -= lowest[:, None]
    mid_hip0 = (out[0, lm.LEFT_HIP] + out[0, lm.RIGHT_HIP]) / 2.0
    out[..., 0] -= mid_hip0[0]
    out[..., 1] -= mid_hip0[1]
    return out


def estimate_root_trajectory(
    image_xy: np.ndarray,
    world: np.ndarray,
    visibility: np.ndarray,
    camera: PinholeCamera,
    fps: float,
    human_height_m: float = DEFAULT_HUMAN_HEIGHT_M,
    pitch_down_rad: float | None = None,
    use_floor: bool = True,
    ray_correct: bool = True,
) -> RootTrajectoryResult:
    """Lift per-frame landmarks into one consistent z-up world trajectory.

    image_xy: (T, 33, 2) full-image pixels. world: (T, 33, 3) MediaPipe world landmarks
    (hip-centred, oriented as seen along the ray to the hips). visibility: (T, 33) in [0, 1].
    ray_correct: rotate the landmarks from that ray frame into camera axes (see module doc).
    """
    if image_xy.shape[:2] != world.shape[:2] or visibility.shape != world.shape[:2]:
        raise ValueError(
            f"shape mismatch: image {image_xy.shape}, world {world.shape}, visibility {visibility.shape}"
        )
    scale = body_scale_factor(world, human_height_m)
    world_scaled = world * scale
    if ray_correct:
        hip_px = smooth_time((image_xy[:, lm.LEFT_HIP] + image_xy[:, lm.RIGHT_HIP]) / 2.0, fps, window_s=0.2)
        world_scaled = np.einsum("tij,tkj->tki", ray_rotations(hip_px, camera), world_scaled)
    weights = np.clip(visibility, 0.0, 1.0)[:, lm.BODY]
    root_cam = np.stack(
        [
            solve_root_translation(image_xy[t, lm.BODY], world_scaled[t, lm.BODY], weights[t], camera)
            for t in range(len(world))
        ]
    )
    root_cam = _smooth_root(root_cam, fps)
    pitch_given = pitch_down_rad is not None
    if pitch_down_rad is None:
        pitch_down_rad = estimate_camera_pitch(world_scaled)
    rotation = world_from_camera(pitch_down_rad)
    points_cam = world_scaled + root_cam[:, None, :]
    floor_normal = camera_height = planted = None
    if use_floor:
        # A pitch given by the user fixes "up"; then only the camera height is fitted.
        floor = fit_floor_plane(points_cam, rotation[2], fit_normal=not pitch_given)
        if floor is not None:
            floor_normal, camera_height = floor
            planted = planted_weights(image_xy, fps)
            points_cam = apply_floor_constraint(points_cam, image_xy, camera, floor_normal, camera_height, planted, fps)
            rotation = world_from_floor_normal(floor_normal)
            pitch_down_rad = float(np.arcsin(np.clip(-floor_normal[2], -1.0, 1.0)))
            root_cam = (points_cam[:, lm.LEFT_HIP] + points_cam[:, lm.RIGHT_HIP]) / 2.0
    keypoints_world = points_cam @ rotation.T
    if planted is not None:
        # Put the fitted floor at z = 0 before snapping planted feet onto it.
        keypoints_world[..., 2] += camera_height
        keypoints_world = snap_planted_feet(keypoints_world, planted, fps)
        keypoints_world = anchor_planted_feet(keypoints_world, planted, fps)
    keypoints_world = ground_and_center(keypoints_world, fps)
    return RootTrajectoryResult(
        keypoints_world, root_cam, scale, pitch_down_rad, camera, floor_normal, camera_height, planted
    )


# ---- floor plane -----------------------------------------------------------------------------

FOOT_POINTS = ((lm.LEFT_ANKLE, lm.LEFT_HEEL, lm.LEFT_FOOT_INDEX), (lm.RIGHT_ANKLE, lm.RIGHT_HEEL, lm.RIGHT_FOOT_INDEX))
MIN_CAMERA_HEIGHT_M = 0.2
MAX_CAMERA_HEIGHT_M = 6.0
# The fitted floor may differ from the torso-based pitch by this much before we distrust it.
MAX_FLOOR_TILT_FROM_TORSO_RAD = np.deg2rad(20.0)
# Planted = the foot's contact point is still in the image, measured in torso lengths / s.
PLANTED_SPEED_TORSO_PER_S = 0.6
SWING_SPEED_TORSO_PER_S = 1.8
# A floor ray this close to horizontal gives a useless depth (camera at floor level).
MIN_RAY_DOWN_COMPONENT = 0.05
MAX_DEPTH_CORRECTION_FRACTION = 0.25
ROOT_FLOOR_WINDOW_S = 1.5
FLOOR_SLOPE_PRIOR_SPREAD_M = 0.25
ANCHOR_WINDOW_S = 2.0
MIN_PLANTED_FRAMES_FOR_BIAS_S = 1.0
KNEE_SHARE_OF_FOOT_CORRECTION = 0.5


def fit_floor_plane(
    points_cam: np.ndarray, normal_init: np.ndarray, fit_normal: bool = True, iterations: int = 5
) -> tuple[np.ndarray, float] | None:
    """Floor plane `n . X + h = 0` (n unit, pointing up; h = camera height) in camera axes.

    Fitted to the lowest foot point of every frame (at least one foot is down) with an
    asymmetric robust weight: points far above the current plane are lifted feet, not floor.
    fit_normal=False keeps `normal_init` and fits only h. Returns None if the fit is
    implausible (then the caller keeps the torso pitch).
    """
    feet = points_cam[:, lm.FEET]
    n0 = np.asarray(normal_init, dtype=np.float64) / np.linalg.norm(normal_init)
    n, h = n0, None
    for _ in range(iterations):
        lowest = feet[np.arange(len(feet)), np.argmin(feet @ n, axis=1)]
        if h is None:
            weights = np.ones(len(lowest))
        else:
            above = lowest @ n + h
            weights = np.where(above > 0.03, (0.03 / np.maximum(above, 1e-9)) ** 2, 1.0)
        if not fit_normal:
            h = -float(np.sum(weights * (lowest @ n)) / np.sum(weights))
            continue
        # n . X + h = 0 written as z' = a x' + b y' + c in a frame whose z is the current n.
        basis = world_from_floor_normal(n)
        local = lowest @ basis.T
        # Ridge rows pull the slopes towards the torso prior when the feet barely move (a
        # fencer drilling on the spot does not span the floor).
        centred = local[:, :2] - np.average(local[:, :2], axis=0, weights=weights)
        a = np.c_[centred, np.ones(len(local))]
        sw = np.sqrt(weights)
        ridge = np.sqrt(FLOOR_SLOPE_PRIOR_SPREAD_M**2 * weights.sum()) * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coef, *_ = np.linalg.lstsq(
            np.vstack([a * sw[:, None], ridge]), np.concatenate([local[:, 2] * sw, [0.0, 0.0]]), rcond=None
        )
        offset = np.average(local[:, :2], axis=0, weights=weights)
        coef[2] -= coef[0] * offset[0] + coef[1] * offset[1]
        up_local = np.array([-coef[0], -coef[1], 1.0])
        up_local /= np.linalg.norm(up_local)
        n = basis.T @ up_local
        h = -float(coef[2] * up_local[2])
    if not (MIN_CAMERA_HEIGHT_M <= h <= MAX_CAMERA_HEIGHT_M):
        return None
    if np.arccos(np.clip(n @ n0, -1.0, 1.0)) > MAX_FLOOR_TILT_FROM_TORSO_RAD:
        return None
    return n, h


def world_from_floor_normal(normal_cam: np.ndarray) -> np.ndarray:
    """Rotation camera -> z-up world with z along `normal_cam` and x along the image's right.

    Same as world_from_camera(pitch) when the normal has no roll component."""
    z = normal_cam / np.linalg.norm(normal_cam)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def planted_weights(image_xy: np.ndarray, fps: float) -> np.ndarray:
    """(T, 2) how planted each foot is, 1 = still in the image, 0 = swinging.

    Image speed of the foot (mean of ankle, heel, toe), in torso lengths per second so it is
    independent of distance to the camera; ramps linearly between the two thresholds.
    """
    mid_hip = (image_xy[:, lm.LEFT_HIP] + image_xy[:, lm.RIGHT_HIP]) / 2.0
    mid_sh = (image_xy[:, lm.LEFT_SHOULDER] + image_xy[:, lm.RIGHT_SHOULDER]) / 2.0
    torso_px = np.maximum(np.median(np.linalg.norm(mid_sh - mid_hip, axis=-1)), 1.0)
    out = np.zeros((len(image_xy), 2))
    for side, ids in enumerate(FOOT_POINTS):
        foot = smooth_time(image_xy[:, list(ids)].mean(axis=1), fps, window_s=0.2)
        speed = np.linalg.norm(np.gradient(foot, 1.0 / fps, axis=0), axis=-1) / torso_px
        ramp = (SWING_SPEED_TORSO_PER_S - speed) / (SWING_SPEED_TORSO_PER_S - PLANTED_SPEED_TORSO_PER_S)
        out[:, side] = np.clip(smooth_time(ramp, fps, window_s=0.2), 0.0, 1.0)
    return out


def _ray_floor_depth(pixel: np.ndarray, camera: PinholeCamera, normal: np.ndarray, height: float) -> np.ndarray:
    """Depth (camera z) where the ray through each pixel meets the floor; NaN if it never does."""
    rays = np.stack(
        [(pixel[..., 0] - camera.cx) / camera.focal_px, (pixel[..., 1] - camera.cy) / camera.focal_px, np.ones(pixel.shape[:-1])],
        axis=-1,
    )
    down = rays @ normal / np.linalg.norm(rays, axis=-1)
    depth = -height / (rays @ normal)
    return np.where(down < -MIN_RAY_DOWN_COMPONENT, depth, np.nan)


def apply_floor_constraint(
    points_cam: np.ndarray,
    image_xy: np.ndarray,
    camera: PinholeCamera,
    normal: np.ndarray,
    height: float,
    planted: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Correct the slow part of the root depth error with the floor, keeping every projection.

    For each planted foot, the ray through the heel-toe midpoint pixel meets the floor at a
    depth; the ratio of that depth to the size-based one says how far the body should move
    along the camera rays (scaling about the camera keeps every pixel fixed). The ratio is
    noisy frame to frame (1 px of foot jitter is 2-4 cm of depth for a waist-high camera), so
    only its low-frequency part (ROOT_FLOOR_WINDOW_S) is applied: it removes bias and drift,
    while the size-based solve keeps the frame-to-frame detail.
    """
    num_frames = len(points_cam)
    frames = np.arange(num_frames)
    ratio = np.ones((num_frames, 2))
    ok = np.zeros((num_frames, 2), dtype=bool)
    for side, (_, heel, toe) in enumerate(FOOT_POINTS):
        pixel = (image_xy[:, heel] + image_xy[:, toe]) / 2.0
        est = (points_cam[:, heel, 2] + points_cam[:, toe, 2]) / 2.0
        floor_depth = _ray_floor_depth(pixel, camera, normal, height)
        r = floor_depth / est
        ok[:, side] = np.isfinite(r) & (np.abs(r - 1.0) < MAX_DEPTH_CORRECTION_FRACTION)
        ratio[:, side] = np.where(ok[:, side], r, 1.0)
    weights = np.where(ok, planted, 0.0)
    total = weights.sum(axis=1)
    has_support = total > 0.2
    if has_support.sum() < 2:
        return points_cam.copy()
    root_ratio = (weights * ratio).sum(axis=1)[has_support] / total[has_support]
    root_ratio = np.interp(frames, frames[has_support], root_ratio)
    root_ratio = median_filter(root_ratio, size=max(3, int(round(0.3 * fps)) | 1), mode="nearest")
    root_ratio = smooth_time(root_ratio, fps, window_s=ROOT_FLOOR_WINDOW_S, polyorder=1)
    return points_cam * root_ratio[:, None, None]


def snap_planted_feet(keypoints_world: np.ndarray, planted: np.ndarray, fps: float) -> np.ndarray:
    """Put each planted foot flat on z = 0 (world, floor already at 0), knee by half as much.

    MediaPipe's depth for the far foot is often off by 10+ cm in oblique views, which after
    the lift shows up as that foot hovering, and its heel sits a few cm above its toe tip on
    a flat foot. A planted foot is on the floor by definition: first each foot's median
    height while well planted is removed from all its frames, then heel and toe of planted
    frames are lowered separately; the ankle follows their mean.
    """
    out = keypoints_world.copy()
    for side, (ankle, heel, toe) in enumerate(FOOT_POINTS):
        knee = lm.LEFT_KNEE if side == 0 else lm.RIGHT_KNEE
        # MediaPipe's error for one foot is largely a constant offset over a clip (the far
        # foot sits too high or too low), so it is removed from every frame, swings included.
        well_planted = planted[:, side] > 0.95
        if well_planted.sum() >= MIN_PLANTED_FRAMES_FOR_BIAS_S * fps:
            bias = float(np.median((out[well_planted, heel, 2] + out[well_planted, toe, 2]) / 2.0))
            out[:, [ankle, heel, toe], 2] -= bias
            out[:, knee, 2] -= KNEE_SHARE_OF_FOOT_CORRECTION * bias
        heel_drop = smooth_time(planted[:, side] * out[:, heel, 2], fps, window_s=0.15)
        toe_drop = smooth_time(planted[:, side] * out[:, toe, 2], fps, window_s=0.15)
        ankle_drop = (heel_drop + toe_drop) / 2.0
        out[:, heel, 2] -= heel_drop
        out[:, toe, 2] -= toe_drop
        out[:, ankle, 2] -= ankle_drop
        out[:, knee, 2] -= KNEE_SHARE_OF_FOOT_CORRECTION * ankle_drop
    return out


def anchor_planted_feet(keypoints_world: np.ndarray, planted: np.ndarray, fps: float) -> np.ndarray:
    """Remove the fast part of the root's horizontal error by pinning planted feet in place.

    A planted foot does not move over the floor, so its frame-to-frame horizontal motion in
    the lifted world is root error (mostly depth noise from the size-based solve). That
    motion is subtracted from the whole body and the correction integrated; its slow part
    (ANCHOR_WINDOW_S) is dropped again, so drift over seconds stays with the image-based
    root and a misdetected contact cannot drag the body away. Frames with no planted foot
    (a lunge's flight, a jump) keep the image-based motion.
    """
    feet = np.stack([keypoints_world[:, [heel, toe], :2].mean(axis=1) for _, heel, toe in FOOT_POINTS], axis=1)
    step = np.diff(feet, axis=0)  # (T-1, 2 feet, 2)
    weight = np.minimum(planted[1:], planted[:-1])
    total = weight.sum(axis=1)
    mean_step = (weight[..., None] * step).sum(axis=1) / np.maximum(total, 1e-9)[:, None]
    delta = -np.clip(total, 0.0, 1.0)[:, None] * mean_step
    correction = np.vstack([np.zeros((1, 2)), np.cumsum(delta, axis=0)])
    correction -= smooth_time(correction, fps, window_s=ANCHOR_WINDOW_S, polyorder=1)
    out = keypoints_world.copy()
    out[..., :2] += correction[:, None, :]
    return out
