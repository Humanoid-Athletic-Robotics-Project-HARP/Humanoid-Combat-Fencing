"""Damped least-squares IK on one G1 leg, used to pin a planted foot.

Only the six joints of that leg move, so the pelvis, torso and arms keep exactly the
trajectory the human made; the foot's orientation is held while its position is moved.
"""

from __future__ import annotations

import mujoco
import numpy as np

from fencing.motion.kinematics import FOOT_SIDES, G1Model, foot_point_world


class LegIK:
    def __init__(self, g1: G1Model, damping: float = 0.03, rotation_weight: float = 0.3):
        self.g1 = g1
        self.data = mujoco.MjData(g1.model)
        self.damping = damping
        self.rotation_weight = rotation_weight
        self._damping_matrix = damping**2 * np.eye(6)
        self.jac_pos = np.zeros((3, g1.model.nv))
        self.jac_rot = np.zeros((3, g1.model.nv))
        self.leg_qpos = {}
        self.leg_dofs = {}
        self.leg_lower = {}
        self.leg_upper = {}
        for side in FOOT_SIDES:
            cols = g1.leg_columns[side]
            self.leg_qpos[side] = g1.qpos_index[cols]
            self.leg_dofs[side] = g1.dof_index[cols]
            self.leg_lower[side] = g1.joint_lower[cols]
            self.leg_upper[side] = g1.joint_upper[cols]

    def _kinematics(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.g1.model, self.data)
        mujoco.mj_comPos(self.g1.model, self.data)

    def foot_pose(self, qpos: np.ndarray, foot_index: int) -> tuple[np.ndarray, np.ndarray]:
        self._kinematics(qpos)
        bid = self.g1.foot_body_ids[foot_index]
        return foot_point_world(self.g1, self.data, foot_index), self.data.xmat[bid].reshape(3, 3).copy()

    def solve(
        self,
        qpos: np.ndarray,
        foot_index: int,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        iterations: int = 20,
        tolerance_m: float = 2e-4,
    ) -> tuple[np.ndarray, float]:
        """Returns (new qpos, final position error in m). qpos is not modified."""
        side = FOOT_SIDES[foot_index]
        qpos = qpos.copy()
        qidx, didx = self.leg_qpos[side], self.leg_dofs[side]
        lower, upper = self.leg_lower[side], self.leg_upper[side]
        bid = int(self.g1.foot_body_ids[foot_index])
        error_norm = np.inf
        for _ in range(iterations):
            self._kinematics(qpos)
            point = foot_point_world(self.g1, self.data, foot_index)
            pos_error = target_pos - point
            current_rot = self.data.xmat[bid].reshape(3, 3)
            # Small-angle orientation error (world frame); exact enough for the few-degree
            # corrections asked of it, and ~50x cheaper than a matrix log.
            delta = target_rot @ current_rot.T
            rot_error = 0.5 * np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
            error_norm = float(np.sqrt(pos_error @ pos_error))
            if error_norm < tolerance_m and rot_error @ rot_error < 1e-6:
                break
            mujoco.mj_jac(self.g1.model, self.data, self.jac_pos, self.jac_rot, point, bid)
            jac = np.vstack([self.jac_pos[:, didx], self.rotation_weight * self.jac_rot[:, didx]])
            err = np.concatenate([pos_error, self.rotation_weight * rot_error])
            # DLS: well-behaved near the straight-knee singularity a deep lunge reaches.
            step = jac.T @ np.linalg.solve(jac @ jac.T + self._damping_matrix, err)
            qpos[qidx] = np.minimum(np.maximum(qpos[qidx] + step, lower), upper)
        return qpos, error_norm
