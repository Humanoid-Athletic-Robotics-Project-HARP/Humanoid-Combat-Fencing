"""Guess which hand holds the sabre from body keypoints.

Two cues, both from en garde:
  * feet: the front foot is on the sword-arm side, and it points along the line from the back
    foot to the front foot, while the back foot is turned out across that line;
  * arms: the sword arm is held more extended than the back arm, which rests near the hip.
Only frames with a fencing-width stance vote. Returns None when the cues disagree or are weak
rather than a coin flip, so a downstream mirror step never acts on a guess.
"""

from __future__ import annotations

import numpy as np

from fencing.pose import landmarks as lm

MIN_STANCE_WIDTH_M = 0.35
ARM_CUE_WEIGHT = 0.5
MIN_CONFIDENT_SCORE = 0.15
MIN_VOTING_FRAMES = 10


def _horizontal_unit(v: np.ndarray) -> np.ndarray:
    h = v.copy()
    h[..., 2] = 0.0
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)


def handedness_score(keypoints_world: np.ndarray) -> tuple[float, int]:
    """Positive = right-handed, negative = left-handed. Also returns the number of voting frames.

    keypoints_world: (T, 33, 3) MediaPipe layout, z-up metres.
    """
    k = keypoints_world
    left_ankle, right_ankle = k[:, lm.LEFT_ANKLE], k[:, lm.RIGHT_ANKLE]
    stance = right_ankle - left_ankle
    stance_width = np.linalg.norm(stance[:, :2], axis=-1)
    voting = stance_width > MIN_STANCE_WIDTH_M
    if voting.sum() < MIN_VOTING_FRAMES:
        return 0.0, int(voting.sum())

    left_to_right = _horizontal_unit(stance)
    right_foot_dir = _horizontal_unit(k[:, lm.RIGHT_FOOT_INDEX] - k[:, lm.RIGHT_HEEL])
    left_foot_dir = _horizontal_unit(k[:, lm.LEFT_FOOT_INDEX] - k[:, lm.LEFT_HEEL])
    right_front = np.sum(right_foot_dir * left_to_right, axis=-1)
    left_front = np.sum(left_foot_dir * -left_to_right, axis=-1)
    foot_vote = (right_front - left_front) / 2.0

    def reach(shoulder: int, elbow: int, wrist: int) -> np.ndarray:
        arm_len = np.linalg.norm(k[:, elbow] - k[:, shoulder], axis=-1) + np.linalg.norm(
            k[:, wrist] - k[:, elbow], axis=-1
        )
        return np.linalg.norm(k[:, wrist] - k[:, shoulder], axis=-1) / np.maximum(arm_len, 1e-9)

    arm_vote = reach(lm.RIGHT_SHOULDER, lm.RIGHT_ELBOW, lm.RIGHT_WRIST) - reach(
        lm.LEFT_SHOULDER, lm.LEFT_ELBOW, lm.LEFT_WRIST
    )
    score = float(np.mean(foot_vote[voting]) + ARM_CUE_WEIGHT * np.mean(arm_vote[voting]))
    return score, int(voting.sum())


def guess_handedness(keypoints_world: np.ndarray) -> str | None:
    """'R', 'L' or None (not enough en garde frames, or the cues disagree)."""
    score, frames = handedness_score(keypoints_world)
    if frames < MIN_VOTING_FRAMES or abs(score) < MIN_CONFIDENT_SCORE:
        return None
    return "R" if score > 0 else "L"
