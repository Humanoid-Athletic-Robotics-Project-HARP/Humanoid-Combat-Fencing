from functools import lru_cache

import numpy as np

from fencing.motion.normalize import mirror_motion, rotate_about_z
from fencing.motion.segment import best_segment, find_segments, hold_pose
from fencing.motion.synthetic import add_jitter, synthesize

SCRIPT = [
    ("en_garde", 1.0), ("advance", 0.6), ("en_garde", 0.6), ("lunge", 1.2), ("en_garde", 0.6),
    ("retreat", 0.6), ("en_garde", 0.5), ("advance", 0.6), ("lunge", 1.2), ("en_garde", 1.0),
]


@lru_cache(maxsize=1)
def session():
    motion, spans = synthesize(SCRIPT)
    # Arbitrary camera heading + noise: segmentation must work in any world frame.
    return add_jitter(rotate_about_z(motion, 1.0, np.zeros(2)), 0.01, 0.003), spans


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / (b[1] - b[0])


def _expected_actions(spans):
    """Ground truth actions, with an advance directly followed by a lunge merged."""
    actions = [s for s in spans if s[2] != "en_garde"]
    merged, i = [], 0
    while i < len(actions):
        if actions[i][2] == "advance" and i + 1 < len(actions) and actions[i + 1][2] == "lunge" and actions[i][1] == actions[i + 1][0]:
            merged.append((actions[i][0], actions[i + 1][1], "advance_lunge"))
            i += 2
        else:
            merged.append(actions[i])
            i += 1
    return merged


def test_finds_and_labels_every_synthetic_action():
    motion, spans = session()
    segments, feats = find_segments(motion)
    actions = [s for s in segments if s.label != "en_garde"]
    expected = _expected_actions(spans)
    assert [s.label for s in actions] == [e[2] for e in expected]
    for seg, exp in zip(actions, expected):
        assert _overlap((seg.start, seg.end), exp) > 0.9, (seg, exp)
        assert seg.confidence >= 0.5
    assert feats.hand == "R"


def test_segments_start_and_end_in_holds():
    motion, spans = session()
    segments, _ = find_segments(motion)
    lunge = best_segment(segments, "lunge")
    truth = next(s for s in spans if s[2] == "lunge")
    assert lunge.start < truth[0] and lunge.end > truth[1]  # padded into the en garde holds


def test_left_handed_session_is_segmented_the_same():
    motion, spans = session()
    segments, feats = find_segments(mirror_motion(motion))
    assert feats.hand == "L"
    assert [s.label for s in segments if s.label != "en_garde"] == [e[2] for e in _expected_actions(spans)]


def test_en_garde_holds_are_reported():
    motion, _ = session()
    segments, _ = find_segments(motion)
    holds = [s for s in segments if s.label == "en_garde"]
    assert holds and all((s.end - s.start) / motion.fps >= 0.8 for s in holds)


def test_hold_pose_adds_static_ends_and_eases_the_junction():
    motion, _ = synthesize([("advance", 0.6)])
    motion.joint_pos[:, 0] = np.linspace(0.0, 0.5, motion.n_frames)  # constant velocity from frame 0
    step = 0.5 / (motion.n_frames - 1)
    held, _ = hold_pose(motion, 0.5, 0.5, ease_s=0.2)
    assert held.n_frames == motion.n_frames + 30
    assert np.abs(np.diff(held.joint_pos[:15], axis=0)).max() == 0
    assert np.abs(np.diff(held.joint_pos[-15:], axis=0)).max() == 0
    # without easing the first step after the hold would be the full `step`
    assert abs(held.joint_pos[16, 0] - held.joint_pos[15, 0]) < 0.3 * step


DRILL = [("en_garde", 1.0)] + [("retreat", 0.45)] * 3 + [("advance", 0.45)] * 3 + [("en_garde", 1.0)]


def _bob(motion, spans, depth_m=0.05):
    """Pelvis dips mid-step, as monocular depth noise gives on real drills (8VWY)."""
    out = motion.copy()
    for a, b, label in spans:
        if label != "en_garde":
            out.root_pos[a:b, 2] -= depth_m * np.sin(np.linspace(0.0, np.pi, b - a))
    return out


def test_back_to_back_steps_are_split_into_single_steps():
    motion, spans = synthesize(DRILL)
    segments, _ = find_segments(add_jitter(rotate_about_z(motion, 2.0, np.zeros(2)), 0.01, 0.003))
    actions = [s for s in segments if s.label != "en_garde"]
    truth = [s for s in spans if s[2] != "en_garde"]
    assert [s.label for s in actions] == [t[2] for t in truth]
    for seg, exp in zip(actions, truth):
        assert _overlap((seg.start, seg.end), exp) > 0.6, (seg, exp)


def test_a_retreat_advance_drill_with_pelvis_bob_is_not_a_lunge():
    motion, spans = synthesize(DRILL)
    segments, _ = find_segments(_bob(motion, spans))
    labels = [s.label for s in segments]
    assert "lunge" not in labels and "advance_lunge" not in labels, segments
    assert labels.count("retreat") == 3 and labels.count("advance") == 3, labels


def test_a_lunge_with_its_recovery_stays_one_segment():
    motion, spans = synthesize([("en_garde", 1.0), ("lunge", 1.2), ("en_garde", 1.0)])
    segments, _ = find_segments(_bob(motion, spans, 0.02))
    actions = [s for s in segments if s.label != "en_garde"]
    assert [s.label for s in actions] == ["lunge"], actions
    assert actions[0].features["recovered"]
