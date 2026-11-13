"""Goal 1 bladework recipes: bases, lunge timing, scripts and publishing."""

from functools import lru_cache

import pytest

from fencing.motion.publish import check_motion_name
from fencing.saber import recipes
from fencing.saber.keyframes import BladeKey


@lru_cache(maxsize=2)
def _base(kind):
    return recipes.load_base(kind, "synthetic")


def test_synthetic_base_is_marked_synthetic_in_its_source():
    base = _base("en_garde")
    assert base.synthetic and "SYNTHETIC" in base.source and base.path is None


def test_auto_base_prefers_newest_published_motion(tmp_path):
    for version in (1, 3):
        (tmp_path / f"en_garde_v{version}.csv").write_text("")
    assert recipes.newest_published("en_garde", tmp_path).name == "en_garde_v3.csv"
    assert recipes.newest_published("advance_lunge", tmp_path) is None


def test_lunge_timing_finds_synthetic_touchdown_and_recovery():
    from fencing.motion.synthetic import footwork_keys

    base = _base("advance_lunge")
    _, spans = footwork_keys(recipes.SYNTHETIC_SCRIPTS["advance_lunge"])
    lunge_start, lunge_end, _ = next(s for s in spans if s[2] == "lunge")
    expected_touchdown = (lunge_start + int(0.30 * (lunge_end - lunge_start))) / base.motion.fps
    touchdown, recovery = recipes.lunge_timing(base.motion)
    assert touchdown == pytest.approx(expected_touchdown, abs=2 / base.motion.fps)
    assert touchdown < recovery <= lunge_end / base.motion.fps + 0.1


@lru_cache(maxsize=1)
def _drag_lunge():
    """Lunge with the back foot dragged forward while the front foot is still in the air
    (advance_lunge_v2 / aY1P3wzvNBM): the stance is widest mid-flight and the front foot
    lands ~0.35 s later. Returns the motion and the landing frame."""
    import numpy as np

    from fencing.motion.filters import smoothstep
    from fencing.motion.synthetic import footwork_keys, solve_keys

    keys, _ = footwork_keys([("en_garde", 3.0)])
    a, land = 30, 50

    def ramp(start, end):
        out = np.zeros(keys.feet.shape[0])
        out[start:end] = smoothstep(np.linspace(0.0, 1.0, end - start))
        out[end:] = 1.0
        return out

    keys.feet[:, 1, 0] += 0.50 * ramp(a, a + 12)  # front foot flies forward...
    flight = np.arange(a, land)
    keys.feet[flight, 1, 2] += 0.08 * np.sin(np.pi * (flight - a) / (land - a))  # ...and lands at `land`
    keys.feet[:, 0, 0] += 0.22 * ramp(a + 6, a + 18)  # back foot dragged
    keys.pelvis[:, 0] += 0.36 * ramp(a, a + 18)
    keys.pelvis[:, 2] -= 0.06 * ramp(a, a + 12)
    keys.feet[:, 1, 0] -= 0.28 * ramp(land + 20, land + 32)  # recovery
    keys.pelvis[:, 0] -= 0.14 * ramp(land + 20, land + 32)
    keys.pelvis[:, 2] += 0.06 * ramp(land + 20, land + 32)
    return solve_keys(keys), land


def test_lunge_timing_waits_for_the_landing_of_a_dragged_lunge():
    motion, land = _drag_lunge()
    touchdown, recovery = recipes.lunge_timing(motion)
    assert touchdown == pytest.approx(land / motion.fps, abs=3 / motion.fps)
    assert touchdown < recovery


def test_static_recipes_start_and_end_in_guard():
    for name in recipes.STATIC_RECIPES:
        script = recipes.recipe_script(name, _base("en_garde"))
        assert script[0] == BladeKey("en_garde", 0.0, recipes.GUARD_HOLD_S)
        assert script[-1].action == "en_garde"



def test_cut_head_recipe_slows_its_arrival_to_the_torque_demand_of_the_other_goal1_moves():
    """cut_head_v3 asked for 0.2 s, and the joint speed cap never stretched it (0.37x motor): its
    straight-arm swing is a long lever, not a fast joint. That reference demanded 1.9x the shoulder
    pitch and 5.2x the wrist pitch effort limit (inverse dynamics), against at most 2.3x for every
    other Goal 1 move, and its tip was still 20 cm out at 10 m/s two policy steps before the key.
    Its key-frame tip error was the worst of the set (4.2 cm sim2sim, 6.1 PhysX).

    Runs on the published video base: on the synthetic one the IK lands in another arm solution,
    which the speed cap already stretches to 0.27 s."""
    from fencing import paths

    base_csv = paths.MOTIONS_DIR / "en_garde_v3.csv"
    if not base_csv.is_file():
        pytest.skip(f"{base_csv} not present")
    base = recipes.load_base("en_garde", base_csv)
    fast = recipes.run_script("cut_head", base, [BladeKey("en_garde", 0.0, recipes.GUARD_HOLD_S),
                                                 BladeKey("cut_head", 0.2, 0.3),
                                                 BladeKey("en_garde", recipes.RECOVER_S, recipes.GUARD_HOLD_S)])
    result = recipes.run_recipe("cut_head", base)
    cut = result.blade.keys[1]
    assert cut.action == "cut_head" and cut.move_s_used >= 0.3
    slow_ratio, fast_ratio = result.blade.summary["max_torque_ratio"], fast.blade.summary["max_torque_ratio"]
    assert slow_ratio < 0.6 * fast_ratio, (slow_ratio, fast_ratio)
    assert slow_ratio <= 2.4, slow_ratio


def test_end_hold_keeps_the_whole_body_still_after_the_last_key():
    """mjlab resamples an env when its motion ends, so a policy only learns to stand as long as the
    reference's end hold; the eval then holds the last frame for 2 s (advance_lunge_v2: recovery 0.00
    with a 0.5 s hold, advance_lunge_v3: 1.00 with 2 s)."""
    import numpy as np

    from fencing.saber.keyframes import BladeConfig

    base = _base("en_garde")
    script = recipes.recipe_script("cut_head", base)
    short = recipes.run_script("cut_head", base, script)
    held = recipes.run_script("cut_head", base, script, BladeConfig(end_hold_s=2.0))
    motion = held.blade.motion
    tail = int(round(2.0 * motion.fps))
    assert motion.n_frames > short.blade.motion.n_frames
    assert np.abs(np.diff(motion.joint_pos[-tail:], axis=0)).max() < 1e-9
    assert np.abs(np.diff(motion.root_pos[-tail:], axis=0)).max() < 1e-9
    assert np.abs(np.diff(motion.root_quat[-tail:], axis=0)).max() < 1e-9


def test_end_hold_starts_once_the_base_body_has_stopped_moving():
    """The advance-lunge base is still recovering when the last guard key is reached; the end hold
    must be measured from the moment the whole body is still, not from the arm's last key."""
    import numpy as np

    from fencing.saber.keyframes import BladeConfig

    base = _base("advance_lunge")
    script = recipes.recipe_script("advance_lunge_cut", base)
    motion = recipes.run_script("advance_lunge_cut", base, script, BladeConfig(end_hold_s=2.0)).blade.motion
    tail = int(round(2.0 * motion.fps))
    for name in ("root_pos", "root_quat", "joint_pos"):
        values = getattr(motion, name)
        assert np.abs(values[-tail - 1:] - values[-1]).max() < 1e-6, name


def test_negative_end_hold_is_rejected():
    from fencing.saber.keyframes import BladeConfig

    with pytest.raises(ValueError, match="end_hold_s"):
        BladeConfig(end_hold_s=-1.0)


def test_advance_lunge_cut_lands_at_touchdown():
    base = _base("advance_lunge")
    script = recipes.recipe_script("advance_lunge_cut", base, cut="cut_flank")
    touchdown, _ = recipes.lunge_timing(base.motion)
    cut = script[1]
    assert cut.action == "cut_flank" and cut.at_s == pytest.approx(touchdown)
    assert script[-1].action == "en_garde" and script[-1].at_s > cut.at_s


def test_unknown_recipe_lists_the_known_ones():
    with pytest.raises(KeyError, match="parry_quinte"):
        recipes.recipe_script("parry_banana", _base("en_garde"))


def test_every_recipe_name_is_publishable():
    for name in recipes.RECIPES:
        check_motion_name(name)


def test_publish_writes_version_and_marks_synthetic_provenance(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Motions\n\n## Provenance\n\n| File | Source | Segment | Hand | Cleanup | Notes |\n|---|---|---|---|---|---|\n| | | | | | |\n"
    )
    result = recipes.run_recipe("parry_quinte", _base("en_garde"))
    assert result.quality["verdict"] == "pass" and result.blade_ok
    path = recipes.publish(result, motions_dir=tmp_path)
    assert path.name == "parry_quinte_v1.csv"
    readme = (tmp_path / "README.md").read_text()
    assert "SYNTHETIC" in readme and "hand-keyframed" in readme and "parry_quinte" in readme
    files = sorted(p.name for p in tmp_path.iterdir())
    assert recipes.publish(result, motions_dir=tmp_path) == path  # same bytes: no parry_quinte_v2
    assert sorted(p.name for p in tmp_path.iterdir()) == files and (tmp_path / "README.md").read_text() == readme


def test_published_sidecar_records_the_scored_key_frame(tmp_path):
    import json

    from fencing.saber import scored_keys

    result = recipes.run_recipe("parry_quinte", _base("en_garde"))
    record = result.record()
    parry = next(k for k in result.blade.keys if k.action == "parry_quinte")
    assert record["key_frames"] == scored_keys.key_frames_from_blade_keys(record["blade"]["keys"],
                                                                         result.blade.motion.fps)
    assert [(k["target"], k["frame"]) for k in record["key_frames"]] == [("parry_quinte", parry.frame)]
    (tmp_path / "README.md").write_text(
        "# Motions\n\n## Provenance\n\n| File | Source | Segment | Hand | Cleanup | Notes |\n|---|---|---|---|---|---|\n| | | | | | |\n"
    )
    path = recipes.publish(result, motions_dir=tmp_path)
    assert json.loads(path.with_suffix(".json").read_text())["key_frames"] == record["key_frames"]


# ---------------------------------------------------------------- the GVHMR guard (en_garde_v4)


@lru_cache(maxsize=4)
def _on_gvhmr_guard(name, blade_path):
    from fencing import paths
    from fencing.saber.keyframes import BladeConfig

    base = recipes.load_base("en_garde", paths.MOTIONS_DIR / "en_garde_v4.csv")
    return recipes.run_recipe(name, base, BladeConfig(end_hold_s=2.0, blade_path=blade_path))


def _tip_stats(motion):
    import numpy as np

    from fencing.motion.kinematics import load_g1
    from fencing.saber import keyframes

    ctx = keyframes._context(keyframes.NOMINAL_SABER, keyframes.SOFT_LIMIT_FACTOR)
    tips = np.array([keyframes._blade_tip(ctx, q)[0] for q in load_g1().qpos_from_motion(motion)])
    return float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum()), float(tips[:, 2].min())


def test_default_blade_path_still_reproduces_the_published_thrust_v4(tmp_path):
    # The joint-space default is what every other published motion was built with; thrust_v4 is
    # the motion the line path changes most, so it is the one checked byte for byte here.
    from fencing import paths
    from fencing.motion.io import write_csv

    out = tmp_path / "thrust.csv"
    write_csv(_on_gvhmr_guard("thrust", "joint").blade.motion, out)
    assert out.read_bytes() == (paths.MOTIONS_DIR / "thrust_v4.csv").read_bytes()


def test_line_blade_path_thrusts_on_the_side_on_gvhmr_guard_without_sweeping_the_floor(tmp_path):
    # thrust_v4 (joint space): tip path 6.38 m, lowest tip 0.07 m. thrust_v3 (MediaPipe guard): 1.30 m, 0.93 m.
    import numpy as np

    from fencing import paths
    from fencing.motion.io import read_csv, write_csv

    result = _on_gvhmr_guard("thrust", "line")
    write_csv(result.blade.motion, tmp_path / "thrust.csv")
    assert (tmp_path / "thrust.csv").read_bytes() == (paths.MOTIONS_DIR / "thrust_v5.csv").read_bytes()
    length, lowest = _tip_stats(result.blade.motion)
    assert length < 1.3 and lowest > 0.9, (length, lowest)
    assert result.blade_ok and result.quality["verdict"] == "pass"
    assert all(move["used"] for move in result.blade.summary["line_moves"])
    guard = read_csv(paths.MOTIONS_DIR / "parry_quinte_v4.csv")
    motion = result.blade.motion
    np.testing.assert_allclose(motion.joint_pos[0], guard.joint_pos[0], atol=1e-5)  # the guard every saber motion starts on
    np.testing.assert_allclose(motion.joint_pos[-1], guard.joint_pos[-1], atol=0.01)


def test_line_blade_path_extends_straight_on_the_gvhmr_guard(tmp_path):
    # extension_v4 (joint space): each move 2.24x / 2.19x the straight tip distance, wrist pitch
    # swinging 2.7 rad. extension_v3 (MediaPipe guard): 1.22x / 1.24x.
    import numpy as np

    from fencing import paths
    from fencing.motion.io import read_csv, write_csv

    result = _on_gvhmr_guard("extension", "line")
    write_csv(result.blade.motion, tmp_path / "extension.csv")
    assert (tmp_path / "extension.csv").read_bytes() == (paths.MOTIONS_DIR / "extension_v5.csv").read_bytes()
    assert result.blade_ok and result.quality["verdict"] == "pass" and not result.quality["reasons"]
    assert all(move["used"] for move in result.blade.summary["line_moves"])
    length, lowest = _tip_stats(result.blade.motion)
    assert length < 0.85 and lowest > 1.0, (length, lowest)
    guard = read_csv(paths.MOTIONS_DIR / "parry_quinte_v4.csv")
    np.testing.assert_allclose(result.blade.motion.joint_pos[0], guard.joint_pos[0], atol=1e-5)
