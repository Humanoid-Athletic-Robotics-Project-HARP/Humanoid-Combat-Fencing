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
