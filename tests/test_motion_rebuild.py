"""fencing.motion.rebuild: the Goal 1 recipe against the published sidecars, recipe validation,
and a small rebuild end to end (scratch output, comparison, publishing as new versions)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from fencing import paths
from fencing.motion import rebuild
from fencing.motion.io import read_csv, write_csv

MOTIONS = paths.MOTIONS_DIR
GOAL1_FINALS = {
    "en_garde": "en_garde_v3", "advance": "advance_v1", "retreat": "retreat_v1", "lunge": "lunge_v5",
    "advance_lunge": "advance_lunge_v4", "extension": "extension_v3", "parry_prime": "parry_prime_v3",
    "parry_seconde": "parry_seconde_v3", "parry_tierce_high": "parry_tierce_high_v3",
    "parry_quinte": "parry_quinte_v3", "parry_sixte": "parry_sixte_v3", "cut_head": "cut_head_v4",
    "cut_flank": "cut_flank_v3", "cut_chest": "cut_chest_v3", "thrust": "thrust_v3",
    "advance_lunge_cut": "advance_lunge_cut_v5", "stand_to_en_garde": "stand_to_en_garde_v1",
}


GVHMR_RECIPE = rebuild.RECIPE_DIR / "goal1_gvhmr.toml"
GVHMR_RECIPE_REL = "motions/recipes/goal1_gvhmr.toml"
STATIC_BLADEWORK = ("extension", "parry_prime", "parry_seconde", "parry_tierce_high", "parry_quinte", "parry_sixte",
                    "cut_head", "cut_flank", "cut_chest", "thrust")


@pytest.fixture(scope="module")
def goal1():
    return rebuild.load_recipe(rebuild.DEFAULT_RECIPE)


@pytest.fixture(scope="module")
def gvhmr():
    return rebuild.load_recipe(GVHMR_RECIPE)


@pytest.fixture(scope="module")
def gvhmr_published(gvhmr):
    """motion -> (newest published path, its sidecar) for the 17 motions of the GVHMR recipe."""
    from fencing.pipeline import latest_versions

    newest = latest_versions(MOTIONS, tuple(gvhmr.motions))
    return {name: (path, _sidecar(path.stem)) for name, path in newest.items()}


def _sidecar(name: str) -> dict:
    return json.loads((MOTIONS / f"{name}.json").read_text())


# ---------------------------------------------------------------- the Goal 1 recipe


def test_goal1_recipe_reproduces_the_17_final_motions(goal1):
    assert {n: m.reproduces for n, m in goal1.motions.items()} == GOAL1_FINALS


def test_goal1_finals_are_published_and_only_the_gvhmr_rebuild_supersedes_them(goal1):
    # The MediaPipe-era finals stay in motions/; the newer versions on top of them are the GVHMR
    # rebuild of the same 17 motions (motions/recipes/goal1_gvhmr.toml), nothing else.
    from fencing.pipeline import latest_versions

    newest = latest_versions(MOTIONS, tuple(goal1.motions))
    for name, m in goal1.motions.items():
        assert (MOTIONS / f"{m.reproduces}.csv").is_file(), name
        if newest[name].stem != m.reproduces:
            assert _sidecar(newest[name].stem)["rebuild"]["recipe"] == GVHMR_RECIPE_REL, name
    for checkpoint in goal1.checkpoints:
        assert (MOTIONS / f"{checkpoint}.csv").is_file(), checkpoint


def test_goal1_builds_bases_before_what_is_built_on_them(goal1):
    order = goal1.order()
    assert len(order) == 17
    for later, earlier in (("lunge", "en_garde"), ("cut_head", "en_garde"), ("stand_to_en_garde", "parry_quinte"),
                           ("advance_lunge_cut", "advance_lunge")):
        assert order.index(earlier) < order.index(later)
    assert goal1.order(["stand_to_en_garde"]) == ["en_garde", "parry_quinte", "stand_to_en_garde"]


def _clean_steps(goal1):
    """(motion, clean step, the published file the clean output was saved as)."""
    for m in goal1.motions.values():
        step = m.steps[0]
        if step.op != "clean":
            continue
        published = step.checkpoint or (m.reproduces if len(m.steps) == 1 else None)
        yield m, step, published


def test_goal1_clean_configs_match_the_published_sidecars(goal1):
    checked = 0
    for m, step, published in _clean_steps(goal1):
        record = _sidecar(published)
        if "config" not in record:  # en_garde_v1 was published from a batch output; its config is in docs
            continue
        effective = asdict(rebuild.clean_config(step.params.get("config") or {}))
        recorded = record["config"]
        for key in set(effective) & set(recorded):
            assert effective[key] == recorded[key], (m.name, key)
        # Keys the sidecar lacks were added to the cleaner later; the recipe must switch them off.
        for key in set(effective) - set(recorded):
            assert key in ("plant_hover_m", "end_hold_s") and effective[key] is None, (m.name, key)
        assert record["segment"]["start"] == step.params["frames"][0], m.name
        assert record["segment"]["end"] == step.params["frames"][1], m.name
        assert record["segment"]["label"] == step.params["label"], m.name
        checked += 1
    assert checked == 4  # advance, retreat, lunge (v2), advance_lunge (v3)


def test_goal1_en_garde_clean_matches_the_documented_batch_run(goal1):
    """en_garde_v1: frames 18-43, 0.5 s holds, start pelvis heading, from the cleanup code of 2026-11-09."""
    step = goal1.motions["en_garde"].steps[0]
    config = rebuild.clean_config(step.params["config"])
    assert step.params["frames"] == [18, 43]
    assert (config.heading, config.straighten, config.contact_rounds, config.plant_hover_m) == ("pelvis", False, 2, None)
    assert (config.hold_s, config.end_hold_s, config.ease_s) == (0.5, None, 0.25)


def test_goal1_derived_steps_match_the_published_sidecars(goal1):
    steps = {s.checkpoint or f"{m.name}:final": s for m in goal1.motions.values() for s in m.steps}
    heading = steps["en_garde_v2"]
    assert rebuild.FOOT_INDEX[heading.params["foot"]] == _sidecar("en_garde_v2")["set_foot_heading"]["foot_index"]
    assert heading.params["target_yaw_deg"] == _sidecar("en_garde_v2")["set_foot_heading"]["target_yaw_deg"]
    assert steps["en_garde:final"].params["seconds"] == _sidecar("en_garde_v3")["hold"]["end_hold_s"]
    assert round(steps["lunge:final"].params["seconds"] * 30) == _sidecar("lunge_v5")["end_hold"]["appended_frames"]
    lunge = goal1.motions["lunge"].steps
    assert _sidecar("lunge_v3")["rear_arm_pose_from"].startswith(lunge[1].params["pose_from"])
    assert lunge[2].params["side"] == _sidecar("lunge_v3")["clear_arm_torso_collision"]["side"]
    for name, published in (("lunge", "lunge_v4"), ("advance_lunge", "advance_lunge_v4")):
        neutral = next(s for s in goal1.motions[name].steps if s.op == "neutralize_wrists")
        joints = [f"{side}_wrist_{axis}_joint" for side in neutral.params["sides"] for axis in ("pitch", "yaw")]
        assert joints == _sidecar(published)["neutralize_wrists"]["joints"]


def test_goal1_blade_scripts_configs_and_bases_match_the_published_sidecars(goal1):
    from fencing.saber import recipes

    checked = 0
    for m in goal1.motions.values():
        step = m.steps[0]
        if step.op != "blade":
            continue
        record = _sidecar(m.reproduces)
        config = {k: v for k, v in asdict(rebuild.blade_config(step.params.get("config") or {})).items() if k != "saber"}
        recorded = record["blade"]["summary"]["config"]
        assert {k: config[k] for k in recorded} == recorded, m.name
        # Keys the sidecar lacks were added to the keyframer later; the recipe must leave them at
        # the behaviour these motions were built with.
        for key in set(config) - set(recorded):
            assert (key, config[key]) == ("blade_path", "joint"), (m.name, key)
        assert Path(record["base"]["path"]).name == f"{goal1.motions[m.source].reproduces}.csv", m.name
        base_csv = MOTIONS / f"{goal1.motions[m.source].reproduces}.csv"
        base = recipes.load_base(recipes.base_kind(step.params["recipe"]), base_csv)
        script = recipes.recipe_script(step.params["recipe"], base, step.params.get("cut", "cut_head"),
                                       step.params.get("hit_at_s"))
        expected = record["blade"]["script"]
        assert len(script) == len(expected), m.name
        for key, want in zip(script, expected):
            assert (key.action, key.move_s, key.hold_s) == (want["action"], want["move_s"], want["hold_s"]), m.name
            assert (key.at_s is None) == (want["at_s"] is None), m.name
            if key.at_s is not None:
                assert abs(key.at_s - want["at_s"]) < 1e-9, m.name
        checked += 1
    assert checked == 11


def test_goal1_entry_targets_the_published_parry_quinte_frame(goal1):
    m = goal1.motions["stand_to_en_garde"]
    record = _sidecar(m.reproduces)["entry"]
    assert record["target"] == f"{goal1.motions[m.source].reproduces}.csv"
    assert record["frame"] == m.steps[0].params["frame"]


def test_goal1_clip_motions_say_what_to_re_choose_on_new_pose_data(goal1):
    for m in goal1.motions.values():
        if m.source_kind == "clip":
            assert m.tuned_on == "fallback_mediapipe", m.name
            assert m.tied_to_mediapipe, m.name


# ---------------------------------------------------------------- the GVHMR recipe and what it published


def test_gvhmr_recipe_builds_the_same_17_motions_from_gvhmr_clips(goal1, gvhmr):
    assert set(gvhmr.motions) == set(goal1.motions)
    assert len(gvhmr.order()) == 17
    for m in gvhmr.motions.values():
        assert m.reproduces is None, m.name  # new data: there is nothing to reproduce
        if m.source_kind == "clip":
            assert m.tuned_on == "gvhmr", m.name
        else:  # the bladework and the entry are built on the same bases as in goal1.toml
            assert (m.source, [s.op for s in m.steps]) == (goal1.motions[m.name].source,
                                                           [s.op for s in goal1.motions[m.name].steps]), m.name


def test_gvhmr_recipe_moves_only_thrust_cut_flank_and_extension_along_the_blade_line(gvhmr, gvhmr_published):
    # thrust_v4 / cut_flank_v4 were joint-space moves on the side-on GVHMR guard (tip down to 0.07 m /
    # 0.34 m) and extension_v4 detoured 2.24x the straight distance (v3: 1.22); their v5s are the
    # blade_path="line" rebuild. The other bladework keeps the joint-space default, so its published
    # files stay what is being trained.
    line = {name for name, m in gvhmr.motions.items()
            if m.steps[0].op == "blade" and rebuild.blade_config(m.steps[0].params.get("config") or {}).blade_path == "line"}
    assert line == {"thrust", "cut_flank", "extension"}
    assert rebuild.blade_paths(gvhmr) == dict.fromkeys(line, "line")
    for name in line:
        path, record = gvhmr_published[name]
        assert path.name == f"{name}_v5.csv", name
        assert record["blade"]["summary"]["config"]["blade_path"] == "line", name
        assert all(move["used"] for move in record["blade"]["summary"]["line_moves"]), name


def test_gvhmr_motions_are_the_newest_versions_with_their_provenance(gvhmr, gvhmr_published):
    assert set(gvhmr_published) == set(gvhmr.motions)
    for name, (path, record) in gvhmr_published.items():
        m = gvhmr.motions[name]
        rec = record["rebuild"]
        assert (rec["recipe"], rec["motion"], rec["pose_method"]) == (GVHMR_RECIPE_REL, name, "gvhmr"), name
        assert rec["source"] == f"{m.source_kind}:{m.source}", name
        if m.source_kind == "clip":
            assert len(rec["source_sha256"]) == 64 and rec["source_csv"].endswith(f"{m.source}.csv"), name
            assert "gvhmr retarget" in record["provenance"]["source_clip"], name
        else:  # built on the GVHMR version of its base, named with its sha256
            base_path, _ = gvhmr_published[m.source]
            assert rec["base_file"] == base_path.name, name
            assert rec["base_sha256"] == _sha256(base_path), name


def test_gvhmr_static_bladework_starts_and_ends_in_one_guard(gvhmr_published):
    # All ten start on en garde's first frame, exactly. They end on its last frame with the arm
    # back on the guard target: the legs settle by up to 0.055 rad over en garde (en_garde_v3:
    # 0.10), so the end guard key is re-solved (keyframes._solve_all) and lands within 0.041 rad
    # of the first, thrust 0.005 rad from the others.
    guard = read_csv(gvhmr_published["parry_quinte"][0])
    assert np.abs(guard.joint_pos[-1] - guard.joint_pos[0]).max() < 0.06
    for name in STATIC_BLADEWORK:
        motion = read_csv(gvhmr_published[name][0])
        np.testing.assert_allclose(motion.joint_pos[0], guard.joint_pos[0], atol=1e-5, err_msg=name)
        np.testing.assert_allclose(motion.joint_pos[-1], guard.joint_pos[-1], atol=0.01, err_msg=name)
        for frame in (0, -1):
            np.testing.assert_allclose(motion.root_pos[frame], guard.root_pos[frame], atol=1e-5, err_msg=name)
            np.testing.assert_allclose(motion.root_quat[frame], guard.root_quat[frame], atol=1e-6, err_msg=name)


def test_gvhmr_entry_ends_exactly_in_the_guard_every_saber_motion_starts_from(gvhmr_published):
    from fencing.motion import entry

    stand, record = gvhmr_published["stand_to_en_garde"]
    motion = read_csv(stand)
    np.testing.assert_allclose(motion.joint_pos[0], entry.standby_joint_positions(), atol=1e-6)
    assert record["entry"]["target"] == gvhmr_published["parry_quinte"][0].name
    for name in STATIC_BLADEWORK:
        first = read_csv(gvhmr_published[name][0])
        np.testing.assert_allclose(motion.joint_pos[-1], first.joint_pos[0], atol=1e-5, err_msg=name)
        np.testing.assert_allclose(motion.root_pos[-1], first.root_pos[0], atol=1e-5, err_msg=name)
        assert abs(float(np.dot(motion.root_quat[-1], first.root_quat[0]))) > 1 - 1e-9, name


def test_gvhmr_motions_pass_the_motion_check_as_the_recipe_allows(gvhmr, gvhmr_published):
    from fencing.motion.quality import quality_report

    for name, (path, _) in gvhmr_published.items():
        verdict = quality_report(read_csv(path))["verdict"]
        assert verdict in (("pass", "warn") if gvhmr.motions[name].allow_warn else ("pass",)), (name, verdict)


def test_gvhmr_footwork_end_holds_are_at_least_the_eval_hold(gvhmr_published):
    # docs/pipeline.md, known errors: a policy only practises standing for as long as the end hold.
    for name in ("en_garde", "advance", "retreat", "lunge", "advance_lunge"):
        motion = read_csv(gvhmr_published[name][0])
        still = np.abs(np.diff(motion.joint_pos, axis=0)).max(axis=1) < 1e-9
        held = int(np.argmin(still[::-1])) if not still.all() else len(still)
        assert held + 1 >= int(round(2.0 * motion.fps)), (name, held)


# ---------------------------------------------------------------- validation


def _write_recipe(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "recipe.toml"
    path.write_text("schema = 1\n" + body)
    return path


@pytest.mark.parametrize("body, message", [
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "teleport"\n', "unknown op"),
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\nspeed = 2\n',
     "does not take"),
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\n'
     '[[motions.lunge.steps]]\nop = "clean"\nlabel = "lunge"\n', "must be the first step"),
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "rear_arm_pose"\nside = "left"\n'
     'pose_from = "nowhere_v2"\n', "neither a motion nor a checkpoint"),
    ('[motions.a]\nsource = "motion:b"\n[[motions.a.steps]]\nop = "neutralize_wrists"\n'
     '[motions.b]\nsource = "motion:a"\n[[motions.b.steps]]\nop = "neutralize_wrists"\n', "cycle"),
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "clean"\nlabel = "lunge"\nframes = [9, 3]\n',
     "frames must be"),
    ('[motions.lunge]\nsource = "clip:x"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\ncheckpoint = "k"\n'
     '[[motions.lunge.steps]]\nop = "neutralize_wrists"\ncheckpoint = "k"\n', "used twice"),
    ('[motions.lunge]\nsource = "video:x"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\n', "source must be"),
    ('[motions.lunge]\nsource = "clip:x"\ncolour = "red"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\n',
     "unknown keys"),
])
def test_bad_recipes_are_refused_with_the_reason(tmp_path, body, message):
    with pytest.raises(rebuild.RecipeError, match=message):
        rebuild.load_recipe(_write_recipe(tmp_path, body))


def test_clean_config_false_switches_a_number_off_and_unknown_keys_fail():
    config = rebuild.clean_config({"plant_hover_m": False, "straighten": False, "end_hold_s": 2.0})
    assert config.plant_hover_m is None and config.straighten is False and config.end_hold_s == 2.0
    with pytest.raises(rebuild.RecipeError, match="unknown clean config keys"):
        rebuild.clean_config({"smoothness": 3})


# ---------------------------------------------------------------- end to end on a synthetic clip

MINI = """
[motions.lunge]
source = "clip:fake_clip"
reproduces = "lunge_v2"
allow_warn = true
tuned_on = "fallback_mediapipe"
tied_to_mediapipe = ["frames picked on MediaPipe"]
[[motions.lunge.steps]]
op = "neutralize_wrists"
checkpoint = "lunge_v1"
[[motions.lunge.steps]]
op = "append_end_hold"
seconds = 0.5

[motions.en_garde]
source = "motion:lunge"
allow_warn = true
[[motions.en_garde.steps]]
op = "rear_arm_pose"
side = "left"
pose_from = "lunge_v1"
"""


@pytest.fixture()
def mini(tmp_path):
    from fencing.saber import recipes

    csv_dir = tmp_path / "csv"
    raw = recipes.load_base("en_garde", "synthetic").motion
    raw.joint_pos[:, rebuild.arm_columns("left")] += np.linspace(0.0, 0.3, raw.n_frames)[:, None]
    write_csv(raw, csv_dir / "fake_clip.csv")
    (csv_dir / "fake_clip.json").write_text(json.dumps({"clip_id": "fake_clip", "pose_method": "gvhmr", "fps": 30}))
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "README.md").write_text("# Motions\n\n## Provenance\n\n| File | Source | Segment | Hand | Cleanup | Notes |\n"
                                       "|---|---|---|---|---|---|\n| | | | | | |\n")
    return {"recipe": rebuild.load_recipe(_write_recipe(tmp_path, MINI)), "csv_dir": csv_dir, "motions": motions,
            "raw": read_csv(csv_dir / "fake_clip.csv"), "out": tmp_path / "out"}


def _snapshot(directory: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_rebuild_writes_scratch_outputs_and_never_touches_motions(mini):
    before = _snapshot(mini["motions"])
    job = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    outcomes = {o.name: o for o in job.run()}
    assert _snapshot(mini["motions"]) == before
    lunge = read_csv(mini["out"] / "lunge.csv")
    assert lunge.n_frames == mini["raw"].n_frames + 15
    wrists = [i for i, n in enumerate(rebuild.JOINT_NAMES) if "wrist_pitch" in n or "wrist_yaw" in n]
    assert np.all(lunge.joint_pos[:, wrists] == 0.0)
    assert (mini["out"] / "checkpoints" / "lunge_v1.csv").is_file()
    assert (mini["out"] / "lunge.report.json").is_file() and (mini["out"] / "summary.json").is_file()
    # The arm pose is the checkpoint's mean as written to CSV, held in every frame.
    checkpoint = read_csv(mini["out"] / "checkpoints" / "lunge_v1.csv")
    en_garde = read_csv(mini["out"] / "en_garde.csv")
    left = rebuild.arm_columns("left")
    np.testing.assert_allclose(en_garde.joint_pos[:, left], np.round(checkpoint.joint_pos[:, left].mean(0), 6)[None].repeat(en_garde.n_frames, 0), atol=1e-6)
    assert outcomes["lunge"].quality["verdict"] in ("pass", "warn", "fail")
    assert outcomes["lunge"].pose_method == "gvhmr"
    assert outcomes["lunge"].retune == ["frames picked on MediaPipe"]  # tuned on MediaPipe, source is GVHMR
    assert outcomes["en_garde"].retune == []


def test_compare_reports_identical_checkpoints_and_the_size_and_place_of_a_difference(mini):
    first = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    first.run()
    (mini["motions"] / "lunge_v1.csv").write_bytes((mini["out"] / "checkpoints" / "lunge_v1.csv").read_bytes())
    final = np.loadtxt(mini["out"] / "lunge.csv", delimiter=",")
    final[7, 10] += 0.001
    np.savetxt(mini["motions"] / "lunge_v2.csv", final, delimiter=",", fmt="%.6f")
    job = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"], compare=True)
    lunge = {o.name: o for o in job.run()}["lunge"]
    checkpoint, whole = lunge.compare
    assert checkpoint["status"] == "identical" and checkpoint["bytes_identical"]
    assert whole["status"] == "differs" and abs(whole["max_abs"] - 0.001) < 1e-6
    assert (whole["worst_frame"], whole["worst_column"]) == (7, rebuild.COLUMNS[10])


def test_publish_adds_new_versions_and_leaves_existing_ones(mini):
    (mini["motions"] / "lunge_v1.csv").write_text("old\n")
    (mini["motions"] / "lunge_v2.csv").write_text("old\n")
    before = _snapshot(mini["motions"])
    job = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    outcomes = job.run()
    if any(o.quality["verdict"] == "fail" for o in outcomes):
        pytest.skip("synthetic clip fails the check; publishing is covered by the refusal test")
    status = job.publish()
    assert (status["lunge"].kind, status["lunge"].path) == ("new", mini["motions"] / "lunge_v3.csv")
    assert (status["en_garde"].kind, status["en_garde"].path) == ("new", mini["motions"] / "en_garde_v1.csv")
    for name, content in before.items():
        if name != "README.md":
            assert (mini["motions"] / name).read_bytes() == content
    assert (mini["motions"] / "lunge_v3.csv").read_bytes() == (mini["out"] / "lunge.csv").read_bytes()
    record = json.loads((mini["motions"] / "lunge_v3.json").read_text())
    assert record["rebuild"]["source"] == "clip:fake_clip" and record["rebuild"]["pose_method"] == "gvhmr"
    assert [s["op"] for s in record["rebuild"]["steps"]] == ["neutralize_wrists", "append_end_hold"]
    readme = (mini["motions"] / "README.md").read_text()
    assert "`lunge_v3.csv`" in readme and "`en_garde_v1.csv`" in readme


MINI_WITH_ADVANCE = MINI + """
[motions.advance]
source = "clip:fake_clip"
allow_warn = true
[[motions.advance.steps]]
op = "append_end_hold"
seconds = 0.2
"""


def _publish(mini, recipe, out):
    job = rebuild.Rebuild(recipe, out, csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    outcomes = job.run()
    for outcome in outcomes:  # the synthetic clip's check verdict is not what these tests are about
        outcome.quality = {**outcome.quality, "verdict": "pass", "reasons": []}
    return job.publish()


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publishing_twice_creates_nothing_the_second_time(mini, tmp_path):
    recipe = rebuild.load_recipe(_write_recipe(tmp_path, MINI_WITH_ADVANCE))
    first = _publish(mini, recipe, mini["out"] / "1")
    assert {n: s.kind for n, s in first.items()} == {"lunge": "new", "en_garde": "new", "advance": "new"}
    before = _snapshot(mini["motions"])
    second = _publish(mini, recipe, mini["out"] / "2")
    assert _snapshot(mini["motions"]) == before  # no new CSV, sidecar or README row
    assert {n: (s.kind, s.path.name) for n, s in second.items()} == {
        "lunge": ("unchanged", "lunge_v1.csv"), "en_garde": ("unchanged", "en_garde_v1.csv"),
        "advance": ("unchanged", "advance_v1.csv")}
    assert str(second["lunge"]) == "unchanged: lunge_v1.csv"


def test_published_bytes_are_the_scratch_bytes_even_where_a_csv_round_trip_is_not_stable(mini, tmp_path):
    # write_csv normalizes the root quaternion, so writing a read-back 6-digit CSV can move a last
    # digit (seen on stand_to_en_garde: qw 0.995992 -> 0.995993). Publishing must write the scratch
    # bytes, or a second --publish finds a "change" and makes a duplicate version.
    raw = np.loadtxt(mini["csv_dir"] / "fake_clip.csv", delimiter=",")
    a = np.linspace(0.2, 1.4, len(raw))
    raw[:, 3:7] = np.stack([np.zeros_like(a), 0.01 * np.sin(5 * a), np.sin(a / 2), np.cos(a / 2)], 1)
    np.savetxt(mini["csv_dir"] / "fake_clip.csv", raw, delimiter=",", fmt="%.9f")  # first 6-digit write is ours
    one_step = ('[motions.lunge]\nsource = "clip:fake_clip"\n[[motions.lunge.steps]]\nop = "neutralize_wrists"\n')
    recipe = rebuild.load_recipe(_write_recipe(tmp_path, one_step))
    status = _publish(mini, recipe, mini["out"] / "1")
    scratch = mini["out"] / "1" / "lunge.csv"
    rewritten = tmp_path / "rewritten.csv"
    write_csv(read_csv(scratch), rewritten)
    assert rewritten.read_bytes() != scratch.read_bytes()  # the round trip is unstable on this data
    assert status["lunge"].path.read_bytes() == scratch.read_bytes()
    second = _publish(mini, recipe, mini["out"] / "2")
    assert {s.kind for s in second.values()} == {"unchanged"}


def test_after_a_change_only_that_motion_and_its_dependants_get_new_versions(mini, tmp_path):
    _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path, MINI_WITH_ADVANCE)), mini["out"] / "1")
    changed = MINI_WITH_ADVANCE.replace("seconds = 0.5", "seconds = 0.7")  # lunge's end hold; en_garde is built on lunge
    (tmp_path / "changed").mkdir()
    status = _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path / "changed", changed)), mini["out"] / "2")
    assert {n: (s.kind, s.path.name) for n, s in status.items()} == {
        "lunge": ("new", "lunge_v2.csv"), "en_garde": ("new", "en_garde_v2.csv"), "advance": ("unchanged", "advance_v1.csv")}
    assert not (mini["motions"] / "advance_v2.csv").exists()
    record = json.loads((mini["motions"] / "en_garde_v2.json").read_text())
    assert record["rebuild"]["base_file"] == "lunge_v2.csv"


def test_output_identical_to_a_superseded_version_is_unchanged_not_republished_as_the_newest(mini, tmp_path):
    # A recipe reverted to an old step (thrust without blade_path) rebuilds an old version byte for
    # byte; publishing it again would make it the newest, the one training picks.
    _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path, MINI_WITH_ADVANCE)), mini["out"] / "1")
    changed = MINI_WITH_ADVANCE.replace("seconds = 0.5", "seconds = 0.7")
    (tmp_path / "changed").mkdir()
    _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path / "changed", changed)), mini["out"] / "2")
    before = _snapshot(mini["motions"])
    status = _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path, MINI_WITH_ADVANCE)), mini["out"] / "3")
    assert _snapshot(mini["motions"]) == before
    assert {n: (s.kind, s.path.name) for n, s in status.items()} == {
        "lunge": ("unchanged", "lunge_v1.csv"), "en_garde": ("unchanged", "en_garde_v1.csv"),
        "advance": ("unchanged", "advance_v1.csv")}


def test_a_dependant_of_an_unchanged_motion_is_built_on_and_names_the_existing_file(mini, tmp_path):
    _publish(mini, mini["recipe"], mini["out"] / "1")
    changed = MINI.replace('pose_from = "lunge_v1"\n', 'pose_from = "lunge_v1"\n[[motions.en_garde.steps]]\n'
                           'op = "append_end_hold"\nseconds = 0.1\n')
    (tmp_path / "changed").mkdir()
    status = _publish(mini, rebuild.load_recipe(_write_recipe(tmp_path / "changed", changed)), mini["out"] / "2")
    assert (status["lunge"].kind, status["lunge"].path.name) == ("unchanged", "lunge_v1.csv")
    assert (status["en_garde"].kind, status["en_garde"].path.name) == ("new", "en_garde_v2.csv")
    record = json.loads((mini["motions"] / "en_garde_v2.json").read_text())
    assert record["rebuild"]["base_file"] == "lunge_v1.csv"
    assert record["rebuild"]["base_sha256"] == _sha256(mini["motions"] / "lunge_v1.csv")
    readme_row = next(line for line in (mini["motions"] / "README.md").read_text().splitlines() if "en_garde_v2.csv" in line)
    assert "lunge_v1.csv" in readme_row


def test_rear_arm_provenance_names_the_published_version_and_the_step_not_the_checkpoint_name(mini):
    status = _publish(mini, mini["recipe"], mini["out"])
    assert status["lunge"].path.name == "lunge_v1.csv" and status["en_garde"].kind == "new"
    record = json.loads((mini["motions"] / "en_garde_v1.json").read_text())
    pose = record["rebuild"]["steps"][0]["info"]["pose_source"]
    checkpoint_file = mini["out"] / "checkpoints" / "lunge_v1.csv"
    assert pose["motion"] == "lunge" and pose["after_step"] == 1 and pose["after_op"] == "neutralize_wrists"
    assert pose["sha256"] == _sha256(checkpoint_file)
    assert pose["motion_published_as"] == "lunge_v1.csv"
    row = next(line for line in (mini["motions"] / "README.md").read_text().splitlines() if "en_garde_v1.csv" in line)
    # The checkpoint is called lunge_v1 in the recipe but is not the published lunge_v1.csv (that has the end hold).
    assert "after step 1 (neutralize_wrists)" in row and "lunge published as lunge_v1.csv" in row
    assert "sha256 " + pose["sha256"][:12] in row
    assert "-> lunge_v1 mean pose" not in row


def test_published_sidecars_record_the_sha256_of_the_source_clip_and_base(mini):
    _publish(mini, mini["recipe"], mini["out"])
    lunge = json.loads((mini["motions"] / "lunge_v1.json").read_text())["rebuild"]
    assert lunge["source_sha256"] == _sha256(mini["csv_dir"] / "fake_clip.csv")
    assert lunge["steps"][0]["recipe_checkpoint"] == "lunge_v1" and "checkpoint" not in lunge["steps"][0]
    en_garde = json.loads((mini["motions"] / "en_garde_v1.json").read_text())["rebuild"]
    assert en_garde["base_file"] == "lunge_v1.csv"
    assert en_garde["base_sha256"] == _sha256(mini["motions"] / "lunge_v1.csv")


def test_publish_refuses_a_failing_check_and_everything_built_on_it(mini, monkeypatch):
    job = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    job.run()
    job.outcomes["lunge"].quality = {"verdict": "fail", "reasons": ["foot slip"]}
    status = job.publish()
    assert str(status["lunge"]).startswith("not published: check fail (foot slip)")
    assert str(status["en_garde"]) == "not published: lunge was not published"
    assert status["lunge"].kind == "refused" and status["en_garde"].path is None
    assert not list(mini["motions"].glob("*.csv"))


def test_from_published_starts_after_that_checkpoint_and_is_never_published(mini):
    plain = rebuild.Rebuild(mini["recipe"], mini["out"] / "a", csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    plain.run()
    published = np.loadtxt(mini["out"] / "a" / "checkpoints" / "lunge_v1.csv", delimiter=",")
    published[:, 7] += 0.01  # left hip pitch: untouched by the steps after the checkpoint
    np.savetxt(mini["motions"] / "lunge_v1.csv", published, delimiter=",", fmt="%.6f")
    job = rebuild.Rebuild(mini["recipe"], mini["out"] / "b", csv_dir=mini["csv_dir"], motions_dir=mini["motions"],
                          anchors=("lunge_v1",))
    lunge = {o.name: o for o in job.run()}["lunge"]
    assert lunge.anchored == "lunge_v1" and [s["op"] for s in lunge.steps] == ["append_end_hold"]
    a = np.loadtxt(mini["out"] / "a" / "lunge.csv", delimiter=",")
    b = np.loadtxt(mini["out"] / "b" / "lunge.csv", delimiter=",")
    assert np.allclose(b[:, 7] - a[:, 7], 0.01, atol=2e-6)
    with pytest.raises(rebuild.RecipeError, match="verification only"):
        job.publish()
    with pytest.raises(rebuild.RecipeError, match="not a checkpoint"):
        rebuild.Rebuild(mini["recipe"], mini["out"], motions_dir=mini["motions"], anchors=("nope_v9",))


def test_a_missing_retarget_csv_fails_that_motion_and_what_is_built_on_it(mini):
    (mini["csv_dir"] / "fake_clip.csv").unlink()
    job = rebuild.Rebuild(mini["recipe"], mini["out"], csv_dir=mini["csv_dir"], motions_dir=mini["motions"])
    outcomes = {o.name: o for o in job.run()}
    assert "fake_clip.csv not found" in outcomes["lunge"].error
    assert outcomes["en_garde"].error == "not built: lunge failed"
    assert "FAILED" in rebuild.summary_table(list(outcomes.values()), mini["recipe"])


def test_output_inside_motions_is_refused(mini):
    with pytest.raises(rebuild.RecipeError, match="inside"):
        rebuild.Rebuild(mini["recipe"], mini["motions"] / "scratch", motions_dir=mini["motions"])


def test_cli_rebuild_rejects_publish_with_from_published(capsys):
    from fencing.motion import cli

    assert cli.main(["rebuild", "--publish", "--from-published", "en_garde_v1"]) == 2
    assert "verification only" in capsys.readouterr().err
