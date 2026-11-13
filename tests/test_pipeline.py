"""fencing.pipeline: task choice, version picking, log rows, and the dry-run stage plan."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pytest

from fencing import pipeline


def test_motion_base_strips_the_version_only():
    assert pipeline.motion_base("parry_quinte_v3") == "parry_quinte"
    assert pipeline.motion_base("en_garde_v12") == "en_garde"
    assert pipeline.motion_base("dance1_subject2") == "dance1_subject2"


def test_bladework_motions_train_with_the_saber_and_footwork_without():
    for name in ("extension_v1", "parry_prime_v2", "cut_head_v1", "thrust_v1", "advance_lunge_cut_v1"):
        assert pipeline.task_for(name) == pipeline.SABER_TASK, name
    for name in ("en_garde_v1", "advance_v1", "retreat_v1", "lunge_v1", "advance_lunge_v1"):
        assert pipeline.task_for(name) == pipeline.paths.TRACKING_TASK, name


def test_latest_versions_picks_the_highest_number_not_the_last_string(tmp_path):
    for name in ("en_garde_v2.csv", "en_garde_v10.csv", "en_garde_v9.csv", "lunge_v1.csv",
                 "not_a_goal1_motion_v4.csv", "advance.csv"):
        (tmp_path / name).write_text("")
    found = pipeline.latest_versions(tmp_path)
    assert found == {"en_garde": tmp_path / "en_garde_v10.csv", "lunge": tmp_path / "lunge_v1.csv"}


def test_advance_lunge_and_advance_lunge_cut_are_kept_apart(tmp_path):
    (tmp_path / "advance_lunge_v1.csv").write_text("")
    (tmp_path / "advance_lunge_cut_v3.csv").write_text("")
    found = pipeline.latest_versions(tmp_path)
    assert found["advance_lunge"].name == "advance_lunge_v1.csv"
    assert found["advance_lunge_cut"].name == "advance_lunge_cut_v3.csv"


def _result(**extra):
    return {"motion": "en_garde_v1", "task": pipeline.paths.TRACKING_TASK, "iterations": 100, "num_envs": 64,
            "commit": "abc1234", "run_dir": "/runs/g1_tracking/2026-11-09_10-00-00_en_garde_v1_pipe",
            "stages": {"train": True, "export": True, "eval": True, "sim2sim": False}, **extra}


def test_summary_names_verdicts_numbers_and_failed_stages():
    metrics = {"verdict": "PASS", "nominal": {"success_rate": 0.98, "num_episodes": 1024,
                                              "joint_angle_error_rad": 0.05, "body_pos_error_m": 0.031,
                                              "foot_slip_m_s": 0.02, "blade_tip_error_m": 0.04}}
    text = pipeline.summarize_result(_result(metrics=metrics, sim2sim={"verdict": "FAIL"}))
    assert "mjlab PASS" in text and "success 0.98 of 1024" in text
    assert "body 3.1 cm" in text and "blade tip 4.0 cm" in text
    assert "sim2sim FAIL" in text and "failed stages: sim2sim" in text


def test_log_row_replaces_the_empty_placeholder_then_appends(tmp_path):
    log_md = tmp_path / "log.md"
    log_md.write_text("## Runs\n\n| Date | Stack |\n|---|---|\n| | | | | | | | |\n")
    pipeline.append_log_row(_result(), log_md)
    pipeline.append_log_row(_result(motion="lunge_v2"), log_md)
    lines = log_md.read_text().splitlines()
    assert "| | | | | | | | |" not in lines
    rows = [line for line in lines if "| mjlab |" in line]
    assert len(rows) == 2
    assert "| en_garde | en_garde_v1 |" in rows[0] and "| lunge | lunge_v2 |" in rows[1]
    assert rows[0].count("|") == 9  # 8 columns, same as the table header


def test_train_dry_run_plans_one_train_per_motion_without_running(tmp_path, capsys):
    csv = tmp_path / "parry_quinte_v1.csv"
    csv.write_text("")
    code = pipeline.main(["--dry-run", "train", str(csv), "--iterations", "7"])
    err = capsys.readouterr().err
    assert code == 0
    assert "train run" in err and "parry_quinte_v1_pipe" in err and pipeline.SABER_TASK in err
    assert "--iterations 7" in err


def test_train_refuses_missing_files(tmp_path):
    assert pipeline.main(["train", str(tmp_path / "nope_v1.csv")]) == 2


def test_blade_args_use_video_bases_when_published(tmp_path, monkeypatch):
    (tmp_path / "en_garde_v2.csv").write_text("")
    monkeypatch.setattr(pipeline.paths, "MOTIONS_DIR", tmp_path)
    cmd = pipeline.blade_args(publish=True)
    assert cmd[:2] == ["motion", "blade-all"]
    assert ["--base-en-garde", str(tmp_path / "en_garde_v2.csv")] == cmd[2:4]
    assert "--base-advance-lunge" not in cmd and cmd[-1] == "--publish"


def test_parser_has_every_subcommand():
    parser = pipeline.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {"data", "train", "all", "status", "queue"}


def test_a_run_that_failed_before_creating_a_run_dir_is_still_logged(tmp_path, monkeypatch):
    csv = tmp_path / "en_garde_v1.csv"
    csv.write_text("")
    failed = {"motion": "en_garde_v1", "task": pipeline.paths.TRACKING_TASK, "iterations": 7, "num_envs": 64,
              "commit": "abc1234", "stages": {"train": False}}
    logged = []
    monkeypatch.setattr(pipeline, "train_one", lambda csv, args: dict(failed))
    monkeypatch.setattr(pipeline, "append_log_row", lambda result: logged.append(result))
    monkeypatch.setattr(pipeline.paths, "EVAL", tmp_path / "eval")
    assert pipeline.main(["train", str(csv)]) == 1
    assert len(logged) == 1 and "failed stages: train" in pipeline.summarize_result(logged[0])


# ------------------------------------------------------------------ queue


def test_the_entry_motion_has_no_default_task():
    # No default is chosen in code; the message names the simulation-recommended task (docs/mtc.md).
    with pytest.raises(ValueError, match="not chosen") as refused:
        pipeline.task_for("stand_to_en_garde_v1")
    assert pipeline.RECOMMENDED_ENTRY_VARIANT == "feetjoint05"
    assert pipeline.ENTRY_VARIANT_TASKS["feetjoint05"] in str(refused.value)
    assert pipeline.ENTRY_VARIANTS == {"tip": pipeline.SABER_TASK, "feet": pipeline.ENTRY_TASK}
    assert "stand_to_en_garde" in pipeline.QUEUE_MOTIONS and len(pipeline.QUEUE_MOTIONS) == 17


def test_train_refuses_the_entry_motion_without_an_explicit_task(tmp_path, capsys):
    csv = tmp_path / "stand_to_en_garde_v1.csv"
    csv.write_text("")
    assert pipeline.main(["--dry-run", "train", str(csv)]) == 2
    assert "not chosen" in capsys.readouterr().err
    assert pipeline.main(["--dry-run", "train", str(csv), "--task", pipeline.ENTRY_TASK]) == 0


def test_entry_task_id_matches_the_registered_task():
    mjlab_task = pytest.importorskip("fencing.saber.mjlab_task")
    assert pipeline.ENTRY_TASK == mjlab_task.ENTRY_TASK_ID
    assert tuple(pipeline.ENTRY_JOINT_VARIANTS.values()) == mjlab_task.ENTRY_JOINT_TASK_IDS


def test_the_entry_joint_variants_are_opt_in_names_for_the_joint_tasks():
    assert pipeline.ENTRY_JOINT_VARIANTS == {
        "feetjoint": "Fencing-Tracking-Flat-G1-Saber-Entry-Joint",
        "feetjoint05": "Fencing-Tracking-Flat-G1-Saber-Entry-Joint05",
    }
    args = pipeline.build_parser().parse_args(["queue", "--entry-variant", "feetjoint05"])
    assert args.entry_variant == "feetjoint05"


def _queue_args(**extra):
    args = pipeline.build_parser().parse_args(["queue"])
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def _run(root, name, stamp, checkpoints, evaluated=(), task=None, csv_sha=None, mtc=None, experiment="g1_tracking"):
    run_dir = root / "runs" / experiment / f"{stamp}_{name}"
    run_dir.mkdir(parents=True)
    meta = {"run_name": name, "task": task or pipeline.paths.TRACKING_TASK, "iterations": 3000}
    if csv_sha:
        meta["motion_csv_sha256"] = csv_sha
    (run_dir / "run.json").write_text(json.dumps(meta))
    for it in checkpoints:
        (run_dir / f"model_{it}.pt").write_text("")
    for it in evaluated:
        out = root / "eval" / run_dir.name / f"model_{it}"
        out.mkdir(parents=True)
        (out / "metrics.json").write_text('{"verdict": "PASS"}')
        (out / "sim2sim.json").write_text('{"verdict": "FAIL"}')
        if mtc:
            (out / "mtc.json").write_text(json.dumps({"verdict": mtc}))
    return run_dir


@pytest.fixture()
def queue_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.paths, "RUNS_LOG_ROOT", tmp_path / "runs")
    monkeypatch.setattr(pipeline.paths, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(pipeline.paths, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "git_short_sha", lambda: "abc1234")
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_queue_state_from_runs_on_disk(queue_dirs):
    args = _queue_args()  # 3000 iterations: finished = model_2999
    csv = queue_dirs / "en_garde_v4.csv"
    assert pipeline.queue_item(csv, args).state == "train"
    _run(queue_dirs, "en_garde_v4_pipe", "2026-11-12_10-00-00", [1000])  # stopped early: not resumed
    assert pipeline.queue_item(csv, args).state == "train"
    trained = _run(queue_dirs, "en_garde_v4_pipe", "2026-11-12_11-00-00", [1000, 2999])
    item = pipeline.queue_item(csv, args)
    assert (item.state, item.run_dir, item.checkpoint.name) == ("evaluate", trained, "model_2999.pt")
    _run(queue_dirs, "en_garde_v4_pipe", "2026-11-12_12-00-00", [2999], evaluated=[2999])
    assert pipeline.queue_item(csv, args).state == "done"
    # A later unfinished run does not hide the finished, evaluated one.
    _run(queue_dirs, "en_garde_v4_pipe", "2026-11-12_13-00-00", [500])
    assert pipeline.queue_item(csv, args).state == "done"
    # More iterations asked than any run has: train again.
    assert pipeline.queue_item(csv, _queue_args(iterations=5000)).state == "train"


def test_resume_finds_an_evaluated_run_whatever_its_tag(queue_dirs):
    csv = queue_dirs / "cut_head_v4.csv"
    run = _run(queue_dirs, "cut_head_v4_tip", "2026-11-11_13-49-44", [2999], evaluated=[2999], task=pipeline.SABER_TASK)
    _run(queue_dirs, "cut_head_v40_pipe", "2026-11-11_14-00-00", [2999], evaluated=[2999], task=pipeline.SABER_TASK)
    item = pipeline.queue_item(csv, _queue_args())
    assert (item.state, item.run_dir) == ("done", run)
    assert item.run_name == "cut_head_v4_pipe"  # what a new run would be called


def test_an_evaluated_run_beats_a_newer_unevaluated_one_of_the_same_motion(queue_dirs):
    csv = queue_dirs / "lunge_v3.csv"
    done = _run(queue_dirs, "lunge_v3_pipe", "2026-11-10_04-02-40", [2999], evaluated=[2999])
    _run(queue_dirs, "lunge_v3_pipe6k", "2026-11-11_03-09-39", [2999])
    assert pipeline.queue_item(csv, _queue_args()).run_dir == done


def test_resume_ignores_runs_of_another_task_or_another_csv(queue_dirs):
    import hashlib

    csv = queue_dirs / "en_garde_v4.csv"
    csv.write_text("1,2,3\n")
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    _run(queue_dirs, "en_garde_v4_tip", "2026-11-12_10-00-00", [2999], evaluated=[2999], task=pipeline.SABER_TASK)
    _run(queue_dirs, "en_garde_v4_pipe", "2026-11-12_11-00-00", [2999], evaluated=[2999], csv_sha="0" * 64)
    assert pipeline.queue_item(csv, _queue_args()).state == "train"
    match = _run(queue_dirs, "en_garde_v4_old", "2026-11-12_09-00-00", [2999], evaluated=[2999], csv_sha=sha)
    assert pipeline.queue_item(csv, _queue_args()).run_dir == match
    no_meta = queue_dirs / "runs" / "g1_tracking" / "2026-11-12_12-00-00_en_garde_v4_x"
    no_meta.mkdir()
    (no_meta / "model_2999.pt").write_text("")  # no run.json: its task is unknown, so it is not used
    assert pipeline.queue_item(csv, _queue_args()).run_dir == match


def test_a_run_is_not_done_unless_its_verdict_files_parse(queue_dirs):
    csv = queue_dirs / "advance_v2.csv"
    run = _run(queue_dirs, "advance_v2_pipe", "2026-11-12_10-00-00", [2999], evaluated=[2999])
    out = queue_dirs / "eval" / run.name / "model_2999"
    (out / "sim2sim.json").write_text('{"verdict": ')  # cut off mid-write
    assert pipeline.queue_item(csv, _queue_args()).state == "evaluate"
    (out / "sim2sim.json").write_text('{"nominal": {}}')  # no verdict
    assert pipeline.queue_item(csv, _queue_args()).state == "evaluate"
    (out / "sim2sim.json").write_text('{"verdict": "PASS"}')
    assert pipeline.queue_item(csv, _queue_args()).state == "done"


def test_the_entry_motion_is_queued_as_both_variants_unless_one_is_named(queue_dirs):
    csv = queue_dirs / "stand_to_en_garde_v1.csv"
    tip = _run(queue_dirs, "stand_to_en_garde_v1_tip", "2026-11-11_15-31-43", [2999], evaluated=[2999],
               task=pipeline.SABER_TASK, experiment="g1_saber_tracking")
    items = pipeline.queue_items([csv], _queue_args())
    assert [(i.label, i.task, i.run_name) for i in items] == [
        ("stand_to_en_garde_v1_tip", pipeline.SABER_TASK, "stand_to_en_garde_v1_tip"),
        ("stand_to_en_garde_v1_feet", pipeline.ENTRY_TASK, "stand_to_en_garde_v1_feet")]
    assert (items[0].state, items[0].run_dir) == ("done", tip)
    assert items[1].state == "train"  # the tip run is on the saber task, not a feet run
    only = pipeline.queue_items([csv], _queue_args(entry_variant="feet"))
    assert [i.label for i in only] == ["stand_to_en_garde_v1_feet"]
    explicit = pipeline.queue_items([csv], _queue_args(task=pipeline.ENTRY_TASK))
    assert [(i.label, i.task) for i in explicit] == [("stand_to_en_garde_v1", pipeline.ENTRY_TASK)]
    # The joint variants are trained only when named: "both" stays tip + feet.
    joint = pipeline.queue_items([csv], _queue_args(entry_variant="feetjoint05"))
    assert [(i.label, i.task, i.run_name) for i in joint] == [
        ("stand_to_en_garde_v1_feetjoint05", pipeline.ENTRY_JOINT_VARIANTS["feetjoint05"],
         "stand_to_en_garde_v1_feetjoint05")]


def test_queue_never_runs_more_than_two_at_once_and_skips_done_motions():
    import threading

    lock, active, peak, ran = threading.Lock(), [0], [0], []

    def work(item):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
            ran.append(item.motion)
        return {"motion": item.motion, "stages": {"train": True}}

    items = [pipeline.QueueItem(Path(f"m{i}_v1.csv"), f"m{i}_v1", "t", f"m{i}_v1_pipe", "train") for i in range(5)]
    items.append(pipeline.QueueItem(Path("done_v1.csv"), "done_v1", "t", "done_v1_pipe", "done"))
    results = pipeline.run_queue(items, work)
    assert peak[0] == 2
    assert sorted(ran) == [f"m{i}_v1" for i in range(5)] and len(results) == 5
    with pytest.raises(ValueError, match="one GPU"):
        pipeline.run_queue(items, work, max_parallel=3)


def test_one_failing_motion_does_not_stop_the_queue():
    def work(item):
        if item.motion == "bad_v1":
            raise RuntimeError("CUDA out of memory")
        return {"motion": item.motion, "stages": {"train": True}}

    items = [pipeline.QueueItem(Path(f"{n}.csv"), n, "t", f"{n}_pipe", "train") for n in ("bad_v1", "good_v1")]
    results = {r["label"]: r for r in pipeline.run_queue(items, work)}
    assert results["good_v1"]["stages"] == {"train": True}
    assert results["bad_v1"]["stages"] == {"queue": False} and "CUDA out of memory" in results["bad_v1"]["error"]


def test_mtc_runs_one_at_a_time_and_only_after_every_training():
    import threading

    lock, events = threading.Lock(), []

    def work(item):
        with lock:
            events.append(("train start", item.motion))
        time.sleep(0.03)
        with lock:
            events.append(("train end", item.motion))
        return {"motion": item.motion, "stages": {"train": True}}

    def mtc_work(item):
        with lock:
            events.append(("mtc start", item.motion))
        time.sleep(0.01)
        with lock:
            events.append(("mtc end", item.motion))
        return {"label": item.label, "mtc": "exit 0", "ok": True}

    items = [pipeline.QueueItem(Path(f"m{i}.csv"), f"m{i}", "t", f"m{i}_pipe", "train") for i in range(4)]
    _, mtc = pipeline.execute_queue(items, work, mtc_work)
    kinds = [kind for kind, _ in events]
    last_train = max(i for i, k in enumerate(kinds) if k == "train end")
    first_mtc = kinds.index("mtc start")
    assert last_train < first_mtc
    assert kinds[first_mtc:] == ["mtc start", "mtc end"] * 4  # never overlapping
    assert len(mtc) == 4
    _, none = pipeline.execute_queue(items, work)  # without --mtc nothing runs after training
    assert none == []


def test_an_mtc_failure_or_invalid_makes_the_queue_fail_and_shows_in_the_table(queue_dirs):
    trained_ok = [{"label": "a_v1", "stages": {"train": True}}]
    assert pipeline.queue_failures(trained_ok, [{"label": "a_v1", "mtc": "exit 0", "ok": True}]) == []
    assert pipeline.queue_failures(trained_ok, [{"label": "a_v1", "mtc": "exit 3", "ok": False}]) == ["a_v1 mtc"]
    assert pipeline.queue_failures([{"label": "b_v1", "stages": {"eval": False}}], []) == ["b_v1"]
    csv = queue_dirs / "lunge_v6.csv"
    run = _run(queue_dirs, "lunge_v6_pipe", "2026-11-12_10-00-00", [2999], evaluated=[2999], mtc="INVALID")
    item = pipeline.queue_item(csv, _queue_args())
    row = pipeline.verdict_rows([item], [], _queue_args(), [{"label": "lunge_v6", "mtc": "exit 3", "ok": False}])[0]
    assert row["mtc"] == "INVALID (exit 3)"
    (queue_dirs / "eval" / run.name / "model_2999" / "mtc.json").unlink()
    row = pipeline.verdict_rows([item], [], _queue_args(), [{"label": "lunge_v6", "mtc": "exit 1", "ok": False}])[0]
    assert row["mtc"] == "failed (exit 1)"


def test_queue_dry_run_prints_the_exact_commands_and_writes_nothing(queue_dirs, capsys):
    fresh = queue_dirs / "parry_quinte_v4.csv"
    done = queue_dirs / "lunge_v6.csv"
    measured = queue_dirs / "retreat_v2.csv"
    entry = queue_dirs / "stand_to_en_garde_v2.csv"
    for csv in (fresh, done, measured, entry):
        csv.write_text("")
    run = _run(queue_dirs, "lunge_v6_pipe", "2026-11-12_10-00-00", [2999], evaluated=[2999])
    _run(queue_dirs, "retreat_v2_pipe", "2026-11-12_10-00-00", [2999], evaluated=[2999], mtc="PASS")
    code = pipeline.main(["--dry-run", "queue", str(fresh), str(done), str(measured), str(entry), "--mtc"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert not (queue_dirs / "logs").exists() and not (queue_dirs / "eval" / "pipeline").exists()
    train = next(line for line in out if "train run" in line and "parry_quinte_v4" in line)
    assert pipeline.SABER_TASK in train and "--iterations 3000" in train and "--num-envs 4096" in train
    assert train.startswith(f"{sys.executable} -m fencing.cli train run")
    assert any("sim2sim '<new run dir of parry_quinte_v4_pipe>' --checkpoint model_2999.pt --video" in line for line in out)
    entry_trains = [line for line in out if "train run" in line and "stand_to_en_garde_v2" in line]
    assert len(entry_trains) == 2
    assert "--name stand_to_en_garde_v2_tip --task " + pipeline.SABER_TASK + " " in entry_trains[0]
    assert "--name stand_to_en_garde_v2_feet --task " + pipeline.ENTRY_TASK + " " in entry_trains[1]
    assert any(line.startswith("# lunge_v6") and "skip" in line for line in out)
    assert not any("train run" in line and ("lunge_v6" in line or "retreat_v2" in line) for line in out)
    mtc_lines = [i for i, line in enumerate(out) if "mtc_sim2sim.sh" in line]
    # parry_quinte, lunge (evaluated, no mtc.json), both entry variants; not retreat (valid mtc.json)
    assert len(mtc_lines) == 4 and min(mtc_lines) > max(i for i, line in enumerate(out) if "fencing.cli" in line)
    assert any(f"{run} --checkpoint 2999" in out[i] for i in mtc_lines)
    assert not any("retreat_v2" in out[i] for i in mtc_lines)


def test_queue_goal1_includes_the_entry_motion(queue_dirs, monkeypatch, capsys):
    motions = queue_dirs / "motions"
    motions.mkdir()
    for name in ("en_garde_v3.csv", "en_garde_v4.csv", "stand_to_en_garde_v1.csv", "cut_head_v4.csv"):
        (motions / name).write_text("")
    monkeypatch.setattr(pipeline.paths, "MOTIONS_DIR", motions)
    assert pipeline.main(["--dry-run", "queue", "--goal1"]) == 0
    headers = [line for line in capsys.readouterr().out.splitlines() if line.startswith("# ") and "(" in line]
    assert [h.split()[1] for h in headers] == ["en_garde_v4", "cut_head_v4", "stand_to_en_garde_v1_tip",
                                               "stand_to_en_garde_v1_feet"]


def test_goal1_picks_the_blade_line_thrust_flank_and_extension_not_the_v4s():
    # thrust_v4 swept its tip to 7 cm above the floor, cut_flank_v4 to 0.34 m, and extension_v4
    # detoured 2.24x the straight distance (docs/saber.md, GVHMR set); `queue --goal1`, `train --goal1`
    # and `all` take the newest version, which must be the blade_path="line" rebuild.
    csvs = pipeline._queue_csvs(argparse.Namespace(motions=[], goal1=True))
    names = {path.stem for path in csvs}
    assert {"thrust_v5", "cut_flank_v5", "extension_v5"} <= names
    assert not names & {f"{n}_v{v}" for n in ("thrust", "cut_flank", "extension") for v in (3, 4)}
    for name in ("thrust", "cut_flank", "extension"):
        record = json.loads((pipeline.paths.MOTIONS_DIR / f"{name}_v5.json").read_text())
        assert record["blade"]["summary"]["config"]["blade_path"] == "line", name


def test_verdict_table_reads_the_verdicts_on_disk(queue_dirs):
    csv = queue_dirs / "lunge_v6.csv"
    _run(queue_dirs, "lunge_v6_pipe", "2026-11-12_10-00-00", [2999], evaluated=[2999])
    item = pipeline.queue_item(csv, _queue_args())
    rows = pipeline.verdict_rows([item], [], _queue_args())
    assert rows[0]["mjlab"] == "PASS" and rows[0]["sim2sim"] == "FAIL" and rows[0]["mtc"] == "-"
    assert rows[0]["this_run"] == "skipped (evaluated before)" and rows[0]["checkpoint"] == "model_2999.pt"
    table = pipeline.format_table(rows)
    assert table.splitlines()[0].split()[:5] == ["motion", "task", "mjlab", "sim2sim", "mtc"]


def test_summary_survives_null_metrics_from_a_run_where_every_env_fell():
    metrics = {"verdict": "FAIL", "nominal": {"success_rate": 0.0, "num_episodes": 8, "joint_angle_error_rad": None,
                                              "body_pos_error_m": None, "foot_slip_m_s": None}}
    text = pipeline.summarize_result(_result(metrics=metrics))
    assert "success 0.00 of 8" in text and "body n/a" in text


def test_pipeline_blade_stage_takes_each_blade_path_from_the_gvhmr_recipe(monkeypatch):
    # `pipeline all` runs this stage with --publish; on the GVHMR guard a joint-space thrust,
    # cut_flank or extension is the superseded v4 again.
    from fencing.motion import cli as motion_cli
    from fencing.motion import rebuild
    from fencing.saber import recipes

    seen = {}

    def fake_run_blade(name, base, args, script, recipe=None):
        seen[name] = motion_cli._blade_config(args, recipe).blade_path
        return 0

    monkeypatch.setattr(motion_cli, "_run_blade", fake_run_blade)
    cmd = pipeline.blade_args(publish=False)
    args = motion_cli.build_parser().parse_args(cmd[1:])
    assert args.func(args) == 0
    from_recipe = rebuild.blade_paths(rebuild.load_recipe(rebuild.GVHMR_RECIPE))
    assert from_recipe == {"thrust": "line", "cut_flank": "line", "extension": "line"}
    assert seen == {name: from_recipe.get(name, "joint") for name in recipes.GOAL1_RECIPES}
    seen.clear()
    args = motion_cli.build_parser().parse_args(cmd[1:] + ["--blade-path", "joint"])  # reproduces the v4s
    assert args.func(args) == 0 and set(seen.values()) == {"joint"}


def test_blade_paths_refuses_a_recipe_that_disagrees_with_itself(tmp_path):
    from fencing.motion import rebuild

    step = '[motions.{0}]\nsource = "motion:en_garde"\n[[motions.{0}.steps]]\nop = "blade"\nrecipe = "thrust"\n' \
           'config = {{ blade_path = "{1}" }}\n'
    text = ('schema = 1\n[motions.en_garde]\nsource = "clip:x"\n[[motions.en_garde.steps]]\nop = "clean"\n'
            'label = "en_garde"\n' + step.format("thrust", "line") + step.format("cut_x", "joint"))
    (tmp_path / "r.toml").write_text(text)
    with pytest.raises(rebuild.RecipeError, match="blade_path"):
        rebuild.blade_paths(rebuild.load_recipe(tmp_path / "r.toml"))
