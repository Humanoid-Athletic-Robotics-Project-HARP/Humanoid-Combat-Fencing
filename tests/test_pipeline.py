"""fencing.pipeline: task choice, version picking, log rows, and the dry-run stage plan."""

from __future__ import annotations

import argparse

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
    assert set(sub.choices) == {"data", "train", "all", "status"}


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


def test_summary_survives_null_metrics_from_a_run_where_every_env_fell():
    metrics = {"verdict": "FAIL", "nominal": {"success_rate": 0.0, "num_episodes": 8, "joint_angle_error_rad": None,
                                              "body_pos_error_m": None, "foot_slip_m_s": None}}
    text = pipeline.summarize_result(_result(metrics=metrics))
    assert "success 0.00 of 8" in text and "body n/a" in text
