"""The chained start in MTC (docs/mtc.md, "Entry from standby"): standby_controller -> entry
policy (walking_controller, R1 + A) -> motion policy (walking_controller1, R1 + X).

One recording holds both policies. The motion is scored from the second switch exactly as a
standby start is scored from the first, with the entry policy in standby's place; the entry is
scored from the first switch up to the second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fencing.sim2sim import mtc


def _recording():
    return {
        "activation_t": np.array([4.0, 4.5, 9.0]),  # walking_controller active (the entry)
        "activity": np.array([[1.0, 1, 0], [4.0, 0, 1], [14.0, 0, 0]], dtype=float),
        "stat_period": np.array([[50.0, 49.0, 51.0, 50.0]]),
        "activation1_t": np.array([14.0, 15.0]),  # walking_controller1 active (the motion)
        "activity_chain": np.array([[4.0, 1, 0], [14.0, 0, 1]], dtype=float),
        "stat1_period": np.array([[50.0, 48.0, 52.0, 50.0]]),
        "joint_t": np.arange(0.0, 20.0, 0.002),
    }


def test_chained_runs_write_their_own_json_and_recordings_never_the_held_or_standby_ones():
    assert mtc.run_outputs(standby=True, chain=True) == ("mtc_entry.json", "runs_entry")
    assert mtc.run_outputs(standby=False) == ("mtc.json", "runs")
    assert mtc.run_outputs(standby=True) == ("mtc_standby.json", "runs_standby")


def test_motion_view_puts_walking_controller1_where_the_scorer_reads_the_policy_and_the_entry_in_standbys_place():
    rec = _recording()
    view = mtc.chain_motion_view(rec)
    np.testing.assert_array_equal(view["activation_t"], [14.0, 15.0])
    np.testing.assert_array_equal(view["stat_period"], rec["stat1_period"])
    # standby_state_report reads column 1 as "the previous controller active": now the entry.
    np.testing.assert_array_equal(view["activity"], rec["activity_chain"])
    assert view["joint_t"] is rec["joint_t"]
    np.testing.assert_array_equal(rec["activation_t"], [4.0, 4.5, 9.0])  # the original is untouched


def test_motion_view_refuses_a_recording_where_walking_controller1_never_became_active():
    rec = _recording()
    rec["activation1_t"] = np.zeros(0)
    with pytest.raises(mtc.RecordingIncomplete, match="walking_controller1"):
        mtc.chain_motion_view(rec)


def test_entry_is_scored_with_the_hold_that_fits_before_the_second_switch():
    # Entry active at 4.0 s, motion at 14.0 s, entry reference 7.5 s: 2.5 s of hold fit, less two
    # policy steps so no scored state comes from after the switch; never more than the asked hold.
    assert mtc.chain_entry_hold_seconds(4.0, 14.0, entry_frames=375, hold_seconds=2.0) == pytest.approx(2.0)
    assert mtc.chain_entry_hold_seconds(4.0, 13.0, entry_frames=375, hold_seconds=2.0) == pytest.approx(1.46)
    with pytest.raises(mtc.RecordingIncomplete, match="before the entry reference ended"):
        mtc.chain_entry_hold_seconds(4.0, 11.0, entry_frames=375, hold_seconds=2.0)


def test_plan_with_an_entry_prints_both_exports_and_the_second_switch_time(tmp_path, capsys, monkeypatch):
    motion, entry = tmp_path / "motion" / "model_2999", tmp_path / "entry" / "model_2999"
    for export, frames in ((motion, 150), (entry, 375)):
        export.mkdir(parents=True)
        (export / "export.json").write_text(f'{{"motion_frames": {frames}}}')
    monkeypatch.setattr("fencing.sim2sim.cli.resolve_export_dir",
                        lambda target, checkpoint: entry if Path(target).parent.name == "entry" else motion)
    assert mtc.main([str(motion), "--plan", "--standby", "--entry", str(entry)]) == 0
    # Motion: 3 s + 2 s hold + 1 s margin past walking_controller1. Second switch: entry 7.5 s +
    # its 2 s hold + 0.5 s, past walking_controller.
    assert capsys.readouterr().out.split() == [str(motion), "6", "mtc_entry.json", "runs_entry", str(entry), "10"]


def test_entry_requires_standby():
    with pytest.raises(SystemExit):
        mtc.main(["x", "--plan", "--entry", "y"])


def _scored(activation_s: float) -> dict:
    """A score_recording result with only the fields main reads."""
    return {
        "timing": {"activation_sim_s": activation_s, "first_policy_tick_after_activation_s": 0.02,
                   "real_time_factor": 1.0},
        "standby": {"start_state": {"still": True, "reasons": []},
                    "convergence": {"line_rad": 0.1, "window_s": 0.2, "step": 0, "after_switch_s": 0.0}},
        "estimator": None, "terminated_at_step": None, "terminated_at_s": None, "termination_terms": [],
        "nominal": {}, "settle_seconds": 0.0, "hold_steps": 100, "motion_steps": 150, "steps_run": 250,
        "recording_complete": True,
    }


def _chained_main(tmp_path, monkeypatch, motion_timing_problems):
    """Run main on a chained recording with the simulator and scorer stubbed; return the report."""
    import json

    motion, entry = tmp_path / "motion" / "model_2999", tmp_path / "entry_run" / "model_2999"
    for export in (motion, entry):
        export.mkdir(parents=True)
        (export / "export.json").write_text('{"motion_frames": 375}')
    (entry / "policy.onnx").write_bytes(b"entry policy bytes")
    monkeypatch.setattr("fencing.sim2sim.cli.resolve_export_dir",
                        lambda target, checkpoint: entry if Path(target).parent.name == "entry_run" else motion)
    monkeypatch.setattr("fencing.sim2sim.cli.summary_line", lambda nominal: "")
    monkeypatch.setattr("fencing.sim2sim.mtc_plant.policy_plant", lambda export_dir: "saber")
    monkeypatch.setattr("fencing.sim2sim.mtc_plant.require_plant", lambda plant, mjcf: None)
    monkeypatch.setattr(mtc, "load_recording", lambda path: _recording())
    monkeypatch.setattr(mtc, "score_recording",
                        lambda export_dir, rec, *a, **k: _scored(float(rec["activation_t"][0])))
    monkeypatch.setattr(mtc, "_judge", lambda result: ([], "PASS"))
    monkeypatch.setattr(mtc, "after_convergence", lambda *a: {"nominal": {}, "verdict": "PASS"})
    monkeypatch.setattr(mtc, "timing_problems",
                        lambda timing: motion_timing_problems if timing["activation_sim_s"] == 14.0 else [])
    out = tmp_path / "mtc_entry.json"
    code = mtc.main([str(motion), "--standby", "--entry", str(entry), "--recording", str(tmp_path / "bag.npz"),
                     "--out", str(out)])
    return code, json.loads(out.read_text())


def test_an_invalid_chained_run_reports_no_judged_verdict_anywhere(tmp_path, monkeypatch):
    code, report = _chained_main(tmp_path, monkeypatch, ["first policy tick 300 ms after activation"])
    assert code == mtc.EXIT_INVALID
    assert report["verdict"] == "INVALID"
    # A reader of the nested blocks alone must not see a PASS the run never earned.
    assert report["entry"]["verdict"] == "INVALID"
    assert report["after_convergence"]["verdict"] == "INVALID"


def test_a_chained_report_names_the_entry_run_and_its_onnx(tmp_path, monkeypatch):
    import hashlib

    code, report = _chained_main(tmp_path, monkeypatch, [])
    assert code == 0
    assert report["verdict"] == report["entry"]["verdict"] == report["after_convergence"]["verdict"] == "PASS"
    assert report["entry"]["run"] == "entry_run"
    assert report["entry"]["onnx_sha256"] == hashlib.sha256(b"entry policy bytes").hexdigest()
