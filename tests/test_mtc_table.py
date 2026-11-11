"""The docs/mtc.md results tables are printed from the mtc.json and sim2sim.json files on disk."""

from __future__ import annotations

import json

from fencing.sim2sim import mtc_table

NOMINAL = {"joint_angle_error_rad": 0.07514, "body_pos_error_m": 0.04082, "blade_tip_error_m": 0.01769,
           "anchor_pos_error_m": 0.02361, "foot_slip_m_s": 0.03661, "success_rate": 1.0, "recovery_rate": 1.0}


def test_metric_cells_are_radians_centimetres_and_metres_per_second():
    assert mtc_table.metric_cells(NOMINAL) == ["0.075", "4.08", "1.77", "2.36", "0.037", "1", "1"]


def test_plain_g1_runs_show_the_blade_tip_as_not_applicable():
    assert mtc_table.metric_cells({**NOMINAL, "blade_tip_error_m": None})[2] == "n/a"


def test_each_eval_dir_gives_an_mtc_row_then_a_python_row(tmp_path):
    d = tmp_path / "2026-11-10_06-14-28_extension_v3_tip" / "model_2999"
    d.mkdir(parents=True)
    (d / "mtc.json").write_text(json.dumps({"nominal": NOMINAL, "verdict": "PASS", "attempt": 2}))
    (d / "sim2sim.json").write_text(json.dumps({"nominal": {**NOMINAL, "joint_angle_error_rad": 0.074},
                                                "verdict": "PASS"}))
    mtc_row, python_row = mtc_table.rows(d)
    assert mtc_row.startswith("| extension_v3_tip (`2026-11-10_06-14-28_extension_v3_tip/model_2999`) | MTC |")
    assert mtc_row.endswith("| PASS |")
    assert python_row == "| | Python | 0.074 | 4.08 | 1.77 | 2.36 | 0.037 | 1 | 1 | PASS |"


def test_an_eval_dir_without_mtc_json_says_so_instead_of_inventing_numbers(tmp_path):
    d = tmp_path / "2026-11-10_06-14-28_extension_v3_tip" / "model_2999"
    d.mkdir(parents=True)
    (d / "sim2sim.json").write_text(json.dumps({"nominal": NOMINAL, "verdict": "PASS"}))
    assert "no mtc.json" in mtc_table.rows(d)[0]
