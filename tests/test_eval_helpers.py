"""Pure helpers behind `fencing train eval` and `fencing sim2sim` (no mjlab, no GPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fencing.sim2sim import cli as sim2sim_cli
from fencing.train import evaluate


class _Steps:
    """Stand-in for a torch tensor: supports += on all/indexed entries and clamp_."""

    def __init__(self, values):
        self.values = list(values)

    def __iadd__(self, other):
        self.values = [v + other for v in self.values]
        return self

    def __getitem__(self, ids):
        return _Indexed(self, ids)

    def __setitem__(self, ids, indexed):
        pass  # _Indexed already wrote through

    def clamp_(self, max):
        self.values = [min(v, max) for v in self.values]


class _Indexed:
    def __init__(self, parent, ids):
        self.parent, self.ids = parent, ids

    def __iadd__(self, other):
        for i in self.ids:
            self.parent.values[i] += other
        return self


def _command(time_steps, pending_forward=False):
    forwards = []
    command = SimpleNamespace(
        time_steps=_Steps(time_steps), _pending_forward=pending_forward,
        _env=SimpleNamespace(sim=SimpleNamespace(forward=lambda: forwards.append(1))),
        relative_updates=0,
    )

    def update_relative_body_poses():
        command.relative_updates += 1

    command.update_relative_body_poses = update_relative_body_poses
    return command, forwards


def test_hold_last_frame_clamps_instead_of_wrapping():
    command, forwards = _command([3, 4, 5])
    evaluate.hold_last_frame(command, last_frame=5)
    command._update_command()
    assert command.time_steps.values == [4, 5, 5]
    command._update_command(env_ids=[0])
    assert command.time_steps.values == [5, 5, 5]
    assert command.relative_updates == 2
    assert forwards == []


def test_hold_last_frame_runs_a_pending_forward_once():
    command, forwards = _command([0], pending_forward=True)
    evaluate.hold_last_frame(command, last_frame=10)
    command._update_command()
    command._update_command()
    assert forwards == [1]
    assert command._pending_forward is False


def test_summary_line_handles_missing_metrics():
    line = evaluate.summary_line({"success_rate": 0.5, "num_envs": 8, "body_pos_error_m": 0.0312})
    assert "success 0.5 (8 envs)" in line
    assert "body 3.12 cm" in line
    assert "blade n/a" in line


def test_mjlab_gap_diffs_shared_metrics(tmp_path):
    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(json.dumps({"seconds": 20.0, "nominal": {
        "num_envs": 256, "success_rate": 0.9, "joint_angle_error_rad": 0.05, "blade_tip_error_m": None}}))
    gap = sim2sim_cli.mjlab_gap({"success_rate": 1.0, "joint_angle_error_rad": 0.08, "blade_tip_error_m": None},
                                metrics_json)
    assert gap["mjlab_seconds"] == 20.0
    assert gap["mjlab_num_envs"] == 256
    assert set(gap["metrics"]) == {"success_rate", "joint_angle_error_rad"}
    assert gap["metrics"]["joint_angle_error_rad"]["diff"] == pytest.approx(0.03)


def test_mjlab_gap_without_metrics_or_nominal(tmp_path):
    assert sim2sim_cli.mjlab_gap({}, tmp_path / "missing.json") is None
    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(json.dumps({"nominal": None, "robustness": {}}))
    assert sim2sim_cli.mjlab_gap({"success_rate": 1.0}, metrics_json)["metrics"] == {}


def test_resolve_export_dir_accepts_an_export_dir(tmp_path):
    (tmp_path / "policy.onnx").write_bytes(b"")
    assert sim2sim_cli.resolve_export_dir(str(tmp_path), None) == tmp_path.resolve()
    with pytest.raises(ValueError, match="--checkpoint"):
        sim2sim_cli.resolve_export_dir(str(tmp_path), "500")
