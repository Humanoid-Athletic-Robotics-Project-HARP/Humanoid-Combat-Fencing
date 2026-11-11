"""run.json provenance, run/NPZ naming and the lock-safe NPZ conversion wrapper."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fencing.train import naming, npz, provenance


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_npz_path_appends_mjlab_suffix_once(tmp_path):
    assert naming.npz_path("en_garde_v1", tmp_path) == tmp_path / "en_garde_v1_mjlab.npz"
    assert naming.npz_path("en_garde_v1_mjlab", tmp_path) == tmp_path / "en_garde_v1_mjlab.npz"


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", " lead", "-flag"])
def test_npz_path_rejects_unsafe_names(tmp_path, bad):
    with pytest.raises(ValueError, match="invalid motion name"):
        naming.npz_path(bad, tmp_path)


def test_motion_name_strips_extension_and_mjlab_suffix():
    assert naming.motion_name_from_path("/x/dance1_subject2.csv") == "dance1_subject2"
    assert naming.motion_name_from_path("/x/dance1_subject2_mjlab.npz") == "dance1_subject2"


def test_run_name_convention_and_run_dir_parsing():
    assert naming.run_name("en_garde", "v1") == "en_garde_v1"
    with pytest.raises(ValueError, match="already contains"):
        naming.run_name("en_garde", "en_garde_v1")
    assert naming.run_name_from_dir("2026-11-09_04-33-56_p12_dance1_subject2") == "p12_dance1_subject2"
    assert naming.run_name_from_dir("2026-11-09_04-33-56") == "2026-11-09_04-33-56"


def test_checkpoints_sort_numerically_and_resolve(tmp_path):
    for i in (0, 1000, 500):
        _write(tmp_path / f"model_{i}.pt", b"x")
    assert [p.name for p in naming.list_checkpoints(tmp_path)] == ["model_0.pt", "model_500.pt", "model_1000.pt"]
    assert naming.resolve_checkpoint(tmp_path, None).name == "model_1000.pt"
    assert naming.resolve_checkpoint(tmp_path, "500").name == "model_500.pt"
    with pytest.raises(FileNotFoundError, match="model_500.pt"):
        naming.resolve_checkpoint(tmp_path, "750")


def test_run_json_records_motion_hashes_and_settings(tmp_path):
    csv = _write(tmp_path / "motion.csv", b"0,0,0.8,0,0,0,1\n")
    motion_npz = _write(tmp_path / "motion_mjlab.npz", b"npz-bytes")
    started = datetime(2026, 11, 9, 4, 33, 56, tzinfo=timezone.utc)
    record = provenance.build_run_record(
        run_name="motion_v1",
        task="Mjlab-Tracking-Flat-Unitree-G1",
        motion_npz=motion_npz,
        motion_csv=csv,
        iterations=3000,
        num_envs=4096,
        started_at=started,
    )
    out = provenance.write_run_json(tmp_path / "run", record)
    data = json.loads(out.read_text())
    assert data["motion_csv_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert data["motion_npz_sha256"] == hashlib.sha256(b"npz-bytes").hexdigest()
    assert data["task"] == "Mjlab-Tracking-Flat-Unitree-G1"
    assert (data["iterations"], data["num_envs"]) == (3000, 4096)
    assert data["started_at"] == "2026-11-09T04:33:56+00:00"
    assert data["repo_commit"]  # the repo is a git checkout
    assert provenance.read_run_json(tmp_path / "run") == data


def test_run_json_fails_loudly_on_missing_motion(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.npz"):
        provenance.build_run_record(
            run_name="x_v1", task="t", motion_npz=tmp_path / "nope.npz", motion_csv=None,
            iterations=1, num_envs=1,
        )


def test_run_json_rejects_nonpositive_iterations(tmp_path):
    motion_npz = _write(tmp_path / "m.npz", b"x")
    with pytest.raises(ValueError, match="iterations"):
        provenance.build_run_record(
            run_name="x_v1", task="t", motion_npz=motion_npz, motion_csv=None, iterations=0, num_envs=1
        )


def test_npz_conversion_is_serialised_by_the_lock_and_copies_its_own_output(tmp_path):
    """Two conversions share mjlab's fixed /tmp/motion.npz; each must get its own bytes."""
    shared_tmp = tmp_path / "motion.npz"
    lock_file = tmp_path / "motion.npz.lock"
    active = []
    overlaps = []

    def fake_converter(csv_path: Path, content: bytes):
        def run() -> None:
            active.append(csv_path)
            if len(active) > 1:
                overlaps.append(tuple(active))
            time.sleep(0.05)
            shared_tmp.write_bytes(content)
            time.sleep(0.05)
            active.remove(csv_path)

        return run

    results = {}

    def convert(name: str, content: bytes) -> None:
        csv = _write(tmp_path / f"{name}.csv", b"csv")
        results[name] = npz.convert_with_lock(
            run_converter=fake_converter(csv, content),
            output=naming.npz_path(name, tmp_path / "npz"),
            tmp_output=shared_tmp,
            lock_path=lock_file,
        )

    threads = [threading.Thread(target=convert, args=(f"m{i}", f"bytes-{i}".encode())) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlaps
    for i in range(3):
        assert results[f"m{i}"].read_bytes() == f"bytes-{i}".encode()
        assert results[f"m{i}"].name == f"m{i}_mjlab.npz"


def test_npz_conversion_fails_if_converter_did_not_write_a_fresh_file(tmp_path):
    shared_tmp = _write(tmp_path / "motion.npz", b"stale from another run")
    with pytest.raises(RuntimeError, match="did not write"):
        npz.convert_with_lock(
            run_converter=lambda: None,
            output=tmp_path / "out_mjlab.npz",
            tmp_output=shared_tmp,
            lock_path=tmp_path / "lock",
        )


_ENV_YAML_ACTIONS = """\
commands:
  motion:
    motion_file: /x/en_garde_mjlab.npz
actions:
  joint_pos:
    entity_name: robot
    clip: null
    scale:
      .*_elbow_joint: 0.43857731392336724
      .*_wrist_pitch_joint: 0.07450087032950714
      right_wrist_yaw_joint: 0.25
    offset: 0.0
    preserve_order: false
events:
  push_robot:
    params:
      scale: 9.0
"""


def test_trained_action_scale_reads_the_joint_pos_scale_block(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text(_ENV_YAML_ACTIONS)
    assert provenance.trained_action_scale(tmp_path) == {
        ".*_elbow_joint": 0.43857731392336724,
        ".*_wrist_pitch_joint": 0.07450087032950714,
        "right_wrist_yaw_joint": 0.25,
    }


def test_trained_action_scale_is_none_without_env_yaml_or_a_dict_scale(tmp_path):
    assert provenance.trained_action_scale(tmp_path) is None
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text("actions:\n  joint_pos:\n    scale: 0.5\n    offset: 0.0\n")
    assert provenance.trained_action_scale(tmp_path) is None
