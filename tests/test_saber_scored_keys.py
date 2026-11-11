"""Scored key frames of bladework motions: derived from blade keys, recorded in the sidecar,
resolved from a training NPZ to policy steps."""

from __future__ import annotations

import hashlib
import json

import pytest

from fencing.saber import scored_keys


def _key(index, action, frame, kind="parry", target=None, time_s=None):
    return {"index": index, "action": action, "target": target or action, "kind": kind, "frame": frame,
            "time_s": round(frame / 30.0, 4) if time_s is None else time_s}


STATIC_KEYS = [_key(0, "en_garde", 0, "guard"), _key(1, "parry_quinte", 27), _key(2, "en_garde", 60, "guard")]


def test_every_key_but_the_rest_guard_and_base_is_scored():
    keys = [_key(0, "en_garde", 0, "guard"), _key(1, "extension", 32, "guard"), _key(2, "base", 50, "base"),
            _key(3, "point_thrust", 70, "thrust", target="point_thrust"), _key(4, "en_garde", 90, "guard")]
    frames = scored_keys.key_frames_from_blade_keys(keys, fps=30.0)
    assert [k["target"] for k in frames] == ["extension", "point_thrust"]
    assert [k["index"] for k in frames] == [1, 3]
    assert frames[0] == {"index": 1, "target": "extension", "kind": "guard", "frame": 32,
                         "time_s": pytest.approx(32 / 30.0)}


def test_key_time_comes_from_the_frame_not_the_rounded_report_time():
    frames = scored_keys.key_frames_from_blade_keys([_key(1, "en_garde_x", 149, time_s=4.9667)], fps=30.0)
    assert frames[0]["time_s"] == pytest.approx(149 / 30.0, abs=1e-6)


def test_with_key_frames_adds_only_the_key_frames_field():
    record = {"fps": 30.0, "frames": 121, "blade": {"keys": STATIC_KEYS}, "file": "parry_quinte_v3.csv"}
    updated = scored_keys.with_key_frames(record)
    assert {k: v for k, v in updated.items() if k != "key_frames"} == record
    assert "key_frames" not in record  # the input is not mutated
    assert [k["target"] for k in updated["key_frames"]] == ["parry_quinte"]


def test_with_key_frames_refuses_a_sidecar_without_blade_keys():
    with pytest.raises(ValueError, match="blade keys"):
        scored_keys.with_key_frames({"fps": 30.0, "file": "lunge_v3.csv"})


def test_key_steps_map_seconds_to_policy_steps_and_drop_keys_past_the_end():
    keys = [{"time_s": 3.1}, {"time_s": 26 / 30.0}, {"time_s": 9.0}]
    assert scored_keys.key_steps(keys, step_dt=0.02, num_frames=355) == [155, 43]


def _published(tmp_path, record, csv_text="0,0,0\n"):
    """A motion CSV + sidecar and the NPZ + NPZ sidecar `fencing train npz` writes for it."""
    csv = tmp_path / "motions" / "parry_quinte_v3.csv"
    csv.parent.mkdir()
    csv.write_text(csv_text)
    csv.with_suffix(".json").write_text(json.dumps(record))
    npz = tmp_path / "npz" / "parry_quinte_v3_mjlab.npz"
    npz.parent.mkdir()
    npz.write_bytes(b"npz")
    npz.with_suffix(".json").write_text(json.dumps({
        "source_csv": str(csv), "source_csv_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
        "input_fps": 30.0, "output_fps": 50}))
    return csv, npz


def test_resolve_follows_the_npz_sidecar_to_the_motion_sidecar(tmp_path):
    record = scored_keys.with_key_frames({"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    csv, npz = _published(tmp_path, record)
    resolved = scored_keys.resolve_for_npz(npz, step_dt=0.02, num_frames=200)
    assert resolved.steps == (45,)
    assert resolved.sidecar == str(csv.with_suffix(".json"))
    assert resolved.report() == {"sidecar": str(csv.with_suffix(".json")), "steps": [45],
                                 "keys": record["key_frames"]}


def test_resolve_needs_recorded_key_frames_not_just_blade_keys(tmp_path):
    """Only the recorded field counts: an un-backfilled sidecar has no key frames."""
    _, npz = _published(tmp_path, {"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    assert scored_keys.resolve_for_npz(npz, step_dt=0.02, num_frames=200) is None


def test_resolve_is_none_without_a_traceable_motion(tmp_path):
    assert scored_keys.resolve_for_npz(tmp_path / "missing_mjlab.npz", 0.02, 100) is None
    assert scored_keys.resolve_for_npz(None, 0.02, 100) is None
    npz = tmp_path / "loose_mjlab.npz"
    npz.write_bytes(b"npz")  # no NPZ sidecar
    assert scored_keys.resolve_for_npz(npz, 0.02, 100) is None


def test_resolve_refuses_a_csv_changed_since_the_npz_was_made(tmp_path, capsys):
    record = scored_keys.with_key_frames({"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    csv, npz = _published(tmp_path, record)
    csv.write_text("1,1,1\n")
    assert scored_keys.resolve_for_npz(npz, 0.02, 200) is None
    assert "sha256" in capsys.readouterr().err


def test_resolve_with_every_key_past_the_motion_end_is_none_like_no_key_frames(tmp_path):
    """None, not an empty list, so eval and sim2sim reports match a run with no key frames."""
    record = scored_keys.with_key_frames({"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    _, npz = _published(tmp_path, record)
    assert scored_keys.resolve_for_npz(npz, step_dt=0.02, num_frames=40) is None


def test_resolve_refuses_an_npz_replaced_since_its_record_was_written(tmp_path, capsys):
    """An older record under a re-converted NPZ name must not hand out another motion's keys."""
    record = scored_keys.with_key_frames({"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    _, npz = _published(tmp_path, record)
    npz_record = json.loads(npz.with_suffix(".json").read_text())
    npz_record["npz_sha256"] = hashlib.sha256(b"npz").hexdigest()
    npz.with_suffix(".json").write_text(json.dumps(npz_record))
    assert scored_keys.resolve_for_npz(npz, 0.02, 200).steps == (45,)
    npz.write_bytes(b"npz from a different csv")
    assert scored_keys.resolve_for_npz(npz, 0.02, 200) is None
    assert "npz_sha256" in capsys.readouterr().err


def test_resolve_without_a_recorded_npz_sha256_keeps_resolving(tmp_path):
    record = scored_keys.with_key_frames({"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    _, npz = _published(tmp_path, record)
    npz.write_bytes(b"changed, but the record never stored a hash")
    assert scored_keys.resolve_for_npz(npz, 0.02, 200).steps == (45,)


def test_backfill_writes_key_frames_and_leaves_everything_else_byte_identical(tmp_path):
    from fencing.motion.io import write_sidecar

    path = write_sidecar(tmp_path / "parry_quinte_v3.json",
                         {"fps": 30.0, "frames": 121, "blade": {"keys": STATIC_KEYS}, "version": 3})
    before = json.loads(path.read_text())
    changed = scored_keys.backfill(path, write=True)
    assert changed
    after = json.loads(path.read_text())
    assert {k: v for k, v in after.items() if k != "key_frames"} == before
    assert after["key_frames"][0]["frame"] == 27
    assert scored_keys.backfill(path, write=True) is False  # idempotent


def test_backfill_dry_run_does_not_write(tmp_path):
    from fencing.motion.io import write_sidecar

    path = write_sidecar(tmp_path / "x_v1.json", {"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    raw = path.read_bytes()
    assert scored_keys.backfill(path, write=False) is True
    assert path.read_bytes() == raw


def test_key_frames_command_reports_then_writes(tmp_path, capsys):
    from fencing.motion import cli
    from fencing.motion.io import write_sidecar

    path = write_sidecar(tmp_path / "parry_quinte_v3.json", {"fps": 30.0, "blade": {"keys": STATIC_KEYS}})
    assert cli.main(["key-frames", str(path)]) == 0
    assert "key_frames" not in json.loads(path.read_text())
    assert "parry_quinte@0.900s" in capsys.readouterr().out
    assert cli.main(["key-frames", str(path), "--write"]) == 0
    assert json.loads(path.read_text())["key_frames"][0]["target"] == "parry_quinte"


def test_newest_bladework_sidecars_pick_the_newest_version_of_each_recipe(tmp_path):
    for name in ("cut_head_v2", "cut_head_v3", "parry_sixte_v1", "lunge_v4"):
        (tmp_path / f"{name}.csv").write_text("")
        (tmp_path / f"{name}.json").write_text("{}")
    found = [p.name for p in scored_keys.newest_bladework_sidecars(tmp_path)]
    assert found == ["cut_head_v3.json", "parry_sixte_v1.json"]
