import shutil

import numpy as np
import pytest

from fencing import paths
from fencing.motion.io import Motion, read_csv
from fencing.motion.publish import Provenance, PublishError, next_version, publish_motion


@pytest.fixture
def motions_dir(tmp_path):
    shutil.copy(paths.MOTIONS_DIR / "README.md", tmp_path / "README.md")
    return tmp_path


def tiny_motion(value=0.0):
    n = 5
    return Motion(np.zeros((n, 3)) + value, np.tile([0, 0, 0, 1.0], (n, 1)), np.zeros((n, 29)))


PROV = Provenance("abc123_1000-2000", "frames 0-5", "R", "smooth", "pass")


def test_versions_auto_increment_and_never_overwrite(motions_dir):
    first = publish_motion(tiny_motion(0.0), "lunge", PROV, motions_dir)
    second = publish_motion(tiny_motion(1.0), "lunge", PROV, motions_dir)
    other = publish_motion(tiny_motion(2.0), "advance", PROV, motions_dir)
    assert (first.name, second.name, other.name) == ("lunge_v1.csv", "lunge_v2.csv", "advance_v1.csv")
    assert read_csv(first).root_pos[0, 0] == 0.0  # v1 untouched by v2
    assert next_version("lunge", motions_dir) == 3
    assert first.with_suffix(".json").exists()


def test_version_skips_gaps_and_ignores_other_names(motions_dir):
    (motions_dir / "lunge_v7.csv").write_text("")
    (motions_dir / "advance_lunge_v9.csv").write_text("")
    assert next_version("lunge", motions_dir) == 8


def test_provenance_rows_are_appended_to_the_table(motions_dir):
    publish_motion(tiny_motion(), "lunge", PROV, motions_dir)
    publish_motion(tiny_motion(), "lunge", Provenance("clip|2", "seg", "L -> mirrored", "x", "y"), motions_dir)
    text = (motions_dir / "README.md").read_text()
    table = text.split("## Provenance", 1)[1].strip().splitlines()
    # The real README may already list published motions; the new rows go at the end, in order.
    assert table[-2].startswith("| `lunge_v1.csv` | abc123_1000-2000 |")
    assert table[-1].startswith("| `lunge_v2.csv` | clip\\|2 |")  # pipes escaped
    assert not any(row.replace("|", "").strip() == "" for row in table)  # placeholder gone
    assert text.split("## Provenance")[0] == (paths.MOTIONS_DIR / "README.md").read_text().split("## Provenance")[0]


@pytest.mark.parametrize("name", ["Lunge", "lunge_v2", "lunge-1", "wave"])
def test_bad_motion_names_are_rejected(motions_dir, name):
    with pytest.raises(PublishError, match=repr(name)):
        publish_motion(tiny_motion(), name, PROV, motions_dir)


def test_parry_and_cut_names_are_allowed(motions_dir):
    assert publish_motion(tiny_motion(), "parry_quinte", PROV, motions_dir).name == "parry_quinte_v1.csv"
