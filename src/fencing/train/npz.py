"""`fencing train npz`: motion CSV -> mjlab NPZ, safe to run concurrently.

mjlab's csv_to_npz always writes the fixed path /tmp/motion.npz (then tries a W&B
upload). Two conversions at once would overwrite each other's output, so every
conversion holds an exclusive flock on /tmp/motion.npz.lock from "delete the old file"
until "copied to paths.NPZ/<name>_mjlab.npz". Anything else that calls csv_to_npz
directly bypasses this lock -- use this command instead.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from fencing import paths
from fencing.train import naming
from fencing.train.provenance import sha256_file

MJLAB_TMP_OUTPUT = Path("/tmp/motion.npz")
LOCK_PATH = Path("/tmp/motion.npz.lock")


@contextmanager
def exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def convert_with_lock(
    run_converter: Callable[[], None],
    output: Path,
    tmp_output: Path = MJLAB_TMP_OUTPUT,
    lock_path: Path = LOCK_PATH,
) -> Path:
    """Run a converter that writes `tmp_output`, then atomically copy it to `output`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(lock_path):
        # Deleting first is how we know the file we copy was written by *this* run.
        tmp_output.unlink(missing_ok=True)
        run_converter()
        if not tmp_output.is_file():
            raise RuntimeError(f"converter did not write {tmp_output}; nothing copied to {output}")
        partial = output.with_suffix(output.suffix + ".partial")
        shutil.copyfile(tmp_output, partial)
        os.replace(partial, output)
    return output


def _mjlab_converter(csv: Path, motion_name: str, input_fps: float, device: str) -> Callable[[], None]:
    def run() -> None:
        cmd = [
            str(paths.venv_python("mjlab")), "-m", "mjlab.scripts.csv_to_npz",
            "--input-file", str(csv),
            "--output-name", motion_name,
            "--input-fps", str(input_fps),
            "--output-fps", "50",
            "--device", device,
        ]
        env = {**os.environ, "WANDB_MODE": "disabled", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
        print("[npz] " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, env=env)
        # The W&B upload after the save can fail without W&B; the save is what matters,
        # and convert_with_lock checks that the file was actually written.
        if result.returncode != 0:
            print(f"[npz] csv_to_npz exited with {result.returncode} (checking whether the NPZ was saved)",
                  file=sys.stderr)

    return run


def write_sidecar(npz_file: Path, csv: Path, input_fps: float) -> Path:
    sidecar = npz_file.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "source_csv": str(csv.resolve()),
        "source_csv_sha256": sha256_file(csv),
        "input_fps": input_fps,
        "output_fps": 50,
        "npz_sha256": sha256_file(npz_file),
    }, indent=2) + "\n")
    return sidecar


def convert_csv(csv: Path, name: str | None = None, input_fps: float = 30.0, device: str = "cuda:0") -> Path:
    if not csv.is_file():
        raise FileNotFoundError(f"motion CSV not found: {csv}")
    if input_fps <= 0:
        raise ValueError(f"--input-fps must be positive, got {input_fps}")
    motion_name = name or naming.motion_name_from_path(csv)
    output = naming.npz_path(motion_name)
    convert_with_lock(_mjlab_converter(csv, motion_name, input_fps, device), output)
    write_sidecar(output, csv, input_fps)
    return output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", type=Path, help="headerless G1 motion CSV (x,y,z,qx,qy,qz,qw,29 joints)")
    parser.add_argument("--name", help="motion name (default: CSV stem); output is paths.NPZ/<name>_mjlab.npz")
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0", help="FK device for csv_to_npz (cuda:0 or cpu)")


def run(args: argparse.Namespace) -> int:
    output = convert_csv(args.csv.expanduser(), args.name, args.input_fps, args.device)
    print(f"[npz] wrote {output}")
    return 0
