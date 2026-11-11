"""Run a worker inside one of the per-stage venvs (pose, gvhmr, gmr) with a per-clip log.

The `fencing` CLI venv only has numpy/scipy/opencv/mujoco. Torch, MediaPipe, GVHMR and GMR
live in their own venvs under HCF_DATA/venvs, so heavy steps are subprocesses.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from fencing import paths

SRC_DIR = paths.REPO_ROOT / "src"
# MediaPipe's C bindings dlopen libGLESv2.so.2, which this machine lacks system-wide and we may
# not apt-install (no sudo). `scripts/setup_linux_gpu.sh pose` extracts it from the Ubuntu
# libgles2 .deb into this directory.
EXTRA_LIB_DIR = paths.DATA / "syslibs" / "root" / "usr" / "lib" / "x86_64-linux-gnu"

SETUP_HINT = {
    "pose": "scripts/setup_linux_gpu.sh pose",
    "gvhmr": "scripts/setup_linux_gpu.sh gvhmr",
    "gmr": "scripts/setup_linux_gpu.sh gmr",
}


class WorkerError(RuntimeError):
    pass


def require_venv(name: str) -> Path:
    python = paths.venv_python(name)
    if not python.exists():
        hint = SETUP_HINT.get(name, "see docs/setup.md")
        raise FileNotFoundError(f"venv '{name}' not found: {python} is missing. Create it with `{hint}`.")
    return python


def run_in_venv(
    venv: str,
    args: list[str],
    log_path: Path,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    extra_pythonpath: list[Path] | None = None,
) -> float:
    """Run `<venv python> <args>`, appending stdout/stderr to log_path. Returns seconds taken.

    Raises WorkerError with the log's tail on a non-zero exit, so batch mode can record it and
    move on to the next clip.
    """
    python = require_venv(venv)
    env = dict(os.environ)
    pythonpath = [str(SRC_DIR)] + [str(p) for p in (extra_pythonpath or [])]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if EXTRA_LIB_DIR.exists():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(EXTRA_LIB_DIR), env.get("LD_LIBRARY_PATH")]))
    env.setdefault("HCF_DATA", str(paths.DATA))
    env.update(extra_env or {})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n$ {python} {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run([str(python), *args], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-25:])
        raise WorkerError(f"{venv} worker failed (exit {proc.returncode}); log {log_path}:\n{tail}")
    return elapsed


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)
