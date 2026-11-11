"""Provisional Goal 1 pass/fail thresholds (docs/evaluation.md explains each number).

Pure python so the fencing venv can test it. A threshold is (metric path, op, value):
the metric path is a dotted key into the metrics dict that `fencing train eval` and
`fencing sim2sim` write, e.g. "nominal.success_rate" or "robustness.push.success_rate".
Checks whose metric is missing or null are reported as "skipped", never as passed, so a
saber-only threshold does not fail a plain-G1 run and an unmeasured metric is visible.

Override with `--thresholds FILE.json` ({"nominal.success_rate": [">=", 0.9], ...}; a bare
number keeps the default operator; null drops a check) or `--threshold KEY=VALUE`.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path

OPERATORS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}

# metric path -> (operator, value). Provisional until Phase 1.2 numbers exist.
DEFAULT_THRESHOLDS: dict[str, tuple[str, float]] = {
    "nominal.success_rate": (">=", 0.95),
    "nominal.joint_angle_error_rad": ("<=", 0.10),
    "nominal.body_pos_error_m": ("<=", 0.05),
    "nominal.anchor_pos_error_m": ("<=", 0.25),
    "nominal.anchor_ori_error_rad": ("<=", 0.20),
    "nominal.foot_slip_m_s": ("<=", 0.10),
    "nominal.recovery_rate": (">=", 0.90),
    "nominal.blade_tip_error_m": ("<=", 0.10),
    "nominal.blade_tip_error_key_m": ("<=", 0.05),
    "robustness.push.success_rate": (">=", 0.80),
    "robustness.obs_noise.success_rate": (">=", 0.90),
    "robustness.friction.success_rate": (">=", 0.90),
    "robustness.saber_mass.success_rate": (">=", 0.90),
}


@dataclass(frozen=True)
class CheckResult:
    metric: str
    op: str
    threshold: float
    value: float | None
    status: str  # "pass" | "fail" | "skipped"

    def line(self) -> str:
        shown = "n/a" if self.value is None else f"{self.value:.4g}"
        return f"{self.status.upper():7s} {self.metric} = {shown} (need {self.op} {self.threshold:g})"


def _parse_entry(key: str, entry, default_op: str | None) -> tuple[str, float] | None:
    if entry is None:
        return None
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        if default_op is None:
            raise ValueError(f"threshold {key!r} is new; give it as [op, value]")
        return default_op, float(entry)
    if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] in OPERATORS:
        return str(entry[0]), float(entry[1])
    raise ValueError(f"threshold {key!r}: expected a number, [op, value] or null, got {entry!r}")


def merge_thresholds(overrides: dict | None = None,
                     base: dict[str, tuple[str, float]] | None = None) -> dict[str, tuple[str, float]]:
    merged = dict(DEFAULT_THRESHOLDS if base is None else base)
    for key, entry in (overrides or {}).items():
        default_op = merged[key][0] if key in merged else None
        parsed = _parse_entry(key, entry, default_op)
        if parsed is None:
            merged.pop(key, None)
        else:
            merged[key] = parsed
    return merged


def parse_cli_overrides(pairs: list[str] | None) -> dict:
    """["nominal.success_rate=0.9", "x.y=<=0.2", "z=none"] -> override dict."""
    overrides: dict = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--threshold expects KEY=VALUE, got {pair!r}")
        value = value.strip()
        if value.lower() in ("none", "null", "off"):
            overrides[key] = None
            continue
        op = next((o for o in (">=", "<=", ">", "<") if value.startswith(o)), None)
        overrides[key] = [op, float(value[len(op):])] if op else float(value)
    return overrides


def load_thresholds(path: Path | None = None, cli_pairs: list[str] | None = None) -> dict[str, tuple[str, float]]:
    file_overrides = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(file_overrides, dict):
        raise ValueError(f"{path}: thresholds file must hold a JSON object")
    return merge_thresholds(parse_cli_overrides(cli_pairs), merge_thresholds(file_overrides))


def lookup(metrics: dict, path: str):
    node = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# Checks that only exist for a motion with scored key frames (fencing.saber.scored_keys). A
# run without key-frame data leaves them out entirely, so its check list and verdict are
# exactly what they were before key frames existed.
KEY_FRAME_CHECKS = frozenset({"nominal.blade_tip_error_key_m"})


def check(metrics: dict, thresholds: dict[str, tuple[str, float]] | None = None) -> list[CheckResult]:
    results = []
    has_key_frames = bool(_flag(metrics, "blade_key_frames"))
    for metric, (op, threshold) in (DEFAULT_THRESHOLDS if thresholds is None else thresholds).items():
        if metric in KEY_FRAME_CHECKS and not has_key_frames:
            continue
        value = lookup(metrics, metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            results.append(CheckResult(metric, op, threshold, None, "skipped"))
            continue
        passed = OPERATORS[op](float(value), threshold)
        results.append(CheckResult(metric, op, threshold, float(value), "pass" if passed else "fail"))
    return results


def _flag(report: dict, key: str):
    """`fencing train eval` writes per-pass flags inside `nominal`; sim2sim writes them at the top level."""
    value = lookup(report, f"nominal.{key}")
    return report.get(key) if value is None else value


def not_applicable(report: dict) -> set[str]:
    """Checks that cannot apply to this run, so their skipping does not make it PARTIAL:
    the blade metric without a saber, and robustness passes the task has no randomization for.
    The key-frame blade check is listed only when the motion has key frames (otherwise
    `check` leaves it out)."""
    skip = set()
    if _flag(report, "has_blade") is False:
        skip.add("nominal.blade_tip_error_m")
        if _flag(report, "blade_key_frames"):
            skip.update(KEY_FRAME_CHECKS)
    for name, result in (report.get("robustness") or {}).items():
        if isinstance(result, dict) and "skipped" in result:
            skip.add(f"robustness.{name}.success_rate")
    return skip


def incomplete_reasons(report: dict) -> list[str]:
    """Why the scored rollout did not cover the whole motion."""
    return ["motion truncated (--seconds)"] if _flag(report, "truncated") else []


def verdict(results: list[CheckResult], not_applicable: set[str] = frozenset(),
            incomplete: list[str] = ()) -> str:
    """FAIL if any check failed or none ran. PARTIAL if every check that ran passed but an
    applicable check was skipped (a pass not run, recovery not evaluated) or the rollout was
    incomplete: a PARTIAL never counts as passing. Otherwise PASS."""
    ran = [r for r in results if r.status != "skipped"]
    if not ran or any(r.status != "pass" for r in ran):
        return "FAIL"
    missing = [r for r in results if r.status == "skipped" and r.metric not in not_applicable]
    return "PARTIAL" if missing or incomplete else "PASS"
