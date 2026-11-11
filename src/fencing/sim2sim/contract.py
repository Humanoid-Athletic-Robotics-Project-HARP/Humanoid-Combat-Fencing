"""The policy contract carried in an mjlab tracking ONNX's metadata.

mjlab's exporter (mjlab.rl.exporter_utils.get_base_metadata plus the tracking runner's
anchor/body names) writes everything a deployment needs to run the policy without mjlab:
joint order, default pose, action scale, PD gains and the actor observation layout. This
is the same contract `motion_tracking_controller` (C++) reads, so sim2sim reads only this.

Lists are comma-separated strings with 3 decimals (mjlab's list_to_csv_str). Three
decimals lose precision on the action scale (0.25 * effort / kp, e.g. 0.43859 -> 0.439),
so `fencing train export` also writes FULL_PRECISION_KEY, a JSON copy of the float lists;
we prefer it when present and say which source was used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

FULL_PRECISION_KEY = "hcf_full_precision"
FLOAT_LIST_KEYS = ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale")
REQUIRED_KEYS = ("joint_names", "default_joint_pos", "action_scale", "observation_names",
                 "anchor_body_name", "body_names")


@dataclass(frozen=True)
class PolicyContract:
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray  # (J,)
    action_scale: np.ndarray  # (J,)
    joint_stiffness: np.ndarray | None  # (J,) kp, None if absent
    joint_damping: np.ndarray | None  # (J,) kd
    observation_names: tuple[str, ...]
    observation_scales: tuple[float | tuple[float, ...], ...]
    observation_clips: tuple[tuple[float, float], ...]
    observation_history_lengths: tuple[int, ...]
    anchor_body_name: str
    body_names: tuple[str, ...]
    full_precision: bool  # floats came from FULL_PRECISION_KEY, not the 3-decimal strings

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


def _split(value: str) -> list[str]:
    return [item for item in value.split(",") if item != ""]


def _float(text: str) -> float:
    text = text.strip()
    if text in ("inf", "+inf"):
        return math.inf
    if text == "-inf":
        return -math.inf
    return float(text)


def _float_list(value: str) -> np.ndarray:
    return np.array([_float(v) for v in _split(value)], dtype=np.float64)


def _scale_entry(text: str) -> float | tuple[float, ...]:
    parts = text.split(";")
    return _float(parts[0]) if len(parts) == 1 else tuple(_float(p) for p in parts)


def _clip_entry(text: str) -> tuple[float, float]:
    parts = text.split(";")
    if len(parts) != 2:
        raise ValueError(f"observation clip entry must be 'min;max', got {text!r}")
    return _float(parts[0]), _float(parts[1])


def parse_metadata(meta: dict[str, str]) -> PolicyContract:
    """Parse onnxruntime's custom_metadata_map into a PolicyContract. Fails loudly."""
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(f"ONNX metadata lacks {missing}; was it exported by mjlab's tracking runner?")
    joint_names = tuple(_split(meta["joint_names"]))
    floats = {k: _float_list(meta[k]) for k in FLOAT_LIST_KEYS if k in meta}
    full_precision = FULL_PRECISION_KEY in meta
    if full_precision:
        precise = json.loads(meta[FULL_PRECISION_KEY])
        for key in FLOAT_LIST_KEYS:
            if key in precise:
                floats[key] = np.asarray(precise[key], dtype=np.float64)
    if floats["action_scale"].size == 1:  # a scalar scale applies to every joint
        floats["action_scale"] = np.full(len(joint_names), floats["action_scale"][0])
    for key, values in floats.items():
        if values.shape != (len(joint_names),):
            raise ValueError(f"metadata {key} has {values.size} values for {len(joint_names)} joints")

    names = tuple(_split(meta["observation_names"]))
    num_terms = len(names)
    scales = tuple(_scale_entry(s) for s in _split(meta.get("observation_terms_scale", "")))
    clips = tuple(_clip_entry(c) for c in _split(meta.get("observation_terms_clip", "")))
    history = tuple(int(float(h)) for h in _split(meta.get("observation_terms_history_length", "")))
    scales = scales or (1.0,) * num_terms
    clips = clips or ((-math.inf, math.inf),) * num_terms
    history = history or (0,) * num_terms
    for label, values in (("scale", scales), ("clip", clips), ("history_length", history)):
        if len(values) != num_terms:
            raise ValueError(f"metadata has {len(values)} observation {label} entries for {num_terms} terms")
    return PolicyContract(
        joint_names=joint_names,
        default_joint_pos=floats["default_joint_pos"],
        action_scale=floats["action_scale"],
        joint_stiffness=floats.get("joint_stiffness"),
        joint_damping=floats.get("joint_damping"),
        observation_names=names,
        observation_scales=scales,
        observation_clips=clips,
        observation_history_lengths=history,
        anchor_body_name=meta["anchor_body_name"],
        body_names=tuple(_split(meta["body_names"])),
        full_precision=full_precision,
    )


def full_precision_metadata(values: dict[str, list[float]]) -> str:
    """The FULL_PRECISION_KEY value `fencing train export` attaches."""
    unknown = set(values) - set(FLOAT_LIST_KEYS)
    if unknown:
        raise ValueError(f"unexpected full-precision keys {sorted(unknown)}")
    return json.dumps({k: [float(x) for x in v] for k, v in values.items()})
