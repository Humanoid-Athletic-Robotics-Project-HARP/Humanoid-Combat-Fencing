"""`fencing train export`: ONNX policy with metadata, plus the exact MuJoCo scene.

Uses mjlab's own tracking exporter (MotionTrackingOnPolicyRunner.export_policy_to_onnx and
the same metadata it attaches when saving during training), so the file is what a
training run would have produced for that checkpoint. Alongside it we write the scene
mjlab simulated (robot + floor, with the solver options it ran with) so `fencing
sim2sim` can run on identical physics in plain CPU MuJoCo. One addition to mjlab's
metadata: `hcf_full_precision`, a JSON copy of the float lists (mjlab rounds them to 3
decimals, which shifts the G1 action scale by up to 0.1%); see fencing.sim2sim.contract.

Output: paths.EVAL/<run dir>/<checkpoint>/policy.onnx, scene/scene.xml, export.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fencing.train import naming

# MjOption fields mjlab sets from SimulationCfg; copied onto the exported spec.
OPTION_FIELDS = (
    "timestep", "iterations", "ls_iterations", "integrator", "cone", "impratio", "tolerance",
    "ls_tolerance", "solver", "jacobian", "gravity", "noslip_iterations", "ccd_iterations",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="run dir, run dir name or run name")
    parser.add_argument("--checkpoint", help="iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--out", type=Path, help="output dir (default: paths.EVAL/<run>/<checkpoint>)")
    parser.add_argument("--device", help="default: cuda:0 if available")


def export_scene(env, out_dir: Path) -> Path:
    """Write the simulated scene with mjlab's runtime solver options baked in."""
    from mjlab.utils.spec import export_spec

    spec = env.unwrapped.scene.spec.copy()
    runtime_opt = env.unwrapped.sim.mj_model.opt
    for name in OPTION_FIELDS:
        value = getattr(runtime_opt, name)
        setattr(spec.option, name, value.copy() if isinstance(value, np.ndarray) else value)
    export_spec(spec, out_dir)
    copy_missing_meshes(out_dir)
    return out_dir / "scene.xml"


def copy_missing_meshes(scene_dir: Path) -> None:
    """export_spec only writes assets held in memory; meshes the spec loaded from disk
    (the G1's STL files) are referenced by name but not copied, so copy them here."""
    import shutil
    import xml.etree.ElementTree as ET

    from fencing import paths

    root = ET.parse(scene_dir / "scene.xml").getroot()
    compiler = root.find("compiler")
    mesh_dir = scene_dir / (compiler.get("meshdir", "") if compiler is not None else "")
    search_dirs = [paths.G1_MJCF.parent / "assets", paths.G1_MJCF.parent]
    for element in root.iter("mesh"):
        file_name = element.get("file")
        if not file_name or (mesh_dir / file_name).is_file():
            continue
        source = next((d / file_name for d in search_dirs if (d / file_name).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"mesh {file_name!r} referenced by the exported scene not found in {search_dirs}")
        (mesh_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, mesh_dir / file_name)


def inspect_onnx(onnx_path: Path) -> dict:
    """What onnxruntime sees: input/output names and shapes plus the metadata keys."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    return {
        "inputs": {i.name: list(i.shape) for i in session.get_inputs()},
        "outputs": {o.name: list(o.shape) for o in session.get_outputs()},
        "metadata_keys": sorted(meta),
        "metadata": meta,
    }


def check_onnx_matches_torch(onnx_path: Path, policy, obs_dim: int, device: str, trials: int = 8) -> float:
    """Max |onnx - torch| action difference on random observations (normalizer included)."""
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        obs = rng.normal(size=(1, obs_dim)).astype(np.float32)
        onnx_actions = session.run(["actions"], {"obs": obs, "time_step": np.zeros((1, 1), np.float32)})[0]
        with torch.no_grad():
            obs_td = TensorDict({"actor": torch.from_numpy(obs).to(device)}, batch_size=[1])
            torch_actions = policy(obs_td).cpu().numpy()
        worst = max(worst, float(np.abs(onnx_actions - torch_actions).max()))
    return worst


def export(run: str, checkpoint: str | None, out: Path | None = None, device: str | None = None) -> dict:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    from fencing.sim2sim.contract import FLOAT_LIST_KEYS, FULL_PRECISION_KEY, full_precision_metadata
    from fencing.train import mjlab_env

    run_dir, checkpoint_path = mjlab_env.resolve(run, checkpoint)
    out_dir = out or naming.eval_dir(run_dir, checkpoint_path)
    device = device or mjlab_env.default_device()
    task = mjlab_env.run_task(run_dir)
    motion_file = mjlab_env.run_motion_file(run_dir)
    env, _, agent_cfg = mjlab_env.make_env(task, motion_file, 1, play=True, device=device, run_dir=run_dir)
    try:
        runner = mjlab_env.load_runner(env, task, agent_cfg, checkpoint_path, device)
        onnx_path = out_dir / "policy.onnx"
        runner.export_policy_to_onnx(str(out_dir), onnx_path.name)
        # Same metadata MotionTrackingOnPolicyRunner.save() attaches during training.
        metadata = get_base_metadata(env.unwrapped, f"local:{run_dir.name}/{checkpoint_path.name}")
        motion_term = env.unwrapped.command_manager.get_term("motion")
        metadata.update({
            "anchor_body_name": motion_term.cfg.anchor_body_name,
            "body_names": list(motion_term.cfg.body_names),
        })
        # mjlab writes floats with 3 decimals; sim2sim needs the action scale exactly.
        metadata[FULL_PRECISION_KEY] = full_precision_metadata({
            key: metadata[key] if isinstance(metadata[key], list) else [metadata[key]]
            for key in FLOAT_LIST_KEYS if key in metadata
        })
        attach_metadata_to_onnx(str(onnx_path), metadata)
        scene_xml = export_scene(env, out_dir / "scene")
        obs_dim = env.get_observations()["actor"].shape[-1]
        max_diff = check_onnx_matches_torch(onnx_path, runner.get_inference_policy(device=device), obs_dim, device)
        info = inspect_onnx(onnx_path)
        summary = {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "iteration": naming.checkpoint_iteration(checkpoint_path),
            "task": task,
            "motion_file": str(motion_file),
            "onnx": str(onnx_path),
            "scene_xml": str(scene_xml),
            "obs_dim": int(obs_dim),
            "onnx_vs_torch_max_abs_action_diff": max_diff,
            "step_dt": float(env.unwrapped.step_dt),
            "decimation": int(env.unwrapped.cfg.decimation),
            "physics_timestep": float(env.unwrapped.sim.mj_model.opt.timestep),
            "motion_frames": int(motion_term.motion.time_step_total),
            **{k: v for k, v in info.items() if k != "metadata"},
        }
    finally:
        env.close()
    mjlab_env.write_json(out_dir / "export.json", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    summary = export(args.run, args.checkpoint, args.out, args.device)
    print(f"[export] {summary['onnx']}")
    print(f"[export] scene: {summary['scene_xml']}")
    print(f"[export] inputs:  {summary['inputs']}")
    print(f"[export] outputs: {summary['outputs']}")
    print(f"[export] metadata keys: {', '.join(summary['metadata_keys'])}")
    diff = summary["onnx_vs_torch_max_abs_action_diff"]
    print(f"[export] onnxruntime vs torch actions: max |diff| = {diff:.2e}")
    if diff > 1e-4:
        print("[export] ERROR: ONNX output does not match the torch policy", flush=True)
        return 1
    return 0
