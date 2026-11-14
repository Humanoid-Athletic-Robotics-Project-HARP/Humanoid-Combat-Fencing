"""Run existing saber policies (unchanged ONNX) on a bending blade in CPU MuJoCo.

Experimental. Reuses fencing.sim2sim's rollout pieces read-only and copies the default path of
`rollout.rollout` (scene PD, no start noise, no video) rather than changing it; a test runs both on the
rigid saber and requires identical metrics. Adds three measurements the repo's rollout does not make:
  * the REAL blade tip (the `blade_tip` site, at the end of the bending chain), scored like the
    repo's blade-tip metrics (same re-anchored reference tip, same +-2 step key window);
  * sword-wrist pitch/yaw actuator torque against its 5 Nm force range;
  * how far the real tip bends away from where a rigid blade's tip would be.

The policy sees exactly what it saw in training: inputs are built from the ONNX contract's 29
joints by name, so the blade hinges are invisible to it and have no actuator.

    scripts/hcf saber flex validate
    scripts/hcf saber flex sweep --policy cut_head_v3 --segments 3 5 10 --f1 3 5 7 10 15

See docs/saber_flex.md for the model, the commands and results.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from fencing import paths
from fencing.saber.flex import (
    FlexBlade,
    flex_joint_ids,
    flexible_scene,
    rigid_scene_for,
    saber_mass_check,
    scene_with_meshes,
    stable_timestep,
)
from fencing.saber.spec import NOMINAL_SABER
from fencing.sim2sim import rollout as ro
from fencing.sim2sim.observation import MujocoState, build_observation
from fencing.train import metrics

POLICIES_DIR = paths.REPO_ROOT / "results" / "policies"
OUT_DIR = paths.DATA / "eval" / "flex"
WRIST_ACTUATORS = ("right_wrist_pitch_joint", "right_wrist_yaw_joint")
# Gates from src/fencing/train/thresholds.py (nominal.*), applied to the real-tip numbers too.
GATES = {"joint_angle_error_rad": 0.10, "body_pos_error_m": 0.05, "blade_tip_error_m": 0.10,
         "blade_tip_error_key_m": 0.05}
SATURATION_FRACTION = 0.99  # |torque| >= 99% of the force range counts as "at the limit"

# The saber policies of docs/hardware_tests.md (blocks B and C), cut_head_v3 (the slower
# cut_head_v4's predecessor, kept as the hardest wrist case) and the recommended entry policy
# (stand to en garde, docs/mtc.md), which has no blade key moment.
SABER_POLICIES = {
    "extension_v3": "2026-11-10_06-14-28_extension_v3_tip",
    "parry_sixte_v3": "2026-11-10_11-42-25_parry_sixte_v3_tip",
    "parry_tierce_high_v3": "2026-11-10_12-26-40_parry_tierce_high_v3_tip",
    "parry_quinte_v3": "2026-11-10_07-03-50_parry_quinte_v3_tip",
    "parry_prime_v3": "2026-11-10_10-09-18_parry_prime_v3_tip",
    "parry_seconde_v3": "2026-11-10_10-54-06_parry_seconde_v3_tip",
    "thrust_v3": "2026-11-10_09-21-15_thrust_v3_tip",
    "cut_chest_v3": "2026-11-10_08-36-30_cut_chest_v3_tip",
    "cut_flank_v3": "2026-11-10_07-47-56_cut_flank_v3_tip",
    "cut_head_v3": "2026-11-10_05-31-03_cut_head_v3_tip",
    "cut_head_v4": "2026-11-11_13-49-44_cut_head_v4_tip",
    "advance_lunge_cut_v5": "2026-11-11_11-15-05_advance_lunge_cut_v5_tip",
    "stand_to_en_garde_v1_feetjoint05": "2026-11-12_07-52-58_stand_to_en_garde_v1_feetjoint05",
}


def export_dir(policy: str) -> Path:
    return POLICIES_DIR / SABER_POLICIES.get(policy, policy) / "model_2999"


def committed_key_steps(directory: Path) -> tuple[int, ...] | None:
    """Key steps as the committed eval recorded them (the motion NPZs they come from are not in
    the repo, so fencing.saber.scored_keys cannot resolve them on a fresh clone)."""
    for name in ("sim2sim.json", "metrics.json"):
        path = directory / name
        if path.is_file():
            record = json.loads(path.read_text())
            keys = (record.get("nominal") or {}).get("key_frames") or record.get("key_frames")
            if keys and keys.get("steps"):
                return tuple(int(s) for s in keys["steps"])
    return None


def run(directory: Path, scene_xml: str, timestep: float, key_steps, hold_seconds: float = 2.0,
        settle_s: float = 1.0, on_step=None) -> dict:
    """rollout.rollout with the scene as a string, a chosen physics timestep and extra measurements.

    on_step(step, frame, model, data, context), if given, runs after every policy step (e.g. to
    render a video); context holds the MujocoState, the reference and the scorer."""
    export = ro.load_export(directory)
    policy = ro.OnnxPolicy(export.onnx)
    contract = policy.contract
    model = mujoco.MjModel.from_xml_string(scene_xml)
    step_dt = float(export.info["step_dt"])
    decimation = round(step_dt / timestep)
    if abs(decimation * timestep - step_dt) > 1e-12:
        raise ValueError(f"timestep {timestep} does not divide step_dt {step_dt}")
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    state = MujocoState(model, contract)
    actuators = ro.actuator_ids(model, state, contract)
    ro.check_gains(model, actuators, contract)
    wrist_act = [model.actuator(f"robot/{name}").id for name in WRIST_ACTUATORS]
    wrist_limit = model.actuator_forcerange[wrist_act, 1].copy()

    reference = policy.reference(int(export.info["motion_frames"]))
    motion_steps = reference.num_frames
    hold_steps = round(hold_seconds / step_dt)
    scorer = ro.TrackingScorer(model, state, reference, motion_steps, hold_steps, step_dt, key_frames=key_steps)
    tip_site = ro.find_site(model, ro.TIP_SITE)
    wrist_index, tip_in_wrist = scorer.wrist_index, scorer.tip_in_wrist
    rng = np.random.default_rng(0)
    ro.reset_to_frame(model, data, state, reference, 0, rng, 0.0, scorer.pelvis_index)

    # Settle the blade into its static droop for the start pose (a real blade is not straight at rest).
    flex = flex_joint_ids(model)
    if flex:
        flex_q = [int(model.jnt_qposadr[j]) for j in flex]
        flex_v = [int(model.jnt_dofadr[j]) for j in flex]
        other_q = np.setdiff1d(np.arange(model.nq), flex_q)
        other_v = np.setdiff1d(np.arange(model.nv), flex_v)
        q0, v0 = data.qpos[other_q].copy(), data.qvel[other_v].copy()
        damping = model.dof_damping[flex_v].copy()
        model.dof_damping[flex_v] = damping * 40  # heavily overdamped while settling
        for _ in range(int(settle_s / timestep)):
            mujoco.mj_step(model, data)
            data.qpos[other_q], data.qvel[other_v] = q0, v0
        model.dof_damping[flex_v] = damping
        data.qvel[flex_v] = 0.0
        data.time = 0.0
        mujoco.mj_forward(model, data)

    key_window = metrics.key_window_frames(key_steps or ())
    real_err, real_key_err, bend = [], [], 0.0
    torque_peak = np.zeros(2)
    at_limit = np.zeros(2)
    substeps = 0
    last_action = np.zeros(contract.num_joints)
    anchor = scorer.anchor_index
    t0 = time.perf_counter()
    for step in range(scorer.total_steps):
        frame = scorer.frame(step)
        inputs = state.inputs(data, reference.joint_pos[frame], reference.joint_vel[frame],
                              reference.body_pos_w[frame, anchor], reference.body_quat_w[frame, anchor], last_action)
        scorer.begin_step(step, inputs.robot_anchor_pos_w, inputs.robot_anchor_quat_w)
        ref_tip = metrics.blade_tip_from_wrist(scorer._ref_rel_pos[wrist_index], scorer._ref_rel_quat[wrist_index],
                                               tip_in_wrist)
        obs = build_observation(inputs, contract)
        action = policy.run(obs, frame)["actions"].astype(np.float64)
        target = contract.default_joint_pos + contract.action_scale * action
        for _ in range(decimation):
            data.ctrl[actuators] = target
            mujoco.mj_step(model, data)
            if step < motion_steps and scorer.alive:
                force = np.abs(data.actuator_force[wrist_act])
                torque_peak = np.maximum(torque_peak, force)
                at_limit += force >= SATURATION_FRACTION * wrist_limit
                substeps += 1
        mujoco.mj_forward(model, data)
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise FloatingPointError(f"simulation unstable at step {step} (timestep {timestep})")
        last_action = action
        alive_before = scorer.alive
        scorer.end_step(step, data)
        if on_step is not None:
            on_step(step, frame, model, data, {"state": state, "reference": reference, "scorer": scorer})
        if step < motion_steps and alive_before:
            real_tip = data.site_xpos[tip_site]
            error = float(np.linalg.norm(real_tip - ref_tip))
            real_err.append(error)
            if frame in key_window:
                real_key_err.append(error)
            wrist_pos = state.body_pos_w(data)[wrist_index]
            wrist_quat = state.body_quat_w(data)[wrist_index]
            rigid_tip = metrics.blade_tip_from_wrist(wrist_pos, wrist_quat, tip_in_wrist)
            bend = max(bend, float(np.linalg.norm(real_tip - rigid_tip)))
        if not scorer.alive:
            break
    result = scorer.result()
    nominal = result["nominal"]
    out = {
        "timestep": timestep, "decimation": decimation, "elapsed_s": round(time.perf_counter() - t0, 2),
        "key_steps": list(key_steps or ()),
        "saber_mass_kg": saber_mass_check(model),
        "terminated_at_s": result["terminated_at_s"], "termination_terms": result["termination_terms"],
        "repo_metrics": {k: nominal.get(k) for k in ("success_rate", "recovery_rate", "joint_angle_error_rad",
                                                     "body_pos_error_m", "foot_slip_m_s", "blade_tip_error_m",
                                                     "blade_tip_error_key_m")},
        "real_tip_error_m": float(np.mean(real_err)) if real_err else None,
        "real_tip_error_key_m": float(np.mean(real_key_err)) if real_key_err else None,
        "max_bend_m": bend,
        "wrist_torque_peak_nm": dict(zip(WRIST_ACTUATORS, torque_peak.round(4).tolist())),
        "wrist_time_at_limit": dict(zip(WRIST_ACTUATORS, (at_limit / max(substeps, 1)).round(4).tolist())),
    }
    real = {"joint_angle_error_rad": nominal.get("joint_angle_error_rad"),
            "body_pos_error_m": nominal.get("body_pos_error_m"),
            "blade_tip_error_m": out["real_tip_error_m"], "blade_tip_error_key_m": out["real_tip_error_key_m"]}
    fails = [k for k, limit in GATES.items() if real[k] is not None and real[k] > limit]
    if result["terminated_at_s"] is not None or (nominal.get("recovery_rate") or 0) < 0.9:
        fails.insert(0, "fell" if result["terminated_at_s"] is not None else "recovery")
    out["verdict"] = "PASS" if not fails else "FAIL: " + ", ".join(fails)
    return out


def rigid_scene(directory: Path) -> str:
    return scene_with_meshes((directory / "scene" / "scene.xml").read_text())


def validate(policy: str) -> dict:
    """This loop on the unmodified rigid scene must match the repo's own rollout and the committed sim2sim.json."""
    directory = export_dir(policy)
    keys = committed_key_steps(directory)
    scene = rigid_scene(directory)
    tmp = OUT_DIR / "tmp" / policy / "scene.xml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(scene)
    export = ro.load_export(directory)
    repo = ro.rollout(replace(export, scene_xml=tmp), config=ro.RolloutConfig(key_frames=keys))["nominal"]
    ours = run(directory, scene, 0.005, keys)
    committed = json.loads((directory / "sim2sim.json").read_text())["nominal"]
    fields = ("success_rate", "joint_angle_error_rad", "body_pos_error_m", "blade_tip_error_m", "blade_tip_error_key_m")
    return {"policy": policy, "key_steps": keys,
            "committed_sim2sim": {k: committed.get(k) for k in fields},
            "repo_rollout_here": {k: repo.get(k) for k in fields},
            "our_loop_rigid": {k: ours["repo_metrics"].get(k) for k in fields},
            "our_loop_real_tip": {"blade_tip_error_m": ours["real_tip_error_m"],
                                  "blade_tip_error_key_m": ours["real_tip_error_key_m"]},
            "our_wrist": {"peak": ours["wrist_torque_peak_nm"], "at_limit": ours["wrist_time_at_limit"]}}


def sweep(policies, segments, f1s, zetas, timesteps_rigid=(0.005,), tag="sweep") -> Path:
    rows = []
    for policy in policies:
        directory = export_dir(policy)
        keys = committed_key_steps(directory)
        base = rigid_scene(directory)
        rigid_cache = {}
        for dt in timesteps_rigid:
            rigid_cache[dt] = run(directory, base, dt, keys)
            rows.append({"policy": policy, "blade": "rigid", **rigid_cache[dt]})
            print(f"{policy:22s} rigid       dt={dt:<7g} {rows[-1]['verdict']}", flush=True)
        for n in segments:
            for f1 in f1s:
                for zeta in zetas:
                    blade = FlexBlade(n, f1, zeta)
                    scene, modes = flexible_scene(base, blade)
                    dt = stable_timestep(modes)
                    if dt not in rigid_cache:  # baseline at the same timestep: separates dt from flex
                        rigid_cache[dt] = run(directory, base, dt, keys)
                        rows.append({"policy": policy, "blade": "rigid", **rigid_cache[dt]})
                    try:
                        r = run(directory, scene, dt, keys)
                    except FloatingPointError as error:
                        r = {"timestep": dt, "verdict": f"UNSTABLE ({error})"}
                    rows.append({"policy": policy, "blade": asdict(blade),
                                 "modes_hz": [round(float(f), 2) for f in modes.frequencies_hz[:3]],
                                 "hinge_stiffness_nm_rad": modes.stiffness, "hinge_damping": modes.damping,
                                 "static_droop_m": modes.static_tip_droop_m, **r})
                    print(f"{policy:22s} N={n:<2} f1={f1:<4g} zeta={zeta:<4g} dt={dt:<7g} {r['verdict']}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    return out


def wobble_from_video(frames: float, cycles: float, fps: float) -> float:
    """First bending frequency (Hz) from slow-motion video: `cycles` oscillations took `frames` frames."""
    if frames <= 0 or cycles <= 0 or fps <= 0:
        raise ValueError("frames, cycles and fps must be positive")
    return cycles * fps / frames


def damping_from_decay(first_amplitude: float, later_amplitude: float, cycles: float) -> float:
    """Damping ratio from two peak amplitudes `cycles` oscillations apart (logarithmic decrement)."""
    if not 0 < later_amplitude < first_amplitude or cycles <= 0:
        raise ValueError("need 0 < later_amplitude < first_amplitude and cycles > 0")
    delta = math.log(first_amplitude / later_amplitude) / cycles
    return delta / math.sqrt(4 * math.pi**2 + delta**2)


def uniform_beam_f1(tip_load_kg: float, deflection_m: float, free_length_m: float, blade_mass_kg: float,
                    blade_length_m: float) -> float:
    """Cross-check only: the f1 a UNIFORM blade would have, from a bend test (load at the tip of a
    free_length_m cantilever). Real blades taper, which raises f1 for the same stiffness (steel
    foils: uniform 3.7-4.3 Hz against a measured 6.0-6.8 Hz), so the video number comes first."""
    ei = tip_load_kg * 9.81 * free_length_m**3 / (3 * deflection_m)
    mu = blade_mass_kg / blade_length_m
    return 1.8751**2 / (2 * math.pi) * math.sqrt(ei / (mu * blade_length_m**4))


def measured(policies, f1: float, zeta: float, saber, segments: int = 3, tag: str = "measured",
             f1_edge: float | None = None) -> Path:
    """Every policy with the measured blade, against a rigid saber of the same mass and length.

    f1 is the wobble towards the flat side, f1_edge towards the edge (None = same as f1)."""
    rows = []
    blade = FlexBlade(segments, f1, zeta, f1_edge)
    for policy in policies:
        directory = export_dir(policy)
        keys = committed_key_steps(directory)
        base = rigid_scene(directory)
        scene, modes = flexible_scene(base, blade, saber)
        dt = stable_timestep(modes)
        rigid = run(directory, rigid_scene_for(base, saber), dt, keys)
        rows.append({"policy": policy, "blade": "rigid", **rigid})
        bent = run(directory, scene, dt, keys)
        rows.append({"policy": policy, "blade": asdict(blade), "saber": asdict(saber),
                     "modes_hz": [round(float(f), 2) for f in modes.frequencies_hz[:3]],
                     "static_droop_m": modes.static_tip_droop_m, **bent})
        print(f"{policy:22s} rigid {rigid['verdict']:28s} bending {bent['verdict']}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    return out


def _cm(x):
    return "—" if x is None else f"{100 * x:.1f}"


def report(paths_in) -> str:
    """Markdown table of one or more sweep JSONs: rigid baselines first, then each blade."""
    header = ("| Policy | Blade | dt (ms) | Verdict (real tip) | Key tip, repo metric (cm) | Key tip, real (cm) | "
              "Mean tip, real (cm) | Max bend (cm) | Joint err (rad) | Wrist pitch peak / at limit | "
              "Wrist yaw peak / at limit |")
    lines = [header,
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in paths_in:
        for r in json.loads(Path(path).read_text()):
            blade = r["blade"]
            name = "rigid" if blade == "rigid" else f"N={blade['segments']}, {blade['f1_hz']:g} Hz, ζ={blade['zeta']:g}"
            if blade != "rigid" and blade.get("f1_edge_hz"):
                name += f", edge {blade['f1_edge_hz']:g} Hz"
            if "repo_metrics" not in r:
                lines.append(f"| {r['policy']} | {name} | {r['timestep'] * 1000:g} | {r['verdict']} |" + " |" * 7)
                continue
            m = r["repo_metrics"]
            peak, lim = r["wrist_torque_peak_nm"], r["wrist_time_at_limit"]
            p, y = WRIST_ACTUATORS
            lines.append(
                f"| {r['policy']} | {name} | {r['timestep'] * 1000:g} | {r['verdict']} | {_cm(m['blade_tip_error_key_m'])} | "
                f"{_cm(r['real_tip_error_key_m'])} | {_cm(r['real_tip_error_m'])} | {_cm(r['max_bend_m'])} | "
                f"{m['joint_angle_error_rad']:.3f} | {peak[p]:.2f} Nm / {100 * lim[p]:.1f}% | "
                f"{peak[y]:.2f} Nm / {100 * lim[y]:.1f}% |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fencing saber flex", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--policy", default="cut_head_v3")
    s = sub.add_parser("sweep")
    s.add_argument("--policy", nargs="+", default=["cut_head_v3"])
    s.add_argument("--segments", type=int, nargs="+", default=[3])
    s.add_argument("--f1", type=float, nargs="+", default=[3, 5, 7, 10, 15])
    s.add_argument("--zeta", type=float, nargs="+", default=[0.05])
    s.add_argument("--rigid-dt", type=float, nargs="+", default=[0.005])
    s.add_argument("--tag", default="sweep")
    ms = sub.add_parser("measured", help="run every policy with a measured blade")
    ms.add_argument("--f1", type=float, help="first wobble frequency towards the flat side (Hz), if known")
    ms.add_argument("--frames", type=float, help="video frames spanning --cycles full oscillations")
    ms.add_argument("--f1-edge", type=float, help="first wobble frequency towards the edge (Hz); default: same")
    ms.add_argument("--frames-edge", type=float, help="as --frames, for the wobble towards the edge")
    ms.add_argument("--cycles", type=float, default=5)
    ms.add_argument("--fps", type=float, default=240)
    ms.add_argument("--decay", type=float, nargs=3, metavar=("A_FIRST", "A_LATER", "CYCLES"),
                    help="two peak amplitudes (any unit) and the cycles between them -> damping ratio")
    ms.add_argument("--zeta", type=float, default=0.05, help="damping ratio if --decay is not given")
    ms.add_argument("--blade-mass-g", type=float, help="blade mass, without grip and guard")
    ms.add_argument("--blade-length-cm", type=float, help="blade length from the guard to the tip")
    ms.add_argument("--bend", type=float, nargs=3, metavar=("LOAD_G", "DEFLECTION_CM", "FREE_LENGTH_CM"),
                    help="optional bend test, printed as a uniform-blade cross-check")
    ms.add_argument("--policy", nargs="+", default=["all"])
    ms.add_argument("--segments", type=int, default=3)
    ms.add_argument("--tag", default="measured")
    rp = sub.add_parser("report")
    rp.add_argument("json", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "measured":
        if args.f1 is None and args.frames is None:
            parser.error("give --f1, or --frames with --cycles and --fps")
        f1 = args.f1 if args.f1 is not None else wobble_from_video(args.frames, args.cycles, args.fps)
        f1_edge = args.f1_edge
        if f1_edge is None and args.frames_edge is not None:
            f1_edge = wobble_from_video(args.frames_edge, args.cycles, args.fps)
        zeta = damping_from_decay(*args.decay) if args.decay else args.zeta
        saber = NOMINAL_SABER
        if args.blade_mass_g is not None:
            saber = replace(saber, blade_mass=args.blade_mass_g / 1000)
        if args.blade_length_cm is not None:
            saber = replace(saber, blade_length=args.blade_length_cm / 100)
        edge = "same" if f1_edge is None else f"{f1_edge:.2f} Hz"
        print(f"blade: f1 {f1:.2f} Hz flatwise, {edge} edgewise, damping ratio {zeta:.3f}, "
              f"{saber.blade_mass * 1000:.0f} g, "
              f"{saber.blade_length * 100:.0f} cm")
        if args.bend:
            load_g, deflection_cm, free_cm = args.bend
            check = uniform_beam_f1(load_g / 1000, deflection_cm / 100, free_cm / 100, saber.blade_mass,
                                    saber.blade_length)
            print(f"bend-test cross-check (uniform blade): f1 ~ {check:.1f} Hz")
        policies = list(SABER_POLICIES) if args.policy == ["all"] else args.policy
        out = measured(policies, f1, zeta, saber, args.segments, args.tag, f1_edge)
        print(out)
        print(report([out]))
        return 0
    if args.cmd == "report":
        print(report(args.json))
    elif args.cmd == "validate":
        print(json.dumps(validate(args.policy), indent=1, default=float))
    else:
        policies = list(SABER_POLICIES) if args.policy == ["all"] else args.policy
        print(sweep(policies, args.segments, args.f1, args.zeta, tuple(args.rigid_dt), args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
