"""`fencing train eval`: Goal 1 metrics for a checkpoint, in mjlab, over N parallel envs.

Every pass rolls the policy out from reference frame 0 over the whole motion (or the
first --seconds), then holds the last reference frame for --hold-seconds to measure
recovery. mjlab's play config wraps the motion around at its end; we clamp the command
at the last frame instead (`hold_last_frame`). Each env is scored until its first
termination (mjlab's own anchor-height, anchor-tilt and end-effector-height checks), with
mjlab's evaluate.py pairing: post-step robot state vs the reference frame of that step.

Passes (docs/evaluation.md):
    nominal     play config: no observation noise, no pushes, and none of the startup
                randomization (encoder bias, CoM offset, foot friction, saber mass)
    push        + mjlab's push_robot interval event from the training config
    obs_noise   + the training observation noise and encoder bias
    friction    + mjlab's foot_friction startup event (per-env friction in --friction-range)
    saber_mass  + the saber task's saber_mass startup event (saber task only)

Output: paths.EVAL/<run dir>/<checkpoint>/metrics.json, plus a PASS/FAIL against the
provisional thresholds in fencing.train.thresholds.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from fencing.train import naming, thresholds

ROBUSTNESS_PASSES = ("push", "obs_noise", "friction", "saber_mass")
ALL_PASSES = ("nominal", *ROBUSTNESS_PASSES)
# Startup/interval events the nominal pass removes from mjlab's play config.
NOMINAL_DROPPED_EVENTS = ("push_robot", "encoder_bias", "base_com", "foot_friction", "saber_mass")
PASS_DESCRIPTIONS = {
    "nominal": "play config, no obs noise, no pushes, no startup randomization",
    "push": "nominal + push_robot (training velocity ranges, every 1-3 s)",
    "obs_noise": "nominal + training observation noise + encoder bias",
    "friction": "nominal + foot_friction (per-env, uniform over the friction range)",
    "saber_mass": "nominal + saber_mass (per-env saber mass over the training range)",
}
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
PELVIS_BODY = "pelvis"
WRIST_BODY = "right_wrist_yaw_link"
TIP_SITE = "blade_tip"
TRAINING_EPISODE_S = 10.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="run dir, run dir name or run name")
    parser.add_argument("--checkpoint", help="iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--num-envs", type=int, default=1024, help="episodes per pass (default 1024)")
    parser.add_argument("--robustness-num-envs", type=int, help="episodes per robustness pass (default: --num-envs)")
    parser.add_argument("--no-robustness", action="store_true", help="nominal pass only")
    parser.add_argument("--passes", nargs="+", choices=ALL_PASSES, help="run only these passes")
    parser.add_argument("--task", help="task id (default: the run's task)")
    parser.add_argument("--motion", type=Path, help="motion NPZ (default: the one the run trained on)")
    parser.add_argument("--seconds", type=float, help="evaluate only the first S seconds of the motion")
    parser.add_argument("--hold-seconds", type=float, default=2.0,
                        help="hold the last reference frame this long to measure recovery (0: skip)")
    parser.add_argument("--friction-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                        help="foot friction range for the friction pass (default: the training range)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="default: cuda:0 if available")
    parser.add_argument("--out", type=Path, help="output JSON (default: paths.EVAL/<run>/<checkpoint>/metrics.json)")
    parser.add_argument("--thresholds", type=Path, help="JSON file overriding the default thresholds")
    parser.add_argument("--threshold", action="append", metavar="KEY=VALUE", help="override one threshold")


def hold_last_frame(command, last_frame: int) -> None:
    """Replace MotionCommand._update_command so time_steps stop at `last_frame` instead of
    wrapping to frame 0 (which would teleport the robot back to the start)."""

    def update(env_ids=None):
        if env_ids is None:
            command.time_steps += 1
        else:
            command.time_steps[env_ids] += 1
        command.time_steps.clamp_(max=last_frame)
        if command._pending_forward:  # a reset path resampled and wrote sim state
            command._pending_forward = False
            command._env.sim.forward()
        command.update_relative_body_poses()

    command._update_command = update


def configure_pass(name: str, task: str, seed: int, friction_range: tuple[float, float] | None):
    """env_cfg editor for one pass; robustness terms are copied from the training config."""
    from mjlab.tasks.registry import load_env_cfg

    from fencing.train import mjlab_env

    mjlab_env.require_task(task)  # the saber task is ours: it is not in mjlab's registry until registered
    train_cfg = load_env_cfg(task, play=False)

    def configure(env_cfg) -> None:
        env_cfg.seed = seed
        for event in NOMINAL_DROPPED_EVENTS:
            env_cfg.events.pop(event, None)
        env_cfg.observations["actor"].enable_corruption = False
        if name == "push":
            env_cfg.events["push_robot"] = copy.deepcopy(train_cfg.events["push_robot"])
        elif name == "obs_noise":
            env_cfg.observations["actor"].enable_corruption = True
            env_cfg.events["encoder_bias"] = copy.deepcopy(train_cfg.events["encoder_bias"])
        elif name == "friction":
            event = copy.deepcopy(train_cfg.events["foot_friction"])
            if friction_range is not None:
                event.params["ranges"] = tuple(friction_range)
            env_cfg.events["foot_friction"] = event
        elif name == "saber_mass":
            env_cfg.events["saber_mass"] = copy.deepcopy(train_cfg.events["saber_mass"])

    return configure, train_cfg


def pass_available(name: str, train_cfg) -> bool:
    if name == "nominal":
        return True
    event = {"push": "push_robot", "obs_noise": "encoder_bias", "friction": "foot_friction",
             "saber_mass": "saber_mass"}[name]
    return event in train_cfg.events


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def evaluate_pass(name: str, task: str, motion_file: Path, checkpoint: Path, num_envs: int, device: str,
                  seconds: float | None, hold_seconds: float, seed: int,
                  friction_range: tuple[float, float] | None, run_dir: Path | None = None) -> dict:
    import mujoco
    import torch

    from fencing.sim2sim.observation import resolve_id
    from fencing.saber import scored_keys
    from fencing.sim2sim.rollout import find_site, tip_offset_in_body
    from fencing.train import metrics, mjlab_env

    configure, train_cfg = configure_pass(name, task, seed, friction_range)
    if not pass_available(name, train_cfg):
        return {"skipped": f"task {task} has no {name} randomization"}
    started = time.time()
    env, env_cfg, agent_cfg = mjlab_env.make_env(task, motion_file, num_envs, play=True, device=device,
                                                 configure=configure, run_dir=run_dir)
    try:
        policy = mjlab_env.load_runner(env, task, agent_cfg, checkpoint, device).get_inference_policy(device=device)
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        termination_manager = base.termination_manager
        failure_terms = [t for t in termination_manager.active_terms
                         if not termination_manager.get_term_cfg(t).time_out]
        step_dt = float(base.step_dt)
        frames = int(command.motion.time_step_total)
        motion_steps = frames
        if seconds is not None:
            motion_steps = max(1, min(frames, int(round(seconds / step_dt))))
        hold_steps = int(round(hold_seconds / step_dt))
        total_steps = motion_steps + hold_steps
        hold_last_frame(command, motion_steps - 1)

        body_names = list(command.cfg.body_names)
        feet = [body_names.index(n) for n in FOOT_BODIES]
        pelvis = body_names.index(PELVIS_BODY)
        ground_z = metrics.estimate_ground_height(_np(command.motion.body_pos_w[:, feet, 2]))
        model = base.sim.mj_model
        tip_site = find_site(model, TIP_SITE)
        tip_in_wrist = wrist = None
        if tip_site is not None and WRIST_BODY in body_names:
            wrist = body_names.index(WRIST_BODY)
            tip_in_wrist = tip_offset_in_body(model, tip_site, resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY))

        key_frames = scored_keys.resolve_for_npz(motion_file, step_dt, frames)
        accumulator = metrics.TrackingAccumulator(num_envs, body_names, feet, ground_z,
                                                  key_frames=key_frames.steps if key_frames else None)
        terminated_at = np.full(num_envs, metrics.NOT_TERMINATED, dtype=np.int64)
        termination_counts = dict.fromkeys(failure_terms, 0)
        final_height_error = final_speed = None
        obs = env.get_observations()
        for step in range(total_steps):
            scoring = step < motion_steps
            if scoring:  # the reference this step is scored against, before the command advances
                ref = {
                    "joint_pos": _np(command.joint_pos), "body_pos_w": _np(command.body_pos_w),
                    "body_pos_rel": _np(command.body_pos_relative_w), "anchor_pos": _np(command.anchor_pos_w),
                    "anchor_quat": _np(command.anchor_quat_w), "body_lin_vel": _np(command.body_lin_vel_w),
                }
                if tip_in_wrist is not None:
                    ref["body_quat_w"] = _np(command.body_quat_w)
                    ref["body_quat_rel"] = _np(command.body_quat_relative_w)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            terminated = termination_manager.terminated.cpu().numpy()
            newly = terminated & (terminated_at == metrics.NOT_TERMINATED)
            if newly.any():
                terminated_at[newly] = step
                for term in failure_terms:
                    termination_counts[term] += int((termination_manager.get_term(term).cpu().numpy() & newly).sum())
            active = terminated_at == metrics.NOT_TERMINATED  # post-reset state of a terminated env is not scored
            if scoring:
                robot_body_pos = _np(command.robot_body_pos_w)
                tips = {}
                if tip_in_wrist is not None:
                    robot_body_quat = _np(command.robot_body_quat_w)
                    tips = {
                        "ref_tip_rel_w": metrics.blade_tip_from_wrist(ref["body_pos_rel"][:, wrist],
                                                                      ref["body_quat_rel"][:, wrist], tip_in_wrist),
                        "ref_tip_w": metrics.blade_tip_from_wrist(ref["body_pos_w"][:, wrist],
                                                                  ref["body_quat_w"][:, wrist], tip_in_wrist),
                        "robot_tip_w": metrics.blade_tip_from_wrist(robot_body_pos[:, wrist],
                                                                    robot_body_quat[:, wrist], tip_in_wrist),
                    }
                accumulator.add(
                    active=active, ref_joint_pos=ref["joint_pos"], robot_joint_pos=_np(command.robot_joint_pos),
                    ref_body_pos_w=ref["body_pos_w"], ref_body_pos_rel_w=ref["body_pos_rel"],
                    robot_body_pos_w=robot_body_pos, ref_anchor_pos_w=ref["anchor_pos"],
                    ref_anchor_quat_w=ref["anchor_quat"], robot_anchor_pos_w=_np(command.robot_anchor_pos_w),
                    robot_anchor_quat_w=_np(command.robot_anchor_quat_w), ref_body_lin_vel_w=ref["body_lin_vel"],
                    robot_body_lin_vel_w=_np(command.robot_body_lin_vel_w), frame=step, **tips,
                )
            if step == total_steps - 1 and hold_steps:
                final_height_error = (_np(command.robot_body_pos_w[:, pelvis, 2])
                                      - _np(command.motion.body_pos_w[motion_steps - 1, pelvis, 2]))
                final_speed = np.linalg.norm(_np(command.robot_body_lin_vel_w[:, pelvis]), axis=-1)
            if not active.any():
                break
            if step % 500 == 0:
                print(f"[eval:{name}] step {step}/{total_steps}: {int(active.sum())}/{num_envs} alive", flush=True)

        window_steps = int(round(TRAINING_EPISODE_S / step_dt))
        outcomes = metrics.episode_outcomes(terminated_at, motion_steps, window_steps, final_height_error, final_speed)
        result = {**metrics.outcome_summary(outcomes, hold_evaluated=hold_steps > 0), **accumulator.summary()}
        result.update({
            "num_envs": num_envs,
            "description": PASS_DESCRIPTIONS[name],
            "termination_counts": termination_counts,
            "mean_termination_time_s": (float(metrics.termination_time_s(terminated_at[~active], step_dt).mean())
                                        if (~active).any() else None),
            "step_dt": step_dt,
            "motion_frames": frames,
            "motion_steps": motion_steps,
            "hold_steps": hold_steps,
            "truncated": motion_steps < frames,
            "window_steps": window_steps,
            "has_blade": tip_in_wrist is not None,
            "elapsed_s": round(time.time() - started, 1),
        })
        if final_height_error is not None and active.any():
            result["final_pelvis_height_error_m"] = float(np.abs(final_height_error[active]).mean())
            result["final_pelvis_speed_m_s"] = float(final_speed[active].mean())
        if key_frames:
            result["key_frames"] = key_frames.report()
        return result
    finally:
        env.close()


def summary_line(nominal: dict) -> str:
    def fmt(key, scale=1.0, unit=""):
        value = nominal.get(key)
        return "n/a" if value is None else f"{value * scale:.3g}{unit}"

    line = (f"success {fmt('success_rate')} ({nominal.get('num_envs')} envs) | joint {fmt('joint_angle_error_rad')} rad"
            f" | body {fmt('body_pos_error_m', 100, ' cm')} (global {fmt('body_pos_error_global_m', 100, ' cm')})"
            f" | anchor {fmt('anchor_pos_error_m', 100, ' cm')} | slip {fmt('foot_slip_m_s')} m/s"
            f" | recovery {fmt('recovery_rate')} | blade {fmt('blade_tip_error_m', 100, ' cm')}")
    if "blade_key_frames" in nominal:
        line += f" | blade at key {fmt('blade_tip_error_key_m', 100, ' cm')}"
    return line


def evaluate(args: argparse.Namespace) -> tuple[dict, Path]:
    from fencing.train import mjlab_env

    run_dir, checkpoint = mjlab_env.resolve(args.run, args.checkpoint)
    task = args.task or mjlab_env.run_task(run_dir)
    motion_file = (args.motion.expanduser() if args.motion else mjlab_env.run_motion_file(run_dir))
    device = args.device or mjlab_env.default_device()
    if args.passes:
        passes = list(args.passes)
    else:
        passes = ["nominal"] if args.no_robustness else list(ALL_PASSES)
    report: dict = {
        "kind": "mjlab_eval",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "iteration": naming.checkpoint_iteration(checkpoint),
        "task": task,
        "motion_file": str(motion_file),
        "seed": args.seed,
        "device": device,
        "seconds": args.seconds,
        "hold_seconds": args.hold_seconds,
        "nominal": None,
        "robustness": {},
    }
    for name in passes:
        num_envs = args.num_envs if name == "nominal" else (args.robustness_num_envs or args.num_envs)
        print(f"[eval] pass {name}: {num_envs} envs ({PASS_DESCRIPTIONS[name]})", flush=True)
        result = evaluate_pass(name, task, motion_file, checkpoint, num_envs, device, args.seconds,
                               args.hold_seconds, args.seed, tuple(args.friction_range) if args.friction_range else None,
                               run_dir=run_dir)
        if name == "nominal":
            report["nominal"] = result
        else:
            report["robustness"][name] = result
        if "skipped" in result:
            print(f"[eval] pass {name} skipped: {result['skipped']}", flush=True)
        else:
            print(f"[eval] pass {name}: success {result['success_rate']:.3f}, "
                  f"terminations {result['termination_counts']} ({result['elapsed_s']} s)", flush=True)
    checks = thresholds.check(report, thresholds.load_thresholds(args.thresholds, args.threshold))
    report["thresholds"] = [c.__dict__ for c in checks]
    report["verdict"] = thresholds.verdict(checks, thresholds.not_applicable(report),
                                           thresholds.incomplete_reasons(report))
    report["summary"] = summary_line(report["nominal"]) if report["nominal"] else None
    out = args.out or naming.eval_dir(run_dir, checkpoint) / "metrics.json"
    mjlab_env.write_json(out, report)
    return report, out


def run(args: argparse.Namespace) -> int:
    report, out = evaluate(args)
    if report["summary"]:
        print(f"[eval] {report['summary']}")
    for check in report["thresholds"]:
        print(f"[eval]   {thresholds.CheckResult(**check).line()}")
    print(f"[eval] {report['verdict']} (provisional thresholds, docs/evaluation.md) -> {out}")
    return 0
