"""`fencing train play --video`: offscreen MP4 of a checkpoint in mjlab, with the reference ghost.

One env in the nominal evaluation setup (play config, no noise, no pushes, no startup
randomization), starting at reference frame 0. The reference is drawn as mjlab's
translucent ghost robot (MotionCommand debug visualisation). Rendering goes through
mjlab's OffscreenRenderer; on WSL2, MUJOCO_GL=egl resolves to Mesa llvmpipe (CPU), so
shadows are off and frames are rendered at --fps rather than every policy step.

Output: paths.EVAL/<run dir>/<checkpoint>/play.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fencing.train import metrics, naming


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="run dir, run dir name or run name")
    parser.add_argument("--video", action="store_true", required=True,
                        help="write an MP4 (the only mode: there is no display here)")
    parser.add_argument("--checkpoint", help="iteration or model_<N>.pt (default: latest)")
    parser.add_argument("--seconds", type=float, default=10.0, help="video length (default 10)")
    parser.add_argument("--start-seconds", type=float, default=0.0,
                        help="skip this much of the motion first (simulated, not rendered)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-ghost", action="store_true", help="do not draw the reference")
    parser.add_argument("--task", help="task id (default: the run's task)")
    parser.add_argument("--motion", type=Path, help="motion NPZ (default: the one the run trained on)")
    parser.add_argument("--device", help="default: cuda:0 if available")
    parser.add_argument("--out", type=Path, help="output MP4 (default: paths.EVAL/<run>/<checkpoint>/play.mp4)")


def play(args: argparse.Namespace) -> dict:
    import torch

    from fencing.sim2sim.video import VideoWriter
    from fencing.train import mjlab_env
    from fencing.train.evaluate import configure_pass, hold_last_frame

    run_dir, checkpoint = mjlab_env.resolve(args.run, args.checkpoint)
    task = args.task or mjlab_env.run_task(run_dir)
    motion_file = args.motion.expanduser() if args.motion else mjlab_env.run_motion_file(run_dir)
    device = args.device or mjlab_env.default_device()
    nominal, _ = configure_pass("nominal", task, seed=0, friction_range=None)

    def configure(env_cfg) -> None:
        nominal(env_cfg)
        env_cfg.viewer.width = args.width
        env_cfg.viewer.height = args.height
        env_cfg.viewer.enable_shadows = False
        env_cfg.commands["motion"].debug_vis = not args.no_ghost

    env, _, agent_cfg = mjlab_env.make_env(task, motion_file, 1, play=True, device=device,
                                           configure=configure, render_mode="rgb_array", run_dir=run_dir)
    out = args.out or naming.eval_dir(run_dir, checkpoint) / "play.mp4"
    started = time.time()
    try:
        policy = mjlab_env.load_runner(env, task, agent_cfg, checkpoint, device).get_inference_policy(device=device)
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        frames = int(command.motion.time_step_total)
        hold_last_frame(command, frames - 1)  # hold the final pose instead of jumping back to frame 0
        step_dt = float(base.step_dt)
        render_every = max(1, int(round(1.0 / (args.fps * step_dt))))
        skip_steps = int(round(args.start_seconds / step_dt))
        video_steps = int(round(args.seconds / step_dt))
        writer = VideoWriter(out, 1.0 / (render_every * step_dt), args.width, args.height)
        terminations = []
        obs = env.get_observations()
        try:
            for step in range(skip_steps + video_steps):
                with torch.no_grad():
                    obs, _, dones, _ = env.step(policy(obs))
                if bool(base.termination_manager.terminated[0]):
                    terminations.append(round(float(metrics.termination_time_s(step, step_dt)), 2))  # play restarts at frame 0
                if step >= skip_steps and (step - skip_steps) % render_every == 0:
                    writer.write(base.render())  # the rsl_rl wrapper does not forward render()
        finally:
            writer.close()
    finally:
        env.close()
    return {"video": str(out), "frames": writer.frames, "fps": 1.0 / (render_every * step_dt),
            "backend": writer.backend, "terminations_s": terminations, "elapsed_s": round(time.time() - started, 1),
            "checkpoint": str(checkpoint)}


def run(args: argparse.Namespace) -> int:
    result = play(args)
    print(f"[play] {result['checkpoint']}: {result['frames']} frames at {result['fps']:g} fps "
          f"({result['backend']}, {result['elapsed_s']} s)")
    if result["terminations_s"]:
        print(f"[play] terminated (and restarted from frame 0) at t = {result['terminations_s']} s")
    print(f"[play] {result['video']}")
    return 0
