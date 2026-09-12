#!/usr/bin/env python3
"""Record a trained tracking (dance) policy and measure how well it dances.

Rolls the policy out in the real play environment, renders the clip, and
writes a provenance-stamped sidecar with the numbers that matter for a
choreography: mean and worst tracked-body position error against the
reference motion, anchor (trunk) error, and how many times the dancer fell
out of the routine (terminations). Works from a torch checkpoint or an
exported ONNX.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import json
import os
import shutil
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck.tasks  # noqa: F401  (register task entry points)
from mjlab_microduck.onnx_policy import OnnxPolicy
from mjlab_microduck.provenance import provenance, repo_relative

TASK_ID = "Mjlab-Tracking-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--output", type=Path, default=Path("clips/dance.mp4"))
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cam-distance", type=float, default=None)
    parser.add_argument("--cam-elevation", type=float, default=None)
    parser.add_argument(
        "--ghost",
        action="store_true",
        help="Keep the translucent reference-motion overlay in the render "
        "(the making-of shot); off by default for clean hero clips.",
    )
    parser.add_argument(
        "--max-mean-error",
        type=float,
        default=None,
        help="Exit non-zero when the mean tracked-body error exceeds this (m).",
    )
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.onnx is None):
        parser.error("Provide exactly one of: a checkpoint path, or --onnx")

    policy_path = (args.onnx or args.checkpoint).resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    if args.cam_distance is not None:
        env_cfg.viewer.distance = args.cam_distance
    if args.cam_elevation is not None:
        env_cfg.viewer.elevation = args.cam_elevation
    env_cfg.commands["motion"].debug_vis = args.ghost

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    if args.onnx is not None:
        policy = OnnxPolicy(policy_path, device=device)
    else:
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(policy_path), load_cfg={"actor": True}, strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()
    motion_cmd = base_env.command_manager.get_term("motion")

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sim_dt = base_env.step_dt
    sim_steps = int(math.ceil(args.seconds / sim_dt))
    nframes = int(round(args.seconds * args.fps))
    next_frame = 0
    body_errors = []
    anchor_errors = []
    falls = 0

    print(
        f"Recording dance rollout: {args.seconds:.1f}s, {args.fps} fps, "
        f"device={device}, seed={args.seed}"
    )
    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as writer, torch.inference_mode():
        for step in range(sim_steps):
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            sim_time = (step + 1) * sim_dt
            err = torch.linalg.norm(
                motion_cmd.robot_body_pos_w - motion_cmd.body_pos_w, dim=-1
            )
            body_errors.append(float(err.mean()))
            anchor_errors.append(
                float(
                    torch.linalg.norm(
                        motion_cmd.robot_anchor_pos_w - motion_cmd.anchor_pos_w
                    )
                )
            )
            falls += int(base_env.reset_terminated[0])
            while next_frame < nframes and next_frame / args.fps <= sim_time:
                frame = base_env.render()
                if frame is None:
                    raise RuntimeError("renderer returned no frame")
                writer.append_data(frame)
                next_frame += 1
    env.close()

    body_errors_t = torch.tensor(body_errors)
    metadata = {
        **provenance(policy_path, args.task),
        "policy_format": "onnx" if args.onnx is not None else "checkpoint",
        "output": repo_relative(args.output),
        "seed": args.seed,
        "seconds": args.seconds,
        "fps": args.fps,
        "mean_body_tracking_error_m": float(body_errors_t.mean()),
        "p95_body_tracking_error_m": float(
            body_errors_t.quantile(0.95)
        ),
        "max_body_tracking_error_m": float(body_errors_t.max()),
        "mean_anchor_error_m": float(torch.tensor(anchor_errors).mean()),
        "falls": falls,
    }
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {args.output} and {sidecar}: "
        f"mean body error {metadata['mean_body_tracking_error_m'] * 1000:.1f} mm, "
        f"p95 {metadata['p95_body_tracking_error_m'] * 1000:.1f} mm, "
        f"falls {falls}"
    )
    if (
        args.max_mean_error is not None
        and metadata["mean_body_tracking_error_m"] > args.max_mean_error
    ):
        raise SystemExit(
            f"Mean tracking error {metadata['mean_body_tracking_error_m']:.4f} m "
            f"exceeds the {args.max_mean_error:.4f} m gate"
        )


if __name__ == "__main__":
    main()
