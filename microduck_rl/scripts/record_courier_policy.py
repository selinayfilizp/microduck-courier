#!/usr/bin/env python3
"""Record a trained courier checkpoint directly from the mjlab environment.

Unlike ``view_apartment.py --demo``, this is an actual policy rollout. The same
61-D observations, grasp latch, rewards, pushes, apartment props, and reader
target used for training are active during recording.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck.tasks  # noqa: F401  (register task entry points)
from mjlab_microduck.onnx_policy import OnnxPolicy
from mjlab_microduck.provenance import provenance


TASK_ID = "Mjlab-Courier-Flat-MicroDuck"


def record(
    checkpoint: Path,
    output: Path,
    seconds: float,
    fps: int,
    width: int,
    height: int,
    seed: int,
    require_success: bool,
    require_failure_before_success: bool,
    story_push_speed: float,
    story_push_time: float,
    require_stumble_recovery: bool,
    task_id: str = TASK_ID,
    use_onnx: bool = False,
    track: bool = False,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Policy not found: {checkpoint}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    if track:
        # Follow the trunk instead of filming from a fixed world point, so
        # every play-mode respawn heading stays in frame for the full clip.
        env_cfg.viewer.origin_type = type(env_cfg.viewer).OriginType.ASSET_BODY
        env_cfg.viewer.entity_name = "robot"
        env_cfg.viewer.lookat = (0.0, 0.0, 0.04)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    if use_onnx:
        policy = OnnxPolicy(checkpoint, device=device)
    else:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg
    output.parent.mkdir(parents=True, exist_ok=True)

    nframes = int(round(seconds * fps))
    sim_dt = base_env.step_dt
    sim_steps = int(math.ceil(seconds / sim_dt))
    next_frame = 0
    saw_grasp = False
    saw_delivery = False
    failure_times: list[float] = []
    grasp_times: list[float] = []
    delivery_times: list[float] = []
    was_grasped = False
    was_delivered = False
    story_push_applied = False
    max_tilt_deg = 0.0
    stumble_time: float | None = None
    recovery_time: float | None = None
    robot = base_env.scene["robot"]

    print(
        f"Recording trained rollout: {seconds:.1f}s, {fps} fps, {width}x{height}, "
        f"device={device}, seed={seed}"
    )
    try:
        with imageio.get_writer(
            output,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer, torch.inference_mode():
            for step in range(sim_steps):
                step_start_time = step * sim_dt
                if (
                    story_push_speed != 0.0
                    and not story_push_applied
                    and step_start_time >= story_push_time
                ):
                    velocity = robot.data.root_link_vel_w.clone()
                    velocity[0, 1] += story_push_speed
                    robot.write_root_link_velocity_to_sim(
                        velocity,
                        env_ids=torch.tensor([0], device=device),
                    )
                    story_push_applied = True
                action = policy(obs)
                obs, _, _, _ = env.step(action)
                sim_time = (step + 1) * sim_dt
                upright_cos = float(
                    torch.clamp(-robot.data.projected_gravity_b[0, 2], -1.0, 1.0)
                )
                tilt_deg = math.degrees(math.acos(upright_cos))
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                if stumble_time is None and tilt_deg >= 45.0:
                    stumble_time = sim_time
                if (
                    stumble_time is not None
                    and recovery_time is None
                    and sim_time >= stumble_time + 0.2
                    and tilt_deg <= 15.0
                ):
                    recovery_time = sim_time
                grasped_now = bool(
                    base_env._courier_grasped[0]
                    | base_env._courier_was_grasped[0]
                )
                delivered_now = bool(
                    base_env._courier_delivered[0]
                    | base_env._courier_was_delivered[0]
                )
                if grasped_now and not was_grasped:
                    grasp_times.append(sim_time)
                if delivered_now and not was_delivered:
                    delivery_times.append(sim_time)
                if bool(base_env.reset_terminated[0]):
                    failure_times.append(sim_time)
                saw_grasp |= grasped_now
                saw_delivery |= delivered_now
                was_grasped = grasped_now
                was_delivered = delivered_now

                while next_frame < nframes and next_frame / fps <= sim_time:
                    frame = base_env.render()
                    if frame is None:
                        raise RuntimeError("mjlab renderer returned no frame")
                    writer.append_data(frame)
                    next_frame += 1

            while next_frame < nframes:
                frame = base_env.render()
                if frame is None:
                    raise RuntimeError("mjlab renderer returned no frame")
                writer.append_data(frame)
                next_frame += 1
    finally:
        env.close()

    print(
        f"Wrote {output} ({nframes} frames); "
        f"grasp={saw_grasp}, delivered={saw_delivery}, "
        f"failures={len(failure_times)}"
    )
    metadata = {
        **provenance(checkpoint, task_id),
        "policy_format": "onnx" if use_onnx else "checkpoint",
        "tracking_camera": track,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "seed": seed,
        "seconds": seconds,
        "fps": fps,
        "frames": nframes,
        "grasp_times_s": grasp_times,
        "failure_times_s": failure_times,
        "delivery_times_s": delivery_times,
        "story_push": {
            "axis": "world_y",
            "time_s": story_push_time,
            "delta_velocity_m_s": story_push_speed,
            "applied": story_push_applied,
        },
        "max_tilt_deg": max_tilt_deg,
        "stumble_time_s": stumble_time,
        "recovery_time_s": recovery_time,
        "stumble_recovery_before_success": bool(
            recovery_time is not None
            and delivery_times
            and recovery_time < delivery_times[0]
        ),
        "failure_before_success": bool(
            delivery_times
            and any(failure < delivery_times[0] for failure in failure_times)
        ),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote rollout metadata: {metadata_path}")
    if require_success and not saw_delivery:
        raise SystemExit(
            "Rollout did not deliver the book. Try another seed/checkpoint before publishing."
        )
    if require_failure_before_success and not metadata["failure_before_success"]:
        raise SystemExit(
            "Rollout did not show a failed episode before delivery. "
            "Try another seed/checkpoint for the intended story."
        )
    if require_stumble_recovery and not metadata["stumble_recovery_before_success"]:
        raise SystemExit(
            "Rollout did not show a 45-degree stumble, upright recovery, and later "
            "delivery. Try another seed or story push."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        help="Torch checkpoint (.pt). Mutually exclusive with --onnx.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Exported deployment ONNX to roll out instead of a checkpoint.",
    )
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Camera follows the trunk so play-mode respawns stay in frame.",
    )
    parser.add_argument("--output", type=Path, default=Path("clips/courier-policy.mp4"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit non-zero when this rollout never reaches the delivered latch",
    )
    parser.add_argument(
        "--require-failure-before-success",
        action="store_true",
        help="Exit non-zero unless a genuine failed episode precedes delivery",
    )
    parser.add_argument(
        "--story-push-speed",
        type=float,
        default=0.0,
        help="One-time world-Y velocity pulse in m/s (0 disables it)",
    )
    parser.add_argument(
        "--story-push-time",
        type=float,
        default=1.4,
        help="Simulation time for the optional one-time story pulse",
    )
    parser.add_argument(
        "--require-stumble-recovery",
        action="store_true",
        help="Require a 45-degree stumble, recovery below 15 degrees, then delivery",
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("seconds, fps, width, and height must all be positive")
    if args.width % 2 or args.height % 2:
        parser.error("width and height must be even for yuv420p video")
    if (args.checkpoint is None) == (args.onnx is None):
        parser.error("Provide exactly one of: a checkpoint path, or --onnx")
    record(
        checkpoint=(args.onnx or args.checkpoint).resolve(),
        task_id=args.task,
        use_onnx=args.onnx is not None,
        track=args.track,
        output=args.output.resolve(),
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        require_success=args.require_success,
        require_failure_before_success=args.require_failure_before_success,
        story_push_speed=args.story_push_speed,
        story_push_time=args.story_push_time,
        require_stumble_recovery=args.require_stumble_recovery,
    )


if __name__ == "__main__":
    main()
