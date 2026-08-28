#!/usr/bin/env python3
"""Evaluate a courier checkpoint across parallel, full-task apartment rollouts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck.tasks  # noqa: F401  (register task entry points)


TASK_ID = "Mjlab-Courier-Flat-MicroDuck"


def evaluate(checkpoint: Path, num_envs: int, seconds: float, seed: int) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    saw_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
    saw_delivery = torch.zeros_like(saw_grasp)
    saw_failure = torch.zeros_like(saw_grasp)
    failure_before_delivery = torch.zeros_like(saw_grasp)
    done_count = torch.zeros(num_envs, dtype=torch.int64, device=device)
    failed_count = torch.zeros_like(done_count)
    first_grasp_step = torch.full(
        (num_envs,), -1, dtype=torch.int64, device=device
    )
    first_delivery_step = torch.full_like(first_grasp_step, -1)
    min_reader_dist = torch.full((num_envs,), float("inf"), device=device)
    steps = int(math.ceil(seconds / base_env.step_dt))

    try:
        with torch.inference_mode():
            for step in range(steps):
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
                grasped_now = base_env._courier_grasped | base_env._courier_was_grasped
                delivered_now = (
                    base_env._courier_delivered | base_env._courier_was_delivered
                )
                failed_now = base_env.reset_terminated
                first_grasp_step[(first_grasp_step < 0) & grasped_now] = step
                first_delivery_step[(first_delivery_step < 0) & delivered_now] = step
                failure_before_delivery |= saw_failure & delivered_now
                saw_grasp |= grasped_now
                saw_delivery |= delivered_now
                saw_failure |= failed_now
                book_xy = base_env.scene["book"].data.root_link_pos_w[:, :2]
                reader_dist = torch.linalg.norm(
                    book_xy - base_env._courier_person_xy, dim=-1
                )
                min_reader_dist = torch.minimum(min_reader_dist, reader_dist)
                done_count += dones.to(dtype=torch.int64)
                failed_count += failed_now.to(dtype=torch.int64)
    finally:
        env.close()

    grasped = int(saw_grasp.sum().item())
    delivered = int(saw_delivery.sum().item())
    grasp_times = first_grasp_step[first_grasp_step >= 0].float() * base_env.step_dt
    delivery_times = (
        first_delivery_step[first_delivery_step >= 0].float() * base_env.step_dt
    )
    return {
        "checkpoint": str(checkpoint),
        "device": device,
        "seed": seed,
        "num_envs": num_envs,
        "seconds": seconds,
        "grasped": grasped,
        "grasp_rate": grasped / num_envs,
        "mean_first_grasp_s": (
            float(grasp_times.mean().item()) if len(grasp_times) else None
        ),
        "delivered": delivered,
        "delivery_rate": delivered / num_envs,
        "mean_first_delivery_s": (
            float(delivery_times.mean().item()) if len(delivery_times) else None
        ),
        "mean_min_reader_distance_m": float(min_reader_dist.mean().item()),
        "best_reader_distance_m": float(min_reader_dist.min().item()),
        "done_events": int(done_count.sum().item()),
        "failed_episode_events": int(failed_count.sum().item()),
        "failure_before_delivery": int(failure_before_delivery.sum().item()),
        "failure_before_delivery_rate": (
            int(failure_before_delivery.sum().item()) / num_envs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit non-zero unless at least one full-task rollout delivers.",
    )
    args = parser.parse_args()
    if args.num_envs <= 0 or args.seconds <= 0:
        parser.error("num-envs and seconds must be positive")

    result = evaluate(
        checkpoint=args.checkpoint.resolve(),
        num_envs=args.num_envs,
        seconds=args.seconds,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.require_success and result["delivered"] == 0:
        raise SystemExit("No full-task delivery observed")


if __name__ == "__main__":
    main()
