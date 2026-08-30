"""Stepped-physics integration tests for the courier tasks.

Every other courier test asserts on cfg dicts or hand-mocked envs. These build
the real environment on CPU, step it, and check what config tests cannot see:
the spawn geometry actually lands where the params say, the 61-D actor obs
assembles at runtime, the courier buffers exist, and nothing goes NaN. This is
exactly the guard to run before paying for a GPU run.
"""

from __future__ import annotations

import math

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (register task entry points)
from mjlab_microduck.tasks import mdp


def _build_env(task_id: str, num_envs: int = 2, seed: int = 7):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _book_polar_in_robot_frame(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    book = env.scene["book"]
    rel = book.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    yaw = _yaw_of(robot.data.root_link_quat_w)
    cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
    local_x = cos_y * rel[:, 0] - sin_y * rel[:, 1]
    local_y = sin_y * rel[:, 0] + cos_y * rel[:, 1]
    radius = torch.sqrt(local_x**2 + local_y**2)
    bearing = torch.atan2(local_y, local_x)
    return radius, bearing


@pytest.mark.parametrize(
    "task_id,radius_lo,radius_hi,bearing_max_deg,expect_wide_bearings",
    [
        ("Mjlab-Courier-Flat-MicroDuck", 0.12, 0.20, 12.0, False),
        ("Mjlab-Courier-Wide-MicroDuck", 0.11, 0.36, 62.0, True),
    ],
)
def test_courier_env_spawn_geometry_and_stepping(
    task_id, radius_lo, radius_hi, bearing_max_deg, expect_wide_bearings
):
    env = _build_env(task_id)
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (2, 61)
        assert obs["critic"].shape[0] == 2

        radii = []
        bearings = []
        person_radii = []
        for _ in range(12):
            obs, _ = env.reset()
            radius, bearing = _book_polar_in_robot_frame(env)
            radii.append(radius)
            bearings.append(bearing)
            # The wide (polar) contract anchors the reader on the ROBOT ROOT;
            # the legacy task anchors it on the env origin.
            if expect_wide_bearings:
                anchor = env.scene["robot"].data.root_link_pos_w[:, :2]
            else:
                anchor = env.scene.terrain.env_origins[:, :2]
            person_radii.append(
                torch.linalg.norm(env._courier_person_xy - anchor, dim=-1)
            )
        radii = torch.cat(radii)
        bearings = torch.cat(bearings)
        person_radius = torch.cat(person_radii)

        assert radii.min() >= radius_lo - 0.01
        assert radii.max() <= radius_hi + 0.01
        assert bearings.abs().max() <= math.radians(bearing_max_deg) + 0.02
        if expect_wide_bearings:
            # 24 draws from +-60 degrees essentially never all land within 15.
            assert bearings.abs().max() > math.radians(15.0)

        # Reader placement: within the configured radius band of its anchor,
        # across every reset (24 samples per task).
        if expect_wide_bearings:
            assert person_radius.min() >= 0.40 - 0.01
            assert person_radius.max() <= 0.90 + 0.01
        else:
            assert (person_radius - 0.55).abs().max() <= 0.05

        # Step real physics with zero actions: obs stay 61-D and finite, the
        # courier buffers exist, and the phase command is the right class.
        cmd = env.command_manager.get_term("twist")
        if expect_wide_bearings:
            assert isinstance(cmd, mdp.CourierPhaseCommand)
        else:
            assert isinstance(cmd, mdp.GroundPickPhaseCommand)
        for _ in range(10):
            out = env.step(torch.zeros(2, 14))
            step_obs = out[0]
            assert step_obs["actor"].shape == (2, 61)
            assert torch.isfinite(step_obs["actor"]).all()
        assert env._courier_grasped.dtype == torch.bool
        assert torch.isfinite(env._courier_person_xy).all()
    finally:
        env.close()
