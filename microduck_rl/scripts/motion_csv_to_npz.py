#!/usr/bin/env python3
"""Convert a keyframe CSV into a tracking-task motion npz, locally.

A microduck-scene port of mjlab's scripts/csv_to_npz.py without the wandb
dependency: kinematically replays the CSV through the real robot model (so
the npz carries forward-kinematics world poses for every body), saves the
npz next to the CSV, optionally renders the replay to mp4, and prints a
feasibility report (per-beat foot lift heights, trunk travel, ground
clearance) so a bad sign convention in the choreography is caught by numbers
before anyone watches a video or pays for a GPU.

CSV row format (author_toosie_reference.py writes this):
  base x, y, z, quat x, y, z, w, then the 14 joints in JOINT_ORDER.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from mjlab_microduck.tasks.microduck_tracking_env_cfg import (
    make_microduck_tracking_env_cfg,
)

from author_toosie_reference import BEAT, JOINT_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("motions/toosie_slide.csv"))
    parser.add_argument("--output", type=Path, default=Path("motions/toosie_slide.npz"))
    parser.add_argument("--input-fps", type=int, default=50)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument(
        "--render", type=Path, default=None, help="Also write an mp4 of the replay."
    )
    args = parser.parse_args()

    device = "cpu"
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / args.output_fps

    scene = Scene(make_microduck_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    renderer = None
    if args.render is not None:
        viewer_cfg = ViewerConfig(
            height=720,
            width=1280,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=0.9,
            elevation=-18.0,
            azimuth=135.0,
        )
        renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
        renderer.initialize()

    motion = MotionLoader(
        motion_file=str(args.input),
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=device,
    )

    robot = scene["robot"]
    joint_indexes = robot.find_joints(JOINT_ORDER, preserve_order=True)[0]
    body_names = list(robot.body_names)
    ankle_l = body_names.index("ankle_left")
    ankle_r = body_names.index("ankle_right")
    trunk = body_names.index("trunk_base")

    log: dict[str, Any] = {
        "fps": [args.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    frames = []
    scene.reset()

    for _ in range(motion.output_frames):
        (
            (
                base_pos,
                base_rot,
                base_lin_vel,
                base_ang_vel,
                dof_pos,
                dof_vel,
            ),
            _reset,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = dof_pos
        joint_vel[:, joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        if renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(
            robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
        )
        log["body_ang_vel_w"].append(
            robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
        )

    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **log)
    print(f"Wrote {args.output} ({motion.output_frames} frames)")

    if renderer is not None and frames:
        import imageio.v2 as imageio

        with imageio.get_writer(
            args.render,
            fps=args.output_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Wrote {args.render}")

    # Feasibility report: numbers first, video second.
    body_pos = log["body_pos_w"]
    t = np.arange(body_pos.shape[0]) / args.output_fps
    foot_l_z = body_pos[:, ankle_l, 2]
    foot_r_z = body_pos[:, ankle_r, 2]
    trunk_y = body_pos[:, trunk, 1]
    rest_l, rest_r = foot_l_z[0], foot_r_z[0]
    print("\nFeasibility report (per beat):")
    print("beat  window          R-foot lift  L-foot lift  trunk y end")
    n_beats = int(t[-1] // BEAT) + 1
    for b in range(n_beats):
        sel = (t >= b * BEAT) & (t < (b + 1) * BEAT)
        if not sel.any():
            continue
        print(
            f"{b + 1:4d}  {b * BEAT:5.2f}-{(b + 1) * BEAT:5.2f} s   "
            f"{(foot_r_z[sel] - rest_r).max() * 1000:8.1f} mm  "
            f"{(foot_l_z[sel] - rest_l).max() * 1000:8.1f} mm  "
            f"{trunk_y[sel][-1] * 1000:8.1f} mm"
        )
    print(
        f"\nGround clearance: min foot-body z delta "
        f"{min((foot_l_z - rest_l).min(), (foot_r_z - rest_r).min()) * 1000:.1f} mm "
        f"(large negative = feet punch through the floor, fix the keyframes)"
    )
    print(
        f"Trunk y travel: {trunk_y.min() * 1000:.1f} to {trunk_y.max() * 1000:.1f} mm"
    )


if __name__ == "__main__":
    main()
