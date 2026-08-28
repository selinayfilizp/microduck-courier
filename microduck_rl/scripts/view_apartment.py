#!/usr/bin/env python3
"""Open or record the apartment scene (no RL / GPU needed).

The duck holds the STAND pose. Drag the camera; Esc to quit.
Pass --record clip.mp4 to write a 20s spinning-camera clip. Add --demo for a
clearly labelled, deterministic fail -> recover -> deliver storyboard. The
storyboard is a scene/demo check, not a trained-policy rollout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, "src/mjlab_microduck/robot/microduck/scene_apartment.xml")

STAND_JOINTS = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float64)


def _load():
    import mujoco

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[trunk])
    data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(model.nu):
        jnt = int(model.actuator_trnid[i, 0])
        data.qpos[int(model.jnt_qposadr[jnt])] = STAND_JOINTS[i]
        data.ctrl[i] = STAND_JOINTS[i]
    book = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "book_free")
    if book >= 0:
        badr = int(model.jnt_qposadr[book])
        data.qpos[badr:badr + 7] = [0.16, 0.0, 0.006, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    qpos0 = data.qpos.copy()
    return mujoco, model, data, qpos0


def _hold(mujoco, model, data, qpos0):
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _storyboard_pose(mujoco, model, data, qpos0, t: float):
    """Pose the scene for the deterministic 20-second demo storyboard."""
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0

    trunk = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root = int(model.jnt_qposadr[trunk])
    book = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "book_free")
    badr = int(model.jnt_qposadr[book])

    x, y, z, pitch = 0.0, 0.0, 0.125, 0.0
    book_xyz = np.array([0.16, 0.0, 0.006], dtype=np.float64)
    gait = 0.0

    if 2.0 <= t < 5.0:  # first attempt; the book slips sideways
        u = _smoothstep((t - 2.0) / 2.0)
        pitch = 1.0 * u
        z = 0.125 - 0.025 * u
        x = 0.025 * u
        if t >= 4.0:
            miss = _smoothstep((t - 4.0) / 0.7)
            book_xyz[1] = 0.055 * miss
    elif 5.0 <= t < 7.0:  # recover upright and line up with the slipped book
        u = _smoothstep((t - 5.0) / 2.0)
        pitch = 1.0 * (1.0 - u)
        z = 0.10 + 0.025 * u
        x = 0.025 * (1.0 - u)
        y = 0.045 * u
        book_xyz[1] = 0.055
    elif 7.0 <= t < 10.0:  # second reach and successful pick
        u = _smoothstep((t - 7.0) / 2.2)
        x, y = 0.025 * u, 0.045
        z = 0.125 - 0.025 * u
        pitch = 1.0 * u
        book_xyz[1] = 0.055
        if t >= 9.1:
            lift = _smoothstep((t - 9.1) / 0.9)
            book_xyz = (1.0 - lift) * book_xyz + lift * np.array(
                [x + 0.105, y + 0.005, z + 0.01]
            )
    elif 10.0 <= t < 16.0:  # carry across the rug with a visible gait
        u = _smoothstep((t - 10.0) / 6.0)
        x, y = 0.025 + 0.355 * u, 0.045 * (1.0 - u)
        rise = _smoothstep((t - 10.0) / 1.0)
        z, pitch = 0.10 + 0.025 * rise, 1.0 * (1.0 - rise)
        gait = np.sin(2.0 * np.pi * 2.2 * (t - 10.0))
        book_xyz = np.array([x + 0.075, y, z + 0.10])
    elif 16.0 <= t < 18.2:  # set the book down at the reader's feet
        u = _smoothstep((t - 16.0) / 1.7)
        x, y = 0.38, 0.0
        z, pitch = 0.125 - 0.025 * u, 0.9 * u
        held = np.array([x + 0.075, y, z + 0.10])
        placed = np.array([0.50, -0.07, 0.006])
        book_xyz = (1.0 - u) * held + u * placed
    elif t >= 18.2:  # clean success hold
        u = _smoothstep((t - 18.2) / 1.0)
        x, y = 0.38, 0.0
        z, pitch = 0.10 + 0.025 * u, 0.9 * (1.0 - u)
        book_xyz = np.array([0.50, -0.07, 0.006])

    data.qpos[root:root + 3] = [x, y, z]
    data.qpos[root + 3:root + 7] = [
        np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0
    ]
    for i in range(model.nu):
        jnt = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[jnt])
        q = STAND_JOINTS[i]
        if i in (2, 11):
            q += (0.18 if i == 2 else -0.18) * gait
        if i in (3, 12):
            q += (-0.22 if i == 3 else 0.22) * gait
        data.qpos[qadr] = q
        data.ctrl[i] = q
    data.qpos[badr:badr + 7] = [*book_xyz, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def _demo_label(t: float) -> tuple[str, tuple[int, int, int]]:
    if t < 2.0:
        return "DELIVER THE BOOK", (245, 232, 190)
    if t < 5.0:
        return "ATTEMPT 1  /  MISS", (244, 114, 101)
    if t < 7.0:
        return "RECOVER + RE-AIM", (249, 196, 86)
    if t < 10.0:
        return "PICK", (104, 201, 150)
    if t < 16.0:
        return "CARRY", (104, 181, 246)
    if t < 18.2:
        return "PLACE", (180, 142, 255)
    return "DELIVERED", (104, 220, 150)


def _annotate_demo(frame: np.ndarray, t: float) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    small_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 42) if os.path.exists(font_path) else ImageFont.load_default()
    small = ImageFont.truetype(small_path, 19) if os.path.exists(small_path) else ImageFont.load_default()
    label, color = _demo_label(t)
    draw.rounded_rectangle((40, 38, 480, 112), radius=16, fill=(15, 13, 12, 205))
    draw.text((62, 52), label, font=font, fill=(*color, 255))
    draw.rounded_rectangle((40, 650, 705, 696), radius=12, fill=(15, 13, 12, 185))
    draw.text(
        (58, 663),
        "SCRIPTED STORYBOARD  •  replace with trained courier rollout after GPU run",
        font=small,
        fill=(235, 228, 215, 255),
    )
    return np.asarray(image)


def _mjpython_path() -> str | None:
    found = shutil.which("mjpython")
    if found:
        return found
    fallback = os.path.expanduser("~/Library/Python/3.9/bin/mjpython")
    return fallback if os.path.exists(fallback) else None


def _relaunch_quiet_on_macos() -> None:
    """mjpython is required for the Cocoa viewer. It also spams

        Task policy set failed: 4 ((os/kern) invalid argument)

    from Apple's thread-QoS APIs. That is not a scene/load failure — filter it.
    """
    if sys.platform != "darwin":
        return
    if "mjpython" in os.path.basename(sys.executable):
        return
    mjpython = _mjpython_path()
    if not mjpython:
        sys.exit(
            "On macOS the MuJoCo viewer needs mjpython.\n"
            "  python3 -m pip install mujoco\n"
            "  mjpython scripts/view_apartment.py"
        )
    print("Opening the apartment window. Drag to look around, Esc to quit.")
    print("(macOS may log a 'Task policy' warning — that's noise, not a crash.)")
    proc = subprocess.Popen(
        [mjpython, os.path.abspath(__file__), *sys.argv[1:]],
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stderr is not None
    for line in proc.stderr:
        if "Task policy set failed" in line or "((os/kern) invalid argument)" in line:
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    raise SystemExit(proc.wait())


def view():
    _relaunch_quiet_on_macos()

    mujoco, model, data, qpos0 = _load()
    import mujoco.viewer

    print("Apartment scene. Duck is standing. Book is in front. Reader is sitting.")
    print("This is the pose viewer — for a walking duck, run ./play_apartment.sh")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _hold(mujoco, model, data, qpos0)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def record(path: str, seconds: float = 20.0, fps: int = 30, demo: bool = False):
    mujoco, model, data, qpos0 = _load()
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg  (needed for --record)")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    nframes = int(seconds * fps)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.28, 0.0, 0.08]
    cam.distance = 0.70
    cam.elevation = -18
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg
    print(f"Recording {seconds:.0f}s → {path}")
    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    ) as writer:
        for i in range(nframes):
            t = i / fps
            cam.azimuth = 105 + 28 * (i / max(nframes - 1, 1))
            cam.lookat[0] = 0.27 + 0.10 * (i / max(nframes - 1, 1))
            if demo:
                _storyboard_pose(mujoco, model, data, qpos0, t)
            else:
                _hold(mujoco, model, data, qpos0)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if demo:
                frame = _annotate_demo(frame, t)
            writer.append_data(frame)
    renderer.close()
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Record a labelled fail/recover/deliver storyboard (not a policy rollout)",
    )
    args = parser.parse_args()
    if not os.path.exists(XML):
        sys.exit(f"Missing scene: {XML}")
    if args.record:
        record(args.record, seconds=args.seconds, demo=args.demo)
    else:
        view()


if __name__ == "__main__":
    main()
