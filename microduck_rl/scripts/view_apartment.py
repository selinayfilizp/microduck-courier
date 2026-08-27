#!/usr/bin/env python3
"""Open the apartment scene in the MuJoCo viewer (no RL / GPU needed).

The duck holds the STAND pose. Drag the camera; Esc to quit.
Pass --record clip.mp4 to write a 20s spinning-camera clip.
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


def record(path: str, seconds: float = 20.0, fps: int = 30):
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
    frames = []
    print(f"Recording {seconds:.0f}s → {path}")
    for i in range(nframes):
        cam.azimuth = 100 + 55 * (i / max(nframes - 1, 1))
        _hold(mujoco, model, data, qpos0)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    imageio.mimsave(path, frames, fps=fps)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not os.path.exists(XML):
        sys.exit(f"Missing scene: {XML}")
    if args.record:
        record(args.record, seconds=args.seconds)
    else:
        view()


if __name__ == "__main__":
    main()
