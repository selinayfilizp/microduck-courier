#!/usr/bin/env python3
"""Author the Toosie Slide reference motion for the Microduck.

Generates a joint-space keyframe CSV (50 fps) for the mjlab tracking pipeline:
base pose (xyz + quat xyzw) plus the 14 servo joints, as deltas around the
HOME standing pose. One choreography cycle is one 82 BPM bar, straight from
the hook: right foot up (beat 1), left foot slide (beat 2), left foot up
(beat 3), right foot slide (beat 4). The clip holds two bars and returns
exactly to its starting pose and position, so the tracking command's restart
at clip end is a seamless loop.

Convert to npz (with a rendered kinematic ghost video and a feasibility
report) via scripts/motion_csv_to_npz.py. Iterate the amplitudes below until
the ghost reads as THE dance, then train; the CSV costs nothing to remake.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Canonical CSV joint order: matches the servo layout (0-4 left leg,
# 5-8 neck/head, 9-13 right leg). motion_csv_to_npz.py uses the same list.
JOINT_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

# HOME standing pose (STAND2, from microduck_constants.HOME_FRAME).
HOME = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

# Joint limits (robot_allcollisions.xml), pre-shrunk by the same 0.9 soft
# factor the robot cfg uses.
LIMITS = {
    "left_hip_yaw": (-0.436, 0.524),
    "left_hip_roll": (-0.384, 0.384),
    "left_hip_pitch": (-1.571, 1.571),
    "left_knee": (-1.571, 1.571),
    "left_ankle": (-1.571, 1.571),
    "neck_pitch": (-1.571, 1.047),
    "head_pitch": (-1.571, 1.571),
    "head_yaw": (-2.967, 2.967),
    "head_roll": (-0.436, 0.436),
    "right_hip_yaw": (-0.524, 0.436),
    "right_hip_roll": (-0.384, 0.384),
    "right_hip_pitch": (-1.571, 1.571),
    "right_knee": (-1.571, 1.571),
    "right_ankle": (-1.571, 1.571),
}

# --- Tunables: iterate these against the ghost replay -----------------------
BPM = 82.0
FPS = 50
BARS = 2
STAND_Z = 0.115  # trunk_base free-joint height at HOME

LIFT_HIP_PITCH = 0.55  # leg-shortening deltas for the foot-up beats
LIFT_KNEE = 0.95
LIFT_ANKLE = 0.15
SWAY_HIP_ROLL = 0.14  # weight-shift lean during a lift (same sign both hips)
SLIDE_Y = 0.06  # lateral base travel per slide beat (meters)
BOUNCE_Z = 0.004  # groove bounce amplitude (meters)
BOB_NECK = 0.14  # per-beat head bob on neck_pitch
HEAD_YAW_SWAY = 0.18  # once-per-bar head yaw sway
# ---------------------------------------------------------------------------

BEAT = 60.0 / BPM
BAR = 4.0 * BEAT


def _bump(t: float, t0: float, t1: float) -> float:
    """Raised-cosine 0 -> 1 -> 0 over [t0, t1]."""
    if t <= t0 or t >= t1:
        return 0.0
    x = (t - t0) / (t1 - t0)
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * x))


def _smoothstep(t: float, t0: float, t1: float) -> float:
    """Smooth 0 -> 1 over [t0, t1], clamped outside."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def _mirrored_lift(deltas: dict, side: str, k: float) -> None:
    """Shorten one leg by k of the lift amplitudes (mirror-signed deltas).

    Sign convention verified against the FK replay report: moving each joint
    AGAINST its HOME sign shortens the leg (the first draft moved with the
    HOME sign and drove the foot 17.5 mm through the floor).
    """
    s = -1.0 if side == "right" else 1.0
    deltas[f"{side}_hip_pitch"] += s * LIFT_HIP_PITCH * k
    deltas[f"{side}_knee"] += s * LIFT_KNEE * k
    deltas[f"{side}_ankle"] += -s * LIFT_ANKLE * k


def frame_at(t: float) -> tuple[list[float], dict]:
    tc = t % BAR  # position within the choreography cycle (one bar)
    deltas = {name: 0.0 for name in JOINT_ORDER}

    # Beat 1: right foot up, weight swaying onto the left leg.
    lift_r = _bump(tc, 0.0, BEAT)
    _mirrored_lift(deltas, "right", lift_r)
    deltas["left_hip_roll"] -= SWAY_HIP_ROLL * lift_r
    deltas["right_hip_roll"] -= SWAY_HIP_ROLL * lift_r

    # Beat 3: left foot up, weight swaying onto the right leg.
    lift_l = _bump(tc, 2.0 * BEAT, 3.0 * BEAT)
    _mirrored_lift(deltas, "left", lift_l)
    deltas["left_hip_roll"] += SWAY_HIP_ROLL * lift_l
    deltas["right_hip_roll"] += SWAY_HIP_ROLL * lift_l

    # Beat 2: slide left (base translates +y); beat 4: slide back right.
    y = SLIDE_Y * _smoothstep(tc, BEAT, 2.0 * BEAT)
    y -= SLIDE_Y * _smoothstep(tc, 3.0 * BEAT, 4.0 * BEAT)

    # Groove: subtle bounce at beat rate, head bob every beat, yaw sway per bar.
    z = STAND_Z - BOUNCE_Z * 0.5 * (1.0 - math.cos(2.0 * math.pi * tc / BEAT))
    # Rise slightly onto the support leg during a lift so the sway does not
    # press the planted foot through the floor (FK report: -8.9 mm without).
    z += 0.006 * (lift_r + lift_l)
    beat_idx = int(tc // BEAT)
    deltas["neck_pitch"] += BOB_NECK * _bump(
        tc, beat_idx * BEAT, (beat_idx + 1) * BEAT
    )
    deltas["head_yaw"] += HEAD_YAW_SWAY * math.sin(2.0 * math.pi * tc / BAR)

    joints = []
    for name in JOINT_ORDER:
        lo, hi = LIMITS[name]
        val = HOME[name] + deltas[name]
        joints.append(min(max(val, 0.9 * lo), 0.9 * hi))
    # Base pose: x fixed, y slides, z bounces, identity orientation (xyzw).
    base = [0.0, y, z, 0.0, 0.0, 0.0, 1.0]
    return base + joints, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("motions/toosie_slide.csv")
    )
    args = parser.parse_args()

    n_frames = round(BARS * BAR * FPS)
    rows = [frame_at(i / FPS)[0] for i in range(n_frames)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.asarray(rows), delimiter=",", fmt="%.6f")

    first, last = np.asarray(rows[0]), np.asarray(rows[-1])
    print(
        f"Wrote {args.output}: {n_frames} frames at {FPS} fps "
        f"({BARS} bars of {BPM} BPM, {n_frames / FPS:.2f} s)"
    )
    print(
        "Loop seam (first vs last frame): "
        f"max joint gap {np.abs(first[7:] - last[7:]).max():.4f} rad, "
        f"base gap {np.abs(first[:3] - last[:3]).max() * 1000:.1f} mm"
    )


if __name__ == "__main__":
    main()
