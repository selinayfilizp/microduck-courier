# From this repo to a real Microduck

Microduck is a real robot: Pollen Robotics opened preorders on 2026-08-27,
first deliveries targeted before Christmas 2026. Its official pipeline is
exactly what this repo uses: train in `microduck_rl` (MuJoCo, PPO, 50 Hz),
export ONNX with the observation normalizer baked in, run it in the
open-source Rust runtime on the robot's RK3566. This document maps what is
already deployment-shaped here, what the runtime must provide, and what is
still open.

## The observation contract

The actor is 61-D and shared across the whole policy family, so policies
hot-swap in the runtime:

| slice | dims | content |
| --- | --- | --- |
| proprio | 48 | base ang vel (3), projected gravity (3), joint pos (14), joint vel (14), last action (14) |
| twist command | 3 | courier: cos/sin of the task phase, 0 |
| head command | 4 | courier: book xyz in base frame, grasp flag |
| body command | 6 | courier: reader xyz in base frame, 3 zero pads |

The proprio half comes from the robot's IMU and encoders and is what every
shipped Pollen policy already consumes. The courier repurposes the two
command slots for object state. In sim those values are ground truth; on the
robot they must come from perception.

## The perception bridge (the real remaining work)

The robot has the sensors for it: a wide-angle camera, an 8x8 ToF array, and
two IMUs.

- Book xyz: detect the book in the camera frame (a fiducial sticker is the
  honest first version; a small detector later), depth from the ToF array or
  known object size, transform into the base frame using the trunk pose
  estimate. Write into the head-command slot at 50 Hz, holding the last
  estimate between detections.
- Grasp flag: the sim latch is kinematic. On hardware the equivalent signal
  is beak-servo position error (the beak closes on something) or the book
  still being visible below the head camera. Start with the beak current
  threshold; it is one number.
- Reader xyz: for a first demo the "reader" is a fixed marker on the floor
  (a tag on a chair leg), which is exactly what the sim's mocap-posed
  mannequin is. Person detection can replace it later.
- The wide task (v2) trains with noise and 0-2 control-step delay on both
  slots, so a perception stack with about 2 cm error and one dropped frame
  in a while is inside the training distribution, not outside it.

## What the sim already models honestly

- BAM actuator model of the XL330 servos (voltage law, back-EMF, friction),
  with per-env friction DR, battery-voltage sag, and command delay wired
  through `MICRODUCK_STANDUP_ROBOT_CFG`.
- IMU misalignment, encoder bias, CoM/mass/armature DR, velocity pushes.
- v2 additionally randomizes servo gains (KP/KD), book mass, friction, and
  spawn yaw, and verifies delivery only after the released book has settled
  on the floor inside the place radius.
- The book is already a beak-scale prop: 7 x 4.5 x 1.2 cm, 40 g. A real
  paperback is roughly half the robot's mass and does not fit the beak; the
  physical demo needs a printed prop book with a cover, which is also the
  charming version.

## Runtime notes

- Export ONLY through `scripts/export.py` (or the auto-export at the end of
  an HF Jobs run): it bakes the observation normalizer into the graph. A
  hand-converted checkpoint works in sim and fails on hardware.
- The exported courier ONNX is (1, 61) in, (1, 14) out, deterministic mean
  actions, exactly like the shipped walk/stand policies, so the runtime's
  existing hot-swap path applies. The runtime side must write the two command
  slots from the perception bridge instead of zeros; all-zero slots mean
  "book at my feet, reader at my feet".
- Policies are unfiltered by design. Do not add smoothing on one side only.

## Open items, in order

1. Perception bridge above (camera + ToF to two slots at 50 Hz).
2. A settle-verified eval on hardware mirroring `evaluate_courier_policy.py`
   (count deliveries over N trials, publish the sidecar with provenance).
3. Battery-length endurance: sim episodes are 14 s; a real fetch cycle with
   search time will run minutes on a roughly 1 h battery.
4. No hardware exists to test on until units ship (preorder lead time is
   4-6 months), which is exactly why the sim evidence stays seeded, stamped,
   and reproducible from the committed artifact.
