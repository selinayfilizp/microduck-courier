# microduck-courier

One-day physical-AI project: teach a [Microduck](https://pollen-robotics.com/microduck/) to deliver *The Courage to Be Disliked* across a tiny apartment.

Pick → carry → place. Reward the delivery, not looking cute.

![The trained policy stumbles to 61 degrees, recovers, grasps the book, and delivers it to the reader](artifacts/courier-policy.gif)

*First episode of the real trained rollout: stumble at 0.6 s, recovery, grasp at 2.9 s, delivery at the reader's feet at 6.0 s. The full 20-second clip and its telemetry are in [Release artifacts](#release-artifacts).*

Clone: `git clone https://github.com/selinayfilizp/microduck-courier.git`

Other agents: read `AGENTS.md` first. This repo is the source of truth (Pollen’s RL stack is vendored under `microduck_rl/`, not a nested clone).

## What’s here

- `microduck_rl/`: Pollen’s RL stack, plus a new task `Mjlab-Courier-Flat-MicroDuck`
- `microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml`: floor, rug, paperback, seated reader
- `policies/`: local 61-D ONNX brains, including the exported courier policy
- `artifacts/`: verified trained-policy clip, telemetry, and deployable ONNX

This Mac has an M4, not CUDA. Walking in the viewer is CPU MuJoCo. Training the courier policy needs a GPU; use Hugging Face Jobs.

Install [uv](https://docs.astral.sh/uv/) once (`brew install uv` or see their docs) before the walking / train commands.

## 1. Duck in the apartment (now)

```bash
cd microduck_rl
python3 scripts/view_apartment.py          # macOS re-launches under mjpython
```

Hold STAND. Drag the camera. Esc to quit.

Walk it with the official gait (arrow keys in the terminal, not the viewer):

```bash
./play_apartment.sh
```

`G` triggers ground-pick (beak to the floor, the pick half of the job).

## 2. Trained courier policy

The selected L4 checkpoint is `model_750.pt` from the private Hugging Face model
repo `selinayfilizp/mjlab-courier-flat-microduck-20260828-134647`. The paid job
was stopped after this checkpoint passed the full task, instead of spending the
remaining budget.

Strict CPU evaluation with play-mode perturbations enabled produced:

- 16/16 deliveries over 8-second rollouts with seed 42.
- 32/32 deliveries over 20-second rollouts with seed 1042.
- Zero failed episodes and zero NaN terminations in the 32-rollout check.

The deployment export is available locally as `policies/courier.onnx` (61 float
inputs, 14 float actions). The repository ignores downloaded/exported ONNX
artifacts, so keep the checkpoint or regenerate the file before moving machines.

### Retrain it

Needs a CUDA GPU. On this laptop:

```bash
cd microduck_rl
WANDB_MODE=disabled uv run train Mjlab-Courier-Flat-MicroDuck \
  --gpu-ids None --env.scene.num-envs 1 \
  --agent.num-steps-per-env 4 --agent.max-iterations 1 \
  --agent.save-interval 1
```

That is a four-step CPU wiring smoke test, not useful training. Real run on an
L4:

```bash
uv run hf auth login
cd ..
./scripts/train_courier_hf.sh
```

Watch `Episode_Reward/place_success` and `Episode_Reward/carry_progress`. Every `Episode_Reward/*penalty*` must stay ≤ 0.

The training reward pays `grasp_edge` and `place_success` once per event; neither
can be farmed by holding a pose. Carry progress is paid only while upright, a
non-delivery termination costs 500, and delivery pays 250 once. The actor remains
61-D: its existing head-command slot is book xyz + grasp flag, and its
body-command slot is reader xyz + padding. A real deployment must populate those
slots from perception.

## 3. The 20-second clip

Standing scene, spinning camera:

```bash
cd microduck_rl
python3 scripts/view_apartment.py --record ../clips/apartment.mp4 --seconds 20
```

Deterministic fail → recover → deliver storyboard (local, CPU, explicitly
watermarked as scripted):

```bash
uv run python scripts/view_apartment.py \
  --record ../clips/courier-storyboard.mp4 --seconds 20 --demo
```

Record a real policy rollout and reject unsuccessful or visually incomplete
seeds automatically:

```bash
uv run python scripts/record_courier_policy.py \
  /path/to/model_750.pt \
  --output ../clips/courier-policy.mp4 --seconds 20 --seed 14 \
  --require-success --require-stumble-recovery
```

The recorder uses the actual mjlab environment and policy observations and emits
a JSON sidecar with grasp, stumble, recovery, and delivery times. Publish that
file as the RL result; keep the watermarked storyboard only as a shot plan.

Play mode resets every 8 seconds with a fresh spawn heading while the camera
stays fixed, so the first episode carries the framed story; later episodes can
drift out of view. Cut to the first 8 seconds when the clip has to stand alone.

## Release artifacts

- [20-second trained-policy rollout](artifacts/courier-policy.mp4)
- [Rollout telemetry](artifacts/courier-policy.json)
- [61-input, 14-action ONNX policy](artifacts/courier-policy.onnx)

The video is a real policy rollout, not the scripted storyboard. The selected
checkpoint itself remains on Hugging Face; the smaller ONNX deployment export is
committed here so a fresh clone contains the usable policy.
