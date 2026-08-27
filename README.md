# microduck-courier

One-day physical-AI project: teach a [Microduck](https://pollen-robotics.com/microduck/) to deliver *The Courage to Be Disliked* across a tiny apartment.

Pick → carry → place. Reward the delivery, not looking cute.

Clone: `git clone https://github.com/selinayfilizp/microduck-courier.git`

Other agents: read `AGENTS.md` first. This repo is the source of truth (Pollen’s RL stack is vendored under `microduck_rl/`, not a nested clone).

## What’s here

- `microduck_rl/` — Pollen’s RL stack, plus a new task `Mjlab-Courier-Flat-MicroDuck`
- `microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml` — floor, rug, paperback, seated reader
- `policies/` — shipped 61-D ONNX brains (walking, stand, ground-pick)

This Mac has an M4, not CUDA. Walking in the viewer is CPU MuJoCo. Training the courier policy needs a GPU — use Hugging Face Jobs.

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

`G` triggers ground-pick (beak to the floor — the pick half of the job).

## 2. Train the courier policy

Needs a CUDA GPU. On this laptop:

```bash
cd microduck_rl
uv run train Mjlab-Courier-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5
```

That’s the smoke test. Real run (~1–2 h on an L4):

```bash
uv run train Mjlab-Courier-Flat-MicroDuck \
  --env.scene.num-envs 4096 --hf-jobs --flavor l4x1 --timeout 3h
```

Watch `Episode_Reward/place_success` and `Episode_Reward/carry_progress`. Every `Episode_Reward/*penalty*` must stay ≤ 0.

## 3. The 20-second clip

Standing scene, spinning camera:

```bash
cd microduck_rl
python3 scripts/view_apartment.py --record ../clips/apartment.mp4 --seconds 20
```

Walking clip: run `./play_apartment.sh` and screen-record. After a courier checkpoint exists, swap it in with `uv run scripts/infer_policy.py --apartment --walking … --new-cmd-obs`.
