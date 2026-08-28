# Agent notes: microduck-courier

Day project: a Microduck delivers *The Courage to Be Disliked* across a tiny apartment.

## Layout

- This repo is the workspace. `microduck_rl/` is Pollen’s stack **vendored in-tree** (not a nested git clone). Courier work lives here so any agent can pick it up.
- Upstream RL: https://github.com/pollen-robotics/microduck_rl
- New task id: `Mjlab-Courier-Flat-MicroDuck` (pick → carry → place). Reward delivery, not cute motion.

## Do next, in order

1. `bash scripts/fetch_policies.sh`: walking / stand / ground-pick ONNX (gitignored, ~3 MB).
2. Pose viewer: `python3 microduck_rl/scripts/view_apartment.py` (macOS: MuJoCo needs `mjpython`; “Task policy set failed” is a Cocoa QoS warning, already filtered).
3. Walk in the apartment: `./play_apartment.sh` (arrow keys in the **terminal**, `G` = ground pick).
4. Train: CUDA GPU or `uv run train Mjlab-Courier-Flat-MicroDuck --hf-jobs …`. This laptop is M4 / no CUDA.

## Files we own

- `microduck_rl/src/mjlab_microduck/robot/microduck/book.xml`
- `microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml`
- `microduck_rl/src/mjlab_microduck/tasks/microduck_courier_env_cfg.py`
- `microduck_rl/src/mjlab_microduck/tasks/mdp.py` (courier_* functions)
- `microduck_rl/scripts/view_apartment.py`
- `microduck_rl/scripts/infer_policy.py` (`--apartment`)
- `microduck_rl/tests/test_courier_cfg.py`

Do not re-clone `microduck_rl` into a nested `.git`. Keep courier changes in this repo so GitHub is the source of truth.
