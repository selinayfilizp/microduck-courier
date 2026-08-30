#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/microduck_rl"

if ! uv run hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face is not authenticated." >&2
  echo "Run: uv run hf auth login" >&2
  exit 1
fi

# Override these without editing the script, for example:
#   COURIER_TASK=Mjlab-Courier-Flat-MicroDuck COURIER_ITERS=4000 \
#     ./scripts/train_courier_hf.sh
COURIER_TASK="${COURIER_TASK:-Mjlab-Courier-Wide-MicroDuck}"
COURIER_HF_FLAVOR="${COURIER_HF_FLAVOR:-l4x1}"
COURIER_HF_TIMEOUT="${COURIER_HF_TIMEOUT:-3h}"
COURIER_ITERS="${COURIER_ITERS:-3000}"

# CPU wiring smoke test before the paid job (AGENTS.md: never launch a long
# run without one). Catches config errors locally for zero dollars.
if [ "${COURIER_SKIP_SMOKE:-0}" != "1" ]; then
  echo "[smoke] CPU wiring test of $COURIER_TASK (COURIER_SKIP_SMOKE=1 skips)"
  WANDB_MODE=disabled uv run train "$COURIER_TASK" \
    --gpu-ids None --env.scene.num-envs 2 \
    --agent.num-steps-per-env 4 --agent.max-iterations 1 \
    --agent.save-interval 1 >/dev/null
  echo "[smoke] ok"
fi

# Invoke the wrapper module directly: `uv run train` can resolve to mjlab's
# own console script (which does not know --hf-jobs) depending on which
# package installed bin/train last.
exec uv run python -m mjlab_microduck.train_cli "$COURIER_TASK" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations "$COURIER_ITERS" \
  --agent.save-interval 250 \
  --hf-jobs \
  --flavor "$COURIER_HF_FLAVOR" \
  --timeout "$COURIER_HF_TIMEOUT" \
  --no-wandb \
  "$@"
