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
#   COURIER_HF_FLAVOR=a10g-large COURIER_HF_TIMEOUT=4h COURIER_ITERS=4000 \
#     ./scripts/train_courier_hf.sh
COURIER_HF_FLAVOR="${COURIER_HF_FLAVOR:-l4x1}"
COURIER_HF_TIMEOUT="${COURIER_HF_TIMEOUT:-3h}"
COURIER_ITERS="${COURIER_ITERS:-3000}"

exec uv run train Mjlab-Courier-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations "$COURIER_ITERS" \
  --agent.save-interval 250 \
  --hf-jobs \
  --flavor "$COURIER_HF_FLAVOR" \
  --timeout "$COURIER_HF_TIMEOUT" \
  --no-wandb \
  "$@"
