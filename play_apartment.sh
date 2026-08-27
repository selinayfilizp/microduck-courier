#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RL="$ROOT/microduck_rl"
POL="$ROOT/policies"

if [[ ! -f "$POL/alpha_walking.onnx" ]]; then
  echo "Missing policies. Run: bash scripts/fetch_policies.sh" >&2
  exit 1
fi

cd "$RL"
exec uv run scripts/infer_policy.py \
  --apartment \
  --walking "$POL/alpha_walking.onnx" \
  --standing "$POL/alpha_stand.onnx" \
  --ground-pick "$POL/alpha_ground_pick.onnx" \
  --new-cmd-obs \
  --current-limit 0 \
  "$@"
