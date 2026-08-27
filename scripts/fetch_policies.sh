#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/policies"
mkdir -p "$DEST"
BASE="https://raw.githubusercontent.com/pollen-robotics/microduck/main/policies"
for f in alpha_walking.onnx alpha_stand.onnx alpha_ground_pick.onnx alpha_sitstand.onnx; do
  echo "fetch $f"
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
done
ls -lh "$DEST"
