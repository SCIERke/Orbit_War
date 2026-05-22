#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_NAME="submission.tar.gz"

SOURCE_FILES=(
  "main.py"
  "agent/n_nearest_planet.py"
  "lib/celestialBody.py"
  "lib/comet.py"
  "lib/planet.py"
  "lib/ship.py"
  "lib/sun.py"
  "game_types/planet.py"
)

python -m py_compile "${SOURCE_FILES[@]}"

rm -f "$ARCHIVE_NAME"
tar -czf "$ARCHIVE_NAME" "${SOURCE_FILES[@]}"

echo "Built $ARCHIVE_NAME with ${#SOURCE_FILES[@]} files"
tar -tzf "$ARCHIVE_NAME"
