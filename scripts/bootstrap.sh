#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and re-run this script."
  exit 1
fi

mkdir -p var/lancedb var/cache models
uv sync

echo "Bootstrap complete."
echo "Next:"
echo "  uv run python scripts/download_model.py"
echo "  cp .env.example .env"
echo "  # edit .env and set ALPHASEARCH_MODEL_PATH=./models/Qwen3-VL-Embedding-2B"
echo "  uv run python scripts/index_data.py --reset"

