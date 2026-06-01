#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/data/models/trellis}"
REPO_ID="microsoft/TRELLIS.2-4B"

echo "=== TRELLIS 2 Model Downloader ==="
echo "Target: $TARGET_DIR"
echo "Model:  $REPO_ID"

mkdir -p "$TARGET_DIR"

if [ -f "$TARGET_DIR/config.json" ]; then
    echo "Resuming download..."
else
    echo "Starting fresh download..."
fi

if ! command -v huggingface-cli &>/dev/null; then
    pip install -q huggingface-hub
fi

huggingface-cli download "$REPO_ID" \
    --local-dir "$TARGET_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "=== Download Complete ==="
du -sh "$TARGET_DIR"
