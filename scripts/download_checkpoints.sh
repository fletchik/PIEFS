#!/usr/bin/env bash
# Download pretrained checkpoints from GitHub Release v1.0
# Usage: bash scripts/download_checkpoints.sh [output_dir]
#
# Downloads 25 checkpoints (5 datasets × 5 metrics, seed 0, best val accuracy).
# Datasets : two_moon  circles  htru2  mnist_binary  mnist_mc
# Metrics  : off  diag  conformal  global_low_rank  local_low_rank

set -euo pipefail

RELEASE_URL="https://github.com/fletchik/PIEFS/releases/download/v1.0"
OUT_DIR="${1:-checkpoints}"

mkdir -p "$OUT_DIR"

DATASETS="two_moon circles htru2 mnist_binary mnist_mc"
METRICS="off diag conformal global_low_rank local_low_rank"

echo "Downloading checkpoints to $OUT_DIR/ ..."
for DS in $DATASETS; do
    for MT in $METRICS; do
        FILE="${DS}_${MT}.pt"
        DEST="$OUT_DIR/$FILE"
        if [ -f "$DEST" ]; then
            echo "  skip (exists): $FILE"
        else
            echo "  $FILE"
            curl -fsSL "$RELEASE_URL/$FILE" -o "$DEST"
        fi
    done
done

echo "Done. $OUT_DIR/ contains $(ls "$OUT_DIR"/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints."
