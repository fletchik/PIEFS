#!/usr/bin/env bash
# HTRU2 binary classification — reproduce Table 1
# K=6, seeds 42 and 123, three metric variants.
set -e
cd "$(dirname "$0")/.."

for SEED in 42 123; do
  for METRIC in off diag lambda_u_sparse; do
    RUN_ID="htru2_K6_${METRIC}_seed${SEED}"
    echo "=== $RUN_ID ==="
    python train.py \
      run_id="$RUN_ID" \
      dataset=htru2 \
      model.metric_type="${METRIC}" \
      model.K=6 \
      model.task=binary \
      trainer.seed="$SEED" \
      writer.mode=online
  done
done
