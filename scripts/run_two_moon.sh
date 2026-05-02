#!/usr/bin/env bash
# Two-moon 2-D experiments — rich visualisation focus
set -e
cd "$(dirname "$0")/.."

for SEED in 42 123; do
  for METRIC in off diag lambda_u_sparse; do
    RUN_ID="twomoon_K6_${METRIC}_seed${SEED}"
    echo "=== $RUN_ID ==="
    python train.py \
      run_id="$RUN_ID" \
      dataset=two_moon \
      model.metric_type="${METRIC}" \
      model.K=6 \
      model.task=binary \
      trainer.seed="$SEED" \
      writer.mode=online
  done
done
