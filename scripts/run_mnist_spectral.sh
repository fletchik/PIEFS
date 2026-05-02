#!/usr/bin/env bash
# MNIST 10-class unsupervised (NeuralEF protocol)
# K=10, no task loss (w_task=0), fit LR on features after training.
set -e
cd "$(dirname "$0")/.."

for SEED in 42 123; do
  for METRIC in off diag lambda_u_sparse; do
    RUN_ID="mnist10_K10_${METRIC}_seed${SEED}"
    echo "=== $RUN_ID ==="
    python train.py \
      run_id="$RUN_ID" \
      dataset=mnist_multiclass \
      model.metric_type="${METRIC}" \
      model.K=10 \
      model.task=multiclass \
      criterion.w_task=0.0 \
      trainer.seed="$SEED" \
      writer.mode=online
  done
done
