#!/usr/bin/env bash
# CIFAR-10 binary — grid search over K, w_gram, w_dirichlet
set -e
cd "$(dirname "$0")/.."

for SEED in 42 123; do
  for K in 6 16 32; do
    for METRIC in off diag; do
      for W_GRAM in 0.05 0.1 0.5; do
        for W_DIR in 1.0 5.0 10.0; do
          RUN_ID="cifar10_K${K}_${METRIC}_wg${W_GRAM}_wd${W_DIR}_seed${SEED}"
          echo "=== $RUN_ID ==="
          python train.py \
            run_id="$RUN_ID" \
            dataset=cifar10_binary \
            model.metric_type="${METRIC}" \
            model.K="$K" \
            model.task=binary \
            criterion.w_gram="$W_GRAM" \
            criterion.w_dirichlet="$W_DIR" \
            trainer.seed="$SEED" \
            writer.mode=online
        done
      done
    done
  done
done
