#!/usr/bin/env bash
# =============================================================================
# run_overnight_B.sh — cifar10_features (all 15 runs: 3 metrics × 5 seeds)
#
# Run with:  caffeinate -i bash scripts/run_overnight_B.sh
#
# Estimated wall time: ~5h (sequential on CPU)
#   cifar10 off s0-s4      ≈0.7h  (d=512, ~4.5ms/step × 120k)
#   cifar10 diag s0-s4     ≈1.0h  (d=512, ~6.4ms/step × 120k)
#   cifar10 trotter s0-s4  ≈3.5h  (d=512, vectorised Trotter ~21ms/step)
#
# Prerequisites: data/cifar10_features/ must exist (run extract_cnn_features.py)
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python3}"
mkdir -p logs
exec >> "logs/overnight_B.log" 2>&1

run_job() {
    local run_id="$1"
    local ds_args="$2"
    local metric="$3"
    local seed="$4"
    local steps="$5"

    local final_ckpt="logs/${run_id}/checkpoint_final.pt"
    local best_ckpt="logs/${run_id}/checkpoint_best_val.pt"

    if [ -f "$final_ckpt" ]; then
        echo "[SKIP] ${run_id} (already complete)"
        return 0
    fi

    local resume_flag=""
    if [ -f "$best_ckpt" ]; then
        resume_flag="+resume=${best_ckpt}"
        echo "[RESUME] ${run_id}"
    else
        echo "[START] ${run_id}"
    fi

    $PYTHON train.py \
        run_id="${run_id}" \
        ${ds_args} \
        model.metric_type="${metric}" \
        trainer.seed="${seed}" \
        trainer.total_steps="${steps}" \
        writer.mode=disabled \
        ${resume_flag}

    echo "[DONE] ${run_id}"
    echo "------------------------------------------------------------"
}

echo "============================================================"
echo "EFDO Overnight B — started $(date)"
echo "cifar10_features: off/diag/trotter × s0-s4  (15 runs)"
echo "============================================================"

# Check that cifar10 features exist
if [ ! -f "data/cifar10_features/X_train.npy" ]; then
    echo "ERROR: data/cifar10_features/X_train.npy not found!"
    echo "Run: .venv/bin/python3 scripts/extract_cnn_features.py first."
    exit 1
fi

CF_ARGS="dataset=cifar10_features_multiclass model.task=multiclass model.K=16"

# off first (fastest: ~8 min/seed)
for seed in 0 1 2 3 4; do
    run_job "grid_cifar10_features_off_s${seed}" \
            "${CF_ARGS}" "off" "${seed}" 120000
done

# diag (medium: ~12 min/seed)
for seed in 0 1 2 3 4; do
    run_job "grid_cifar10_features_diag_s${seed}" \
            "${CF_ARGS}" "diag" "${seed}" 120000
done

# trotter last (slowest: ~42 min/seed, vectorised so feasible)
for seed in 0 1 2 3 4; do
    run_job "grid_cifar10_features_lambda_u_trotter_s${seed}" \
            "${CF_ARGS}" "lambda_u_trotter" "${seed}" 120000
done

echo "============================================================"
echo "Overnight B COMPLETE — $(date)"
echo "Run: .venv/bin/python3 scripts/collect_grid_results.py to aggregate."
echo "============================================================"
