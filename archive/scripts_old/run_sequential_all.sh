#!/usr/bin/env bash
# =============================================================================
# run_sequential_all.sh — ALL remaining runs, strictly ONE process at a time.
#
# Run with:  caffeinate -i bash scripts/run_sequential_all.sh
#
# Why sequential? Parallel runs caused OOM (2× d=784 models in RAM at once).
# Each run: 1-4 GB RSS. Sequential keeps peak RAM at ~4 GB.
#
# Skip logic  : if checkpoint_final.pt exists  → already done, skip.
# Resume logic: if checkpoint_best_val.pt exists → resume from that checkpoint.
#
# Estimated wall time (~10 ms/step on M-series CPU, all sequential):
#   Finish partials                         ~  15 min
#   mnist_mc off/diag/trotter  s0-s4        ~  90 min  (60k steps each × 15 runs)
#   fashion_mnist trotter      s0-s4        ~ 100 min  (120k steps each × 5 runs)
#   cifar10_features off/diag/trotter s0-s4 ~ 270 min  (120k steps each × 15 runs)
#   ─────────────────────────────────────────────────────
#   Total                                   ~ 7-8 h
#
# Prerequisites:
#   data/cifar10_features/X_train.npy  (run scripts/extract_cnn_features.py once)
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python3}"
LOG="logs/sequential_all.log"
mkdir -p logs
exec >> "$LOG" 2>&1

# ── Helper: run one job with skip/resume ────────────────────────────────────
run_job() {
    local run_id="$1"
    local ds_args="$2"
    local metric="$3"
    local seed="$4"
    local steps="$5"

    local final_ckpt="logs/${run_id}/checkpoint_final.pt"
    local best_ckpt="logs/${run_id}/checkpoint_best_val.pt"

    if [ -f "$final_ckpt" ]; then
        echo "[SKIP]   ${run_id}"
        return 0
    fi

    local resume_flag=""
    if [ -f "$best_ckpt" ]; then
        resume_flag="+resume=${best_ckpt}"
        echo "[RESUME] ${run_id}"
    else
        echo "[START]  ${run_id}"
    fi

    $PYTHON train.py \
        run_id="${run_id}" \
        ${ds_args} \
        model.metric_type="${metric}" \
        trainer.seed="${seed}" \
        trainer.total_steps="${steps}" \
        writer.mode=disabled \
        ${resume_flag}

    echo "[DONE]   ${run_id}"
    echo "------------------------------------------------------------"
}

echo "============================================================"
echo "Sequential ALL — started $(date)"
echo "============================================================"

# ── Prereq check ──────────────────────────────────────────────────────────────
if [ ! -f "data/cifar10_features/X_train.npy" ]; then
    echo "ERROR: data/cifar10_features/X_train.npy not found!"
    echo "Run: $PYTHON scripts/extract_cnn_features.py first."
    exit 1
fi

# ── Args shortcuts ────────────────────────────────────────────────────────────
MC_ARGS="dataset=mnist_multiclass model.task=multiclass model.K=16"
FM_ARGS="dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"
CF_ARGS="dataset=cifar10_features_multiclass model.task=multiclass model.K=16"

# ── Block 1: Finish partials first (fast wins) ────────────────────────────────
echo "--- Block 1: finish partials ---"
run_job "grid_cifar10_features_off_s2"          "${CF_ARGS}" "off"              2  120000
run_job "grid_mnist_mc_off_s1"                  "${MC_ARGS}" "off"              1   60000
run_job "grid_mnist_mc_diag_s1"                 "${MC_ARGS}" "diag"             1   60000
run_job "grid_fashion_mnist_lambda_u_trotter_s0" "${FM_ARGS}" "lambda_u_trotter" 0 120000

# ── Block 2: MNIST mc — off s2-s4 ─────────────────────────────────────────────
echo "--- Block 2: mnist_mc off s2-s4 ---"
for seed in 2 3 4; do
    run_job "grid_mnist_mc_off_s${seed}" "${MC_ARGS}" "off" "${seed}" 60000
done

# ── Block 3: MNIST mc — diag s2-s4 ───────────────────────────────────────────
echo "--- Block 3: mnist_mc diag s2-s4 ---"
for seed in 2 3 4; do
    run_job "grid_mnist_mc_diag_s${seed}" "${MC_ARGS}" "diag" "${seed}" 60000
done

# ── Block 4: MNIST mc — trotter s0-s4 ────────────────────────────────────────
echo "--- Block 4: mnist_mc trotter s0-s4 ---"
for seed in 0 1 2 3 4; do
    run_job "grid_mnist_mc_lambda_u_trotter_s${seed}" \
            "${MC_ARGS}" "lambda_u_trotter" "${seed}" 60000
done

# ── Block 5: Fashion-MNIST trotter s1-s4 ──────────────────────────────────────
echo "--- Block 5: fashion_mnist trotter s1-s4 ---"
for seed in 1 2 3 4; do
    run_job "grid_fashion_mnist_lambda_u_trotter_s${seed}" \
            "${FM_ARGS}" "lambda_u_trotter" "${seed}" 120000
done

# ── Block 6: CIFAR-10 features — off s3-s4 ────────────────────────────────────
echo "--- Block 6: cifar10_features off s3-s4 ---"
for seed in 3 4; do
    run_job "grid_cifar10_features_off_s${seed}" "${CF_ARGS}" "off" "${seed}" 120000
done

# ── Block 7: CIFAR-10 features — diag s0-s4 ──────────────────────────────────
echo "--- Block 7: cifar10_features diag s0-s4 ---"
for seed in 0 1 2 3 4; do
    run_job "grid_cifar10_features_diag_s${seed}" "${CF_ARGS}" "diag" "${seed}" 120000
done

# ── Block 8: CIFAR-10 features — trotter s0-s4 ────────────────────────────────
echo "--- Block 8: cifar10_features trotter s0-s4 ---"
for seed in 0 1 2 3 4; do
    run_job "grid_cifar10_features_lambda_u_trotter_s${seed}" \
            "${CF_ARGS}" "lambda_u_trotter" "${seed}" 120000
done

echo "============================================================"
echo "Sequential ALL COMPLETE — $(date)"
echo "Run: $PYTHON scripts/collect_grid_results.py to aggregate."
echo "============================================================"
