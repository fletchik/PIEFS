#!/usr/bin/env bash
# =============================================================================
# run_priority.sh  — MNIST mc + CIFAR-10 features first, fashion_mnist last.
#
# Run with:  caffeinate -i bash scripts/run_priority.sh
#
# Strictly ONE process at a time to avoid OOM on 4 GB machine.
# Skip/resume via checkpoint_final.pt / checkpoint_best_val.pt.
#
# Estimated wall time (one process, ~10 ms/step):
#   mnist_mc   off   s1-s4  resume/fresh   ~  40 min
#   mnist_mc   diag  s1-s4  resume/fresh   ~  40 min
#   mnist_mc   trotter s0-s4               ~  50 min
#   cifar10    off   s2-s4                 ~  45 min
#   cifar10    diag  s0-s4                 ~ 110 min
#   cifar10    trotter s0-s4               ~ 110 min
#   fashion_mnist trotter s0-s4            ~ 100 min
#   ──────────────────────────────────────────────────
#   Total                                  ~  ~8 h
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python3}"
LOG="logs/priority_run.log"
mkdir -p logs
exec >> "$LOG" 2>&1

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
echo "Priority run — started $(date)"
echo "ORDER: mnist_mc (all) → cifar10_features (all) → fashion_mnist trotter"
echo "============================================================"

MC_ARGS="dataset=mnist_multiclass model.task=multiclass model.K=16"
FM_ARGS="dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"
CF_ARGS="dataset=cifar10_features_multiclass model.task=multiclass model.K=16"

# ── 1. MNIST mc — off (s0 done; resume s1, fresh s2-s4) ──────────────────────
echo "=== MNIST mc: off ==="
for seed in 0 1 2 3 4; do
    run_job "grid_mnist_mc_off_s${seed}" "${MC_ARGS}" "off" "${seed}" 60000
done

# ── 2. MNIST mc — diag (s0 done; resume s1, fresh s2-s4) ─────────────────────
echo "=== MNIST mc: diag ==="
for seed in 0 1 2 3 4; do
    run_job "grid_mnist_mc_diag_s${seed}" "${MC_ARGS}" "diag" "${seed}" 60000
done

# ── 3. MNIST mc — trotter s0-s4 ──────────────────────────────────────────────
echo "=== MNIST mc: trotter ==="
for seed in 0 1 2 3 4; do
    run_job "grid_mnist_mc_lambda_u_trotter_s${seed}" \
            "${MC_ARGS}" "lambda_u_trotter" "${seed}" 60000
done

# ── 4. CIFAR-10 features — off (s0,s1 done; resume s2, fresh s3-s4) ──────────
if [ ! -f "data/cifar10_features/X_train.npy" ]; then
    echo "ERROR: data/cifar10_features/X_train.npy not found — skipping cifar10 blocks"
else
    echo "=== CIFAR-10 features: off ==="
    for seed in 0 1 2 3 4; do
        run_job "grid_cifar10_features_off_s${seed}" "${CF_ARGS}" "off" "${seed}" 120000
    done

    # ── 5. CIFAR-10 features — diag s0-s4 ────────────────────────────────────
    echo "=== CIFAR-10 features: diag ==="
    for seed in 0 1 2 3 4; do
        run_job "grid_cifar10_features_diag_s${seed}" "${CF_ARGS}" "diag" "${seed}" 120000
    done

    # ── 6. CIFAR-10 features — trotter s0-s4 ─────────────────────────────────
    echo "=== CIFAR-10 features: trotter ==="
    for seed in 0 1 2 3 4; do
        run_job "grid_cifar10_features_lambda_u_trotter_s${seed}" \
                "${CF_ARGS}" "lambda_u_trotter" "${seed}" 120000
    done
fi

# ── 7. Fashion-MNIST — trotter s0-s4 (last: heaviest runs) ───────────────────
echo "=== Fashion-MNIST: trotter ==="
for seed in 0 1 2 3 4; do
    run_job "grid_fashion_mnist_lambda_u_trotter_s${seed}" \
            "${FM_ARGS}" "lambda_u_trotter" "${seed}" 120000
done

echo "============================================================"
echo "Priority run COMPLETE — $(date)"
echo "Run: $PYTHON scripts/collect_grid_results.py to aggregate."
echo "============================================================"
