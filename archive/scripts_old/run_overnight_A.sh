#!/usr/bin/env bash
# =============================================================================
# run_overnight_A.sh — fashion_mnist trotter (s0-s4) + mnist_mc remaining
#
# Run with:  caffeinate -i bash scripts/run_overnight_A.sh
#
# Estimated wall time: ~14h (sequential on CPU)
#   fashion_mnist trotter s0-s4  ≈5.3h  (vectorised Trotter ~32ms/step)
#   mnist_mc off s1-s4           ≈1.5h  (s1 resumes from checkpoint)
#   mnist_mc diag s1-s4          ≈2.5h  (s1 resumes from checkpoint)
#   mnist_mc trotter s0-s4       ≈4.0h  (vectorised Trotter ~50ms/step)
#
# Skip logic: if checkpoint_final.pt exists → skip
# Resume logic: if checkpoint_best_val.pt exists → +resume flag
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python3}"
mkdir -p logs
exec >> "logs/overnight_A.log" 2>&1

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
echo "EFDO Overnight A — started $(date)"
echo "fashion_mnist trotter (5) + mnist_mc remaining (13 runs)"
echo "============================================================"

# ── Block 1: fashion_mnist trotter s0-s4 (120k steps) ────────────────────────
FM_ARGS="dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"
for seed in 0 1 2 3 4; do
    run_job "grid_fashion_mnist_lambda_u_trotter_s${seed}" \
            "${FM_ARGS}" "lambda_u_trotter" "${seed}" 120000
done

# ── Block 2: mnist_mc off s1-s4 (60k steps; s1 resumes from ~75%) ────────────
MC_ARGS="dataset=mnist_multiclass model.task=multiclass model.K=16"
for seed in 1 2 3 4; do
    run_job "grid_mnist_mc_off_s${seed}" \
            "${MC_ARGS}" "off" "${seed}" 60000
done

# ── Block 3: mnist_mc diag s1-s4 (60k steps; s1 resumes from ~25%) ───────────
for seed in 1 2 3 4; do
    run_job "grid_mnist_mc_diag_s${seed}" \
            "${MC_ARGS}" "diag" "${seed}" 60000
done

# ── Block 4: mnist_mc trotter s0-s4 (60k steps, vectorised Trotter) ──────────
for seed in 0 1 2 3 4; do
    run_job "grid_mnist_mc_lambda_u_trotter_s${seed}" \
            "${MC_ARGS}" "lambda_u_trotter" "${seed}" 60000
done

echo "============================================================"
echo "Overnight A COMPLETE — $(date)"
echo "Run: .venv/bin/python3 scripts/collect_grid_results.py to aggregate."
echo "============================================================"
