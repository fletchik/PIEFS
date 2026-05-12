#!/bin/bash
# ============================================================
# PIEFS Diagnostic Experiments — CIKM 2026
# ============================================================
# Run from project root:
#   bash scripts/run_experiments.sh [group]
#
# Groups:
#   D0  — quick sanity: use_gram_squared comparison on htru2 (~5 min CPU)
#   D1  — metric ablation on two_moon + htru2 (all 7 metric types)
#   D2  — metric ablation on mnist (off, conformal, diag, global_low_rank)
#   D3  — rank ablation (r=1,3,5,9,16) on htru2
#   D4  — use_gram_squared ablation on mnist
#   D5  — three-phase curriculum ablation on htru2
#   all — run all groups sequentially (long!)
# ============================================================

set -e
PYTHON="${PYTHON:-python3}"
SEEDS=(42 1337 0 7 99)
LOG_DIR="logs"
WRITER_MODE="${WRITER_MODE:-disabled}"   # set to 'online' for WandB

run() {
    echo ""
    echo ">>> $@"
    $PYTHON train.py "$@" writer.mode=$WRITER_MODE trainer.seed=${SEED:-42}
}

GROUP="${1:-D0}"

# ============================================================
# D0: use_gram_squared comparison on htru2 — FASTEST diagnostic
# Answers: does the g_k² fix actually improve things?
# Expected: use_gram_squared=true (fixed) > false (original bug)
# ============================================================
if [[ "$GROUP" == "D0" || "$GROUP" == "all" ]]; then
    echo "=== D0: g_k² bug fix comparison (htru2) ==="
    for SEED in 42 1337 0; do
        run run_id=D0_htru2_gk_fixed_s${SEED} \
            dataset=htru2 model.metric_type=diag model.K=6 \
            criterion.dynamic_weighting=true criterion.use_gram_squared=true \
            trainer.total_steps=60000 trainer.seed=${SEED}

        run run_id=D0_htru2_gk_orig_s${SEED} \
            dataset=htru2 model.metric_type=diag model.K=6 \
            criterion.dynamic_weighting=true criterion.use_gram_squared=false \
            trainer.total_steps=60000 trainer.seed=${SEED}
    done
fi

# ============================================================
# D1: Metric ablation — two_moon (fast, 2D visualization)
# All 7 metric types; 3 seeds each
# Expected: conformal > off, global_low_rank > diag
# ============================================================
if [[ "$GROUP" == "D1" || "$GROUP" == "all" ]]; then
    echo "=== D1: Metric ablation — two_moon ==="
    for MTYPE in off conformal diag lambda_u_trotter global_low_rank local_low_rank fisher_diag; do
        for SEED in 42 1337 0; do
            EXTRA_ARGS=""
            if [[ "$MTYPE" == "global_low_rank" || "$MTYPE" == "local_low_rank" ]]; then
                EXTRA_ARGS="model.low_rank_r=1"  # binary: r=1 is optimal
            fi
            run run_id=D1_two_moon_${MTYPE}_s${SEED} \
                dataset=two_moon model.metric_type=$MTYPE model.K=6 \
                criterion.dynamic_weighting=true criterion.use_gram_squared=true \
                trainer.total_steps=60000 trainer.seed=${SEED} $EXTRA_ARGS
        done
    done
fi

# ============================================================
# D2: Metric ablation — mnist_multiclass (main table)
# Key metrics only (off, conformal, diag, global_low_rank); 5 seeds
# ============================================================
if [[ "$GROUP" == "D2" || "$GROUP" == "all" ]]; then
    echo "=== D2: Metric ablation — mnist_multiclass ==="
    for MTYPE in off conformal diag global_low_rank local_low_rank; do
        for SEED in "${SEEDS[@]}"; do
            EXTRA_ARGS=""
            if [[ "$MTYPE" == "global_low_rank" || "$MTYPE" == "local_low_rank" ]]; then
                EXTRA_ARGS="model.low_rank_r=9"  # 10 classes: r=C-1=9
                if [[ "$MTYPE" == "global_low_rank" ]]; then
                    EXTRA_ARGS="$EXTRA_ARGS curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000"
                fi
            fi
            run run_id=D2_mnist_${MTYPE}_s${SEED} \
                dataset=mnist_multiclass model.metric_type=$MTYPE model.K=16 \
                criterion.dynamic_weighting=true criterion.use_gram_squared=true \
                trainer.total_steps=60000 trainer.seed=${SEED} $EXTRA_ARGS
        done
    done
fi

# ============================================================
# D3: Rank ablation — htru2
# Vary r ∈ {1, 2, 4, 8, 16} for global_low_rank
# Tests: is r=C-1=1 optimal for binary? Or does extra rank help?
# ============================================================
if [[ "$GROUP" == "D3" || "$GROUP" == "all" ]]; then
    echo "=== D3: Rank ablation — htru2, global_low_rank ==="
    for R in 1 2 4 8 16; do
        for SEED in 42 1337 0; do
            run run_id=D3_htru2_glr_r${R}_s${SEED} \
                dataset=htru2 model.metric_type=global_low_rank model.K=6 \
                model.low_rank_r=${R} \
                criterion.dynamic_weighting=true criterion.use_gram_squared=true \
                curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000 \
                trainer.total_steps=60000 trainer.seed=${SEED}
        done
    done
fi

# ============================================================
# D4: use_gram_squared ablation — mnist_multiclass
# Full comparison: fixed vs. original on MNIST
# ============================================================
if [[ "$GROUP" == "D4" || "$GROUP" == "all" ]]; then
    echo "=== D4: use_gram_squared ablation — mnist ==="
    for GSQ in true false; do
        for SEED in 42 1337 0; do
            run run_id=D4_mnist_gsq${GSQ}_s${SEED} \
                dataset=mnist_multiclass model.metric_type=off model.K=16 \
                criterion.dynamic_weighting=true criterion.use_gram_squared=${GSQ} \
                trainer.total_steps=60000 trainer.seed=${SEED}
        done
    done
fi

# ============================================================
# D5: Three-phase curriculum ablation — htru2, global_low_rank
# Compare: no curriculum (p1=p2=0) vs. 50/75% schedule
# ============================================================
if [[ "$GROUP" == "D5" || "$GROUP" == "all" ]]; then
    echo "=== D5: Three-phase curriculum — htru2, global_low_rank ==="
    for SEED in 42 1337 0; do
        # No curriculum
        run run_id=D5_htru2_glr_nocurr_s${SEED} \
            dataset=htru2 model.metric_type=global_low_rank model.K=6 \
            model.low_rank_r=1 \
            criterion.dynamic_weighting=true criterion.use_gram_squared=true \
            curriculum.phase1_end_step=0 curriculum.phase2_end_step=0 \
            trainer.total_steps=60000 trainer.seed=${SEED}

        # With curriculum (50%/75%)
        run run_id=D5_htru2_glr_curr_s${SEED} \
            dataset=htru2 model.metric_type=global_low_rank model.K=6 \
            model.low_rank_r=1 \
            criterion.dynamic_weighting=true criterion.use_gram_squared=true \
            curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000 \
            trainer.total_steps=60000 trainer.seed=${SEED}
    done
fi

echo ""
echo "=== Experiments complete. Results in $LOG_DIR/ ==="
echo "    Collect results: python scripts/collect_grid_results.py --log_dir $LOG_DIR"
