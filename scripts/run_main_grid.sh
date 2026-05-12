#!/usr/bin/env bash
# =============================================================================
# run_main_grid.sh  —  Main experiment grid for PIEFS paper
#
# Grid:  4 datasets × 3 metrics × 5 seeds  =  60 runs
#
# Datasets:   two_moon  |  circles  |  htru2  |  mnist_mc
# Metrics:    off       |  diag     |  lambda_u_trotter
# Seeds:      0  1  2  3  4
#
# Each run trains for 60 000 steps (matches all prior experiments).
# Total time estimate (CPU): ~10 min/run × 60 = ~10 h
#                  (Kaggle GPU P100): ~2 min/run × 60 = ~2 h
#
# Usage
# -----
#   # Full grid (sequential):
#   bash scripts/run_main_grid.sh
#
#   # Specific subset:
#   DATASETS="two_moon circles" METRICS="off diag" SEEDS="0 1" bash scripts/run_main_grid.sh
#
#   # Parallel (background, use with care on single machine):
#   PARALLEL=1 bash scripts/run_main_grid.sh
#
# Output
# ------
#   logs/grid_<dataset>_<metric>_s<seed>/  — one dir per run
#   logs/grid_<dataset>_<metric>_s<seed>/metrics.jsonl
#   logs/grid_<dataset>_<metric>_s<seed>/checkpoint_final.pt
# =============================================================================

set -euo pipefail

# ── Configurable variables (override via env) ────────────────────────────────
DATASETS="${DATASETS:-two_moon circles htru2 mnist_mc}"
METRICS="${METRICS:-off diag lambda_u_trotter}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PARALLEL="${PARALLEL:-0}"          # set to 1 for background jobs
PYTHON="${PYTHON:-.venv/bin/python3}"
STEPS="${STEPS:-60000}"            # total training steps per run

# ── Dataset-specific config overrides ────────────────────────────────────────
# mnist_mc needs multiclass head and more K; htru2 uses binary with K=6.
ds_config() {
    local ds=$1
    case "$ds" in
        two_moon)       echo "dataset=two_moon  model.task=binary  model.K=6";;
        circles)        echo "dataset=circles   model.task=binary  model.K=6";;
        htru2)          echo "dataset=htru2     model.task=binary  model.K=6";;
        mnist_mc)       echo "dataset=mnist_multiclass  model.task=multiclass  model.K=16  trainer.total_steps=120000";;
        fashion_mnist)      echo "dataset=fashion_mnist_multiclass  model.task=multiclass  model.K=16  trainer.total_steps=120000";;
        cifar10_features)   echo "dataset=cifar10_features_multiclass  model.task=multiclass  model.K=16  trainer.total_steps=120000";;
        *)                  echo "dataset=${ds}";;
    esac
}

# ── Counters ──────────────────────────────────────────────────────────────────
total=0
started=0
pids=()

for ds in $DATASETS; do
    for metric in $METRICS; do
        for seed in $SEEDS; do
            total=$((total + 1))
        done
    done
done

echo "============================================================"
echo "PIEFS Main Grid  —  $total runs total"
echo "  datasets : $DATASETS"
echo "  metrics  : $METRICS"
echo "  seeds    : $SEEDS"
echo "  steps    : $STEPS"
echo "  parallel : $PARALLEL"
echo "============================================================"
echo ""

# ── Main loop ────────────────────────────────────────────────────────────────
for ds in $DATASETS; do
    for metric in $METRICS; do
        for seed in $SEEDS; do
            started=$((started + 1))
            run_id="grid_${ds}_${metric}_s${seed}"
            ds_args=$(ds_config "$ds")

            echo "[$started/$total] run_id=$run_id"
            echo "  → $PYTHON train.py run_id=$run_id $ds_args"
            echo "    model.metric_type=$metric  trainer.seed=$seed"

            cmd="$PYTHON train.py \
                run_id=$run_id \
                $ds_args \
                model.metric_type=$metric \
                trainer.seed=$seed \
                trainer.total_steps=$STEPS \
                writer.mode=${WANDB_MODE:-online}"

            if [[ "$PARALLEL" == "1" ]]; then
                eval "$cmd" &>"logs/${run_id}.log" &
                pids+=($!)
                echo "  → spawned pid=$!"
            else
                eval "$cmd"
                echo "  → done"
                echo ""
            fi
        done
    done
done

# ── Wait for parallel jobs ────────────────────────────────────────────────────
if [[ "$PARALLEL" == "1" ]]; then
    echo ""
    echo "Waiting for ${#pids[@]} background jobs..."
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$((failed + 1))
    done
    echo "All jobs finished. Failures: $failed"
fi

echo ""
echo "============================================================"
echo "Grid complete.  Results in logs/grid_*/"
echo "Run scripts/collect_grid_results.py to aggregate metrics."
echo "============================================================"
