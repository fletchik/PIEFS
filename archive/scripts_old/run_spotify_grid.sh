#!/usr/bin/env bash
# =============================================================================
# run_spotify_grid.sh — EFDO grid on Spotify Songs dataset
#
# PREREQUISITE (run once on the cluster):
#   pip install kaggle
#   kaggle datasets download -d mrmorj/dataset-of-songs-in-spotify -p data/spotify
#   unzip data/spotify/dataset-of-songs-in-spotify.zip -d data/spotify
#
# Grid: 2 tasks × 3 metrics × 5 seeds = 30 runs
#
# Tasks:
#   spotify_mc     — 10-class music genre classification (K=16, 120k steps)
#   spotify_bin    — binary mode classification         (K=6,  60k steps)
#
# Metrics: off | diag | lambda_u_trotter
# Seeds:   0 1 2 3 4
#
# Usage:
#   # Full grid (sequential):
#   bash scripts/run_spotify_grid.sh
#
#   # Multiclass only:
#   TASKS="spotify_mc" bash scripts/run_spotify_grid.sh
#
#   # Parallel (background):
#   PARALLEL=1 bash scripts/run_spotify_grid.sh
# =============================================================================

set -euo pipefail

TASKS="${TASKS:-spotify_mc spotify_bin}"
METRICS="${METRICS:-off diag lambda_u_trotter}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PARALLEL="${PARALLEL:-0}"
PYTHON="${PYTHON:-.venv/bin/python3}"
WANDB_MODE="${WANDB_MODE:-disabled}"

task_config() {
    local task=$1
    case "$task" in
        spotify_mc)
            echo "dataset=spotify_multiclass model.task=multiclass model.K=16 trainer.total_steps=120000";;
        spotify_bin)
            echo "dataset=spotify_binary model.task=binary model.K=6 trainer.total_steps=60000";;
        *)
            echo "dataset=${task}";;
    esac
}

total=0
for t in $TASKS; do for m in $METRICS; do for s in $SEEDS; do total=$((total+1)); done; done; done

echo "============================================================"
echo "EFDO Spotify Grid — $total runs"
echo "  tasks   : $TASKS"
echo "  metrics : $METRICS"
echo "  seeds   : $SEEDS"
echo "============================================================"

pids=()
started=0

for task in $TASKS; do
    for metric in $METRICS; do
        for seed in $SEEDS; do
            started=$((started+1))
            run_id="grid_${task}_${metric}_s${seed}"
            task_args=$(task_config "$task")

            echo "[$started/$total] $run_id"
            cmd="$PYTHON train.py \
                run_id=$run_id \
                $task_args \
                model.metric_type=$metric \
                trainer.seed=$seed \
                writer.mode=$WANDB_MODE"

            if [[ "$PARALLEL" == "1" ]]; then
                mkdir -p logs
                eval "$cmd" &>"logs/${run_id}.log" &
                pids+=($!)
                echo "  → spawned pid=$!"
            else
                eval "$cmd"
                echo "  → done"
            fi
        done
    done
done

if [[ "$PARALLEL" == "1" ]]; then
    echo "Waiting for ${#pids[@]} jobs ..."
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed+1)); done
    echo "Done. Failures: $failed"
fi

echo "Grid complete. Run scripts/collect_grid_results.py to aggregate."
