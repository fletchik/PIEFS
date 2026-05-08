#!/usr/bin/env bash
# run_overnight.sh — fashion_mnist + mnist_mc + cifar10, one job at a time
# Usage: caffeinate -i bash scripts/run_overnight.sh
set -euo pipefail
PYTHON=".venv/bin/python3"
mkdir -p logs
exec >> "logs/overnight.log" 2>&1

run_job() {
    local run_id=$1 ds_args=$2 metric=$3 seed=$4 steps=$5
    local final="logs/$run_id/checkpoint_final.pt"
    local best="logs/$run_id/checkpoint_best_val.pt"
    if [ -f "$final" ]; then echo "[SKIP] $run_id"; return; fi
    local resume_flag=""
    if [ -f "$best" ]; then resume_flag="+resume=$best"; echo "[RESUME] $run_id"
    else echo "[START] $run_id"; fi
    $PYTHON train.py run_id="$run_id" $ds_args \
        model.metric_type="$metric" trainer.seed="$seed" \
        trainer.total_steps="$steps" writer.mode=disabled $resume_flag
    echo "[DONE] $run_id"
}

echo "=== Overnight start: $(date) ==="
echo "Plan: fashion_mnist (8 runs) + mnist_mc (13 runs) + cifar10 (15 runs)"

FM="dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"
for s in 2 3 4;     do run_job "grid_fashion_mnist_diag_s${s}"             "$FM" diag             $s 120000; done
for s in 0 1 2 3 4; do run_job "grid_fashion_mnist_lambda_u_trotter_s${s}" "$FM" lambda_u_trotter $s 120000; done

MC="dataset=mnist_multiclass model.task=multiclass model.K=16"
for s in 1 2 3 4;   do run_job "grid_mnist_mc_off_s${s}"                  "$MC" off              $s 120000; done
for s in 1 2 3 4;   do run_job "grid_mnist_mc_diag_s${s}"                 "$MC" diag             $s 120000; done
for s in 0 1 2 3 4; do run_job "grid_mnist_mc_lambda_u_trotter_s${s}"     "$MC" lambda_u_trotter $s 120000; done

CF="dataset=cifar10_features_multiclass model.task=multiclass model.K=16"
for s in 0 1 2 3 4; do run_job "grid_cifar10_features_off_s${s}"                  "$CF" off             $s 120000; done
for s in 0 1 2 3 4; do run_job "grid_cifar10_features_diag_s${s}"                 "$CF" diag            $s 120000; done
for s in 0 1 2 3 4; do run_job "grid_cifar10_features_lambda_u_trotter_s${s}"     "$CF" lambda_u_trotter $s 120000; done

echo "=== Overnight DONE: $(date) ==="
