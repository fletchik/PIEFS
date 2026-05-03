#!/usr/bin/env bash
# =============================================================================
# Overnight experiment suite — run with caffeinate to prevent sleep
#
# Usage (from EFDO/ directory):
#   bash scripts/run_overnight.sh
#
# This script runs all experiments SEQUENTIALLY (safer than parallel for
# memory and reproducibility). Total estimated time: ~3.5-4 hours.
#
# Each experiment writes to logs/<run_id>/ and produces:
#   - checkpoint_final.pt (model weights)
#   - train.log (full training log)
#   - After training: eval_eigenfeatures.py results (Table 1 protocol)
#
# Design rationale:
#   - Simple datasets (two_moon, circles): K=4, 60k steps — enough
#   - HTRU2 (8D, Table 1 dataset): K=6, 120k steps — main A(x) test
#   - MNIST multiclass (784D, 10-class): K=10, 240k steps — stress test
#   - GL pretraining on/off pairs — direct comparison of convergence
#   - Augmentation ablation — test overfitting prevention
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Python from project venv
PYTHON=".venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate venv first."
    exit 1
fi

LOG_FILE="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

run_train() {
    local run_id="$1"
    shift
    log ">>> START: $run_id"
    local start_time=$(date +%s)

    $PYTHON train.py run_id="$run_id" writer.mode=disabled "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    if [ $exit_code -eq 0 ]; then
        log "<<< DONE: $run_id (${elapsed}s)"
        # Run Table 1 evaluation if checkpoint exists
        local ckpt="logs/$run_id/checkpoint_final.pt"
        if [ -f "$ckpt" ]; then
            log "    Evaluating eigenfeatures (Table 1 protocol)..."
            $PYTHON scripts/eval_eigenfeatures.py --checkpoint "$ckpt" --device cpu \
                --output "logs/${run_id}_table1.json" 2>&1 | tee -a "$LOG_FILE"
        fi
    else
        log "!!! FAILED: $run_id (exit code $exit_code, ${elapsed}s)"
    fi
    echo "" >> "$LOG_FILE"
}

# =============================================================================
log "============================================================"
log "OVERNIGHT EXPERIMENT SUITE"
log "Started: $(date)"
log "Working directory: $PROJECT_DIR"
log "Python: $($PYTHON --version)"
log "============================================================"

# =============================================================================
# GROUP 1: HTRU2 (8D, binary) — KEY Table 1 dataset
# K=6, 120k steps (20k per function)
# 3 metrics × with/without GL pretraining = 6 runs
# Estimated: ~40 min total
# =============================================================================
log ""
log "===== GROUP 1: HTRU2 (8D, K=6, 120k steps) ====="

# 1a. Without GL pretraining
run_train "night_htru2_off_s42" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "night_htru2_diag_s42" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "night_htru2_sparse_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# 1b. With GL pretraining (same configs, so we can compare directly)
run_train "night_htru2_off_s42_gl" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

run_train "night_htru2_diag_s42_gl" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

run_train "night_htru2_sparse_s42_gl" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

# =============================================================================
# GROUP 2: MNIST multiclass (784D, 10-class) — MAIN experiment
# K=10, 240k steps (24k per function)
# off and diag × with/without GL pretraining = 4 runs
# + 1 unsupervised run (w_task=0) for pure spectral features
# + 1 with augmentation
# Estimated: ~2 hours total
# =============================================================================
log ""
log "===== GROUP 2: MNIST multiclass (784D, K=10, 240k steps) ====="

# 2a. Supervised: off metric, no GL
run_train "night_mnist_mc_off_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# 2b. Supervised: diag metric, no GL
run_train "night_mnist_mc_diag_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# 2c. Supervised: off metric, WITH GL pretraining
run_train "night_mnist_mc_off_s42_gl" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

# 2d. Supervised: diag metric, WITH GL pretraining
run_train "night_mnist_mc_diag_s42_gl" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

# 2e. UNSUPERVISED spectral only (w_task=0): pure eigenfeatures, no classification signal
#     This is the closest to paper's protocol for Table 1 comparison
run_train "night_mnist_mc_spectral_off_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    criterion.w_task=0.0 \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# 2f. Supervised + augmentation: test if noise prevents eigenfunction overfitting
run_train "night_mnist_mc_off_s42_aug" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    augmentation.noise_std=0.03 augmentation.wide_normal_fraction=0.1 \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# =============================================================================
# GROUP 3: MNIST binary 0vs1 (784D, 2-class) — comparison with multiclass
# K=6, 120k steps (sufficient for binary)
# Estimated: ~25 min total
# =============================================================================
log ""
log "===== GROUP 3: MNIST binary (784D, K=6, 120k steps) ====="

# 3a. off, no GL
run_train "night_mnist_bin_off_s42" \
    dataset=mnist_binary model.K=6 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# 3b. off, WITH GL
run_train "night_mnist_bin_off_s42_gl" \
    dataset=mnist_binary model.K=6 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

# 3c. diag, no GL
run_train "night_mnist_bin_diag_s42" \
    dataset=mnist_binary model.K=6 model.metric_type=diag \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# 3d. diag, WITH GL
run_train "night_mnist_bin_diag_s42_gl" \
    dataset=mnist_binary model.K=6 model.metric_type=diag \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000

# =============================================================================
# GROUP 4: Two-moon augmentation ablation
# K=4 (sufficient for 2D binary), 60k steps
# Compare: baseline vs noise vs noise+wide_normal
# Estimated: ~10 min total
# =============================================================================
log ""
log "===== GROUP 4: Two-moon augmentation ablation (2D, K=4, 60k steps) ====="

# 4a. Baseline (no augmentation)
run_train "night_twomoon_off_s42_baseline" \
    dataset=two_moon model.K=4 model.metric_type='off' \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42

# 4b. Noise only
run_train "night_twomoon_off_s42_noise" \
    dataset=two_moon model.K=4 model.metric_type='off' \
    augmentation.noise_std=0.05 \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42

# 4c. Noise + wide normal
run_train "night_twomoon_off_s42_aug_full" \
    dataset=two_moon model.K=4 model.metric_type='off' \
    augmentation.noise_std=0.05 augmentation.wide_normal_fraction=0.15 \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42

# =============================================================================
# GROUP 5: Circles augmentation + GL pretraining
# K=4 (2D), 60k steps
# Verifies if GL and augmentation help with the ordering problem
# Estimated: ~10 min total
# =============================================================================
log ""
log "===== GROUP 5: Circles with GL pretraining (2D, K=4, 60k steps) ====="

run_train "night_circles_off_s42_gl" \
    dataset=circles model.K=4 model.metric_type='off' \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=1000 pretrain.distill_steps=1000

run_train "night_circles_off_s42_aug_gl" \
    dataset=circles model.K=4 model.metric_type='off' \
    augmentation.noise_std=0.03 augmentation.wide_normal_fraction=0.1 \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42 \
    pretrain.graph_laplacian=true pretrain.n_points=1000 pretrain.distill_steps=1000

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "ALL EXPERIMENTS COMPLETE"
log "Finished: $(date)"
log "============================================================"
log ""
log "Total experiments run: 20"
log ""
log "Results summary — check:"
log "  logs/night_*_table1.json   — Table 1 evaluation results"
log "  logs/night_*/train.log     — training logs"
log "  logs/night_*_results.md    — experiment summaries"
log ""
log "To compare results:"
log "  for f in logs/night_*_table1.json; do echo \"=== \$f ===\"; cat \$f | python3 -m json.tool | grep -E 'rf_accuracy|lr_accuracy'; done"
