#!/usr/bin/env bash
# =============================================================================
# Extra overnight experiments — fills gaps from run_overnight.sh
#
# Covers:
#   - Second seed (s123) for key experiments
#   - Unsupervised HTRU2 (w_task=0, paper Table 1 protocol)
#   - lambda_u_sparse on MNIST multiclass (784D stress test for A(x))
#   - Augmentation on HTRU2
#   - K ablation on HTRU2 (K=3 vs K=6 vs K=10)
#
# Estimated: ~3-4 hours additional
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found."
    exit 1
fi

LOG_FILE="logs/overnight_extra_$(date +%Y%m%d_%H%M%S).log"
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

log "============================================================"
log "EXTRA OVERNIGHT EXPERIMENTS"
log "Started: $(date)"
log "============================================================"

# =============================================================================
# GROUP E1: HTRU2 second seed (s123) — statistical significance
# =============================================================================
log ""
log "===== E1: HTRU2 seed=123 ====="

run_train "night_htru2_off_s123" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123

run_train "night_htru2_diag_s123" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123

run_train "night_htru2_sparse_s123" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123

# =============================================================================
# GROUP E2: HTRU2 unsupervised (w_task=0) — paper Table 1 protocol
# Pure spectral eigenfeatures, no classification signal during training
# =============================================================================
log ""
log "===== E2: HTRU2 unsupervised (w_task=0) ====="

run_train "night_htru2_spectral_off_s42" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    criterion.w_task=0.0 \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "night_htru2_spectral_diag_s42" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    criterion.w_task=0.0 \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "night_htru2_spectral_sparse_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    criterion.w_task=0.0 \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# =============================================================================
# GROUP E3: HTRU2 augmentation
# =============================================================================
log ""
log "===== E3: HTRU2 with augmentation ====="

run_train "night_htru2_off_s42_aug" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    augmentation.noise_std=0.03 augmentation.wide_normal_fraction=0.1 \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# =============================================================================
# GROUP E4: HTRU2 K ablation (K=3 vs K=10)
# Checks if K=6 is optimal or we're over/under-fitting
# =============================================================================
log ""
log "===== E4: HTRU2 K ablation ====="

# K=3: 40k per function
run_train "night_htru2_off_s42_K3" \
    dataset=htru2 model.K=3 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# K=10: 12k per function
run_train "night_htru2_off_s42_K10" \
    dataset=htru2 model.K=10 model.metric_type='off' \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# =============================================================================
# GROUP E5: MNIST multiclass second seed + sparse
# =============================================================================
log ""
log "===== E5: MNIST multiclass seed=123 + sparse ====="

run_train "night_mnist_mc_off_s123" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=123

# sparse on MNIST — stress test for A(x) on 784D
# NOTE: this will be slow (784x784 metric matrix); but important to test
run_train "night_mnist_mc_sparse_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_sparse \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# =============================================================================
log ""
log "============================================================"
log "ALL EXTRA EXPERIMENTS COMPLETE"
log "Finished: $(date)"
log "Total extra experiments: 11"
log "============================================================"
