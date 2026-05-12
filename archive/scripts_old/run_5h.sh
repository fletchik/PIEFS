#!/usr/bin/env bash
# =============================================================================
# 5-HOUR OPTIMISED RUN  (~4.2–4.7h on CPU)
#
# Priority order (highest scientific value first):
#
#   P2.  Static weighting ablation — HTRU2 (4 runs × ~8 min = 0.5h)
#          dynamic_weighting=false vs overnight default (=true)
#          Ablation of paper eq. 9–10 on small dataset.
#
#   P5.  Full pipeline: PINN + dynamic_weighting + GL (5 runs × ~14 min = 1.2h)
#          Paper's complete method — all three components together.
#          HTRU2 ×2, MNIST mc ×1, Circles ×1, TwoMoon ×1.
#
#   P2b. Static weighting ablation — MNIST mc (3 runs × ~50 min = 2.5h)
#          Same ablation on hard 10-class dataset.
#
#   TOTAL: ~4.2h  (safe within 5h budget)
#
# Skipped (not enough time):
#   P3  — big hidden dims [256,256,256] MNIST mc (~3h)
#   P4  — 480k steps MNIST mc (~5h)
#   These can be run in a separate overnight session.
#
# Pre-condition: caffeinate -s already running (or wrap with it).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate venv first."
    exit 1
fi

LOG_FILE="logs/run_5h_$(date +%Y%m%d_%H%M%S).log"
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

# Shared args
GL_ARGS="pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000"

log "============================================================"
log "5-HOUR OPTIMISED EXPERIMENTS"
log "Started: $(date)"
log "Order: P2 (ablation HTRU2) → P5 (full pipeline) → P2b (ablation MNIST mc)"
log "============================================================"

# =============================================================================
# GROUP P2: Static weighting ablation — HTRU2
#   Compares dynamic_weighting=false vs true (overnight default).
#   Measures contribution of adaptive hierarchy (paper eq. 9–10).
#   ~8 min × 4 = 32 min
# =============================================================================
log ""
log "===== P2: STATIC weighting ablation — HTRU2 all metrics ====="

run_train "pinn_htru2_off_static_s42" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    criterion.dynamic_weighting=false \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_diag_static_s42" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    criterion.dynamic_weighting=false \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_sparse_static_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    criterion.dynamic_weighting=false \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_pinn_static_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=false \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

# =============================================================================
# GROUP P5: Full pipeline — PINN + dynamic_weighting + GL
#   Paper's complete method. All three improvements combined.
#   Datasets: HTRU2 ×2 seeds, MNIST mc ×1, Circles ×1, TwoMoon ×1.
#   ~70 min total.
# =============================================================================
log ""
log "===== P5: Full pipeline (PINN + dynamic + GL) — all datasets ====="

run_train "pinn_htru2_full_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 $GL_ARGS

run_train "pinn_htru2_full_s123" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123 $GL_ARGS

run_train "pinn_mnist_mc_full_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42 $GL_ARGS

run_train "pinn_circles_full_s42" \
    dataset=circles model.K=4 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42 $GL_ARGS

run_train "pinn_twomoon_full_s42" \
    dataset=two_moon model.K=4 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=60000 trainer.log_step=15000 trainer.save_period=30000 \
    trainer.seed=42

# =============================================================================
# GROUP P2b: Static weighting ablation — MNIST multiclass
#   Same comparison on harder 10-class task.
#   ~50 min × 3 = 150 min
# =============================================================================
log ""
log "===== P2b: STATIC weighting ablation — MNIST multiclass ====="

run_train "pinn_mnist_mc_off_static_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    criterion.dynamic_weighting=false \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_diag_static_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    criterion.dynamic_weighting=false \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_pinn_static_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=false \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# =============================================================================
log ""
log "============================================================"
log "5-HOUR RUN COMPLETE"
log "Finished: $(date)"
log "Total experiments: 12"
log "============================================================"
log ""
log "Quick results summary:"
log "  python3 - << 'EOF'"
log "  import json, glob, os"
log "  runs = sorted(glob.glob('logs/pinn_*_table1.json'))"
log "  for f in runs:"
log "      d = json.load(open(f))"
log "      r = d['results']"
log "      ef = r.get('eigen_100%', {}).get('lr_accuracy', '?')"
log "      raw = r.get('raw_100%', {}).get('lr_accuracy', '?')"
log "      name = os.path.basename(f).replace('_table1.json','')"
log "      if isinstance(ef, float) and isinstance(raw, float):"
log "          print(f'{name:50s}  EF={ef:.4f}  Raw={raw:.4f}  D={ef-raw:+.4f}')"
log "  EOF"
