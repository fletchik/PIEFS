#!/usr/bin/env bash
# =============================================================================
# PINN + Dynamic weighting experiments  (~10-14 hours total on CPU)
#
# Covers experiments NEVER run before:
#   P1. lambda_u_pinn on all datasets (first correct run after architecture fix)
#         PINN pretrain converges: MSE 0.50 → <0.001 in 5000 steps (~20s overhead)
#   P2. dynamic_weighting=True ablation (paper eq. 9-10, never tested before)
#   P3. Larger hidden dims for MNIST mc (default [64,64] → [256,256,256])
#   P4. More steps for MNIST mc (240k → 480k, 48k/function)
#   P5. PINN + dynamic weighting combined ("full paper pipeline")
#   P6. PINN + GL pretraining
#
# Estimated wall time per run type (CPU, no GPU):
#   HTRU2 120k steps:       ~8 min   (+20s PINN pretrain)
#   MNIST binary 120k:      ~8 min
#   MNIST mc 240k:          ~22 min  (+20s PINN pretrain)
#   MNIST mc 480k:          ~44 min
#   Circles/twomoon 60k:    ~2 min
#
# Total: ~22 runs × avg 15min ≈ 5.5h (short runs) + 6 slow runs × 44min ≈ 4.4h
# Grand total: ~10h
#
# Pre-condition: start with caffeinate -s before calling this script.
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

LOG_FILE="logs/run_pinn_dynamic_$(date +%Y%m%d_%H%M%S).log"
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

# Shared GL args
GL_ARGS="pretrain.graph_laplacian=true pretrain.n_points=2000 pretrain.distill_steps=2000"

log "============================================================"
log "PINN + DYNAMIC WEIGHTING EXPERIMENTS"
log "Started: $(date)"
log "============================================================"

# =============================================================================
# GROUP P1: lambda_u_pinn — first correct run after 2-bug architecture fix
#   Bug 1 fixed: PINN input (x,v0) → (omega_vec,v0), pretrain on random omega
#   Bug 2 fixed: Ag_pinn stored separately from A to avoid double-multiply
# =============================================================================
log ""
log "===== P1: lambda_u_pinn — HTRU2 (d=8, K=6) ====="
# ~8 min per run × 4 = 32 min

run_train "pinn_htru2_pinn_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_pinn_s123" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123

run_train "pinn_htru2_pinn_s42_gl" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 $GL_ARGS

run_train "pinn_htru2_pinn_s123_gl" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123 $GL_ARGS

log ""
log "===== P1b: lambda_u_pinn — MNIST binary (d=784, K=6) ====="
# ~8 min per run × 4 = 32 min

run_train "pinn_mnist_bin_pinn_s42" \
    dataset=mnist_binary model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_mnist_bin_pinn_s123" \
    dataset=mnist_binary model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123

run_train "pinn_mnist_bin_pinn_s42_gl" \
    dataset=mnist_binary model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42 $GL_ARGS

run_train "pinn_mnist_bin_pinn_s123_gl" \
    dataset=mnist_binary model.K=6 model.metric_type=lambda_u_pinn \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=123 $GL_ARGS

log ""
log "===== P1c: lambda_u_pinn — MNIST multiclass (d=784, K=10) ====="
# ~22 min per run × 3 = 66 min
# NOTE: LambdaUSparse impractical on d=784 (784×784 matrix).
# LambdaUPinn uses apply_to() → one PINN call → O(d), feasible.

run_train "pinn_mnist_mc_pinn_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_pinn_s123" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=123

run_train "pinn_mnist_mc_pinn_s42_gl" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42 $GL_ARGS

# =============================================================================
# GROUP P2: Dynamic weighting ablation (paper eq. 9-10)
#   dynamic_weighting=True: w_task *= exp(-gram/t_orth)
#                            w_mde  *= exp(-max(gram/t_orth, task/t_class))
#   Enforces the training hierarchy: L_orth first → L_class → L_mde (Dirichlet)
#   NEVER tested before — this is the paper's main algorithmic contribution!
# =============================================================================
log ""
log "===== P2: Dynamic weighting — HTRU2 all metrics ====="
# ~8 min per run × 4 = 32 min

run_train "pinn_htru2_off_dyn_s42" \
    dataset=htru2 model.K=6 model.metric_type='off' \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_diag_dyn_s42" \
    dataset=htru2 model.K=6 model.metric_type=diag \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_sparse_dyn_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_sparse \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

run_train "pinn_htru2_pinn_dyn_s42" \
    dataset=htru2 model.K=6 model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=120000 trainer.log_step=30000 trainer.save_period=60000 \
    trainer.seed=42

log ""
log "===== P2b: Dynamic weighting — MNIST mc all metrics ====="
# ~22 min per run × 4 = 88 min

run_train "pinn_mnist_mc_off_dyn_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    criterion.dynamic_weighting=true \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_diag_dyn_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    criterion.dynamic_weighting=true \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_pinn_dyn_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    criterion.dynamic_weighting=true \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# =============================================================================
# GROUP P3: Larger hidden dims for MNIST mc
#   Current: hidden_dims=[64,64]  → EF-LR=94.0%, grammar=0.12
#   New:     hidden_dims=[256,256,256] → 4× more parameters per eigenfunction
#   Hypothesis: bigger backbone → more expressive eigenfunctions → beat raw
# =============================================================================
log ""
log "===== P3: Larger hidden dims [256,256,256] — MNIST mc ====="
# ~30 min per run × 3 = 90 min

run_train "pinn_mnist_mc_big_off_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    'model.hidden_dims=[256,256,256]' \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_big_diag_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    'model.hidden_dims=[256,256,256]' \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

run_train "pinn_mnist_mc_big_off_dyn_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    'model.hidden_dims=[256,256,256]' criterion.dynamic_weighting=true \
    trainer.total_steps=240000 trainer.log_step=60000 trainer.save_period=120000 \
    trainer.seed=42

# =============================================================================
# GROUP P4: More steps for MNIST mc (480k = 48k/function, 2× current)
#   Current 240k: gram_error ~0.12-0.13, still improving at step 240k.
#   With 480k each function gets more time to converge & reduce gram error.
# =============================================================================
log ""
log "===== P4: More steps (480k) — MNIST mc ====="
# ~44 min per run × 3 = 132 min

run_train "pinn_mnist_mc_off_480k_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type='off' \
    trainer.total_steps=480000 trainer.log_step=120000 trainer.save_period=240000 \
    trainer.seed=42

run_train "pinn_mnist_mc_diag_480k_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=diag \
    trainer.total_steps=480000 trainer.log_step=120000 trainer.save_period=240000 \
    trainer.seed=42

run_train "pinn_mnist_mc_pinn_480k_s42" \
    dataset=mnist_multiclass model.K=10 model.task=multiclass model.metric_type=lambda_u_pinn \
    trainer.total_steps=480000 trainer.log_step=120000 trainer.save_period=240000 \
    trainer.seed=42

# =============================================================================
# GROUP P5: Full pipeline (PINN + dynamic weighting + GL)
#   This is the paper's intended complete method: all three improvements together.
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
log ""
log "============================================================"
log "ALL PINN+DYNAMIC EXPERIMENTS COMPLETE"
log "Finished: $(date)"
log "Total experiments: 31"
log "============================================================"
log ""
log "Quick comparison of EF-LR accuracy:"
log "  python3 - << 'EOF'"
log "  import json, glob, os"
log "  for f in sorted(glob.glob('logs/pinn_*_table1.json')):"
log "      d = json.load(open(f))"
log "      r = d['results']"
log "      ef = r.get('eigen_100%', {}).get('lr_accuracy', '?')"
log "      raw = r.get('raw_100%', {}).get('lr_accuracy', '?')"
log "      print(f\"{os.path.basename(f).replace('_table1.json',''):45s}  EF={ef:.4f}  Raw={raw:.4f}\")"
log "  EOF"
