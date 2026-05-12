#!/usr/bin/env bash
# Reproduce all experiments from the PIEFS paper.
# Run from the PIEFS/ directory:
#   cd /path/to/PIEFS
#   bash scripts/reproduce_all.sh
#
# Each group can be run independently by setting GROUPS env var:
#   GROUPS="A B" bash scripts/reproduce_all.sh

set -euo pipefail

# Default: use local .venv if present, else fall back to 'python'
if [ -x "$(pwd)/.venv/bin/python3" ]; then
  PYTHON="${PYTHON:-.venv/bin/python3}"
else
  PYTHON="${PYTHON:-python3}"
fi
GROUPS="${GROUPS:-0 A B C D E F}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# Group 0 — Sanity checks (run first, short)
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "0"; then
  log "=== GROUP 0: Sanity checks ==="

  # 0.1 Smoke test: Two-moon K=3 OFF 3000 steps
  log "Step 0.1: Two-moon smoke test (3000 steps)"
  $PYTHON train.py run_id=sanity_smoke \
    dataset=two_moon model.K=3 model.metric_type=off \
    trainer.total_steps=3000 trainer.seed=42 writer.mode=disabled

  # 0.2 Unit circle (uses dedicated script)
  log "Step 0.2: Unit circle eigenfunctions"
  mkdir -p logs/sanity
  $PYTHON scripts/verify_circle.py 2>&1 | tee logs/sanity/circle_verify.txt

  # 0.3 Sequential vs parallel comparison: Two-moon K=6 OFF/DIAG
  log "Step 0.3a: Two-moon K=6 OFF seed=42"
  $PYTHON train.py run_id=sanity_twomoon_K6_off_s42 \
    dataset=two_moon model.K=6 model.metric_type=off \
    trainer.seed=42 writer.mode=disabled

  log "Step 0.3b: Two-moon K=6 OFF seed=123"
  $PYTHON train.py run_id=sanity_twomoon_K6_off_s123 \
    dataset=two_moon model.K=6 model.metric_type=off \
    trainer.seed=123 writer.mode=disabled

  log "Step 0.3c: Two-moon K=6 DIAG seed=42"
  $PYTHON train.py run_id=sanity_twomoon_K6_diag_s42 \
    dataset=two_moon model.K=6 model.metric_type=diag \
    trainer.seed=42 writer.mode=disabled

  log "Step 0.3d: Two-moon K=6 DIAG seed=123"
  $PYTHON train.py run_id=sanity_twomoon_K6_diag_s123 \
    dataset=two_moon model.K=6 model.metric_type=diag \
    trainer.seed=123 writer.mode=disabled

  # 0.4 Matrix variant smoke tests (3000 steps each)
  log "Step 0.4a: LambdaUSparse Two-moon K=3 3000 steps"
  $PYTHON train.py run_id=sanity_sparse_K3 \
    dataset=two_moon model.K=3 model.metric_type=lambda_u_sparse \
    trainer.total_steps=3000 trainer.seed=42 writer.mode=disabled

  log "Step 0.4b: LambdaUPinn Two-moon K=3 3000 steps"
  $PYTHON train.py run_id=sanity_pinn_K3 \
    dataset=two_moon model.K=3 model.metric_type=lambda_u_pinn \
    trainer.total_steps=3000 trainer.seed=42 writer.mode=disabled

  # 0.5 Multiclass head smoke test
  log "Step 0.5: MNIST 10-class MulticlassHead 3000 steps"
  $PYTHON train.py run_id=sanity_mnist_multiclass \
    dataset=mnist_multiclass model.K=6 model.task=multiclass \
    trainer.total_steps=3000 trainer.seed=42 writer.mode=disabled

  # 0.6 Fixed vs dynamic weighting comparison (Two-moon K=6, 20k steps)
  # Determines which training regime to use for Groups A-F.
  log "Step 0.6a: Two-moon K=6 OFF — FIXED weights (ablation)"
  $PYTHON train.py run_id=cmp_fixed_off_s42 \
    dataset=two_moon model.K=6 model.metric_type='off' \
    criterion.dynamic_weighting=false \
    trainer.total_steps=20000 trainer.seed=42 writer.mode=disabled

  log "Step 0.6b: Two-moon K=6 OFF — DYNAMIC weights (paper eq.10)"
  $PYTHON train.py run_id=cmp_dynamic_off_s42 \
    dataset=two_moon model.K=6 model.metric_type='off' \
    criterion.dynamic_weighting=true criterion.t_orth=0.1 criterion.t_class=0.5 \
    trainer.total_steps=20000 trainer.seed=42 writer.mode=disabled

  log "Step 0.6c: Two-moon K=6 DIAG — FIXED weights"
  $PYTHON train.py run_id=cmp_fixed_diag_s42 \
    dataset=two_moon model.K=6 model.metric_type=diag \
    criterion.dynamic_weighting=false \
    trainer.total_steps=20000 trainer.seed=42 writer.mode=disabled

  log "Step 0.6d: Two-moon K=6 DIAG — DYNAMIC weights"
  $PYTHON train.py run_id=cmp_dynamic_diag_s42 \
    dataset=two_moon model.K=6 model.metric_type=diag \
    criterion.dynamic_weighting=true criterion.t_orth=0.1 criterion.t_class=0.5 \
    trainer.total_steps=20000 trainer.seed=42 writer.mode=disabled

  log "=== Group 0 complete ==="
fi

# ---------------------------------------------------------------------------
# Group A — Two-moon and Circles (60k steps, 3 metrics × 2 seeds × 2 datasets)
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "A"; then
  log "=== GROUP A: Two-moon and Circles ==="
  for DATASET in two_moon circles; do
    for METRIC in off diag lambda_u_sparse; do
      for SEED in 42 123; do
        RUN_ID="groupA_${DATASET}_${METRIC}_s${SEED}"
        log "Running: $RUN_ID"
        METRIC_ARG="model.metric_type=off"
        if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
        if [ "$METRIC" = "lambda_u_sparse" ]; then METRIC_ARG="model.metric_type=lambda_u_sparse"; fi
        $PYTHON train.py run_id="$RUN_ID" \
          dataset="$DATASET" model.K=6 "$METRIC_ARG" \
          trainer.seed="$SEED" writer.mode=disabled
      done
    done
  done
  log "=== Group A complete ==="
fi

# ---------------------------------------------------------------------------
# Group B — MNIST binary 0 vs 1
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "B"; then
  log "=== GROUP B: MNIST binary ==="
  for METRIC in off diag lambda_u_sparse; do
    for K in 6 16; do
      for SEED in 42 123; do
        RUN_ID="groupB_mnist_binary_${METRIC}_K${K}_s${SEED}"
        log "Running: $RUN_ID"
        METRIC_ARG="model.metric_type=off"
        if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
        if [ "$METRIC" = "lambda_u_sparse" ]; then METRIC_ARG="model.metric_type=lambda_u_sparse"; fi
        $PYTHON train.py run_id="$RUN_ID" \
          dataset=mnist_binary model.K="$K" "$METRIC_ARG" \
          trainer.seed="$SEED" writer.mode=disabled
      done
    done
  done
  log "=== Group B complete ==="
fi

# ---------------------------------------------------------------------------
# Group C — MNIST 10-class unsupervised (spectral_only, match NeuralEF Table 1)
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "C"; then
  log "=== GROUP C: MNIST 10-class unsupervised ==="
  for METRIC in off diag lambda_u_sparse; do
    for SEED in 42 123; do
      RUN_ID="groupC_mnist_spectral_${METRIC}_s${SEED}"
      log "Running: $RUN_ID"
      METRIC_ARG="model.metric_type=off"
      if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
      if [ "$METRIC" = "lambda_u_sparse" ]; then METRIC_ARG="model.metric_type=lambda_u_sparse"; fi
      $PYTHON train.py run_id="$RUN_ID" \
        dataset=mnist_multiclass model.K=10 model.task=multiclass "$METRIC_ARG" \
        criterion.w_task=0.0 trainer.seed="$SEED" writer.mode=disabled
    done
  done
  log "=== Group C complete ==="
fi

# ---------------------------------------------------------------------------
# Group D — MNIST 10-class supervised (multiclass CE end-to-end)
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "D"; then
  log "=== GROUP D: MNIST 10-class supervised ==="
  for METRIC in off diag; do
    for SEED in 42 123; do
      RUN_ID="groupD_mnist_supervised_${METRIC}_s${SEED}"
      log "Running: $RUN_ID"
      METRIC_ARG="model.metric_type=off"
      if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
      $PYTHON train.py run_id="$RUN_ID" \
        dataset=mnist_multiclass model.K=10 model.task=multiclass "$METRIC_ARG" \
        trainer.seed="$SEED" writer.mode=disabled
    done
  done
  log "=== Group D complete ==="
fi

# ---------------------------------------------------------------------------
# Group E — HTRU2 (spectral_only, match paper Table 1)
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "E"; then
  log "=== GROUP E: HTRU2 ==="
  for METRIC in off diag lambda_u_sparse; do
    for SEED in 42 123; do
      RUN_ID="groupE_htru2_${METRIC}_s${SEED}"
      log "Running: $RUN_ID"
      METRIC_ARG="model.metric_type=off"
      if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
      if [ "$METRIC" = "lambda_u_sparse" ]; then METRIC_ARG="model.metric_type=lambda_u_sparse"; fi
      $PYTHON train.py run_id="$RUN_ID" \
        dataset=htru2 model.K=6 "$METRIC_ARG" \
        criterion.w_task=0.0 trainer.seed="$SEED" writer.mode=disabled
    done
  done
  log "=== Group E complete ==="
fi

# ---------------------------------------------------------------------------
# Group F — CIFAR-10 binary
# ---------------------------------------------------------------------------
if echo "$GROUPS" | grep -qw "F"; then
  log "=== GROUP F: CIFAR-10 binary ==="
  # Grid search first (OFF K=6 seed=42 only)
  for WGRAM in 0.05 0.1 0.5; do
    for WDIR in 1.0 5.0 10.0; do
      RUN_ID="groupF_grid_wg${WGRAM}_wd${WDIR}"
      log "Grid: $RUN_ID"
      $PYTHON train.py run_id="$RUN_ID" \
        dataset=cifar10_binary model.K=6 model.metric_type=off \
        criterion.w_gram="$WGRAM" criterion.w_dirichlet="$WDIR" \
        trainer.seed=42 writer.mode=disabled
    done
  done

  # Main runs with OFF and DIAG
  for METRIC in off diag; do
    for K in 6 16 32; do
      for SEED in 42 123; do
        RUN_ID="groupF_cifar10_${METRIC}_K${K}_s${SEED}"
        log "Running: $RUN_ID"
        METRIC_ARG="model.metric_type=off"
        if [ "$METRIC" = "diag" ]; then METRIC_ARG="model.metric_type=diag"; fi
        $PYTHON train.py run_id="$RUN_ID" \
          dataset=cifar10_binary model.K="$K" "$METRIC_ARG" \
          trainer.seed="$SEED" writer.mode=disabled
      done
    done
  done
  log "=== Group F complete ==="
fi

log "=== All requested groups complete ==="
