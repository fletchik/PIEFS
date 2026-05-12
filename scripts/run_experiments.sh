#!/usr/bin/env bash
# ============================================================
# PIEFS — Experiment runner for CIKM 2026
# ============================================================
# Run from the project root:
#   bash scripts/run_experiments.sh [group]
#
# Groups (in recommended CPU order):
#   D0  — sanity smoke test (~5 min)
#   D5  — curriculum ablation on htru2 (~1.5 h)
#   D3  — rank ablation on htru2 (~2 h)
#   D1  — metric ablation on 2D datasets (~3 h)
#   D4  — dynamic weighting ablation on htru2 (~1.5 h)
#   D2  — main table: htru2 full metric ablation (~2 h)
#   D6  — main table: MNIST multiclass  [GPU recommended, ~8 h GPU / ~30 h CPU]
#   all — all groups sequentially
#
# Set SEEDS to run fewer seeds for a quicker check:
#   SEEDS="42 1337" bash scripts/run_experiments.sh D2
# ============================================================

set -euo pipefail

# ── Python interpreter ──────────────────────────────────────
# Walk up from the project root looking for a .venv, then fall back to
# PYTHON env-var or system python3. This handles both a checkout at the
# repo root and a git-worktree layout (project root inside .claude/worktrees/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=""
SEARCH_DIR="$PROJECT_ROOT"
for _i in 1 2 3 4; do
    if [ -x "$SEARCH_DIR/.venv/bin/python3" ]; then
        PYTHON="$SEARCH_DIR/.venv/bin/python3"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done
if [ -z "$PYTHON" ]; then
    PYTHON="${PYTHON_OVERRIDE:-python3}"
fi
echo "[run_experiments] Using Python: $PYTHON"

# ── Seeds (override with: SEEDS="42 1337" bash ...) ─────────
SEEDS="${SEEDS:-42 1337 0}"
LOG_DIR="${LOG_DIR:-logs}"
WRITER_MODE="${WRITER_MODE:-disabled}"   # set WRITER_MODE=online for WandB

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    # Usage: run run_id=<id> [extra hydra args...]
    log ">>> $*"
    "$PYTHON" "$PROJECT_ROOT/train.py" "$@" writer.mode="$WRITER_MODE"
    log "    done."
}

GROUP="${1:-D0}"

# ============================================================
# D0 — Sanity smoke test (~5 min)
# Just confirms the pipeline runs end-to-end before long runs.
# ============================================================
if [[ "$GROUP" == "D0" || "$GROUP" == "all" ]]; then
    log "=== D0: Sanity smoke (two_moon K=3, 3000 steps) ==="
    run run_id=D0_smoke \
        dataset=two_moon model.K=3 model.metric_type=off \
        trainer.total_steps=3000 trainer.seed=42
    log "=== D0 complete — pipeline OK ==="
fi

# ============================================================
# D5 — Curriculum ablation on htru2 (~1.5 h, 6 runs)
# Question: does the 3-phase schedule help global_low_rank?
# No curriculum (p1=p2=0)  vs.  50%/75% schedule
# Expected: curriculum wins (Dirichlet energy activated gradually)
# ============================================================
if [[ "$GROUP" == "D5" || "$GROUP" == "all" ]]; then
    log "=== D5: Curriculum ablation — htru2, global_low_rank ==="
    for SEED in $SEEDS; do
        run run_id=D5_htru2_glr_nocurr_s"${SEED}" \
            dataset=htru2 model.K=6 \
            model.metric_type=global_low_rank model.low_rank_r=1 \
            criterion.dynamic_weighting=true \
            curriculum.phase1_end_step=0 curriculum.phase2_end_step=0 \
            trainer.total_steps=60000 trainer.seed="${SEED}"

        run run_id=D5_htru2_glr_curr_s"${SEED}" \
            dataset=htru2 model.K=6 \
            model.metric_type=global_low_rank model.low_rank_r=1 \
            criterion.dynamic_weighting=true \
            curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000 \
            trainer.total_steps=60000 trainer.seed="${SEED}"
    done
    log "=== D5 complete ==="
fi

# ============================================================
# D3 — Rank ablation on htru2 (~2 h, 12 runs)
# Question: is r=C-1=1 really optimal for binary classification?
# Tests r ∈ {1, 2, 4, 8} with global_low_rank + curriculum
# Expected: r=1 ≥ r=2 > r=4 > r=8 (LDA theorem: rank C-1 = 1)
# ============================================================
if [[ "$GROUP" == "D3" || "$GROUP" == "all" ]]; then
    log "=== D3: Rank ablation — htru2, global_low_rank ==="
    for R in 1 2 4 8; do
        for SEED in $SEEDS; do
            run run_id=D3_htru2_glr_r"${R}"_s"${SEED}" \
                dataset=htru2 model.K=6 \
                model.metric_type=global_low_rank model.low_rank_r="${R}" \
                criterion.dynamic_weighting=true \
                curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000 \
                trainer.total_steps=60000 trainer.seed="${SEED}"
        done
    done
    log "=== D3 complete ==="
fi

# ============================================================
# D1 — Metric ablation on 2D datasets (~3 h, 20 runs)
# All 5 metrics on two_moon and circles; 2 seeds each
# Goal: eigenfunction visualizations for the paper
# Expected: global_low_rank ≥ conformal > diag > off
# ============================================================
if [[ "$GROUP" == "D1" || "$GROUP" == "all" ]]; then
    log "=== D1: Metric ablation — two_moon + circles ==="
    for DATASET in two_moon circles; do
        for MTYPE in off diag conformal global_low_rank local_low_rank; do
            EXTRA=""
            if [[ "$MTYPE" == "global_low_rank" || "$MTYPE" == "local_low_rank" ]]; then
                EXTRA="model.low_rank_r=1"   # binary: r = C-1 = 1
            fi
            for SEED in $SEEDS; do
                run run_id=D1_"${DATASET}"_"${MTYPE}"_s"${SEED}" \
                    dataset="${DATASET}" model.K=6 \
                    model.metric_type="${MTYPE}" \
                    criterion.dynamic_weighting=true \
                    trainer.total_steps=60000 trainer.seed="${SEED}" \
                    $EXTRA
            done
        done
    done
    log "=== D1 complete ==="
fi

# ============================================================
# D4 — Dynamic weighting ablation on htru2 (~1.5 h, 6 runs)
# Question: does the adaptive w_task / w_mde schedule matter?
# dynamic_weighting=true (paper eq. 10) vs. false (fixed weights)
# Expected: dynamic > static (weights prevent metric from suppressing
#           Gram term before orthogonality is established)
# ============================================================
if [[ "$GROUP" == "D4" || "$GROUP" == "all" ]]; then
    log "=== D4: Dynamic weighting ablation — htru2, off ==="
    for SEED in $SEEDS; do
        run run_id=D4_htru2_off_static_s"${SEED}" \
            dataset=htru2 model.K=6 model.metric_type=off \
            criterion.dynamic_weighting=false \
            trainer.total_steps=60000 trainer.seed="${SEED}"

        run run_id=D4_htru2_off_dynamic_s"${SEED}" \
            dataset=htru2 model.K=6 model.metric_type=off \
            criterion.dynamic_weighting=true \
            trainer.total_steps=60000 trainer.seed="${SEED}"
    done
    log "=== D4 complete ==="
fi

# ============================================================
# D2 — Main table: htru2, all 5 metrics (~2 h, 15 runs)
# The clean ablation table for the paper.
# ============================================================
if [[ "$GROUP" == "D2" || "$GROUP" == "all" ]]; then
    log "=== D2: Main table — htru2, 5 metrics ==="
    for MTYPE in off diag conformal global_low_rank local_low_rank; do
        EXTRA=""
        if [[ "$MTYPE" == "global_low_rank" ]]; then
            EXTRA="model.low_rank_r=1 curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000"
        elif [[ "$MTYPE" == "local_low_rank" ]]; then
            EXTRA="model.low_rank_r=1"
        fi
        for SEED in $SEEDS; do
            run run_id=D2_htru2_"${MTYPE}"_s"${SEED}" \
                dataset=htru2 model.K=6 \
                model.metric_type="${MTYPE}" \
                criterion.dynamic_weighting=true \
                trainer.total_steps=60000 trainer.seed="${SEED}" \
                $EXTRA
        done
    done
    log "=== D2 complete ==="
fi

# ============================================================
# D6 — Main table: MNIST 10-class, all 5 metrics
# [!] GPU strongly recommended — ~8 h on A100, ~30 h on CPU.
# Runs 3 seeds to keep runtime manageable.
# global_low_rank: r=9 (= C-1 = 10-1), three-phase curriculum
# ============================================================
if [[ "$GROUP" == "D6" || "$GROUP" == "all" ]]; then
    log "=== D6: Main table — MNIST multiclass, 5 metrics ==="
    log "    [!] This is long — GPU recommended."
    for MTYPE in off diag conformal global_low_rank local_low_rank; do
        EXTRA=""
        if [[ "$MTYPE" == "global_low_rank" ]]; then
            EXTRA="model.low_rank_r=9 curriculum.phase1_end_step=30000 curriculum.phase2_end_step=45000"
        elif [[ "$MTYPE" == "local_low_rank" ]]; then
            EXTRA="model.low_rank_r=9"
        fi
        for SEED in $SEEDS; do
            run run_id=D6_mnist_"${MTYPE}"_s"${SEED}" \
                dataset=mnist_multiclass model.K=16 model.task=multiclass \
                model.metric_type="${MTYPE}" \
                criterion.dynamic_weighting=true \
                trainer.total_steps=60000 trainer.seed="${SEED}" \
                $EXTRA
        done
    done
    log "=== D6 complete ==="
fi

log "=== All requested groups complete ==="
log "    Collect results: python scripts/collect_grid_results.py --log_dir $LOG_DIR"
