# PIEFS — Physics-Informed Eigenfunction Features with Learnable Scaling

Sequential neural eigenfunction learning with a learnable Riemannian metric.

## Setup

```bash
cd /path/to/PIEFS

# Option A — virtualenv (recommended)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yaml
conda activate piefs
```

## Quick start

```bash
# Two-moon, K=6, identity metric (off), seed=42
python train.py run_id=exp01

# Circles, K=6, diagonal metric
python train.py run_id=exp02 dataset=circles model.metric_type=diag

# HTRU2, K=6, global low-rank metric (recommended)
python train.py run_id=exp03 dataset=htru2 model.metric_type=global_low_rank model.low_rank_r=1

# MNIST binary 0-vs-1, K=16
python train.py run_id=exp04 dataset=mnist_binary model.K=16

# MNIST 10-class supervised, K=10
python train.py run_id=exp05 dataset=mnist_multiclass model.K=10 model.task=multiclass

# Custom steps and batch size
python train.py run_id=exp06 dataset=circles trainer.total_steps=10000 trainer.batch_size=512
```

## Dataset configs

| Config key                 | Description                                  | `input_dim` |
|----------------------------|----------------------------------------------|-------------|
| `two_moon` (default)       | sklearn make\_moons, 10k samples             | 2           |
| `circles`                  | sklearn make\_circles, 10k samples           | 2           |
| `htru2`                    | HTRU2 pulsar detection, 17 898 samples       | 8           |
| `mnist_binary`             | MNIST 0-vs-1, flat pixels                    | 784         |
| `mnist_multiclass`         | MNIST 10 classes, flat pixels                | 784         |
| `cifar10_binary`           | CIFAR-10 0-vs-1, flat pixels                 | 3072        |
| `cifar10_features_multiclass` | CIFAR-10 10 classes, pretrained features  | 512         |
| `fashion_mnist_multiclass` | Fashion-MNIST 10 classes, flat pixels        | 784         |

## Metric variants

Override with `model.metric_type=<key>` on the command line.

| `metric_type`      | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| `off` (default)    | Identity metric A = I — plain Dirichlet energy, no scaling.           |
| `diag`             | A(x) = Λ(x) diagonal, det = 1 — axis-aligned anisotropy.             |
| `conformal`        | A(x) = σ(x)·I — isotropic x-dependent scaling (scalar MLP).          |
| `global_low_rank`  | A = I + U·D·Vᵀ — global rank-r perturbation. **Recommended.**        |
| `local_low_rank`   | A(x) = I + U(x)·Λ(x)·V(x)ᵀ — x-dependent rank-r perturbation.      |
| `fisher_diag`      | A(x) = diag(√F(x)) — diagonal Fisher Information Metric (MLP approx).|
| `lambda_u_trotter` | A(x) = Λ(x)·U(ω(x)) — Givens rotations via Trotter product formula.  |

`global_low_rank` is the recommended starting point for most datasets. Set
`model.low_rank_r` to `num_classes − 1` (e.g. `1` for binary, `9` for MNIST-10).

## Evaluate from checkpoint

```bash
# Evaluate the final checkpoint (same dataset used for training)
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --split test

# Evaluate on a different dataset
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --dataset two_moon --split test
```

Reported metrics: accuracy, ROC-AUC, gram\_error\_final, gram\_error\_offdiag,
gram\_error\_diag, eigenvalue ordering, wall time per function.

## Resume from checkpoint

```bash
python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
```

## Reproduce all experiments

```bash
# Full suite (Groups 0, A–F) — takes many hours on CPU
bash scripts/reproduce_all.sh

# Specific groups only
GROUPS="0 A" bash scripts/reproduce_all.sh

# Single group
GROUPS="E" bash scripts/reproduce_all.sh
```

## Verification scripts

Run these before experiments to confirm the implementation is correct:

```bash
# 1. LambdaUTrotter geometric properties (orthogonality, det)
python scripts/verify_pinn_rotation.py

# 2. Sequential gram_error on Two-moon (K=3, 5 000 steps)
python scripts/verify_gram.py

# 3. Unit circle eigenfunctions vs cos/sin (K=4, 60 000 steps)
python scripts/verify_circle.py
```

## Aggregate grid results

```bash
python scripts/collect_grid_results.py --log_dir logs --out_dir results
```

Reads `logs/grid_*/metrics.jsonl`, prints a markdown + LaTeX table of
mean ± std over seeds.

## Logs and outputs

```
logs/
  <run_id>/
    checkpoint_15k.pt        # periodic checkpoint
    checkpoint_30k.pt
    checkpoint_45k.pt
    checkpoint_60k.pt
    checkpoint_best_val.pt   # best validation accuracy
    checkpoint_final.pt      # alias to final step
  <run_id>_config.md         # written before training starts
  <run_id>_results.md        # written after training finishes
  sanity/
    SANITY_REPORT.md         # Group 0 pass/fail summary
  final_report.md            # aggregated results across all groups
```

## Paper

Source lives in `paper_0/`. The compiled PDF is not tracked (it is in
`.gitignore`). To compile:

```bash
cd paper_0
pdflatex main.tex
bibtex main
pdflatex main.tex && pdflatex main.tex
```

Or if a `Makefile` is present: `make -C paper_0`.

## Expected results

| Dataset            | Metric           | K  | Val Acc  | gram\_error\_final |
|--------------------|------------------|----|----------|--------------------|
| Two-moon           | off              | 6  | ~1.000   | < 0.05             |
| Two-moon           | diag             | 6  | ~1.000   | < 0.05             |
| HTRU2              | off              | 6  | ~0.966   | < 0.10             |
| MNIST binary       | diag             | 6  | ~0.999   | < 0.10             |
| MNIST 10-class     | off (LR probe)   | 10 | ~0.848   | —                  |

## Architecture notes

Sequential training trains φ₁ then freezes it, then φ₂, ..., φ_K.
This differs from NeuralEF which trains all K functions jointly.

Key differences vs NeuralEF:
- **NeuralEF**: joint/parallel training with EigenGame objective, kernel-based.
- **PIEFS**: sequential training with Gram + Dirichlet loss, learnable metric A(x).
- NeuralEF MNIST uses a CNN backbone → 84.98 % LR accuracy.
- PIEFS on flat 784-dim pixels → lower accuracy expected (backbone gap, not a bug).
