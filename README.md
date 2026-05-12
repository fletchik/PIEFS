# PIEFS — Physics-Informed Eigenfunction Features with Learnable Scaling

Sequential neural eigenfunction learning using a learnable Riemannian metric.

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
# Two-moon, K=6, identity metric (OFF), seed=42
python train.py run_id=exp01

# HTRU2, K=6, diagonal metric, seed=42
python train.py run_id=exp02 dataset=htru2 'model/metric=diag'

# MNIST binary 0-vs-1, K=16, sparse metric
python train.py run_id=exp03 dataset=mnist_binary model.K=16 'model/metric=lambda_u_sparse'

# MNIST 10-class supervised, K=10
python train.py run_id=exp04 dataset=mnist_multiclass model.K=10 model.task=multiclass

# Custom steps and batch
python train.py run_id=exp05 dataset=circles trainer.total_steps=10000 trainer.batch_size=512
```

## Dataset configs

| Config key        | Description                         | input_dim |
|-------------------|-------------------------------------|-----------|
| `two_moon`        | sklearn make_moons (10k samples)    | 2         |
| `circles`         | sklearn make_circles (10k samples)  | 2         |
| `lissajous`       | Unit circle (50k samples)           | 2         |
| `htru2`           | HTRU2 pulsar detection (17898)      | 8         |
| `mnist_binary`    | MNIST 0-vs-1 (flat pixels)          | 784       |
| `mnist_multiclass`| MNIST 10 classes (flat pixels)      | 784       |
| `cifar10_binary`  | CIFAR-10 0-vs-1 (flat pixels)       | 3072      |

## Metric variants

| `model/metric=...`   | Description                                    |
|----------------------|------------------------------------------------|
| (default) `off`      | Identity metric A = I                          |
| `diag`               | Diagonal A(x) = Λ(x), det=1                   |
| `lambda_u_sparse`    | Full A(x) = U·Λ, U=expm(sparse skew ω)        |
| `lambda_u_pinn`      | Full A(x) = U_pinn·Λ, U approx. by PINN       |

## Resume from checkpoint

```bash
python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
```

## Evaluate checkpoint on any dataset

```bash
# Evaluate the final checkpoint on the test split
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --split test

# Evaluate on a different dataset than the one used for training
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --dataset two_moon --split test
```

Reports: accuracy, ROC-AUC, gram_error_final, gram_error_offdiag, gram_error_diag,
eigenvalue ordering, wall time per function.

## Reproduce all experiments

```bash
# Full suite (Groups 0, A–F) — takes many hours
bash scripts/reproduce_all.sh

# Specific groups only
GROUPS="0 A" bash scripts/reproduce_all.sh

# Single group
GROUPS="E" bash scripts/reproduce_all.sh
```

## Verification scripts (run before experiments)

```bash
# Check 1: LambdaUSparse/Pinn geometric properties
python scripts/verify_pinn_rotation.py

# Check 2: Sequential gram_error on Two-moon (K=3, 5000 steps)
python scripts/verify_gram.py

# Check 4: Unit circle eigenfunctions vs cos/sin
python scripts/verify_circle.py
```

## Logs and outputs

```
logs/
  <run_id>/
    checkpoint_15k.pt       # step 15k
    checkpoint_30k.pt       # step 30k
    checkpoint_45k.pt       # step 45k
    checkpoint_60k.pt       # step 60k
    checkpoint_best_val.pt  # best val accuracy
    checkpoint_final.pt     # alias to final
  <run_id>_config.md        # written BEFORE training
  <run_id>_results.md       # written AFTER training
  sanity/
    SANITY_REPORT.md        # Group 0 pass/fail summary
  final_report.md           # aggregated results all groups
```

## Expected results

| Dataset | Metric | K | Val Acc | gram_error_final | Ref |
|---------|--------|---|---------|-----------------|-----|
| Two-moon | OFF | 6 | ~1.000 | < 0.05 | old codebase: 1.000 |
| Two-moon | DIAG | 6 | ~1.000 | < 0.05 | old codebase: 1.000 |
| HTRU2 | OFF | 6 | ~0.966 (LR) | < 0.10 | paper Table 1 |
| MNIST binary | DIAG | 6 | ~0.999 | < 0.10 | old codebase: 0.999 |
| MNIST 10-class (unsupervised LR) | OFF | 10 | ~0.848 | — | paper Table 1 |
| NeuralEF MNIST (CNN-GP, K=10) | — | — | 0.8498 | — | NeuralEF Table 1 |

## Architecture notes

Sequential training trains φ_1, then freezes it, then φ_2, ..., φ_K.
This differs from NeuralEF which trains all K functions jointly (parallel).

Key architectural differences vs NeuralEF:
- NeuralEF: joint/parallel training with EigenGame objective, kernel-based
- PIEFS: sequential training with Gram + Dirichlet loss, Riemannian metric A(x)
- NeuralEF MNIST: CNN backbone → 84.98% LR accuracy
- PIEFS MNIST: flat MLP 784-dim → lower accuracy expected (backbone gap)
