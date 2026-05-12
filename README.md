# PIEFS — Physics-Informed Eigenfunction Features with Learnable Scaling

> Sequential neural eigenfunction learning with a learnable Riemannian metric.

📄 **[Paper (PDF)](paper_0/main.pdf)**

---

## What is PIEFS?

**PIEFS** learns a set of smooth, orthonormal functions φ₁, …, φ_K on the data manifold,
analogous to Laplacian eigenfunctions but adapted to the geometry of the dataset.

The core idea is to minimise a *modified Dirichlet energy* with a **learnable Riemannian
metric A(x)** that captures the local geometry of the data:

```
L = ‖A(x) ∇φₖ‖² + w_orth · ‖ΦᵀΦ/N − I‖²_F + w_class · L_CE
```

| Term | Role |
|------|------|
| `‖A(x) ∇φₖ‖²` | Modified Dirichlet energy — encourages smooth functions that respect data geometry |
| `‖ΦᵀΦ/N − I‖²_F` | Gram penalty — enforces approximate orthonormality across the basis |
| `L_CE` | Cross-entropy — makes features discriminative for the downstream task |

Functions are trained **sequentially**: φ₁ is optimised first and then frozen,
then φ₂ is added, and so on up to φ_K. This guarantees that each new function
is orthogonal to all previous ones by construction.

The resulting feature map **Φ(x) = [φ₁(x), …, φ_K(x)]** can be used directly
as input to a linear classifier (logistic regression probe).

## How does it differ from NeuralEF?

| Aspect            | **PIEFS** (this work)                    | NeuralEF (Deng et al., 2022)       |
|-------------------|------------------------------------------|------------------------------------|
| Training order    | Sequential — one φₖ at a time           | Parallel (EigenGame objective)     |
| Metric            | Learnable **A(x)** (Riemannian)         | Fixed kernel                       |
| MNIST backbone    | Flat MLP (784-dim input)                | CNN — lower accuracy gap expected  |
| Orthogonality     | Gram penalty (finite-batch)             | EigenGame constraint               |

## Learnable Metric Variants

The metric **A(x)** can be parametrised in several ways.
Override with `model.metric_type=<key>`.

| `metric_type`        | Formula                                              | Notes                                              |
|----------------------|------------------------------------------------------|----------------------------------------------------|
| `off` *(default)*    | **A = I**                                            | Plain Dirichlet energy, no scaling                 |
| `diag`               | **A(x) = Λ(x)**, det = 1                            | Axis-aligned anisotropy, MLP output                |
| `conformal`          | **A(x) = σ(x) · I**                                 | Isotropic x-dependent scaling (scalar MLP)         |
| `global_low_rank` ⭐  | **A = I + U·D·Vᵀ** (global, rank-r)                 | No MLP bottleneck. **Recommended starting point**  |
| `local_low_rank`     | **A(x) = I + U(x)·Λ(x)·V(x)ᵀ** (x-dependent)      | Full rank-r coverage, shared MLP backbone          |
| `fisher_diag`        | **A(x) = diag(√F(x))**                              | Diagonal Fisher Information Metric (MLP approx.)   |
| `lambda_u_trotter`   | **A(x) = Λ(x) · U(ω(x))**                           | Givens rotations via Trotter product formula       |

> **Tip — `global_low_rank`** is the recommended starting point.
> Set `model.low_rank_r` to `num_classes − 1`:
> `1` for binary tasks, `9` for MNIST-10 / CIFAR-10.

## Supported Datasets

| Config key                      | Description                                  | `input_dim` | Task        |
|---------------------------------|----------------------------------------------|:-----------:|-------------|
| `two_moon` *(default)*          | sklearn `make_moons`, 10 000 samples         | 2           | binary      |
| `circles`                       | sklearn `make_circles`, 10 000 samples       | 2           | binary      |
| `htru2`                         | HTRU2 pulsar detection, 17 898 samples       | 8           | binary      |
| `mnist_binary`                  | MNIST digits 0 vs 1, flat pixels             | 784         | binary      |
| `mnist_multiclass`              | MNIST 10 classes, flat pixels                | 784         | multiclass  |
| `cifar10_binary`                | CIFAR-10 classes 0 vs 1, flat pixels         | 3 072       | binary      |
| `cifar10_features_multiclass`   | CIFAR-10 10 classes, pretrained features     | 512         | multiclass  |
| `fashion_mnist_multiclass`      | Fashion-MNIST 10 classes, flat pixels        | 784         | multiclass  |

## Expected Results

| Dataset            | Metric           |  K  | Val Acc  | `gram_error_final` |
|--------------------|------------------|:---:|:--------:|:------------------:|
| Two-moon           | `off`            |  6  | ~1.000   | < 0.05             |
| Two-moon           | `diag`           |  6  | ~1.000   | < 0.05             |
| HTRU2              | `off`            |  6  | ~0.966   | < 0.10             |
| MNIST binary       | `diag`           |  6  | ~0.999   | < 0.10             |
| MNIST 10-class     | `off` (LR probe) | 10  | ~0.848   | —                  |

> A residual `gram_error > 0` is expected — it reflects finite-batch estimation
> of L² inner products and does not affect classification performance in practice.

---

## Setup

```bash
git clone https://github.com/fletchik/PIEFS.git
cd PIEFS

# Option A — virtualenv (recommended)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yaml
conda activate piefs
```

## Quick Start

All experiments go through `train.py` with [Hydra](https://hydra.cc) config overrides.

```bash
# Two-moon, K=6, identity metric — fastest sanity check
python train.py run_id=exp01

# Circles, diagonal metric
python train.py run_id=exp02 dataset=circles model.metric_type=diag

# HTRU2 (8-dim tabular), global low-rank metric (recommended)
python train.py run_id=exp03 dataset=htru2 \
    model.metric_type=global_low_rank model.low_rank_r=1

# MNIST binary (0 vs 1), K=16
python train.py run_id=exp04 dataset=mnist_binary model.K=16

# MNIST 10-class supervised, K=10
python train.py run_id=exp05 dataset=mnist_multiclass \
    model.K=10 model.task=multiclass

# Resume from a checkpoint
python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
```

## Evaluation

```bash
# Evaluate final checkpoint (same dataset as training)
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --split test

# Evaluate on a different dataset
python scripts/eval_from_checkpoint.py \
    --checkpoint logs/exp01/checkpoint_final.pt \
    --dataset htru2 --split test
```

Reported metrics: **accuracy**, **ROC-AUC**, `gram_error_final`, `gram_error_offdiag`,
`gram_error_diag`, eigenvalue ordering, wall time per function.

## Reproduce Experiments

```bash
# Full suite (Groups 0, A–F) — takes many hours on CPU
bash scripts/reproduce_all.sh

# Specific groups only
GROUPS="0 A" bash scripts/reproduce_all.sh

# Aggregate results into markdown + LaTeX table
python scripts/collect_grid_results.py --log_dir logs --out_dir results
```

## Verification

Run before experiments to confirm the implementation is correct:

```bash
# 1. LambdaUTrotter: orthogonality and det = 1
python scripts/verify_pinn_rotation.py

# 2. Gram error convergence on Two-moon (K=3, 5 000 steps)
python scripts/verify_gram.py

# 3. Unit-circle eigenfunctions vs cos/sin (K=4, 60 000 steps)
python scripts/verify_circle.py
```

All three should print `OVERALL: PASS`.

---

## Project Structure

```
PIEFS/
├── train.py                          # Main entry point (Hydra)
├── requirements.txt
├── environment.yaml
│
├── src/
│   ├── configs/
│   │   ├── train.yaml                # Root config (trainer, model, writer)
│   │   ├── criterion/spectral.yaml   # Loss weights (w_gram, w_dirichlet, w_task)
│   │   ├── dataset/                  # One yaml per dataset
│   │   ├── model/metric/             # One yaml per metric variant
│   │   └── optimizer/adam.yaml
│   │
│   ├── dataset/                      # Dataset classes + collate utilities
│   │   ├── sklearn_cls.py            # Two-moon, Circles
│   │   ├── htru2.py
│   │   ├── torchvision_flat.py       # MNIST, Fashion-MNIST, CIFAR-10
│   │   ├── pretrained_features.py    # CIFAR-10 with pretrained embeddings
│   │   ├── collate.py
│   │   └── utils.py
│   │
│   ├── loss/
│   │   └── spectral_loss.py          # SpectralDirichletLoss
│   │
│   ├── model/
│   │   ├── basis/
│   │   │   ├── basis_net.py          # Single eigenfunction network φₖ(x)
│   │   │   └── basis_set.py          # BasisSet — manages K nets + active index
│   │   ├── metric/
│   │   │   ├── metric_net.py         # build_metric() dispatcher
│   │   │   ├── diag_metric.py
│   │   │   ├── conformal_metric.py
│   │   │   ├── global_low_rank.py    # ⭐ Recommended
│   │   │   ├── local_low_rank.py
│   │   │   ├── fisher_diag.py
│   │   │   └── lambda_u_trotter.py
│   │   └── spectral_model.py         # SpectralModel, BinaryHead, MulticlassHead
│   │
│   ├── pretrain/
│   │   └── graph_laplacian.py        # Graph-Laplacian warm-start (optional)
│   │
│   ├── trainer/
│   │   └── sequential_trainer.py     # Sequential training loop
│   │
│   └── logger/
│       ├── experiment_logger.py      # Markdown run reports
│       ├── wandb_writer.py           # Weights & Biases integration (optional)
│       └── plots.py                  # Heatmaps, Gram matrix, decision boundary
│
├── scripts/
│   ├── verify_gram.py                # Sanity: gram_error convergence
│   ├── verify_circle.py              # Sanity: unit-circle eigenfunctions
│   ├── eval_from_checkpoint.py       # Offline checkpoint evaluation
│   ├── collect_grid_results.py       # Aggregate logs → table
│   ├── gen_figures.py                # Reproduce paper figures
│   ├── run_sklearn_baselines.py      # Random Forest + Logistic Regression
│   ├── extract_cnn_features.py       # Pretrained features from CIFAR-10
│   └── reproduce_all.sh              # Full experiment suite
│
├── paper_0/
│   ├── main.tex                      # Paper source (LaTeX)
│   ├── main.pdf                      # Compiled paper
│   ├── bibliobase.bib
│   └── figures/                      # All figures used in main.tex
│
└── tests/
    └── test_metrics.py
```

## Logs & Outputs

```
logs/
├── <run_id>/
│   ├── metrics.jsonl              # Step-level metrics (loss, val_acc, gram_error, …)
│   ├── checkpoint_15k.pt
│   ├── checkpoint_30k.pt
│   ├── checkpoint_45k.pt
│   ├── checkpoint_60k.pt
│   ├── checkpoint_best_val.pt     # Best validation accuracy
│   └── checkpoint_final.pt
├── <run_id>_config.md             # Config snapshot (written before training)
├── <run_id>_results.md            # Results summary (written after training)
└── sanity/
    └── SANITY_REPORT.md
```
