# PIEFS — Physics-Informed Eigenfunction Features with Learnable Scaling

> Sequential neural eigenfunction learning with a learnable Riemannian metric.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

**PIEFS** learns an orthonormal basis of eigenfunctions φ₁, …, φ_K on the data manifold by
minimising a *modified Dirichlet energy* with a learnable Riemannian metric **A(x)**:

```
L = ‖A(x) ∇φₖ‖² + w_orth · ‖ΦᵀΦ/N − I‖²_F + w_class · L_CE
```

Functions are trained *sequentially*: φ₁ is learned first, then frozen, then φ₂ is added, and so on.
The resulting feature map Φ(x) = [φ₁(x), …, φ_K(x)] can be evaluated with a simple linear probe.

---

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Metric Variants](#metric-variants)
- [Evaluation](#evaluation)
- [Reproduce Experiments](#reproduce-experiments)
- [Verification](#verification)
- [Project Structure](#project-structure)
- [Logs & Outputs](#logs--outputs)
- [Paper](#paper)
- [Expected Results](#expected-results)

---

## Setup

```bash
# Clone
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

---

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

# Custom steps and batch size
python train.py run_id=exp06 dataset=circles \
    trainer.total_steps=10000 trainer.batch_size=512

# Resume from a checkpoint
python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
```

---

## Datasets

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

Pass the key via `dataset=<key>`, e.g. `dataset=htru2`.

---

## Metric Variants

Override with `model.metric_type=<key>`.

| `metric_type`        | Formula                                             | Notes                                            |
|----------------------|-----------------------------------------------------|--------------------------------------------------|
| `off` *(default)*    | **A = I**                                           | Plain Dirichlet energy, no scaling               |
| `diag`               | **A(x) = Λ(x)**, det = 1                           | Axis-aligned anisotropy, MLP output              |
| `conformal`          | **A(x) = σ(x) · I**                                | Isotropic x-dependent scaling (scalar MLP)       |
| `global_low_rank` ⭐ | **A = I + U·D·Vᵀ** (global, rank-r)                | No MLP bottleneck. **Recommended starting point**|
| `local_low_rank`     | **A(x) = I + U(x)·Λ(x)·V(x)ᵀ** (x-dependent)     | Full rank-r coverage, shared MLP backbone        |
| `fisher_diag`        | **A(x) = diag(√F(x))**                             | Diagonal Fisher Information Metric (MLP approx.) |
| `lambda_u_trotter`   | **A(x) = Λ(x) · U(ω(x))**                          | Givens rotations via Trotter product formula     |

> **Tip — `global_low_rank`** is the recommended starting point for most datasets.
> Set `model.low_rank_r` to `num_classes − 1`:
> `1` for binary classification, `9` for MNIST-10 / CIFAR-10.

---

## Evaluation

```bash
# Evaluate final checkpoint on the same dataset used for training
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

---

## Reproduce Experiments

```bash
# Full suite (Groups 0, A–F) — takes many hours on CPU
bash scripts/reproduce_all.sh

# Specific groups only
GROUPS="0 A" bash scripts/reproduce_all.sh

# Single group
GROUPS="E" bash scripts/reproduce_all.sh

# Aggregate results into a table (markdown + LaTeX)
python scripts/collect_grid_results.py --log_dir logs --out_dir results
```

---

## Verification

Run these before experiments to confirm the implementation is correct:

```bash
# 1. LambdaUTrotter: geometric properties (orthogonality, det = 1)
python scripts/verify_pinn_rotation.py

# 2. Sequential gram_error on Two-moon (K=3, 5 000 steps)
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
│   │   ├── train.yaml                # Root config (defaults, trainer, model, writer)
│   │   ├── criterion/spectral.yaml   # Loss weights (w_gram, w_dirichlet, w_task)
│   │   ├── dataset/                  # One yaml per dataset (two_moon, htru2, …)
│   │   ├── model/metric/             # One yaml per metric variant
│   │   └── optimizer/adam.yaml
│   │
│   ├── dataset/                      # Dataset classes + collate utilities
│   │   ├── sklearn_cls.py            # Two-moon, Circles (sklearn)
│   │   ├── htru2.py
│   │   ├── torchvision_flat.py       # MNIST, Fashion-MNIST, CIFAR-10
│   │   ├── pretrained_features.py    # CIFAR-10 with pretrained embeddings
│   │   ├── spotify.py
│   │   ├── collate.py
│   │   └── utils.py
│   │
│   ├── loss/
│   │   └── spectral_loss.py          # SpectralDirichletLoss (Gram + Dirichlet + CE)
│   │
│   ├── model/
│   │   ├── basis/
│   │   │   ├── basis_net.py          # Single eigenfunction network φₖ(x)
│   │   │   └── basis_set.py          # BasisSet — manages K basis nets + active index
│   │   ├── metric/
│   │   │   ├── metric_net.py         # build_metric() dispatcher
│   │   │   ├── diag_metric.py
│   │   │   ├── conformal_metric.py
│   │   │   ├── global_low_rank.py
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
│       └── plots.py                  # Heatmaps, decision boundaries, Gram matrix
│
├── scripts/
│   ├── verify_gram.py                # Sanity: gram_error convergence
│   ├── verify_circle.py              # Sanity: unit-circle eigenfunctions
│   ├── eval_from_checkpoint.py       # Offline evaluation of saved checkpoints
│   ├── collect_grid_results.py       # Aggregate grid-search logs → table
│   ├── gen_figures.py                # Reproduce paper figures
│   ├── gen_additional_figures.py
│   ├── run_sklearn_baselines.py      # Random Forest + Logistic Regression baselines
│   ├── extract_cnn_features.py       # Extract pretrained features from CIFAR-10
│   └── reproduce_all.sh              # Full experiment suite
│
├── paper_0/
│   ├── main.tex                      # Paper source
│   ├── bibliobase.bib
│   └── figures/                      # All figures referenced in main.tex
│
└── tests/
    └── test_metrics.py
```

---

## Logs & Outputs

```
logs/
├── <run_id>/
│   ├── metrics.jsonl              # Step-level metrics (loss, val_acc, gram_error, …)
│   ├── checkpoint_15k.pt          # Periodic checkpoints
│   ├── checkpoint_30k.pt
│   ├── checkpoint_45k.pt
│   ├── checkpoint_60k.pt
│   ├── checkpoint_best_val.pt     # Best validation accuracy
│   └── checkpoint_final.pt        # Final step
├── <run_id>_config.md             # Config snapshot (written before training)
├── <run_id>_results.md            # Results summary (written after training)
├── sanity/
│   └── SANITY_REPORT.md           # Group-0 verification pass/fail
└── final_report.md                # Aggregated results across all groups
```

---

## Paper

Source is in `paper_0/`. The compiled PDF is excluded from the repository
(listed in `.gitignore`). To compile locally:

```bash
cd paper_0
pdflatex main.tex
bibtex main
pdflatex main.tex && pdflatex main.tex
# Output: paper_0/main.pdf
```

---

## Expected Results

| Dataset            | Metric           |  K  | Val Acc  | `gram_error_final` |
|--------------------|------------------|:---:|:--------:|:------------------:|
| Two-moon           | `off`            |  6  | ~1.000   | < 0.05             |
| Two-moon           | `diag`           |  6  | ~1.000   | < 0.05             |
| HTRU2              | `off`            |  6  | ~0.966   | < 0.10             |
| MNIST binary       | `diag`           |  6  | ~0.999   | < 0.10             |
| MNIST 10-class     | `off` (LR probe) | 10  | ~0.848   | —                  |

> **Note on gram\_error.** A residual ‖C − I‖_F > 0 is expected and acceptable —
> it reflects finite-batch estimation of L² inner products and does not affect
> classification performance in practice.

---

## Architecture Notes

Sequential training trains φ₁ then freezes it, trains φ₂, and so on.
This differs from NeuralEF which trains all K functions jointly.

| Aspect            | PIEFS (this work)                       | NeuralEF (Deng et al., 2022)      |
|-------------------|-----------------------------------------|-----------------------------------|
| Training order    | Sequential, one φₖ at a time           | Parallel (EigenGame objective)    |
| Metric            | Learnable A(x) (Riemannian)            | Fixed kernel                      |
| MNIST backbone    | Flat MLP (784-dim input)               | CNN → lower accuracy gap expected |
| Orthogonality     | Gram penalty (finite-batch)            | Eigengame constraint              |
