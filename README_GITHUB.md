# Physics-Informed Eigenfunction Features with Learnable Scaling (PIEFS)

Official repository for **"Physics-Informed Eigenfunction Features with Learnable Scaling"** submitted to AI4Physics 2026.

## Overview

PIEFS is a method for learning physics-informed eigenfunction bases on data manifolds. It combines:
- **Sequential eigenfunction learning**: Learn eigenfunctions φ₁, φ₂, ..., φₖ one-at-a-time with cyclic activation
- **Modified Dirichlet Energy (MDE) loss**: Minimize ‖A(x)∇φₖ‖² where A(x) is a learnable metric
- **Learnable metric A(x) = Λ(x)·U(ω(x))**: Diagonal scaling Λ(x) plus Givens rotation matrix U with bounded angles
- **Orthogonality constraints**: Batch Gram penalty ‖Φ^T Φ/N - I‖²_F ensures eigenfunctions remain orthonormal
- **Stop-gradient mechanism**: Prevent j≠k parameters from receiving updates during φₖ training

The method achieves competitive accuracy on benchmark datasets (Two Moons, Circles, HTRU2, MNIST, CIFAR-10) while learning geometrically interpretable features.

## Key Results

| Dataset | RF | LR | NeuralEF* | EFDO-off | EFDO-diag | EFDO-trotter |
|---------|----|----|-----------|----------|-----------|--------------|
| Two Moons | 99.77±0.04 | 87.80±0.00 | --- | **100.00±0.00** | 99.97±0.04 | 99.99±0.03 |
| Circles | 98.53±0.11 | 50.40±0.00 | --- | 78.23±14.90 | 79.16±4.82 | **83.59±15.70** |
| HTRU2 | 98.07±0.07 | 98.14±0.00 | --- | 97.52±0.08 | 97.48±0.04 | **97.71±0.06** |
| MNIST | 97.12±0.03 | 91.84±0.00 | 82.52±0.29 | **94.53±0.33** | 93.63±0.34 | 93.99±0.25 |
| CIFAR-10 | --- | --- | --- | **85.50±0.53** | 84.98±0.33 | † |

*NeuralEF numbers are test accuracy from original benchmark. † Pending final runs.

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.11+
- scikit-learn
- matplotlib
- numpy, scipy

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/piefs.git
cd piefs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training on Two Moons

```bash
python train.py \
  --dataset two_moons \
  --variant off \
  --k 3 \
  --steps 60000 \
  --batch_size 256 \
  --seed 42
```

### Training on MNIST

```bash
python train.py \
  --dataset mnist \
  --variant diag \
  --k 16 \
  --steps 60000 \
  --batch_size 256 \
  --seed 42
```

### Training on CIFAR-10 (ResNet-18 embeddings)

```bash
python train.py \
  --dataset cifar10_resnet18 \
  --variant trotter \
  --k 16 \
  --steps 120000 \
  --batch_size 256 \
  --seed 42
```

### Evaluating from Checkpoint

```bash
python scripts/eval_from_checkpoint.py \
  --checkpoint results/checkpoints/two_moons_off_k3_seed42.pt \
  --dataset two_moons \
  --split test
```

## Project Structure

```
piefs/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train.py                           # Main training script
├── src/
│   ├── __init__.py
│   ├── configs/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── base.py                   # Base configuration class
│   │   ├── two_moons.py              # Two Moons config
│   │   ├── cifar10.py                # CIFAR-10 config
│   │   └── mnist.py                  # MNIST config
│   ├── dataset/                       # Dataset utilities
│   │   ├── __init__.py
│   │   ├── loader.py                 # PyTorch DataLoader creation
│   │   ├── synthetic.py              # Synthetic data (Two Moons, Circles)
│   │   └── preprocess.py             # Data preprocessing
│   ├── logger/                        # Logging and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                # Metric computation
│   │   └── wandb_logger.py           # W&B integration (optional)
│   ├── loss/                          # Loss functions
│   │   ├── __init__.py
│   │   ├── dirichlet.py              # Modified Dirichlet Energy
│   │   ├── gram.py                   # Gram matrix constraints
│   │   └── combined.py               # Combined loss with weighting
│   ├── model/                         # Model architectures
│   │   ├── __init__.py
│   │   ├── basis_net.py              # Eigenfunction network φₖ(x)
│   │   ├── metric_net.py             # Metric network A(x)
│   │   ├── trotter.py                # Trotter fixed-point method
│   │   ├── givens.py                 # Givens rotations
│   │   └── efdo.py                   # Main PIEFS model
│   ├── pretrain/                      # Graph-Laplacian pretraining
│   │   ├── __init__.py
│   │   ├── graphlaplacian.py         # Graph Laplacian eigenmaps
│   │   └── warm_start.py             # Warm-start utilities
│   └── trainer/                       # Training loops
│       ├── __init__.py
│       ├── base_trainer.py           # Base training logic
│       ├── standard_trainer.py       # Standard (off/diag) training
│       └── trotter_trainer.py        # Trotter variant training
├── scripts/
│   ├── gen_figures.py                 # Generate paper figures
│   ├── gen_additional_figures.py      # K-ablation & Gram convergence
│   ├── collect_grid_results.py        # Aggregate hyperparameter sweeps
│   ├── eval_from_checkpoint.py        # Evaluate saved models
│   ├── run_sklearn_baselines.py       # Random Forest & Logistic Regression
│   └── extract_cnn_features.py        # Extract ResNet-18 features from CIFAR-10
├── results/
│   ├── checkpoints/                   # Saved model weights (.pt files)
│   ├── logs/                          # Training logs (metrics.jsonl)
│   ├── figures/                       # Generated visualizations
│   └── tables/                        # Aggregated results tables
└── tests/
    └── test_metrics.py                # Unit tests for metrics
```

## Methodology

### Core Algorithm

PIEFS learns eigenfunctions sequentially:

1. **Warm-up stage** (k=1 only): Cycle through loss terms with weight schedule to escape local minima
2. **For k = 1 to K**:
   - Initialize φₖ(x) and metric A(x) randomly
   - Minimize combined loss: L_mde + w_orth · L_gram + w_class · L_class
   - Stop-gradient on θⱼ (j ≠ k) to prevent interference
   - Dynamic weight dampening: w decreases as loss saturates
3. **Output**: Orthonormal basis Φ = [φ₁, ..., φₖ]

### Loss Function

```
L_total = ‖A(x)∇φₖ‖² + w_orth · ‖Φ^T Φ/N - I‖²_F + w_class · L_class
```

Where:
- **Modified Dirichlet Energy**: ‖A(x)∇φₖ‖² encourages smooth eigenfunctions respecting data geometry
- **Gram penalty**: Forces approximate orthogonality (finite-batch estimation acceptable)
- **Classification loss**: Encourages discriminative features (crossentropy or margin)
- **Dynamic weights**: w_orth and w_class decay exponentially during training

### Metric Parametrization

Three variants:

1. **EFDO-off**: A(x) = I (identity, no scaling)
2. **EFDO-diag**: A(x) = Λ(x) with Λᵢᵢ(x) ∈ [0.1, 10]
3. **EFDO-trotter**: A(x) = Λ(x)·U_Trotter(ω(x)) with Givens rotations via Trotter fixed-point

## Hyperparameters

### Training Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `k` | 3 | 1-32 | Number of eigenfunctions |
| `steps` | 60000 | 10k-200k | Gradient steps per seed |
| `batch_size` | 256 | 32-512 | Training batch size |
| `lr` | 0.001 | 1e-4-1e-2 | Initial learning rate |
| `T_orth` | 0.1 | 0.05-0.2 | Orth weight dampening |
| `T_class` | 0.5 | 0.2-1.0 | Class weight dampening |
| `seed` | 42 | any | Random seed for reproducibility |

### Network Architecture

- **Basis network φₖ**: 3 hidden layers, width 64, ReLU activations
- **Metric network A(x)**: 3 hidden layers, width 64, ReLU activations (outputs diagonal or angles)
- **Classifier**: Logistic regression on [φ₁(x), ..., φₖ(x)]

## Reproducing Results

### Single Run with Fixed Seed

```bash
python train.py \
  --dataset mnist \
  --variant off \
  --k 16 \
  --steps 60000 \
  --batch_size 256 \
  --seed 42
```

This produces:
- `results/checkpoints/mnist_off_k16_seed42.pt`: Model weights
- `results/logs/mnist_off_k16_seed42/metrics.jsonl`: Training metrics (JSON lines format)

### Hyperparameter Sweep (5 seeds)

```bash
for seed in 1 2 3 4 5; do
  python train.py \
    --dataset mnist \
    --variant off \
    --k 16 \
    --steps 60000 \
    --seed $seed
done
```

### Aggregate Results

```bash
python scripts/collect_grid_results.py \
  --log_dir results/logs \
  --output results/tables/grid_summary.csv
```

### Generate Paper Figures

```bash
# Training curves, eigenfunction visualizations, Gram matrix
python scripts/gen_figures.py --log_dir results/logs

# K-ablation and Gram convergence curves
python scripts/gen_additional_figures.py --log_dir results/logs
```

## Evaluation Metrics

### Per-Epoch Metrics (in metrics.jsonl)

```json
{
  "epoch": 0,
  "train_loss": 0.523,
  "train_acc": 0.876,
  "val_acc": 0.855,
  "gram_error": 0.370,
  "grad_norm": 1.234,
  "learning_rate": 0.001
}
```

### Final Evaluation

- **Classification accuracy**: Validation accuracy on held-out test set
- **Gram orthogonality**: ‖Φ^T Φ/N - I‖²_F (lower is better)
- **Eigenfunction smoothness**: Average ‖A(x)∇φₖ‖² across dataset
- **Eigenvalue spectrum**: Implied Rayleigh quotient λₖ ≈ (L_mde / variance)

## Comparison with Baselines

### Classical Baselines
- **Random Forest**: 200 trees, scikit-learn defaults
- **Logistic Regression**: scikit-learn with L2 regularization

### Learning-based Baselines
- **NeuralEF** (Deng et al., ICML 2022): Supervised spectral features via manifold regularization

### Evaluation Protocol
- PIEFS and classical baselines: validation set accuracy
- NeuralEF: test accuracy from published benchmark (not directly comparable)
- All results: mean ± std over 5 random seeds

## Important Notes

### Computational Cost
- **CPU**: ~2 hours per eigenfunction on commodity CPU at quoted mesh resolution
- **GPU**: Few minutes per coordinate on modern accelerator (RTX3090, A100)
- **Bottleneck**: Gradient computation for Givens rotations in Trotter variant

### Finite-Batch Orthogonality
- Gram matrix residual ‖C - I‖_F ≈ 0.370 on MNIST is **expected** and **acceptable**
- Comes from finite-batch estimation of L² inner products
- Larger batch sizes or longer training reduce residual further
- Does not impact classification performance in practice

### Known Limitations
1. EFDO learns task-dependent coordinates (not eigenmodes of fixed operator)
2. Finite-batch Gram penalties only approximate global L²-orthogonality
3. Warm-up stage is critical but can be data-dependent
4. CPU optimization time remains a practical drawback (future work)

## Citation

If you use PIEFS in your research, please cite:

```bibtex
@inproceedings{nazarenko2026piefs,
  title={{Physics-Informed Eigenfunction Features with Learnable Scaling}},
  author={Nazarenko, Varvara},
  booktitle={AI4Physics Workshop, ICML 2026},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub or contact the authors.

---

**Last updated**: May 2026  
**Paper status**: Submitted to AI4Physics Workshop @ ICML 2026  
**Reproducibility**: All results reproducible with provided code and hyperparameters
