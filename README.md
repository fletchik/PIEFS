Neural network method for constructing spectral features for classification tasks.
Replaces the fixed Euclidean Dirichlet energy with a modified energy with a learnable
metric matrix, which allows adapting the smoothness regularizer to the structure of
the data. Feature functions are trained sequentially: each new function is optimized
with an explicit orthogonality penalty to the already constructed functions.

## Installation

```bash
git clone https://github.com/fletchik/PIEFS.git
cd PIEFS

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py run_id=exp01
```

Override config via Hydra:

```bash
python train.py run_id=exp02 dataset=mnist_multiclass model.K=10 model.metric_type=global_low_rank
```

Available datasets: `two_moon`, `circles`, `htru2`, `mnist_binary`, `mnist_multiclass`, `cifar10_binary`, `cifar10_multiclass`.

Available `metric_type`: `off`, `diag`, `lambda_u_trotter`, `global_low_rank`, `local_low_rank`.

## Evaluate

```bash
python scripts/eval_from_checkpoint.py --checkpoint logs/exp01/checkpoint_final.pt --split test
```

## Pretrained checkpoints

Checkpoints for 5 datasets × 5 metric types (seed 0, best validation accuracy) are available in [Releases](https://github.com/fletchik/PIEFS/releases/tag/v1.0).

Download all at once:

```bash
bash scripts/download_checkpoints.sh        # saves to checkpoints/
```

Then evaluate:

```bash
python scripts/eval_from_checkpoint.py \
    --checkpoint checkpoints/htru2_global_low_rank.pt --split test
```

## Results

Validation accuracy (%) averaged over 5 seeds, K=10 eigenfunctions. **Bold** = best per dataset.

| Dataset    | off | diag | conformal | global\_low\_rank | local\_low\_rank |
|------------|-----|------|-----------|-------------------|-----------------|
| Two Moons  | 100.00 | 99.99 | 100.00 | 100.00 | 100.00 |
| Circles    | 81.49 ± 11.92 | 85.56 ± 3.79 | 92.08 ± 3.68 | 95.75 ± 1.85 | **97.55 ± 0.28** |
| HTRU2      | 97.34 | 97.48 | **97.73** | 97.68 | 97.71 |
| MNIST-bin  | **99.79** | 99.78 | 99.79 | 99.75 | 99.75 |
| MNIST-10   | 94.48 | 94.33 | 94.66 | **94.69** | 94.52 |
