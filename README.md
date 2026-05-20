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
