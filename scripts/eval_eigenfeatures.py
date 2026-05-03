"""Evaluate eigenfeatures via sklearn classifiers (paper Table 1 protocol).

Protocol from the paper:
  1. Load trained EFDO model (eigenfunctions φ₁...φ_K)
  2. Compute eigenfeatures: x → [φ₁(x), ..., φ_K(x)]  for train and test
  3. Train sklearn Random Forest (RF) and Logistic Regression (LR) on eigenfeatures
  4. Compare accuracy with RF/LR trained on raw features
  5. Also test with only 5% of labels (semi-supervised scenario)

Usage:
    python scripts/eval_eigenfeatures.py --checkpoint logs/groupE_htru2_off_s42/checkpoint_final.pt
    python scripts/eval_eigenfeatures.py --checkpoint logs/groupD_mnist_supervised_off_s42/checkpoint_final.pt \
        --label-fractions 0.05 0.1 0.5 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_model_from_ckpt(ckpt_path: str, device: str):
    """Load model from checkpoint. Returns (model, config)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    ds_cfg = cfg.get('dataset', {})

    K = model_cfg.get('K', 6)
    input_dim = ds_cfg.get('input_dim', None)
    if input_dim is None:
        raise ValueError('checkpoint config missing dataset.input_dim')
    hidden_dims = list(model_cfg.get('hidden_dims', [64, 64, 64]))
    metric_type = model_cfg.get('metric_type', 'off')
    metric_hidden_dims = list(model_cfg.get('metric_hidden_dims', [64, 64]))
    task = model_cfg.get('task', 'binary')
    num_classes = ds_cfg.get('num_classes', 2)

    from src.model.basis.basis_set import BasisSet
    from src.model.metric.metric_net import build_metric
    from src.model.spectral_model import BinaryHead, MulticlassHead, SpectralModel

    basis_set = BasisSet(K=K, input_dim=input_dim, hidden_dims=hidden_dims)
    metric = build_metric(metric_type, input_dim, metric_hidden_dims)
    if metric is not None:
        metric = metric.to(device)

    if task == 'binary':
        head = BinaryHead(K)
    else:
        head = MulticlassHead(K, num_classes)

    model = SpectralModel(basis_set, metric, head).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def _build_dataset(ds_cfg: dict, split: str, seed: int = 42):
    """Reconstruct dataset from config."""
    name = ds_cfg.get('name', 'two_moon')

    if name in ('two_moon', 'circles'):
        from src.dataset.sklearn_cls import SklearnDataset
        return SklearnDataset(
            name=name, split=split,
            n_samples=ds_cfg.get('n_samples', 10000),
            noise=ds_cfg.get('noise', 0.1),
            train_fraction=ds_cfg.get('train_fraction', 0.7),
            standardize=ds_cfg.get('standardize', True),
            seed=seed,
        )
    if name in ('mnist', 'cifar10'):
        from src.dataset.torchvision_flat import TorchvisionFlatDataset
        return TorchvisionFlatDataset(
            name=name, split=split,
            root=ds_cfg.get('root', f'data/{name}'),
            task=ds_cfg.get('task', 'multiclass'),
            binary_classes=tuple(ds_cfg.get('binary_classes', [0, 1])),
            val_fraction=ds_cfg.get('val_fraction', 0.1),
            standardize=ds_cfg.get('standardize', True),
            seed=seed,
        )
    if name == 'htru2':
        from src.dataset.htru2 import HTRU2Dataset
        return HTRU2Dataset(
            root=ds_cfg.get('root', 'data/htru2'),
            split=split,
            train_fraction=ds_cfg.get('train_fraction', 0.7),
            standardize=ds_cfg.get('standardize', True),
            seed=seed,
        )
    raise ValueError(f'Unknown dataset: {name}')


@torch.no_grad()
def extract_eigenfeatures(model, dataset, device: str, batch_size: int = 512):
    """Extract eigenfeatures [φ₁(x), ..., φ_K(x)] for all samples.

    Returns:
        features: (N, K) numpy array of eigenfeature values
        labels: (N,) numpy array of labels
        raw_features: (N, d) numpy array of original input features
    """
    from src.dataset.collate import collate_fn as CollateFn
    from src.dataset.utils import make_loader

    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=CollateFn(use_label=True),
    )

    phi_list, label_list, raw_list = [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['labels']

        # Compute all K eigenfunctions
        fns = model.basis_set.functions
        phi = torch.cat([fn.predict(x) for fn in fns], dim=1)  # (B, K)

        phi_list.append(phi.cpu().numpy())
        label_list.append(y.numpy())
        raw_list.append(x.cpu().numpy())

    features = np.concatenate(phi_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    raw_features = np.concatenate(raw_list, axis=0)
    return features, labels, raw_features


def evaluate_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_fraction: float = 1.0,
    seed: int = 42,
) -> dict[str, float]:
    """Train RF and LR on features, report accuracy.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        label_fraction: Fraction of training labels to use (for semi-supervised)
        seed: Random seed

    Returns:
        Dict with rf_accuracy, lr_accuracy, and optionally roc_auc values
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    rng = np.random.RandomState(seed)

    # Subsample labels if fraction < 1
    n_train = len(y_train)
    if label_fraction < 1.0:
        n_use = max(2, int(n_train * label_fraction))
        idx = rng.choice(n_train, size=n_use, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train_sub, y_train_sub)
    rf_preds = rf.predict(X_test)
    results['rf_accuracy'] = float(accuracy_score(y_test, rf_preds))
    try:
        rf_proba = rf.predict_proba(X_test)
        if rf_proba.shape[1] == 2:
            results['rf_roc_auc'] = float(roc_auc_score(y_test, rf_proba[:, 1]))
        else:
            results['rf_roc_auc'] = float(
                roc_auc_score(y_test, rf_proba, multi_class='ovr', average='macro')
            )
    except Exception:
        pass

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    lr.fit(X_train_sub, y_train_sub)
    lr_preds = lr.predict(X_test)
    results['lr_accuracy'] = float(accuracy_score(y_test, lr_preds))
    try:
        lr_proba = lr.predict_proba(X_test)
        if lr_proba.shape[1] == 2:
            results['lr_roc_auc'] = float(roc_auc_score(y_test, lr_proba[:, 1]))
        else:
            results['lr_roc_auc'] = float(
                roc_auc_score(y_test, lr_proba, multi_class='ovr', average='macro')
            )
    except Exception:
        pass

    return results


def compute_gram_error(features: np.ndarray) -> dict[str, float]:
    """Compute orthonormality metrics for eigenfeatures."""
    N, K = features.shape
    C = (features.T @ features) / N  # (K, K)
    I_K = np.eye(K)
    diff = C - I_K
    gram_error = float(np.linalg.norm(diff, 'fro'))
    off_diag_mask = ~np.eye(K, dtype=bool)
    max_offdiag = float(np.abs(C[off_diag_mask]).max())
    diag_error = float(np.linalg.norm(np.diag(C) - 1))
    return {
        'gram_error_fro': gram_error,
        'max_offdiag_overlap': max_offdiag,
        'diag_norm_error': diag_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate eigenfeatures via sklearn (paper Table 1 protocol)'
    )
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--label-fractions', nargs='+', type=float,
                        default=[1.0, 0.05],
                        help='Fraction of labels to use (default: 1.0 and 0.05)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default=None, help='Save results to file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device != 'auto':
        device = args.device

    # Load model
    print(f'Loading checkpoint: {args.checkpoint}')
    model, cfg = _load_model_from_ckpt(args.checkpoint, device)
    ds_cfg = cfg.get('dataset', {})
    ds_name = ds_cfg.get('name', '?')
    K = model.K
    metric_type = cfg.get('model', {}).get('metric_type', '?')
    print(f'Dataset: {ds_name}, K={K}, metric_type={metric_type}')

    # Build datasets
    print('Loading datasets...')
    train_ds = _build_dataset(ds_cfg, 'train', args.seed)
    val_ds = _build_dataset(ds_cfg, 'val', args.seed)

    # Extract eigenfeatures
    print('Extracting eigenfeatures (train)...')
    phi_train, y_train, raw_train = extract_eigenfeatures(model, train_ds, device, args.batch_size)
    print(f'  Train: {phi_train.shape[0]} samples, {phi_train.shape[1]} features')

    print('Extracting eigenfeatures (val)...')
    phi_val, y_val, raw_val = extract_eigenfeatures(model, val_ds, device, args.batch_size)
    print(f'  Val: {phi_val.shape[0]} samples, {phi_val.shape[1]} features')

    # Gram error on train set
    gram = compute_gram_error(phi_train)
    print(f'\nOrthonormality (train): gram_error={gram["gram_error_fro"]:.4f}, '
          f'max_offdiag={gram["max_offdiag_overlap"]:.4f}, '
          f'diag_error={gram["diag_norm_error"]:.4f}')

    # Table 1 evaluation
    print('\n' + '=' * 80)
    print('TABLE 1 RESULTS (paper protocol: eigenfeatures → sklearn RF/LR)')
    print('=' * 80)

    header = f'{"Features":>20} {"Label%":>8} {"RF acc":>10} {"LR acc":>10} {"RF AUC":>10} {"LR AUC":>10}'
    print(header)
    print('-' * len(header))

    all_results = {}
    for frac in args.label_fractions:
        frac_key = f'{frac:.0%}' if frac < 1 else '100%'

        # Eigenfeatures
        res_eigen = evaluate_sklearn(phi_train, y_train, phi_val, y_val,
                                     label_fraction=frac, seed=args.seed)
        rf_a = res_eigen.get('rf_accuracy', float('nan'))
        lr_a = res_eigen.get('lr_accuracy', float('nan'))
        rf_auc = res_eigen.get('rf_roc_auc', float('nan'))
        lr_auc = res_eigen.get('lr_roc_auc', float('nan'))
        print(f'{"Eigenfeatures":>20} {frac_key:>8} {rf_a:>10.4f} {lr_a:>10.4f} {rf_auc:>10.4f} {lr_auc:>10.4f}')
        all_results[f'eigen_{frac_key}'] = res_eigen

        # Raw features (baseline)
        res_raw = evaluate_sklearn(raw_train, y_train, raw_val, y_val,
                                   label_fraction=frac, seed=args.seed)
        rf_a = res_raw.get('rf_accuracy', float('nan'))
        lr_a = res_raw.get('lr_accuracy', float('nan'))
        rf_auc = res_raw.get('rf_roc_auc', float('nan'))
        lr_auc = res_raw.get('lr_roc_auc', float('nan'))
        print(f'{"Raw features":>20} {frac_key:>8} {rf_a:>10.4f} {lr_a:>10.4f} {rf_auc:>10.4f} {lr_auc:>10.4f}')
        all_results[f'raw_{frac_key}'] = res_raw

    print('=' * 80)

    # Save results
    if args.output:
        import json
        output = {
            'checkpoint': args.checkpoint,
            'dataset': ds_name,
            'K': K,
            'metric_type': metric_type,
            'gram_error': gram,
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to {args.output}')

    # Also print eigenvalue history from checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'eigenvalue_history' in ckpt and ckpt['eigenvalue_history']:
        print('\nDirichlet energies (eigenvalue proxies):')
        for i, e in enumerate(ckpt['eigenvalue_history']):
            print(f'  φ_{i+1}: {e:.6f}')

    if 'wall_time_per_function' in ckpt and ckpt['wall_time_per_function']:
        total = sum(ckpt['wall_time_per_function'])
        print(f'\nTotal training time: {total:.0f}s ({total/60:.1f}min)')


if __name__ == '__main__':
    main()
