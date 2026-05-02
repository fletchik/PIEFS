"""Standalone evaluation from a saved EFDO checkpoint.

Usage:
    python scripts/eval_from_checkpoint.py --checkpoint logs/exp01/checkpoint_final.pt
    python scripts/eval_from_checkpoint.py --checkpoint logs/exp01/checkpoint_final.pt \
        --dataset two_moon --split test

Loads only model weights (no training state needed), evaluates on any dataset,
and reports accuracy, ROC-AUC, gram_error, per-class accuracy (multiclass),
and full orthogonality table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_model_from_ckpt(ckpt_path: str, device: str) -> tuple:
    """Load model from checkpoint. Returns (model, config)."""
    ckpt = torch.load(ckpt_path, map_location=device)
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
    model.set_active_k(K)
    return model, cfg


def _build_dataset(dataset_name: str, cfg: dict, split: str, seed: int = 42):
    """Reconstruct dataset from config."""
    ds_cfg = cfg.get('dataset', {})
    name = dataset_name or ds_cfg.get('name', 'two_moon')

    if name in ('two_moon', 'circles'):
        from src.dataset.sklearn_cls import SklearnDataset
        return SklearnDataset(
            name=name,
            split=split,
            n_samples=ds_cfg.get('n_samples', 10000),
            noise=ds_cfg.get('noise', 0.1),
            train_fraction=ds_cfg.get('train_fraction', 0.7),
            standardize=ds_cfg.get('standardize', True),
            seed=seed,
        )
    if name in ('mnist', 'cifar10'):
        from src.dataset.torchvision_flat import TorchvisionFlatDataset
        return TorchvisionFlatDataset(
            name=name,
            split=split,
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
def evaluate(model, loader, device: str) -> dict:
    """Run inference and compute all metrics."""
    model.eval()
    all_probs, all_labels, phi_all = [], [], []

    K = model.K
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['labels'].to(device)

        # Build full φ matrix (all K functions).
        fns = model.basis_set.functions
        phi = torch.cat([fn.predict(x) for fn in fns], dim=1)  # (B, K)
        phi_all.append(phi.cpu())

        # Head inference (set active_k=K so head gets full phi).
        phi_full = torch.zeros(x.shape[0], K, device=device)
        phi_full[:, :K] = phi
        head_out = model.head(phi_full, y)
        probs = head_out['probs']

        if probs.dim() == 1:
            all_probs.extend(probs.cpu().tolist())
        else:
            all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(y.cpu().tolist())

    # Gram matrix on full eval set.
    Phi = torch.cat(phi_all, dim=0)  # (N, K)
    N = Phi.shape[0]
    C = (Phi.T @ Phi) / N
    I_K = torch.eye(K)
    diff = C - I_K
    gram_error_final = torch.norm(diff, p='fro').item()
    off_diag_mask = ~torch.eye(K, dtype=torch.bool)
    gram_error_offdiag = C[off_diag_mask].abs().max().item()
    gram_error_diag = (torch.diag(C) - 1).abs().norm().item()

    labels_arr = np.array(all_labels)
    metrics = {
        'gram_error_final': gram_error_final,
        'gram_error_offdiag': gram_error_offdiag,
        'gram_error_diag': gram_error_diag,
        'N_eval': N,
    }

    try:
        from sklearn.metrics import accuracy_score, roc_auc_score

        if isinstance(all_probs[0], list):
            probs_arr = np.array(all_probs)
            preds = probs_arr.argmax(axis=1)
            metrics['accuracy'] = float(accuracy_score(labels_arr, preds))
            try:
                metrics['roc_auc'] = float(
                    roc_auc_score(labels_arr, probs_arr, multi_class='ovr', average='macro')
                )
            except Exception:
                pass
            # Per-class accuracy.
            classes = np.unique(labels_arr)
            for c in classes:
                mask = labels_arr == c
                metrics[f'acc_class_{c}'] = float(accuracy_score(labels_arr[mask], preds[mask]))
        else:
            probs_arr = np.array(all_probs)
            preds = (probs_arr >= 0.5).astype(int)
            metrics['accuracy'] = float(accuracy_score(labels_arr, preds))
            try:
                metrics['roc_auc'] = float(roc_auc_score(labels_arr, probs_arr))
            except Exception:
                pass
    except Exception as e:
        print(f'Warning: could not compute classification metrics: {e}')

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate EFDO model from checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--dataset', default=None, help='Dataset name override')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device != 'auto':
        device = args.device

    print(f'Loading checkpoint: {args.checkpoint}')
    model, cfg = _load_model_from_ckpt(args.checkpoint, device)
    print(f'Model K={model.K}, metric={cfg.get("model",{}).get("metric_type","?")}')

    ds = _build_dataset(args.dataset, cfg, args.split, args.seed)
    from src.dataset.collate import collate_fn as CollateFn
    from src.dataset.utils import make_loader
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=CollateFn(use_label=True))

    print(f'\nEvaluating on {args.split} split ({len(ds)} samples)...')
    results = evaluate(model, loader, device)

    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')
    print('=' * 50)

    # Also report from checkpoint if available.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'eigenvalue_history' in ckpt and ckpt['eigenvalue_history']:
        print('\nEigenvalue history (Dirichlet energies):')
        for i, e in enumerate(ckpt['eigenvalue_history']):
            print(f'  φ_{i+1}: {e:.6f}')
        ordering = all(
            ckpt['eigenvalue_history'][i] <= ckpt['eigenvalue_history'][i+1] * 1.05
            for i in range(len(ckpt['eigenvalue_history']) - 1)
        )
        print(f'  ordering_satisfied: {ordering}')

    if 'wall_time_per_function' in ckpt and ckpt['wall_time_per_function']:
        print('\nWall time per function:')
        for i, t in enumerate(ckpt['wall_time_per_function']):
            print(f'  φ_{i+1}: {t:.1f}s')
        print(f'  total: {sum(ckpt["wall_time_per_function"]):.1f}s')


if __name__ == '__main__':
    main()
