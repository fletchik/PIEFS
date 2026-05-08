"""Evaluate sklearn baselines on all EFDO benchmark datasets.

Baselines
---------
Raw features (no embedding):
  - LogisticRegression (linear baseline)
  - RandomForest (nonlinear, no embedding)

Dimensionality-reduction / embedding methods (Tier A):
  - PCA                           (linear, det reduction)
  - KernelPCA (rbf, poly, cosine) (nonlinear kernel methods)
  - SpectralEmbedding             (graph-Laplacian eigenvectors — closest to EFDO)
  - TruncatedSVD                  (linear, sparse-friendly)

Classifier applied on top of embeddings: LogisticRegression + L-BFGS.

Usage
-----
    .venv/bin/python3 scripts/run_sklearn_baselines.py
    .venv/bin/python3 scripts/run_sklearn_baselines.py --datasets two_moon circles
    .venv/bin/python3 scripts/run_sklearn_baselines.py --n_components 6 --seeds 0 1 2 3 4

Output
------
  results/sklearn_baselines.json   — machine-readable (all seeds)
  results/sklearn_baselines.txt    — human-readable table (mean ± std)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── allow 'import src...' from project root ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Dataset loaders (reuse project loaders for consistency)
# ---------------------------------------------------------------------------

def _load_htru2(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.dataset.htru2 import HTRU2Dataset
    kw = dict(root='data/htru2', train_fraction=0.7, standardize=True)
    tr = HTRU2Dataset(split='train', **kw)
    te = HTRU2Dataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    lk = 'labels' if 'labels' in tr[0] else 'label'
    y_tr = np.array([s[lk].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s[lk].item() for s in te])
    return X_tr, y_tr, X_te, y_te


def _load_sklearn_2d(
    name: str,
    seed: int,
    n_samples: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.dataset.sklearn_cls import SklearnDataset
    noise = 0.1 if name == 'two_moon' else 0.05
    kw = dict(name=name, n_samples=n_samples, noise=noise,
              train_fraction=0.7, standardize=True)
    tr = SklearnDataset(split='train', **kw)
    te = SklearnDataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    lk = 'labels' if 'labels' in tr[0] else 'label'
    y_tr = np.array([s[lk].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s[lk].item() for s in te])
    return X_tr, y_tr, X_te, y_te


def _load_mnist(seed: int, task: str = 'multiclass') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.dataset.torchvision_flat import TorchvisionFlatDataset
    kw = dict(name='mnist', root='data/mnist', task=task,
              val_fraction=0.1, standardize=True)
    tr = TorchvisionFlatDataset(split='train', **kw)
    te = TorchvisionFlatDataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    lk = 'labels' if 'labels' in tr[0] else 'label'
    y_tr = np.array([s[lk].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s[lk].item() for s in te])
    return X_tr, y_tr, X_te, y_te


def _load_fashion_mnist(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.dataset.torchvision_flat import TorchvisionFlatDataset
    kw = dict(name='fashion_mnist', root='data/fashion_mnist', task='multiclass',
              val_fraction=0.1, standardize=True)
    tr = TorchvisionFlatDataset(split='train', **kw)
    te = TorchvisionFlatDataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    lk = 'labels' if 'labels' in tr[0] else 'label'
    y_tr = np.array([s[lk].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s[lk].item() for s in te])
    return X_tr, y_tr, X_te, y_te


def _load_cifar10_features(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted ResNet-18 CIFAR-10 features (512-dim).

    Requires that scripts/extract_cnn_features.py has been run first.
    """
    from src.dataset.pretrained_features import PretrainedFeaturesDataset
    kw = dict(root='data/cifar10_features', val_fraction=0.1, standardize=True)
    tr = PretrainedFeaturesDataset(split='train', **kw)
    te = PretrainedFeaturesDataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    y_tr = np.array([s['label'].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s['label'].item() for s in te])
    return X_tr, y_tr, X_te, y_te


def _load_spotify(seed: int, task: str = 'multiclass') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Spotify songs dataset (requires data/spotify/genres_v2.csv)."""
    from src.dataset.spotify import SpotifyDataset
    kw = dict(root='data/spotify', task=task, train_fraction=0.7,
              val_fraction=0.1, standardize=True, seed=seed)
    tr = SpotifyDataset(split='train', **kw)
    te = SpotifyDataset(split='test',  **kw)
    X_tr = np.array([s['x'].numpy() for s in tr])
    y_tr = np.array([s['label'].item() for s in tr])
    X_te = np.array([s['x'].numpy() for s in te])
    y_te = np.array([s['label'].item() for s in te])
    return X_tr, y_tr, X_te, y_te


DATASET_LOADERS: dict[str, Any] = {
    'two_moon':          lambda seed: _load_sklearn_2d('two_moon', seed),
    'circles':           lambda seed: _load_sklearn_2d('circles',  seed),
    'htru2':             lambda seed: _load_htru2(seed),
    'mnist_mc':          lambda seed: _load_mnist(seed, task='multiclass'),
    'fashion_mnist':     lambda seed: _load_fashion_mnist(seed),
    'cifar10_features':  lambda seed: _load_cifar10_features(seed),
    'spotify_mc':        lambda seed: _load_spotify(seed, task='multiclass'),
    'spotify_bin':       lambda seed: _load_spotify(seed, task='binary'),
}

# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _build_pipelines(n_components: int, n_features: int | None = None) -> dict[str, Any]:
    """Return a dict of {name: sklearn Pipeline}.

    n_features: clamps n_components so PCA/KPCA don't exceed the feature dim.
    """
    from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import SpectralEmbedding
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if n_features is not None:
        n_components = min(n_components, n_features)

    lr = dict(
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        C=1.0,
    )

    pipelines: dict[str, Any] = {
        # ── Raw features ────────────────────────────────────────────────────
        'LogReg_raw': Pipeline([
            ('lr', LogisticRegression(**lr)),
        ]),
        # ── PCA → LogReg ────────────────────────────────────────────────────
        'PCA+LogReg': Pipeline([
            ('pca',  PCA(n_components=n_components, random_state=42)),
            ('lr',   LogisticRegression(**lr)),
        ]),
        # ── KernelPCA (RBF) → LogReg ────────────────────────────────────────
        'KPCA_rbf+LogReg': Pipeline([
            ('kpca', KernelPCA(n_components=n_components, kernel='rbf', random_state=42)),
            ('lr',   LogisticRegression(**lr)),
        ]),
        # ── KernelPCA (poly) → LogReg ────────────────────────────────────────
        'KPCA_poly+LogReg': Pipeline([
            ('kpca', KernelPCA(n_components=n_components, kernel='poly', degree=3,
                               random_state=42)),
            ('lr',   LogisticRegression(**lr)),
        ]),
        # ── KernelPCA (cosine) → LogReg ─────────────────────────────────────
        'KPCA_cos+LogReg': Pipeline([
            ('kpca', KernelPCA(n_components=n_components, kernel='cosine', random_state=42)),
            ('lr',   LogisticRegression(**lr)),
        ]),
        # ── TruncatedSVD → LogReg (handles high-d like MNIST efficiently) ───
        'TruncSVD+LogReg': Pipeline([
            ('svd', TruncatedSVD(n_components=n_components, random_state=42)),
            ('lr',  LogisticRegression(**lr)),
        ]),
    }

    # SpectralEmbedding is expensive for large N → skip for MNIST-scale.
    # Add a cheap version limited to n_samples=3000 for speed.
    pipelines['SpectralEmb+LogReg'] = 'spectral'   # placeholder, handled separately

    return pipelines


def _run_spectral(X_tr, y_tr, X_te, y_te, n_components: int) -> float:
    """SpectralEmbedding fitted on train only (no transform for test — transductive).

    Because sklearn's SpectralEmbedding is transductive, we use a workaround:
    embed train+test jointly, then split.  Limit to 5000 samples for speed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import SpectralEmbedding

    MAX_N = 5000
    N_tr = len(X_tr)
    N_te = len(X_te)
    if N_tr + N_te > MAX_N:
        idx_tr = np.random.choice(N_tr, min(N_tr, MAX_N * N_tr // (N_tr + N_te)), replace=False)
        idx_te = np.random.choice(N_te, min(N_te, MAX_N * N_te // (N_tr + N_te)), replace=False)
        X_all = np.vstack([X_tr[idx_tr], X_te[idx_te]])
        y_tr_sub, y_te_sub = y_tr[idx_tr], y_te[idx_te]
        n_tr_sub = len(idx_tr)
    else:
        X_all = np.vstack([X_tr, X_te])
        y_tr_sub, y_te_sub = y_tr, y_te
        n_tr_sub = N_tr

    emb = SpectralEmbedding(n_components=n_components, random_state=42)
    Z_all = emb.fit_transform(X_all)
    Z_tr, Z_te = Z_all[:n_tr_sub], Z_all[n_tr_sub:]

    clf = LogisticRegression(solver='lbfgs',
                             max_iter=1000, random_state=42)
    clf.fit(Z_tr, y_tr_sub)
    return float(clf.score(Z_te, y_te_sub))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    datasets: list[str],
    seeds: list[int],
    n_components: int,
) -> dict[str, dict[str, list[float]]]:
    """Run all pipelines × datasets × seeds.

    Returns:
        results[dataset][method] = list of per-seed accuracies
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    results: dict[str, dict[str, list[float]]] = {}

    for ds_name in datasets:
        print(f'\n{"="*60}')
        print(f'Dataset: {ds_name}')
        print(f'{"="*60}')
        loader = DATASET_LOADERS[ds_name]
        results[ds_name] = {}

        # Add RandomForest separately (sklearn pipelines use same interface)
        rf_pipe = Pipeline([
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, random_state=42,
                n_jobs=-1,
            )),
        ])

        for seed in seeds:
            np.random.seed(seed)
            print(f'  seed={seed} ...', end=' ', flush=True)

            X_tr, y_tr, X_te, y_te = loader(seed)
            n_features = X_tr.shape[1]
            print(f'train={len(X_tr)}, test={len(X_te)}, d={n_features}', end='  ')

            # Rebuild pipelines with feature-clipped n_components
            pipelines = _build_pipelines(n_components, n_features=n_features)

            # RandomForest
            rf_pipe.set_params(rf__random_state=seed)
            rf_pipe.fit(X_tr, y_tr)
            acc = float(rf_pipe.score(X_te, y_te))
            results[ds_name].setdefault('RandomForest', []).append(acc)
            print(f'RF={acc:.4f}', end='  ')

            for name, pipe in pipelines.items():
                if pipe == 'spectral':
                    acc = _run_spectral(X_tr, y_tr, X_te, y_te, n_components)
                    results[ds_name].setdefault(name, []).append(acc)
                    print(f'Spectral={acc:.4f}', end='  ')
                    continue
                try:
                    pipe.fit(X_tr, y_tr)
                    acc = float(pipe.score(X_te, y_te))
                except Exception as exc:
                    print(f'  WARNING: {name} failed: {exc}')
                    acc = float('nan')
                results[ds_name].setdefault(name, []).append(acc)
            print()

    return results


def print_table(
    results: dict[str, dict[str, list[float]]],
    datasets: list[str],
) -> str:
    """Format results as a markdown-style table."""
    # Collect all method names preserving order
    methods: list[str] = []
    for ds in datasets:
        for m in results.get(ds, {}):
            if m not in methods:
                methods.append(m)

    # Header
    col_w = 24
    lines = []
    header = f"{'Method':<{col_w}}" + ''.join(f'{d[:14]:>18}' for d in datasets)
    lines.append(header)
    lines.append('-' * len(header))

    for method in methods:
        row = f'{method:<{col_w}}'
        for ds in datasets:
            accs = results.get(ds, {}).get(method, [])
            if accs:
                valid = [a for a in accs if not (a != a)]  # filter NaN
                if valid:
                    row += f'{np.mean(valid)*100:>12.2f}±{np.std(valid)*100:>4.2f}'
                else:
                    row += f'{"N/A":>18}'
            else:
                row += f'{"—":>18}'
        lines.append(row)

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Run sklearn baselines for EFDO comparison.')
    parser.add_argument(
        '--datasets', nargs='+',
        default=['two_moon', 'circles', 'htru2'],
        choices=list(DATASET_LOADERS.keys()),
        help='Which datasets to evaluate (default: two_moon circles htru2)',
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
        help='Random seeds (default: 0 1 2 3 4)',
    )
    parser.add_argument(
        '--n_components', type=int, default=6,
        help='Number of components / embedding dim (default: 6, matches EFDO K)',
    )
    parser.add_argument(
        '--out_dir', default='results',
        help='Output directory for JSON + txt (default: results/)',
    )
    args = parser.parse_args()

    results = evaluate_all(args.datasets, args.seeds, args.n_components)

    # ── Print table ─────────────────────────────────────────────────────────
    table = print_table(results, args.datasets)
    print('\n\nRESULTS (accuracy % mean ± std over seeds)\n')
    print(table)

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'sklearn_baselines.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {json_path}')

    txt_path = out_dir / 'sklearn_baselines.txt'
    with open(txt_path, 'w') as f:
        f.write('sklearn baselines  (accuracy % mean ± std, 5 seeds)\n\n')
        f.write(table + '\n')
    print(f'Saved: {txt_path}')


if __name__ == '__main__':
    main()
