"""Collect and aggregate results from run_main_grid.sh runs.

Reads logs/grid_*/metrics.jsonl, picks the best val_acc per run,
then prints a LaTeX + markdown table of  mean ± std over 5 seeds.

Usage
-----
    .venv/bin/python3 scripts/collect_grid_results.py
    .venv/bin/python3 scripts/collect_grid_results.py --log_dir logs --out_dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> dict | None:
    """Return a summary dict for one grid run, or None if no metrics found."""
    jsonl = run_dir / 'metrics.jsonl'
    if not jsonl.exists():
        return None

    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not rows:
        return None

    # Best val accuracy across all logged steps.
    best_acc = max((r.get('val_acc', 0.0) or 0.0) for r in rows)
    last = rows[-1]

    return {
        'best_val_acc': best_acc,
        'final_gram_error': last.get('gram_error', float('nan')),
        'final_loss': last.get('loss', float('nan')),
        'wall_sec': last.get('wall', float('nan')),
        'n_steps': last.get('step', float('nan')),
    }


KNOWN_DATASETS = [
    'two_moon', 'circles', 'htru2', 'mnist_mc', 'mnist_binary',
    'lissajous', 'cifar10_binary',
]
KNOWN_METRICS = [
    'off', 'diag', 'lambda_u_sparse', 'lambda_u_pinn', 'lambda_u_trotter',
]


def parse_run_id(run_id: str) -> tuple[str, str, int] | None:
    """Parse 'grid_<dataset>_<metric>_s<seed>' → (dataset, metric, seed).

    Uses known dataset / metric name lists to resolve ambiguity when either
    name contains underscores (e.g. 'lambda_u_trotter', 'mnist_mc').
    """
    if not run_id.startswith('grid_'):
        return None
    rest = run_id[len('grid_'):]

    # Extract seed from tail: '_s<digits>'
    import re
    m = re.search(r'_s(\d+)$', rest)
    if not m:
        return None
    seed = int(m.group(1))
    ds_metric = rest[: m.start()]   # everything before _s<seed>

    # Try to match known dataset + metric combos
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if ds_metric.startswith(ds + '_'):
            metric = ds_metric[len(ds) + 1:]
            if metric in KNOWN_METRICS:
                return ds, metric, seed

    return None


def collect(log_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Returns results[dataset][metric] = list of best_val_acc per seed."""
    results: dict[str, dict[str, list[float]]] = {}

    for run_dir in sorted(log_dir.iterdir()):
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        ds, metric, seed = parsed
        summary = load_run(run_dir)
        if summary is None:
            continue
        results.setdefault(ds, {}).setdefault(metric, []).append(summary['best_val_acc'])

    return results


def markdown_table(
    results: dict[str, dict[str, list[float]]],
    datasets: list[str],
    metrics: list[str],
) -> str:
    """Produce a markdown table: rows=datasets, cols=metrics."""
    col_w = 26
    lines = []

    header = f"{'Dataset':<20}" + ''.join(f'{m:>{col_w}}' for m in metrics)
    lines.append(header)
    lines.append('-' * len(header))

    for ds in datasets:
        row = f'{ds:<20}'
        for metric in metrics:
            accs = results.get(ds, {}).get(metric, [])
            if accs:
                row += f'{np.mean(accs)*100:>14.2f} ± {np.std(accs)*100:>4.2f}%'
            else:
                row += f'{"—":>{col_w}}'
        lines.append(row)

    return '\n'.join(lines)


def latex_table(
    results: dict[str, dict[str, list[float]]],
    datasets: list[str],
    metrics: list[str],
    metric_labels: dict[str, str] | None = None,
) -> str:
    """Produce a LaTeX tabular environment."""
    labels = metric_labels or {m: m for m in metrics}
    n_cols = 1 + len(metrics)
    col_spec = 'l' + 'c' * len(metrics)
    lines = [
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{EFDO test accuracy (\\%) — mean $\\pm$ std over 5 seeds.}',
        '\\label{tab:main_grid}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
    ]

    header = 'Dataset & ' + ' & '.join(labels[m] for m in metrics) + ' \\\\'
    lines.append(header)
    lines.append('\\midrule')

    for ds in datasets:
        cells = [ds.replace('_', '\\_')]
        for metric in metrics:
            accs = results.get(ds, {}).get(metric, [])
            if accs:
                cells.append(f'{np.mean(accs)*100:.2f} $\\pm$ {np.std(accs)*100:.2f}')
            else:
                cells.append('—')
        lines.append(' & '.join(cells) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--out_dir', default='results')
    parser.add_argument(
        '--datasets', nargs='+',
        default=['two_moon', 'circles', 'htru2', 'mnist_mc'],
    )
    parser.add_argument(
        '--metrics', nargs='+',
        default=['off', 'diag', 'lambda_u_trotter'],
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect(log_dir)

    if not results:
        print(f'No grid runs found in {log_dir}/grid_*/')
        return

    print('EFDO Main Grid Results (accuracy % mean ± std, 5 seeds)\n')
    md = markdown_table(results, args.datasets, args.metrics)
    print(md)

    metric_labels = {
        'off': 'No metric',
        'diag': 'Diag $\\Lambda$',
        'lambda_u_trotter': 'Trotter $U\\Lambda$',
    }
    tex = latex_table(results, args.datasets, args.metrics, metric_labels)

    md_path = out_dir / 'grid_results.md'
    tex_path = out_dir / 'grid_results.tex'
    json_path = out_dir / 'grid_results.json'

    with open(md_path, 'w') as f:
        f.write('# EFDO Main Grid Results\n\n')
        f.write('Accuracy % (mean ± std, 5 seeds, test split)\n\n')
        f.write(md + '\n')

    with open(tex_path, 'w') as f:
        f.write(tex + '\n')

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nSaved: {md_path}, {tex_path}, {json_path}')


if __name__ == '__main__':
    main()
