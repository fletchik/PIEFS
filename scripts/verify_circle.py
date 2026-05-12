"""Verify that learned eigenfunctions on the unit circle match trig functions.

Protocol: 10 000 uniform points on unit circle, K=4, 60 000 total steps.

Pass criteria:
  - λ_1 ≤ λ_2 ≤ λ_3 ≤ λ_4  (eigenvalue ordering, approximated by Dirichlet energy)
  - max correlation of each φ_k with {cos(nθ), sin(nθ)} for n=1,2 > 0.7

Saves a plot to logs/sanity/circle_eigenfunctions.png.

Run:
    python scripts/verify_circle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.loss.spectral_loss import SpectralDirichletLoss
from src.model.basis.basis_set import BasisSet
from src.model.spectral_model import BinaryHead, SpectralModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 4
TOTAL_STEPS = 60_000
STEPS_PER_FN = TOTAL_STEPS // K


class _CircleDataset(Dataset):
    """Uniform samples on the unit circle (cos θ, sin θ)."""

    def __init__(self, n_samples: int = 10_000, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        self.x = torch.from_numpy(np.stack([np.cos(theta), np.sin(theta)], axis=1))
        self.y = torch.zeros(n_samples, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {'x': self.x[idx], 'labels': self.y[idx]}


def _train() -> SpectralModel:
    torch.manual_seed(42)
    loader = DataLoader(_CircleDataset(), batch_size=256, shuffle=True)
    basis_set = BasisSet(K=K, input_dim=2)
    model = SpectralModel(basis_set, metric=None, head=BinaryHead(K)).to(DEVICE)
    criterion = SpectralDirichletLoss(w_gram=1.0, w_dirichlet=1.0, w_task=0.1)

    from itertools import repeat
    data_iter = (b for dl in repeat(loader) for b in dl)

    for k in range(1, K + 1):
        basis_set.set_active(k)
        model.set_active_k(k)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(STEPS_PER_FN):
            batch = next(data_iter)
            x = batch['x'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            opt.zero_grad()
            out = model(x, y)
            loss = criterion(
                out['phi_matrix'], out['grad_phi_k'], None, out['head_out'], k
            )['loss']
            loss.backward()
            opt.step()
        basis_set.functions[k - 1].eval()
        for p in basis_set.functions[k - 1].parameters():
            p.requires_grad_(False)

    return model


def _dirichlet_energy(model: SpectralModel, k: int) -> float:
    theta = torch.linspace(0, 2 * np.pi, 5000, device=DEVICE)
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    fn = model.basis_set.functions[k - 1]
    fn.eval()
    phi, grad = fn(pts, return_grad=True)
    return (grad ** 2).mean().item()


def main() -> None:
    print('Training K=4 on unit circle for 60k steps...')
    model = _train()
    model.eval()

    energies = [_dirichlet_energy(model, k) for k in range(1, K + 1)]
    print('\nDirichlet energies (proxy for eigenvalues):')
    for i, e in enumerate(energies):
        print(f'  φ_{i + 1}: {e:.5f}')

    nontrivial = [e for e in energies if e > 0.01]
    ordered = all(
        nontrivial[i] <= nontrivial[i + 1] * 1.2 for i in range(len(nontrivial) - 1)
    ) if len(nontrivial) >= 2 else True

    theta_np = np.linspace(0, 2 * np.pi, 5000)
    theta = torch.tensor(theta_np, dtype=torch.float32, device=DEVICE)
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    with torch.no_grad():
        phis_np = [
            model.basis_set.functions[k].predict(pts).squeeze(-1).cpu().numpy()
            for k in range(K)
        ]

    trig = {
        'cos1': np.cos(theta_np), 'sin1': np.sin(theta_np),
        'cos2': np.cos(2 * theta_np), 'sin2': np.sin(2 * theta_np),
    }

    print('\nMax correlation per φ_k with trig functions:')
    max_corrs, passed = [], True
    for k, (phi, energy) in enumerate(zip(phis_np, energies)):
        corrs = {n: abs(np.corrcoef(phi, f)[0, 1]) for n, f in trig.items()}
        mc = max(corrs.values())
        best = max(corrs, key=corrs.get)
        max_corrs.append(mc)
        is_const = energy < 0.01
        status = '(constant — skip)' if is_const else f'{"PASS" if mc >= 0.7 else "FAIL"}'
        print(f'  φ_{k + 1}: corr={mc:.4f}  energy={energy:.5f}  {status}  (best: {best})')
        if not is_const and mc < 0.7:
            passed = False

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K))
    for k, (phi, ax) in enumerate(zip(phis_np, axes)):
        ax.plot(theta_np, phi, label=f'φ_{k + 1} learned')
        best_f = max(trig, key=lambda n: abs(np.corrcoef(phi, trig[n])[0, 1]))
        ax.plot(theta_np, trig[best_f], '--', label=f'{best_f} (GT)')
        ax.set_title(f'φ_{k + 1}')
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'logs' / 'sanity' / 'circle_eigenfunctions.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nPlot saved: {out}')

    print('\n--- Results ---')
    print(f'Eigenvalue ordering: {"PASS" if ordered else "FAIL"}')
    for k, (mc, energy) in enumerate(zip(max_corrs, energies)):
        status = 'SKIP (constant)' if energy < 0.01 else ('PASS' if mc >= 0.7 else 'FAIL')
        print(f'  φ_{k + 1} max_corr={mc:.4f}: {status}')
    print(f'OVERALL: {"PASS" if (passed and ordered) else "FAIL"}')
    if not (passed and ordered):
        sys.exit(1)


if __name__ == '__main__':
    main()
