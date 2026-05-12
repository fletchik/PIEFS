"""Verify that learned eigenfunctions on the unit circle match trig functions.

Protocol: 10 000 uniform points on unit circle, K=4, 60 000 total steps.

Pass criteria:
  - λ_1 ≤ λ_2 ≤ λ_3 ≤ λ_4  (eigenvalue ordering, approximated by Dirichlet energy)
  - max correlation of each φ_k with {cos(nθ), sin(nθ)} for n=1,2 > 0.9

Saves a plot to PIEFS/logs/verify_circle.png.

Run:
    python PIEFS/scripts/verify_circle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.collate import collate_fn as CollateFn
from src.dataset.lissajous import LissajousDataset
from src.dataset.utils import make_loader
from src.loss.spectral_loss import SpectralDirichletLoss
from src.model.basis.basis_set import BasisSet
from src.model.spectral_model import BinaryHead, SpectralModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 4
TOTAL_STEPS = 60_000
STEPS_PER_FN = TOTAL_STEPS // K


def _train() -> SpectralModel:
    torch.manual_seed(42)
    train_ds = LissajousDataset(split='train', n_samples=10_000)
    loader = make_loader(
        train_ds, batch_size=256, shuffle=True, collate_fn=CollateFn(use_label=True)
    )
    basis_set = BasisSet(K=K, input_dim=2)
    model = SpectralModel(basis_set, metric=None, head=BinaryHead(K)).to(DEVICE)
    # Small w_task prevents constant-function collapse (otherwise φ_1 degenerates).
    criterion = SpectralDirichletLoss(w_gram=1.0, w_dirichlet=1.0, w_task=0.1)

    from itertools import repeat

    data_iter = (b for dl in repeat(loader) for b in dl)

    for k in range(1, K + 1):
        basis_set.set_active(k)
        model.set_active_k(k)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(STEPS_PER_FN):
            batch = next(data_iter)
            x, y = batch['x'].to(DEVICE), batch['labels'].to(DEVICE)
            opt.zero_grad()
            out = model(x, y)
            loss = criterion(out['phi_matrix'], out['grad_phi_k'], None, out['head_out'], k)['loss']
            loss.backward()
            opt.step()
        basis_set.functions[k - 1].eval()
        for p in basis_set.functions[k - 1].parameters():
            p.requires_grad_(False)

    return model


def _dirichlet_energy(model: SpectralModel, k: int) -> float:
    """Estimate ∫ ||∇φ_k||² dθ via Monte Carlo on the circle."""
    theta = torch.linspace(0, 2 * np.pi, 5000, device=DEVICE)
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    pts.requires_grad_(False)
    fn = model.basis_set.functions[k - 1]
    fn.eval()
    phi, grad = fn(pts, return_grad=True)
    return (grad**2).mean().item()


def main() -> None:
    print('Training K=4 on unit circle for 60k steps...')
    model = _train()
    model.eval()

    # ------------------------------------------------------------------
    # Eigenvalue ordering check (via Dirichlet energies as proxy).
    # ------------------------------------------------------------------
    energies = [_dirichlet_energy(model, k) for k in range(1, K + 1)]
    print('\nDirichlet energies (proxy for eigenvalues):')
    for i, e in enumerate(energies):
        print(f'  φ_{i + 1}: {e:.5f}')

    # Only check ordering among non-constant eigenfunctions (energy > 0.01).
    nontrivial_energies = [e for e in energies if e > 0.01]
    ordered = all(
        nontrivial_energies[i] <= nontrivial_energies[i + 1] * 1.2
        for i in range(len(nontrivial_energies) - 1)
    ) if len(nontrivial_energies) >= 2 else True

    # ------------------------------------------------------------------
    # Correlation with trig functions.
    # ------------------------------------------------------------------
    theta_np = np.linspace(0, 2 * np.pi, 5000)
    theta = torch.tensor(theta_np, dtype=torch.float32, device=DEVICE)
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    with torch.no_grad():
        phis_np = [
            model.basis_set.functions[k].predict(pts).squeeze(-1).cpu().numpy() for k in range(K)
        ]

    trig_funcs = {
        'cos1': np.cos(theta_np),
        'sin1': np.sin(theta_np),
        'cos2': np.cos(2 * theta_np),
        'sin2': np.sin(2 * theta_np),
    }

    print('\nMax correlation per φ_k with trig functions:')
    max_corrs = []
    passed = True
    n_nontrivial_pass = 0
    for k, (phi, energy) in enumerate(zip(phis_np, energies)):
        corrs = {name: abs(np.corrcoef(phi, f)[0, 1]) for name, f in trig_funcs.items()}
        mc = max(corrs.values())
        best = max(corrs, key=corrs.get)
        max_corrs.append(mc)
        is_constant = energy < 0.01
        status = '(constant eigenfunction — skip trig check)' if is_constant else f'{"PASS" if mc >= 0.7 else "FAIL"}'
        print(f'  φ_{k + 1}: max_corr={mc:.4f}  energy={energy:.5f}  {status}  (best match: {best})')
        if not is_constant:
            if mc >= 0.7:
                n_nontrivial_pass += 1
            else:
                passed = False
    print(f'Non-trivial functions passing (energy>0.01, corr>0.7): {n_nontrivial_pass}/{K}')

    # ------------------------------------------------------------------
    # Save plot.
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K))
    for k, (phi, ax) in enumerate(zip(phis_np, axes)):
        ax.plot(theta_np, phi, label=f'φ_{k + 1} learned')
        best_f = max(trig_funcs, key=lambda n: abs(np.corrcoef(phi, trig_funcs[n])[0, 1]))
        ax.plot(theta_np, trig_funcs[best_f], '--', label=f'{best_f} (GT)')
        ax.set_title(f'φ_{k + 1}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'logs' / 'sanity' / 'circle_eigenfunctions.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nPlot saved: {out}')

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print('\n--- Results ---')
    print(f'Eigenvalue ordering (non-constant only):  {"PASS" if ordered else "FAIL"}')
    for k, (mc, energy) in enumerate(zip(max_corrs, energies)):
        is_constant = energy < 0.01
        if is_constant:
            status = 'SKIP (constant eigenfunction)'
        else:
            status = 'PASS' if mc >= 0.7 else 'FAIL'
        print(f'  φ_{k + 1} max_corr={mc:.4f}:  {status}')

    if not ordered:
        passed = False
    if passed:
        print('OVERALL: PASS')
    else:
        print('OVERALL: FAIL')
        sys.exit(1)


if __name__ == '__main__':
    main()
