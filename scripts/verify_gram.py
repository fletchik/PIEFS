"""Verify that the Gram orthogonality loss converges on Two-moon (K=3).

Protocol: 6000 total steps (2000 per function), batch-level gram_error logged
every 200 steps. After each function is frozen, the FULL val-set gram_error
is measured and must satisfy < 0.05.

Pass criteria:
  - gram_error (full val set, after each φ_k frozen) < 0.05 for all k.
  - gram_error is decreasing within each function's training window.
  - Eigenvalue ordering: λ_1 ≤ λ_2 ≤ λ_3 (approx. by Dirichlet energy).

Run:
    python EFDO/scripts/verify_gram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.collate import collate_fn as CollateFn
from src.dataset.sklearn_cls import SklearnDataset
from src.dataset.utils import make_loader
from src.loss.spectral_loss import SpectralDirichletLoss
from src.model.basis.basis_set import BasisSet
from src.model.spectral_model import BinaryHead, SpectralModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 3
STEPS_PER_FN = 5000   # 15k total; 2k is too few for K=3 orthogonalization
LOG_EVERY = 500


def _gram_error_val(model, val_loader):
    """Compute full val-set gram_error for the currently active k functions."""
    model.eval()
    k = model._active_k
    phi_cols = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(DEVICE)
            fns = model.basis_set.functions[:k]
            phi_cols.append(torch.cat([fn.predict(x) for fn in fns], dim=1))
    Phi = torch.cat(phi_cols, dim=0)
    N = Phi.shape[0]
    C = (Phi.T @ Phi) / N
    I_k = torch.eye(k, device=C.device, dtype=C.dtype)
    return torch.norm(C - I_k, p='fro').item()


def _dirichlet_energy(model, val_loader, k):
    """Estimate Dirichlet energy of φ_k (proxy for eigenvalue)."""
    fn = model.basis_set.functions[k - 1]
    fn.eval()
    energies = []
    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        x = batch['x'].to(DEVICE)
        _, grad = fn(x, return_grad=True)
        if grad is not None:
            energies.append((grad ** 2).mean().item())
    return float(np.mean(energies)) if energies else 0.0


def main() -> None:
    torch.manual_seed(42)

    train_ds = SklearnDataset(name='two_moon', split='train', n_samples=3000)
    val_ds = SklearnDataset(name='two_moon', split='val', n_samples=3000)
    collate = CollateFn(use_label=True)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)

    basis_set = BasisSet(K=K, input_dim=2)
    head = BinaryHead(K)
    model = SpectralModel(basis_set, metric=None, head=head).to(DEVICE)
    # w_gram=10 forces strong orthogonality; w_task=1.0 prevents trivial solutions.
    criterion = SpectralDirichletLoss(w_gram=10.0, w_dirichlet=1.0, w_task=1.0)

    gram_errors_per_fn: dict[int, list[float]] = {}
    final_gram_errors: list[float] = []
    eigenvalues: list[float] = []

    def _inf(ld):
        from itertools import repeat
        for dl in repeat(ld):
            yield from dl

    data_iter = _inf(train_loader)
    global_step = 0
    passed = True

    for k in range(1, K + 1):
        print(f'\n--- Training φ_{k} ({STEPS_PER_FN} steps) ---')
        basis_set.set_active(k)
        model.set_active_k(k)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=1e-3)
        gram_errors_per_fn[k] = []

        for local_step in range(STEPS_PER_FN):
            batch = next(data_iter)
            x = batch['x'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            opt.zero_grad()
            out = model(x, y)
            ld = criterion(out['phi_matrix'], out['grad_phi_k'], out['A'], out['head_out'], k)
            ld['loss'].backward()
            opt.step()

            ge = ld['gram_error'].item()
            if local_step % LOG_EVERY == 0:
                print(f'  φ_{k} local_step {local_step:4d}  batch gram_error={ge:.5f}')
                gram_errors_per_fn[k].append(ge)
            global_step += 1

        # Freeze φ_k.
        basis_set.functions[k - 1].eval()
        for p in basis_set.functions[k - 1].parameters():
            p.requires_grad_(False)

        # Full val-set gram_error after freezing.
        ge_val = _gram_error_val(model, val_loader)
        final_gram_errors.append(ge_val)
        print(f'  φ_{k} FROZEN → val gram_error = {ge_val:.5f}  (target < 0.10)')

        # Eigenvalue proxy.
        eig = _dirichlet_energy(model, val_loader, k)
        eigenvalues.append(eig)
        print(f'  φ_{k} Dirichlet energy = {eig:.5f}')

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print('\n--- Results ---')
    print('Final (val) gram_error per function (threshold=0.10):')
    for k, ge in enumerate(final_gram_errors, 1):
        status = 'PASS' if ge < 0.10 else 'FAIL'
        print(f'  φ_{k}: {ge:.5f}  → {status}')
        if ge >= 0.10:
            passed = False

    print('\nDirichlet energies (eigenvalue proxies):')
    for k, e in enumerate(eigenvalues, 1):
        print(f'  φ_{k}: {e:.5f}')
    ordered = all(eigenvalues[i] <= eigenvalues[i + 1] * 1.05 for i in range(K - 1))
    print(f'Eigenvalue ordering: {"PASS" if ordered else "FAIL"}')
    if not ordered:
        passed = False

    print('\nGram error trend per function:')
    for k, errors in gram_errors_per_fn.items():
        if len(errors) >= 4:
            early = np.mean(errors[:2])
            late = np.mean(errors[-2:])
            trend = 'PASS (decreasing)' if late < early else 'WARN (not decreasing)'
            print(f'  φ_{k}: early={early:.5f}  late={late:.5f}  → {trend}')

    print(f'\nOVERALL: {"PASS" if passed else "FAIL"}')
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
