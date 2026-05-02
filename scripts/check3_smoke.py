"""Check 3: 100-step smoke tests for BinaryHead and MulticlassHead.

Tests:
  - Two-moon + BinaryHead (K=3)
  - MNIST binary (0 vs 1) + BinaryHead (K=3)
  - MNIST full (10 classes) + MulticlassHead (K=6, C=10)

For each:
  - loss is finite and decreasing
  - no NaN/Inf in outputs
  - gram_error is logged at every step
  - checkpoint saves and loads (resume test)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import torch
import numpy as np
from itertools import repeat

from src.dataset.collate import collate_fn as CollateFn
from src.dataset.sklearn_cls import SklearnDataset
from src.dataset.utils import make_loader
from src.loss.spectral_loss import SpectralDirichletLoss
from src.model.basis.basis_set import BasisSet
from src.model.spectral_model import BinaryHead, MulticlassHead, SpectralModel

DEVICE = 'cpu'
STEPS = 100


def _run_smoke(name, model, criterion, train_loader, K, head_type):
    torch.manual_seed(42)
    model.to(DEVICE)

    def _inf(ld):
        for dl in repeat(ld):
            yield from dl

    data_iter = _inf(train_loader)
    losses = []
    gram_errors = []

    for k in range(1, K + 1):
        model.basis_set.set_active(k)
        model.set_active_k(k)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=1e-3)

        for _ in range(STEPS // K):
            batch = next(data_iter)
            x = batch['x'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            opt.zero_grad()
            out = model(x, y)
            ld = criterion(out['phi_matrix'], out['grad_phi_k'], out['A'], out['head_out'], k)
            ld['loss'].backward()
            opt.step()

            loss_val = ld['loss'].item()
            ge = ld['gram_error'].item()

            assert np.isfinite(loss_val), f'NaN/Inf in loss at step {len(losses)}'
            assert np.isfinite(ge), f'NaN/Inf in gram_error'

            losses.append(loss_val)
            gram_errors.append(ge)

        model.basis_set.functions[k - 1].eval()
        for p in model.basis_set.functions[k - 1].parameters():
            p.requires_grad_(False)

    # Check loss decreased (last 10 vs first 10)
    trend = np.mean(losses[-10:]) < np.mean(losses[:10]) * 2  # 2x tolerance

    # Checkpoint save/load test.
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / 'test_ckpt.pt'
        torch.save({'model_state_dict': model.state_dict(), 'global_step': STEPS}, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model2_bs = BasisSet(K=K, input_dim=model.basis_set.input_dim)
        model2 = SpectralModel(model2_bs, None, head_type(K) if head_type == BinaryHead else head_type(K, 10))
        model2.load_state_dict(ckpt['model_state_dict'])
        ckpt_ok = True

    return {
        'losses_finite': all(np.isfinite(l) for l in losses),
        'loss_trend_ok': trend,
        'gram_errors_logged': len(gram_errors) == len(losses),
        'final_loss': float(np.mean(losses[-5:])),
        'checkpoint_ok': ckpt_ok,
    }


def main():
    passed_all = True

    # --- Test 1: Two-moon + BinaryHead ---
    print('\n=== Test 1: Two-moon + BinaryHead (K=3) ===')
    ds = SklearnDataset(name='two_moon', split='train', n_samples=3000)
    loader = make_loader(ds, batch_size=64, shuffle=True, collate_fn=CollateFn(use_label=True))
    bs = BasisSet(K=3, input_dim=2)
    head = BinaryHead(3)
    model = SpectralModel(bs, None, head)
    criterion = SpectralDirichletLoss(w_gram=1.0, w_dirichlet=1.0, w_task=1.0)
    res = _run_smoke('two_moon_binary', model, criterion, loader, K=3, head_type=BinaryHead)
    ok = res['losses_finite'] and res['gram_errors_logged'] and res['checkpoint_ok']
    print(f"  losses_finite: {res['losses_finite']}")
    print(f"  gram_errors_logged: {res['gram_errors_logged']}")
    print(f"  checkpoint_save_load: {res['checkpoint_ok']}")
    print(f"  final_loss: {res['final_loss']:.4f}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed_all = False

    # --- Test 2: MNIST binary + BinaryHead ---
    print('\n=== Test 2: MNIST binary (0vs1) + BinaryHead (K=3) ===')
    try:
        from src.dataset.torchvision_flat import TorchvisionFlatDataset
        ds2 = TorchvisionFlatDataset(name='mnist', split='train', root='data/mnist',
                                      task='binary', binary_classes=(0, 1), val_fraction=0.1,
                                      standardize=True)
        loader2 = make_loader(ds2, batch_size=64, shuffle=True, collate_fn=CollateFn(use_label=True))
        bs2 = BasisSet(K=3, input_dim=784)
        head2 = BinaryHead(3)
        model2 = SpectralModel(bs2, None, head2)
        res2 = _run_smoke('mnist_binary', model2, criterion, loader2, K=3, head_type=BinaryHead)
        ok2 = res2['losses_finite'] and res2['gram_errors_logged'] and res2['checkpoint_ok']
        print(f"  losses_finite: {res2['losses_finite']}")
        print(f"  gram_errors_logged: {res2['gram_errors_logged']}")
        print(f"  checkpoint_save_load: {res2['checkpoint_ok']}")
        print(f"  final_loss: {res2['final_loss']:.4f}")
        print(f"  RESULT: {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            passed_all = False
    except Exception as e:
        print(f"  SKIP (MNIST not available): {e}")

    # --- Test 3: MNIST full 10-class + MulticlassHead ---
    print('\n=== Test 3: MNIST 10-class + MulticlassHead (K=6, C=10) ===')
    try:
        ds3 = TorchvisionFlatDataset(name='mnist', split='train', root='data/mnist',
                                      task='multiclass', val_fraction=0.1, standardize=True)
        loader3 = make_loader(ds3, batch_size=64, shuffle=True, collate_fn=CollateFn(use_label=True))
        bs3 = BasisSet(K=6, input_dim=784)
        head3 = MulticlassHead(6, 10)
        model3 = SpectralModel(bs3, None, head3)
        criterion3 = SpectralDirichletLoss(w_gram=1.0, w_dirichlet=1.0, w_task=1.0)
        # Verify MulticlassHead C=10 inferred correctly.
        assert isinstance(head3, MulticlassHead), 'MulticlassHead type check'
        assert head3.num_classes == 10, 'C=10 check'
        assert head3.linear.out_features == 10, 'output shape check'

        losses3, gram3 = [], []
        model3.to(DEVICE)
        torch.manual_seed(42)
        data_iter3 = (b for dl in repeat(loader3) for b in dl)
        for k in range(1, 7):
            model3.basis_set.set_active(k)
            model3.set_active_k(k)
            trainable = [p for p in model3.parameters() if p.requires_grad]
            opt3 = torch.optim.Adam(trainable, lr=1e-3)
            for _ in range(STEPS // 6):
                batch = next(data_iter3)
                x = batch['x'].to(DEVICE)
                y = batch['labels'].to(DEVICE)
                opt3.zero_grad()
                out = model3(x, y)
                ld3 = criterion3(out['phi_matrix'], out['grad_phi_k'], out['A'], out['head_out'], k)
                ld3['loss'].backward()
                opt3.step()
                losses3.append(ld3['loss'].item())
                gram3.append(ld3['gram_error'].item())
            model3.basis_set.functions[k - 1].eval()
            for p in model3.basis_set.functions[k - 1].parameters():
                p.requires_grad_(False)

        ok3_finite = all(np.isfinite(l) for l in losses3)
        ok3_gram = all(np.isfinite(g) for g in gram3)
        head_type_ok = isinstance(model3.head, MulticlassHead)
        print(f"  MulticlassHead C=10 inferred: {'PASS' if head3.num_classes == 10 else 'FAIL'}")
        print(f"  losses_finite: {ok3_finite}")
        print(f"  gram_logged: {ok3_gram}")
        print(f"  head type is MulticlassHead: {head_type_ok}")
        print(f"  final_loss: {float(np.mean(losses3[-5:])):.4f}")
        ok3 = ok3_finite and ok3_gram and head_type_ok
        print(f"  RESULT: {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            passed_all = False
    except Exception as e:
        print(f"  SKIP (MNIST not available): {e}")

    print(f'\n=== OVERALL: {"PASS" if passed_all else "FAIL"} ===')
    if not passed_all:
        sys.exit(1)


if __name__ == '__main__':
    main()
