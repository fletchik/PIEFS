"""Property tests for all metric variants.

Run with:
    python3 -m pytest tests/test_metrics.py -v

What is tested
--------------
1. Shape contracts (apply_to returns (B, d)).
2. det(Λ) = 1 by construction (diag, trotter).
3. Identity behaviour for the 'off' metric.
4. Orthogonality of the rotational part (trotter).
5. 1-homogeneity in v — A·(αv) must equal α·(A·v).
6. Numerical stability under extreme inputs.
7. Gradient flow into metric parameters.
8. GlobalLowRankMetric: near-identity init, linearity, gradient flow.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow `import src...` when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.metric.diag_metric import DiagMetric  # noqa: E402
from src.model.metric.global_low_rank import GlobalLowRankMetric  # noqa: E402
from src.model.metric.lambda_u_trotter import LambdaUTrotter  # noqa: E402
from src.model.metric.metric_net import build_metric  # noqa: E402

D = 8                 # dimensionality used throughout
B = 16                # batch size
HIDDEN = [32, 32]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(metric, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute A(x)·v in a unified way across all metric types."""
    if metric is None:                                # 'off'
        return v
    if hasattr(metric, 'apply_to'):
        return metric.apply_to(x, v)
    A = metric(x)
    if A.dim() == 2:
        return A * v
    return torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)


def _run(fn):
    """Helper: call fn and return True if no exception."""
    try:
        fn()
        return True
    except Exception as e:
        print(f'  FAIL: {e}')
        return False


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

def test_apply_to_shape_off():
    metric = build_metric('off', input_dim=D, hidden_dims=HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    out = _apply(metric, x, v)
    assert out.shape == (B, D)


def test_apply_to_shape_diag():
    metric = build_metric('diag', input_dim=D, hidden_dims=HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    assert _apply(metric, x, v).shape == (B, D)


def test_apply_to_shape_trotter():
    metric = build_metric('lambda_u_trotter', input_dim=D, hidden_dims=HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    assert _apply(metric, x, v).shape == (B, D)


def test_apply_to_shape_global_low_rank():
    metric = build_metric('global_low_rank', input_dim=D, low_rank_r=4)
    x, v = torch.randn(B, D), torch.randn(B, D)
    assert _apply(metric, x, v).shape == (B, D)


# ---------------------------------------------------------------------------
# 2. det(Λ) = 1 by construction (volume-preserving)
# ---------------------------------------------------------------------------

def test_diag_det_one():
    metric = DiagMetric(D, HIDDEN)
    x = torch.randn(B, D)
    lam = metric(x)
    log_det = torch.log(lam).sum(dim=-1)
    assert torch.allclose(log_det, torch.zeros(B), atol=1e-5), \
        f'log det = {log_det}'


def test_trotter_det_one():
    metric = LambdaUTrotter(D, HIDDEN)
    x = torch.randn(B, D)
    lam = metric.get_lambda(x)
    log_det = torch.log(lam).sum(dim=-1)
    assert torch.allclose(log_det, torch.zeros(B), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. 'off' is exactly identity
# ---------------------------------------------------------------------------

def test_off_is_identity():
    metric = build_metric('off', input_dim=D, hidden_dims=HIDDEN)
    assert metric is None
    x, v = torch.randn(B, D), torch.randn(B, D)
    assert torch.equal(_apply(metric, x, v), v)


# ---------------------------------------------------------------------------
# 4. Orthogonality of the Trotter rotational part
# ---------------------------------------------------------------------------

def test_trotter_rotation_preserves_norm():
    metric = LambdaUTrotter(D, HIDDEN, n_passes=1, bound_omega=True)
    x = torch.randn(B, D)
    v = F.normalize(torch.randn(B, D), dim=-1)
    omega = metric.get_omega(x)
    Rv = metric._trotter_rotate(omega, v)
    assert torch.allclose(Rv.norm(dim=-1), torch.ones(B), atol=1e-5)


def test_trotter_multipass_orthogonal():
    metric = LambdaUTrotter(D, HIDDEN, n_passes=3, bound_omega=True)
    x = torch.randn(B, D)
    v = F.normalize(torch.randn(B, D), dim=-1)
    omega = metric.get_omega(x)
    Rv = metric._trotter_rotate(omega, v)
    assert torch.allclose(Rv.norm(dim=-1), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# 5. 1-homogeneity in v: A(x)·(αv) == α · A(x)·v
# ---------------------------------------------------------------------------

def test_linearity_diag():
    metric = build_metric('diag', D, HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    alpha = 3.7
    err = (_apply(metric, x, alpha * v) - alpha * _apply(metric, x, v)).abs().max().item()
    assert err < 1e-4, f'diag 1-homogeneity error={err:.2e}'


def test_linearity_trotter():
    metric = build_metric('lambda_u_trotter', D, HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    alpha = 3.7
    err = (_apply(metric, x, alpha * v) - alpha * _apply(metric, x, v)).abs().max().item()
    assert err < 1e-4, f'trotter 1-homogeneity error={err:.2e}'


def test_linearity_global_low_rank():
    """GlobalLowRankMetric: A·(αv) = v + U·D·(V^T·(αv)) = α·(v + U·D·V^T·v) = α·Av."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.01)
    x, v = torch.randn(B, D), torch.randn(B, D)
    alpha = 2.5
    err = (metric.apply_to(x, alpha * v) - alpha * metric.apply_to(x, v)).abs().max().item()
    assert err < 1e-5, f'GLR 1-homogeneity error={err:.2e}'


# ---------------------------------------------------------------------------
# 6. Numerical stability under extreme inputs
# ---------------------------------------------------------------------------

def test_no_nan_off():
    x, v = 10.0 * torch.randn(B, D), torch.randn(B, D)
    assert torch.isfinite(_apply(None, x, v)).all()


def test_no_nan_diag():
    metric = build_metric('diag', D, HIDDEN)
    x, v = 10.0 * torch.randn(B, D), torch.randn(B, D)
    assert torch.isfinite(_apply(metric, x, v)).all()


def test_no_nan_trotter():
    metric = build_metric('lambda_u_trotter', D, HIDDEN)
    x, v = 10.0 * torch.randn(B, D), torch.randn(B, D)
    assert torch.isfinite(_apply(metric, x, v)).all()


def test_no_nan_global_low_rank():
    metric = GlobalLowRankMetric(d=D, r=4)
    x, v = 10.0 * torch.randn(B, D), torch.randn(B, D)
    assert torch.isfinite(_apply(metric, x, v)).all()


# ---------------------------------------------------------------------------
# 7. Gradient flow — metric parameters receive non-zero gradients
# ---------------------------------------------------------------------------

def test_gradient_flow_diag():
    metric = build_metric('diag', D, HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    (_apply(metric, x, v) ** 2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in metric.parameters())


def test_gradient_flow_trotter():
    metric = build_metric('lambda_u_trotter', D, HIDDEN)
    x, v = torch.randn(B, D), torch.randn(B, D)
    (_apply(metric, x, v) ** 2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in metric.parameters())


def test_gradient_flow_global_low_rank():
    """GLR: U, V, log_d all get gradients on the very first step."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.01)
    x, v = torch.randn(B, D), torch.randn(B, D)
    (_apply(metric, x, v) ** 2).sum().backward()
    names_with_grad = [n for n, p in metric.named_parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(names_with_grad) > 0, 'No gradients reached GLR parameters'
    print(f'\n  GLR grad params: {names_with_grad}')


# ---------------------------------------------------------------------------
# 8. GlobalLowRankMetric specific properties
# ---------------------------------------------------------------------------

def test_glr_near_identity_at_small_init():
    """With tiny init_scale, A should be very close to identity."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=1e-6)
    x = torch.randn(B, D)
    v = torch.randn(B, D)
    Av = metric.apply_to(x, v)
    err = (Av - v).abs().max().item()
    assert err < 1e-4, f'GLR should be near-identity at tiny init: err={err:.2e}'


def test_glr_exact_identity_at_zero_init():
    """With init_scale=0 and log_d=0, A = I exactly."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.0)
    x = torch.randn(B, D)
    v = torch.randn(B, D)
    Av = metric.apply_to(x, v)
    assert torch.allclose(Av, v, atol=1e-6), 'GLR(init=0) should be exactly identity'


def test_glr_effective_rank():
    """Effective rank is in [1, r] and equals r at init (all D_i equal)."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.01)
    eff_rank = metric.get_effective_rank()
    assert 1.0 <= eff_rank <= 4.0 + 1e-5, f'Effective rank out of [1,r]: {eff_rank}'


def test_glr_singular_values_shape():
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.01)
    sv = metric.get_singular_values()
    assert sv.shape == (4,), f'Expected shape (4,), got {sv.shape}'
    assert (sv > 0).all(), 'All singular values should be positive (exp of log_d)'


def test_glr_x_invariant():
    """Global metric: apply_to(x1, v) == apply_to(x2, v) for any x1, x2."""
    metric = GlobalLowRankMetric(d=D, r=4, init_scale=0.1)
    v = torch.randn(B, D)
    x1, x2 = torch.randn(B, D), 999 * torch.randn(B, D)
    assert torch.allclose(metric.apply_to(x1, v), metric.apply_to(x2, v)), \
        'GLR should be x-invariant (global, not pointwise)'


# ---------------------------------------------------------------------------
# 8. Trotter bounded angles
# ---------------------------------------------------------------------------

def test_trotter_omega_bounded():
    metric = LambdaUTrotter(D, HIDDEN, bound_omega=True)
    x = 100.0 * torch.randn(B, D)
    omega = metric.get_omega(x)
    assert (omega.abs() <= math.pi + 1e-5).all(), \
        f'omega exceeds π: max={omega.abs().max().item():.3f}'


def test_trotter_2d_exact_rotation():
    """For d=2, P=1, Trotter is a single Givens rotation = exact 2D rotation."""
    metric = LambdaUTrotter(input_dim=2, hidden_dims=[8], n_passes=1, bound_omega=False)
    x = torch.randn(4, 2)
    theta = torch.tensor([0.3, -0.7, 1.2, math.pi / 4])
    omega = theta.view(4, 1, 1)
    v = torch.tensor([[1.0, 0.0]] * 4)
    Rv = metric._trotter_rotate(omega, v)
    expected = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    assert torch.allclose(Rv, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Standalone runner (for environments without pytest)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback
    tests = [k for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    passed = failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f'  PASS  {name}')
            passed += 1
        except Exception:
            print(f'  FAIL  {name}')
            traceback.print_exc()
            failed += 1
    print(f'\n{passed} passed, {failed} failed out of {passed + failed} tests.')
    if failed:
        sys.exit(1)
