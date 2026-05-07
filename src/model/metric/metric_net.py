from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .diag_metric import DiagMetric
from .lambda_u_pinn import LambdaUPinn
from .lambda_u_sparse import LambdaUSparse
from .lambda_u_trotter import LambdaUTrotter

MetricType = Literal[
    'off', 'diag', 'lambda_u_sparse', 'lambda_u_pinn', 'lambda_u_trotter',
]


def build_metric(
    metric_type: MetricType,
    input_dim: int,
    hidden_dims: list[int] | None = None,
    pinn_hidden_dims: list[int] | None = None,
    trotter_passes: int = 1,
    trotter_bound_omega: bool = True,
) -> nn.Module | None:
    """Construct a metric module from its type string.

    Dispatcher for the metric variants:
    - "off": No metric (A = I). Returns None; loss uses plain ∇φ norm.
    - "diag": Λ(x) diagonal, det = 1.
    - "lambda_u_sparse": U(x)·Λ(x), U = expm(sparse ω). O(d³) per sample.
    - "lambda_u_pinn":   U(x)·Λ(x), U approximated by a pretrained MLP.
                         WARNING: audit §1.5 found this is mathematically
                         unsound (the apply_to rescale assumes linearity in
                         v that the PINN does not have). Prefer 'lambda_u_trotter'.
    - "lambda_u_trotter": U(x)·Λ(x), U = product of Givens rotations.
                          Exact orthogonality, exact 1-homogeneity in v,
                          O(B·d) cost. Recommended replacement for PINN.
                          See src/model/metric/lambda_u_trotter.py.

    Args:
        metric_type: see above.
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden widths for Λ/ω networks.
        pinn_hidden_dims: Hidden widths for the PINN (only "lambda_u_pinn").
        trotter_passes: Number of Trotter sweeps (only "lambda_u_trotter").
        trotter_bound_omega: Bound ω to [-π, π] via tanh (only Trotter).
    Returns:
        Metric module, or None for "off".
    """
    match metric_type:
        case 'off':
            return None
        case 'diag':
            return DiagMetric(input_dim, hidden_dims)
        case 'lambda_u_sparse':
            return LambdaUSparse(input_dim, hidden_dims)
        case 'lambda_u_pinn':
            return LambdaUPinn(input_dim, hidden_dims, pinn_hidden_dims)
        case 'lambda_u_trotter':
            return LambdaUTrotter(
                input_dim,
                hidden_dims,
                n_passes=trotter_passes,
                bound_omega=trotter_bound_omega,
            )
        case _:
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                "Expected one of: 'off', 'diag', 'lambda_u_sparse', "
                "'lambda_u_pinn', 'lambda_u_trotter'."
            )
