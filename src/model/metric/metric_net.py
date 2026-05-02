from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .diag_metric import DiagMetric
from .lambda_u_pinn import LambdaUPinn
from .lambda_u_sparse import LambdaUSparse

MetricType = Literal['off', 'diag', 'lambda_u_sparse', 'lambda_u_pinn']


def build_metric(
    metric_type: MetricType,
    input_dim: int,
    hidden_dims: list[int] | None = None,
    pinn_hidden_dims: list[int] | None = None,
) -> nn.Module | None:
    """Construct a metric module from its type string.

    Dispatcher for the four metric variants:
    - "off": No metric (A = I). Returns None; loss uses plain ∇φ norm.
    - "diag": Λ(x) diagonal, det = 1.
    - "lambda_u_sparse": U(x)·Λ(x), U = expm(sparse ω).
    - "lambda_u_pinn": U(x)·Λ(x), U approximated by a pretrained PINN.

    Args:
        metric_type: One of "off", "diag", "lambda_u_sparse", "lambda_u_pinn".
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden widths for Λ/ω networks.
        pinn_hidden_dims: Hidden widths for the PINN (only used with "lambda_u_pinn").
    Returns:
        Metric module, or None for "off".
    Raises:
        ValueError: If metric_type is not one of the four supported values.
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
        case _:
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                "Expected one of: 'off', 'diag', 'lambda_u_sparse', 'lambda_u_pinn'."
            )
