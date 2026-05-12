from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .diag_metric import DiagMetric
from .global_low_rank import GlobalLowRankMetric
from .lambda_u_trotter import LambdaUTrotter

MetricType = Literal[
    'off', 'diag', 'lambda_u_trotter', 'global_low_rank',
]


def build_metric(
    metric_type: MetricType,
    input_dim: int,
    hidden_dims: list[int] | None = None,
    pinn_hidden_dims: list[int] | None = None,  # kept for backward compat
    trotter_passes: int = 1,
    trotter_bound_omega: bool = True,
    low_rank_r: int = 16,
    low_rank_init_scale: float = 0.01,
) -> nn.Module | None:
    """Construct a metric module from its type string.

    Dispatcher for the metric variants:
    - "off":              No metric (A = I). Returns None; loss uses plain ∇φ norm.
    - "diag":             Λ(x) diagonal, det = 1 (x-dependent, MLP-based).
    - "lambda_u_trotter": A(x) = Λ(x)·U_Trotter(ω(x)), U = product of Givens
                          rotations. Exact orthogonality, O(B·d) cost.
                          Note: covers only (d-1)-param subgroup of SO(d);
                          suffers from MLP bottleneck at high d.
    - "global_low_rank":  A = I + U·D·V^T — global (NOT x-dependent) low-rank
                          perturbation of identity. No MLP, no bottleneck.
                          Gradient flows directly to U, V, D parameters.
                          Identity recovery guaranteed at init (U=V≈0, D≈0).
                          Optimal rank for classification: r = num_classes - 1
                          (LDA / Fisher metric connection).
                          *** RECOMMENDED for new experiments ***

    Deprecated (moved to archive/src_deprecated/):
    - "lambda_u_pinn":   Buggy (audit §1.5, §1.6, §2.12). Files archived.
    - "lambda_u_sparse": Superseded by 'lambda_u_trotter'. Files archived.

    Args:
        metric_type: One of the strings above.
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden widths for diag/trotter networks.
        pinn_hidden_dims: Ignored (kept for old config compatibility).
        trotter_passes: Number of Trotter sweeps ("lambda_u_trotter" only).
        trotter_bound_omega: Bound ω ∈ [−π, π] via tanh ("lambda_u_trotter").
        low_rank_r: Rank r of perturbation ("global_low_rank" only).
            Recommended: num_classes − 1 (e.g. 9 for MNIST, 1 for binary).
            Default 16 for a safe over-parameterised choice.
        low_rank_init_scale: Init std for U, V ("global_low_rank" only).
            0.0 = exact identity start; 0.01 = near-identity with nonzero grad.

    Returns:
        Metric nn.Module, or None for "off".
    """
    match metric_type:
        case 'off':
            return None
        case 'diag':
            return DiagMetric(input_dim, hidden_dims)
        case 'lambda_u_trotter':
            return LambdaUTrotter(
                input_dim,
                hidden_dims,
                n_passes=trotter_passes,
                bound_omega=trotter_bound_omega,
            )
        case 'global_low_rank':
            return GlobalLowRankMetric(
                d=input_dim,
                r=low_rank_r,
                init_scale=low_rank_init_scale,
            )
        case _:
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                "Expected one of: 'off', 'diag', 'lambda_u_trotter', "
                "'global_low_rank'."
            )
