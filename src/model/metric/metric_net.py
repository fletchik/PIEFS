from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .conformal_metric import ConformalMetric
from .diag_metric import DiagMetric
from .fisher_diag import FisherDiagMetric
from .global_low_rank import GlobalLowRankMetric
from .lambda_u_trotter import LambdaUTrotter
from .local_low_rank import LocalLowRankMetric

MetricType = Literal[
    'off', 'diag', 'lambda_u_trotter', 'global_low_rank',
    'conformal', 'local_low_rank', 'fisher_diag',
]


def build_metric(
    metric_type: MetricType,
    input_dim: int,
    hidden_dims: list[int] | None = None,
    pinn_hidden_dims: list[int] | None = None,  # kept for backward compat
    trotter_passes: int = 1,
    trotter_bound_omega: bool = True,
    low_rank_r: int = 8,
    low_rank_init_scale: float = 0.01,
    normalize_det: bool = False,
) -> nn.Module | None:
    """Construct a metric module from its type string.

    Dispatcher for all metric variants.

    ── Implemented ─────────────────────────────────────────────────────────────
    "off":             No metric (A = I). Returns None; loss uses plain ∇φ norm.
    "diag":            Λ(x) diagonal, det = 1 (x-dependent MLP). Axis-aligned
                       anisotropy only.
    "conformal":       A(x) = σ(x)·I — isotropic x-dependent scaling (scalar MLP).
                       Simplest x-dependent baseline: adds location-sensitivity
                       without any directional structure.
    "lambda_u_trotter":A(x) = Λ(x)·U_Trotter(ω(x)). U = product of (d-1)
                       Givens rotations. Exact orthogonality, O(B·d) cost.
                       Note: covers only (d-1)-param subgroup of SO(d).
    "global_low_rank": A = I + U·D·Vᵀ — global (not x-dependent) low-rank
                       perturbation of identity. No MLP bottleneck.
                       Optimal rank: r = num_classes - 1 (LDA connection).
                       *** RECOMMENDED for new experiments ***
    "local_low_rank":  A(x) = I + U(x)·Λ(x)·V(x)ᵀ — x-dependent rank-r
                       perturbation. Full rank-r coverage (vs Trotter's subgroup).
                       Shares backbone MLP for U, V, λ heads.
    "fisher_diag":     A(x) = diag(sqrt(F_diag(x))) — diagonal Fisher Information
                       Metric approximation. Theoretically optimal for classification
                       (Information geometry, Amari 1985). MLP approximation of FIM.

    ── Deprecated (moved to archive/src_deprecated/) ────────────────────────
    "lambda_u_pinn":   Nonlinearity bug in apply_to() — archived.
    "lambda_u_sparse": Superseded by 'lambda_u_trotter'. Archived.

    Args:
        metric_type: One of the implemented strings above.
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden widths for x-dependent metrics.
        pinn_hidden_dims: Ignored (kept for old config compatibility).
        trotter_passes: Number of Trotter sweeps ("lambda_u_trotter" only).
        trotter_bound_omega: Bound ω ∈ [−π, π] via tanh ("lambda_u_trotter").
        low_rank_r: Rank r of the perturbation ("global_low_rank", "local_low_rank").
            Recommended: num_classes − 1 (e.g. 9 for MNIST-10, 1 for binary).
        low_rank_init_scale: Init std for U, V ("global_low_rank", "local_low_rank").
            0.0 = exact identity start; 0.01 = near-identity with nonzero grad.
        normalize_det: For "conformal" and "fisher_diag": whether to normalise
            A so that det(A) = 1. Default False.

    Returns:
        Metric nn.Module, or None for "off".
    """
    match metric_type:
        case 'off':
            return None

        case 'diag':
            return DiagMetric(input_dim, hidden_dims)

        case 'conformal':
            return ConformalMetric(
                input_dim,
                hidden_dims=hidden_dims,
                normalize_det=normalize_det,
            )

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

        case 'local_low_rank':
            return LocalLowRankMetric(
                input_dim=input_dim,
                r=low_rank_r,
                hidden_dims=hidden_dims,
                init_scale=low_rank_init_scale,
            )

        case 'fisher_diag':
            return FisherDiagMetric(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                normalize_det=normalize_det,
            )

        case _:
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                "Expected one of: 'off', 'diag', 'conformal', "
                "'lambda_u_trotter', 'global_low_rank', "
                "'local_low_rank', 'fisher_diag'."
            )
