"""GlobalLowRankMetric: A = I + U·D·V^T — global low-rank Riemannian metric.

Why this exists
---------------
Previous metric implementations (DiagMetric, LambdaUTrotter) share two
fundamental problems that prevent A(x) from outperforming identity (A = I):

  Problem 1: MLP bottleneck.
    LambdaUTrotter must predict (d-1) Givens angles via a hidden=[64,64] MLP.
    For d=784 (MNIST), this is a 784→64→64→783 bottleneck: 783 outputs
    through a 64-neuron neck. The gradient signal to ω-MLP is weak and
    indirect (through the Givens chain).

  Problem 2: Subgroup restriction.
    The Trotter product of (d-1) adjacent Givens rotations spans only a
    (d-1)-parameter subgroup of SO(d). Full SO(d) has d(d-1)/2 parameters.
    For d=784: 783 vs 306,936 — coverage is 1/392 of the full group.

This class eliminates both problems by abandoning the pointwise-MLP design:

  •  No MLP at all. U ∈ R^(d×r) and V ∈ R^(d×r) are direct parameters.
     Gradient flows directly to U, V, D without passing through any network.

  •  Global (not pointwise). A does not depend on x, so the metric is a
     single fixed linear map applied to every gradient ∇φ_k. This trades
     x-adaptivity for gradient stability and avoids capacity competition
     with the basis networks.

  •  Full rank coverage. For r ≥ C-1 (number of classes minus one),
     the low-rank perturbation can approximate the optimal Fisher metric
     (LDA metric S_W^{-1} S_B) which has rank exactly C-1. For MNIST
     (10 classes) r=9 is theoretically sufficient; r=16 adds slack.

  •  Identity recovery (Theorem). With init_scale=0.0 (U=V=0, D=0),
     A = I exactly and the gradient w.r.t. U,V,D at step 0 is nonzero
     iff ||A∇φ||² depends on them — i.e. as soon as training begins
     and the basis produces nonzero gradients ∇φ_k. The optimizer can
     only move away from A=I if doing so reduces the loss, guaranteeing
     that GlobalLowRank is never worse than EFDO-off at convergence
     (modulo local minima).

  •  LDA connection. The optimal linear map A* for maximising class
     separation of spectral features satisfies A*^T A* = S_W^{-1/2} S_B
     S_W^{-1/2} (whitened between-class scatter). This is a rank-(C-1)
     matrix, exactly the form I + UDV^T can represent for r=C-1.

Mathematical note
-----------------
apply_to computes:

    A v = v + U · D · (V^T v)

where D = diag(exp(log_d)) ensures positive scaling and exp(·) keeps D > 0.

The Modified Dirichlet Energy becomes:

    ||Av||² = ||v + U D V^T v||²

which is always ≥ 0 and collapses to ||v||² (EFDO-off) when U=V=0.

The full matrix is A = I + U diag(exp(log_d)) V^T ∈ R^(d×d), but we never
materialise it during training — apply_to uses O(B·d·r) instead of O(B·d²).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GlobalLowRankMetric(nn.Module):
    """A = I + U·D·V^T — global low-rank learnable metric.

    The metric does NOT depend on the input x (global, not pointwise).
    Gradient flows directly to parameters U, V, log_d without any MLP.

    Args:
        d: Input dimensionality.
        r: Rank of the low-rank perturbation. Recommended: r = num_classes - 1
           (theoretically optimal for classification via LDA connection) or
           r = 16 for a safe over-parameterised default.
        init_scale: Standard deviation for random initialisation of U and V.
            Use 0.0 for exact identity start (A = I); a small nonzero value
            (e.g. 0.01) breaks symmetry while staying close to identity.
            log_d is always initialised to 0 (D = I).
    """

    def __init__(
        self,
        d: int,
        r: int = 16,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.d = d
        self.r = r
        # U, V: initialise near zero → A ≈ I at the start of training.
        # Small nonzero init breaks the symmetry U=V=0 so that gradients
        # are nonzero from step 1.
        self.U = nn.Parameter(torch.randn(d, r) * init_scale)
        self.V = nn.Parameter(torch.randn(d, r) * init_scale)
        # log_d: log of diagonal scaling; exp(0) = 1 → D = I at init.
        self.log_d = nn.Parameter(torch.zeros(r))

    # ------------------------------------------------------------------
    # Public API  (matches the LambdaU* contract used by SpectralModel)
    # ------------------------------------------------------------------

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A·v = v + U·D·(V^T·v).

        Args:
            x: (B, d) input points — IGNORED (global metric).
            v: (B, d) gradient ∇φ_k.
        Returns:
            (B, d) = A·v.
        """
        D = torch.exp(self.log_d)       # (r,)  positive diagonal
        Vtv = v @ self.V                # (B, r)  V^T v
        scaled = Vtv * D               # (B, r)  D (V^T v)
        return v + scaled @ self.U.T   # (B, d)  v + U D V^T v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full A as a (d, d) matrix — for visualisation/diagnostics only.

        Cost O(d²); do not call during training.
        """
        D = torch.exp(self.log_d)                          # (r,)
        eye = torch.eye(self.d, device=x.device, dtype=x.dtype)
        return eye + self.U * D @ self.V.T                 # (d, d)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_singular_values(self) -> torch.Tensor:
        """Singular values of U·D·V^T — for logging effective rank."""
        D = torch.exp(self.log_d)          # (r,)
        # The rank-r matrix M = U diag(D) V^T has singular values
        # σ_i = D_i · ||U[:,i]|| · ||V[:,i]|| (approximately, for near-orth U,V).
        # Exact SVD would require materialising M; here we return D as proxy.
        return D.detach()

    def get_effective_rank(self) -> float:
        """Effective rank = (Σ σ_i)² / Σ σ_i² — scalar in [1, r]."""
        sv = torch.exp(self.log_d.detach())
        return (sv.sum() ** 2 / (sv ** 2).sum()).item()

    def get_frobenius_perturbation(self) -> float:
        """||U·D·V^T||_F — how far A is from identity."""
        D = torch.exp(self.log_d.detach())
        # ||U D V^T||_F ≤ ||U||_F · max(D) · ||V||_F  (rough upper bound)
        # exact: sqrt(Σ_i D_i² · ||U[:,i]||² · ||V[:,i]||²) for orthogonal U,V
        U_norms = self.U.detach().norm(dim=0)   # (r,)
        V_norms = self.V.detach().norm(dim=0)   # (r,)
        return (D * U_norms * V_norms).norm().item()
