"""ConformalMetric: A(x) = σ(x) · I — isotropic x-dependent scaling.

Mathematical background
-----------------------
A conformal metric is the simplest non-trivial x-dependent metric:
    M(x) = A(x)ᵀ A(x) = σ(x)² I

where σ(x) > 0 is a scalar function learned by a small MLP.

The Modified Dirichlet Energy becomes:
    D_A[φ] = E[σ(x)² ||∇φ||²]

This is equivalent to importance-weighting the gradient norm by position:
points with high σ(x) contribute more to the energy → eigenfunctions are
forced to be smooth primarily in high-σ regions.

Why useful as a baseline
------------------------
If ConformalMetric > Identity (metric_type=off), spatial position matters.
If ConformalMetric ≈ DiagMetric, anisotropy does not help.
If DiagMetric >> ConformalMetric, directional structure is key.

This comparison cleanly decomposes A(x) contributions:
    off → conformal (add x-dependence)
    conformal → diag (add anisotropy)
    diag → low_rank / trotter (add rotation)

Note on det constraint
----------------------
Unlike DiagMetric, we do NOT enforce det(A) = 1. Conformal scaling does not
preserve volume. This is intentional — the scale σ(x) is part of the signal.
The loss has a natural scale invariance at the total_loss level (weights can
compensate), so training remains stable.

If you want a volume-preserving conformal metric, use:
    A(x) = σ(x)^(1/d) · I with σ(x) = exp(MLP(x))
and set normalize_det=True.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConformalMetric(nn.Module):
    """A(x) = σ(x) · I — isotropic location-dependent scaling.

    The metric amplifies/suppresses gradient norm based on position x.
    No directional structure — purely isotropic.

    Args:
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden layer widths for σ(x). Default [64, 64].
        normalize_det: If True, output σ(x)^(1/d) instead of σ(x) so that
            det(A) = 1 (volume-preserving conformal transform). Default False.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        normalize_det: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.d = input_dim
        self.normalize_det = normalize_det

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def _sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Compute σ(x) ∈ (0, ∞) with shape (B, 1)."""
        raw = self.net(x)                        # (B, 1)
        sigma = torch.nn.functional.softplus(raw) + 1e-6  # positive
        if self.normalize_det:
            # σ^(1/d) so that det(A)^(1/d) = σ^(1/d) → det(A)=1 iff σ=1
            sigma = sigma ** (1.0 / self.d)
        return sigma                              # (B, 1)

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A(x)·v = σ(x) · v.

        Args:
            x: (B, d) input points.
            v: (B, d) gradient ∇φ_k.
        Returns:
            (B, d) = σ(x) * v.
        """
        sigma = self._sigma(x)      # (B, 1)
        return sigma * v            # (B, d)  broadcast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full A as a (B, d) diagonal vector (same as σ(x) for all dims).

        This follows the DiagMetric convention: forward returns the diagonal
        λ(x) such that A(x)·v = λ(x) ⊙ v. For conformal, λ_i = σ(x) ∀i.
        """
        sigma = self._sigma(x)          # (B, 1)
        return sigma.expand(-1, self.d) # (B, d)  same scalar for each dim

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_sigma_stats(self, x: torch.Tensor) -> dict[str, float]:
        """Return min/mean/max of σ(x) for logging."""
        with torch.no_grad():
            sigma = self._sigma(x).squeeze(-1)  # (B,)
        return {
            'sigma_min': sigma.min().item(),
            'sigma_mean': sigma.mean().item(),
            'sigma_max': sigma.max().item(),
            'sigma_std': sigma.std().item(),
        }
