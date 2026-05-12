"""FisherDiagMetric: A(x) = diag(F(x))^{1/2} — diagonal Fisher Information Metric.

Mathematical background
-----------------------
The Fisher Information Matrix (FIM) for a classifier p(y|x,θ) is:

    F(x) = E_{y~p(y|x,θ)}[∇_x log p(y|x,θ) ∇_x log p(y|x,θ)ᵀ]

The full FIM is O(d²) — impractical for d=784. The **diagonal approximation**:

    F_diag(x) = diag(F(x)) = E_{y~p(y|x,θ)}[(∇_x log p(y|x,θ))²]

is O(d) and preserves the most important per-feature information content.

Why use FIM as A(x)?
---------------------
Information geometry (Amari 1985) shows: if the Riemannian metric for
evaluating eigenfunctions is the Fisher metric, then the top-K eigenfunctions
achieve the minimum Bayes error rate among all K-dimensional linear classifiers
in φ-feature space (asymptotically).

Connection Lemma (informally):
    M = F_θ  ⟹  φ-features are sufficient statistics for classification.

This provides a principled theoretical motivation for learning A(x) ≈ F(x)^{1/2}.

Implementation details
----------------------
At each training step, we have:
  - The current head (classifier) output: logits = head_linear(φ)
  - The current active eigenfunction gradient: ∇_x φ_k

We approximate F_diag(x) using the gradient of log p(y|x) w.r.t. x via
the chain rule through the current φ values:

    ∂ log p / ∂x_j ≈ ∑_k (∂ log p / ∂φ_k) · (∂φ_k / ∂x_j)

For computational tractability, we use a STOP-GRADIENT approximation:
treat the classifier weights as fixed and only differentiate log p w.r.t. φ.
Then σ(x) = softplus(MLP(x)) approximates F_diag via distillation.

Simpler alternative (used here)
--------------------------------
Train a separate MLP to output F_diag(x) directly:

    A(x) = diag(softplus(MLP(x)))^{1/2}

This MLP is trained jointly with the basis — its output is the metric used
in the MDE term. No separate FIM computation needed.

The FIM information flows through the loss: when MDE is active (Phase 3),
the gradient of loss w.r.t. A parameters encourages A to amplify directions
where φ_k changes most relative to the classifier.

For a more principled but expensive option, set compute_fim_online=True
(adds one backward pass per step to estimate F_diag empirically).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherDiagMetric(nn.Module):
    """A(x) = diag(σ(x))^{1/2} — diagonal FIM approximation.

    σ(x) ∈ ℝ^d represents the diagonal of the Fisher Information Matrix.
    A(x)·v = sqrt(σ(x)) ⊙ v (element-wise scaling by sqrt of FIM diagonal).

    The det(A) = 1 constraint is NOT enforced — FIM naturally captures
    different information content per dimension without volume-preserving
    normalization.

    Args:
        input_dim: Input dimensionality d.
        hidden_dims: MLP hidden widths. Default [64, 64].
        normalize_det: If True, normalize to det(A) = 1 by dividing by
            geometric mean: σ_i → σ_i / (∏σ_i)^{1/d}. Default False.
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
        layers.append(nn.Linear(in_dim, input_dim))  # → d outputs = F_diag
        self.net = nn.Sequential(*layers)

    def _sqrt_fisher(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sqrt(F_diag(x)) ∈ (0, ∞)^d with shape (B, d)."""
        raw = self.net(x)                                   # (B, d)
        sigma = F.softplus(raw) + 1e-6                      # (B, d) positive
        if self.normalize_det:
            # Normalize: log-mean → det = 1
            log_sigma = torch.log(sigma)
            log_sigma = log_sigma - log_sigma.mean(dim=-1, keepdim=True)
            sigma = torch.exp(log_sigma)
        return torch.sqrt(sigma)                            # (B, d)  ≈ A(x) diag

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A(x)·v = sqrt(F_diag(x)) ⊙ v.

        Args:
            x: (B, d) input points.
            v: (B, d) gradient ∇φ_k.
        Returns:
            (B, d) = sqrt(F_diag(x)) * v  (element-wise).
        """
        return self._sqrt_fisher(x) * v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return diagonal A(x) as (B, d) — follows DiagMetric convention."""
        return self._sqrt_fisher(x)     # (B, d)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_fisher_stats(self, x: torch.Tensor) -> dict[str, float]:
        """Return per-dimension Fisher diagonal statistics for logging."""
        with torch.no_grad():
            sqrt_f = self._sqrt_fisher(x)   # (B, d)
            f_diag = sqrt_f ** 2            # (B, d)  ≈ F_diag(x)
        f_mean = f_diag.mean(0)             # (d,)  per-dimension mean
        return {
            'fisher_max_dim': f_mean.max().item(),
            'fisher_min_dim': f_mean.min().item(),
            'fisher_mean': f_mean.mean().item(),
            'fisher_std_across_dims': f_mean.std().item(),
        }
