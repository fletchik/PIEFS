from __future__ import annotations

import torch
import torch.nn as nn


def _make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    """Build a small MLP with Tanh activations and a linear output layer."""
    dims = [input_dim] + hidden_dims
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.Tanh()])
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class DiagMetric(nn.Module):
    """Diagonal Riemannian metric Λ(x) = diag(λ(x)).

    λ(x) = exp(raw(x) − mean(raw(x))) enforces det(Λ) = 1 (Σ log λ_i = 0).

    Args:
        input_dim: Dimensionality d of input x.
        hidden_dims: Hidden layer widths of the MLP. Defaults to [64, 64].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.d = input_dim
        self.mlp = _make_mlp(input_dim, hidden_dims, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return diagonal scaling vectors for a batch.

        Returns λ(x) as a vector (B, d) rather than a full (B, d, d) matrix.
        The loss uses Ag = λ ⊙ ∇φ (element-wise), which is equivalent to
        diag(λ) @ ∇φ but avoids allocating a huge (B, d, d) tensor.

        Args:
            x: (B, d) input.
        Returns:
            lam: (B, d) diagonal scaling, det = 1 (Σ log λ_i = 0 by construction).
        """
        raw = self.mlp(x)  # (B, d)
        raw = raw - raw.mean(dim=1, keepdim=True)  # Σ raw_i = 0 ⟹ Σ log λ_i = 0
        return torch.exp(raw)  # (B, d), positive, det = 1
