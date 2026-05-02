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
        """Return diagonal metric matrices for a batch.

        Args:
            x: (B, d) input.
        Returns:
            A: (B, d, d) diagonal metric matrices with det = 1.
        """
        raw = self.mlp(x)  # (B, d)
        raw = raw - raw.mean(dim=1, keepdim=True)  # Σ raw_i = 0 ⟹ Σ log λ_i = 0
        lam = torch.exp(raw)  # (B, d), positive, det = 1
        return torch.diag_embed(lam)  # (B, d, d)
