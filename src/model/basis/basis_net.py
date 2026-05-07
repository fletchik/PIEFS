from __future__ import annotations

import torch
import torch.nn as nn


class BasisNet(nn.Module):
    """MLP: input_dim → [hidden_dims] → 1 with Tanh activations.

    Args:
        input_dim: Dimensionality of input x.
        hidden_dims: Hidden layer widths. Defaults to [64, 64, 64].
        output_bias: Whether the final linear layer has a bias term.
            Set to False (recommended) so that Σ φ_k(x) ≠ const is not
            an admissible solution — the constant function φ ≡ c satisfies
            L_gram but carries no classification signal, and with bias=True
            it can appear as the "first eigenfunction".
            Backward-compat default True (matches all checkpoints before
            commit fix(basis): output_bias).  New experiments should use False.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-1], 1, bias=output_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, 1) scalar output.
        """
        return self.net(x)
