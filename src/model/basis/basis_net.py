from __future__ import annotations

import torch
import torch.nn as nn


class BasisNet(nn.Module):
    """MLP: input_dim → [hidden_dims] → 1 with Tanh activations.

    Args:
        input_dim: Dimensionality of input x.
        hidden_dims: Hidden layer widths. Defaults to [64, 64, 64].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-1], 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, 1) scalar output.
        """
        return self.net(x)
