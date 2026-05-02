from __future__ import annotations

import torch
import torch.nn as nn

from .basis_net import BasisNet


class BasisFunction(nn.Module):
    """Scalar eigenfunction φ_k: R^d → R.

    Wraps BasisNet. Provides gradient-enabled forward pass for Dirichlet loss
    and a no-grad predict method for building the frozen Gram columns.

    Args:
        input_dim: Input dimensionality d.
        hidden_dims: Hidden layer widths. Defaults to [64, 64, 64].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.net = BasisNet(input_dim, hidden_dims)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate φ_k(x) without building a computation graph.

        Used for frozen functions when assembling the Gram matrix.

        Args:
            x: (B, d)
        Returns:
            (B, 1) scalar values, detached.
        """
        return self.net(x)

    def forward(
        self,
        x: torch.Tensor,
        return_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evaluate φ_k(x) and optionally compute ∇_x φ_k(x).

        Args:
            x: (B, d) input tensor.
            return_grad: If True, compute ∇φ_k via autograd.
        Returns:
            phi: (B, 1) function values.
            grad: (B, d) gradient w.r.t. x, or None if return_grad=False.
        """
        if not return_grad:
            return self.net(x), None

        # Clone-detach x so gradient flows into net parameters but not upstream.
        x_ = x.detach().requires_grad_(True)
        phi = self.net(x_)
        grad = torch.autograd.grad(
            outputs=phi,
            inputs=x_,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]
        return phi, grad
