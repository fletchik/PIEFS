from __future__ import annotations

import torch
import torch.nn as nn

from .basis_function import BasisFunction


class BasisSet(nn.Module):
    """Manages K scalar basis functions φ_1, …, φ_K.

    All functions are allocated at construction. The sequential trainer
    unfreezes them one at a time via set_active(k).

    Args:
        K: Total number of basis functions.
        input_dim: Input dimensionality d.
        hidden_dims: Hidden layer widths for each BasisNet.
    """

    def __init__(
        self,
        K: int,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.functions: nn.ModuleList = nn.ModuleList(
            [BasisFunction(input_dim, hidden_dims, output_bias=output_bias)
             for _ in range(K)]
        )
        self.freeze_all()

    def set_active(self, k: int) -> None:
        """Freeze all functions except the k-th (1-indexed).

        Switches function k to train mode and enables its gradients.
        All others are frozen in eval mode.

        Args:
            k: 1-based index of the function to unfreeze.
        """
        for i, fn in enumerate(self.functions):
            active = i == k - 1
            fn.train(active)
            for p in fn.parameters():
                p.requires_grad_(active)

    def freeze_all(self) -> None:
        """Freeze all functions (eval mode, no gradients)."""
        for fn in self.functions:
            fn.eval()
            for p in fn.parameters():
                p.requires_grad_(False)

    def get_phi_matrix(
        self,
        x: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Φ_{1..k}(x) and ∇φ_k(x).

        Frozen functions (i < k) are evaluated via predict() — no graph.
        The active function (i == k) is evaluated with return_grad=True.

        Args:
            x: (B, d) input.
            k: Current training index (1-based). Functions 1..k-1 are frozen.
        Returns:
            phi_matrix: (B, k) — columns i < k are detached, column k has grad.
            grad_phi_k: (B, d) gradient of φ_k w.r.t. x.
        """
        cols: list[torch.Tensor] = []
        grad_phi_k: torch.Tensor | None = None

        for i in range(k):
            fn = self.functions[i]
            if i == k - 1:
                phi_i, grad_phi_k = fn(x, return_grad=True)
            else:
                phi_i = fn.predict(x)
            cols.append(phi_i)

        phi_matrix = torch.cat(cols, dim=1)  # (B, k)
        return phi_matrix, grad_phi_k  # type: ignore[return-value]
