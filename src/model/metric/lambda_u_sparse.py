from __future__ import annotations

import torch
import torch.nn as nn

from .diag_metric import _make_mlp


class LambdaUSparse(nn.Module):
    """Full Riemannian metric A(x) = U(x) · Λ(x).

    Λ(x) = diag(exp(raw − mean(raw))) — volume-preserving diagonal part.
    U(x) = expm(ω(x)) — rotation from a sparse skew-symmetric matrix.

    ω(x) has exactly 2(d-1) non-zero entries (first super- and sub-diagonal),
    giving O(d) parameters. expm of a skew-symmetric matrix is orthogonal,
    so U^T U = I analytically.

    Args:
        input_dim: Dimensionality d of input x.
        hidden_dims: Hidden layer widths. Defaults to [64, 64].
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
        self._lam_mlp = _make_mlp(input_dim, hidden_dims, input_dim)
        self._omega_mlp = _make_mlp(input_dim, hidden_dims, input_dim - 1)

    def _build_omega(self, v: torch.Tensor) -> torch.Tensor:
        """Construct a sparse skew-symmetric matrix from v ∈ R^(d-1).

        Only the first super-diagonal and sub-diagonal are non-zero.

        Args:
            v: (B, d-1)
        Returns:
            omega: (B, d, d) skew-symmetric, ||ω + ω^T||_F = 0.
        """
        B, d = v.shape[0], self.d
        omega = torch.zeros(B, d, d, device=v.device, dtype=v.dtype)
        idx = torch.arange(d - 1, device=v.device)
        omega[:, idx, idx + 1] = v
        omega[:, idx + 1, idx] = -v
        return omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A(x) = U(x) · Λ(x) for a batch.

        Args:
            x: (B, d)
        Returns:
            A: (B, d, d)
        """
        raw = self._lam_mlp(x)  # (B, d)
        raw = raw - raw.mean(dim=1, keepdim=True)
        lam = torch.exp(raw)  # (B, d)
        Lambda = torch.diag_embed(lam)  # (B, d, d)

        v = self._omega_mlp(x)  # (B, d-1)
        omega = self._build_omega(v)  # (B, d, d)
        U = torch.linalg.matrix_exp(omega)  # (B, d, d), U^T U = I

        return torch.bmm(U, Lambda)  # (B, d, d)
