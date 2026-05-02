from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diag_metric import _make_mlp

logger = logging.getLogger(__name__)


class _RotationPINN(nn.Module):
    """MLP approximating v(1) = U(x) · v_0.

    Input: (x, v_0) concatenated → Output: v_hat ≈ expm(ω(x)) · v_0.

    Args:
        x_dim: Dimensionality of x.
        v_dim: Dimensionality of v (same as x_dim = d).
        hidden_dims: Hidden layer widths.
    """

    def __init__(
        self,
        x_dim: int,
        v_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]
        self.net = _make_mlp(x_dim + v_dim, hidden_dims, v_dim)

    def forward(self, x: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, x_dim)
            v0: (B, v_dim) initial vector.
        Returns:
            v_hat: (B, v_dim) approximation of U(x) · v_0.
        """
        return self.net(torch.cat([x, v0], dim=-1))


class LambdaUPinn(nn.Module):
    """Full Riemannian metric A(x) = U(x) · Λ(x).

    Λ(x) is the same volume-preserving diagonal part as LambdaUSparse.
    U(x) is approximated by a PINN: (x, v_0) → U(x)·v_0, pretrained to match
    the exact matrix exponential expm(ω(x)) · v_0 on random (x, v_0) pairs.

    ω(x) uses the same sparse structure as LambdaUSparse (first off-diagonal).

    Call pretrain() before main training to supervise the PINN.

    Args:
        input_dim: Dimensionality d of input x.
        hidden_dims: Hidden widths for Λ and ω MLPs. Defaults to [64, 64].
        pinn_hidden_dims: Hidden widths for the PINN. Defaults to [128, 128, 128].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        pinn_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.d = input_dim
        self._lam_mlp = _make_mlp(input_dim, hidden_dims, input_dim)
        self._omega_mlp = _make_mlp(input_dim, hidden_dims, input_dim - 1)
        self._pinn = _RotationPINN(input_dim, input_dim, pinn_hidden_dims)

    def _build_omega(self, v: torch.Tensor) -> torch.Tensor:
        B, d = v.shape[0], self.d
        omega = torch.zeros(B, d, d, device=v.device, dtype=v.dtype)
        idx = torch.arange(d - 1, device=v.device)
        omega[:, idx, idx + 1] = v
        omega[:, idx + 1, idx] = -v
        return omega

    def pretrain(
        self,
        steps: int = 5000,
        lr: float = 1e-3,
        batch_size: int = 256,
        x_scale: float = 1.0,
        w_ortho: float = 0.1,
    ) -> None:
        """Pretrain the PINN to match expm(ω(x)) · v_0.

        Draws random (x, v_0) pairs, computes the exact rotation via
        torch.linalg.matrix_exp, and supervises the PINN with MSE loss.
        Also adds orthogonality regularization: for random orthogonal pairs
        (v_1, v_2), enforces ⟨PINN(x,v_1), PINN(x,v_2)⟩ ≈ 0.
        The Λ and ω MLPs are frozen during PINN pretraining.

        Args:
            steps: Number of gradient steps.
            lr: Adam learning rate.
            batch_size: Samples per step.
            x_scale: Std of the random x samples drawn during pretraining.
            w_ortho: Weight for orthogonality regularization loss.
        """
        device = next(self._pinn.parameters()).device
        dtype = next(self._pinn.parameters()).dtype

        for p in list(self._lam_mlp.parameters()) + list(self._omega_mlp.parameters()):
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(self._pinn.parameters(), lr=lr)

        for step in range(steps):
            x = torch.randn(batch_size, self.d, device=device, dtype=dtype) * x_scale
            v0 = torch.randn(batch_size, self.d, device=device, dtype=dtype)
            v0 = F.normalize(v0, dim=-1)

            with torch.no_grad():
                omega_v = self._omega_mlp(x)
                omega = self._build_omega(omega_v)
                U_exact = torch.linalg.matrix_exp(omega)  # (B, d, d)
                target = torch.bmm(U_exact, v0.unsqueeze(-1)).squeeze(-1)  # (B, d)

            v_hat = self._pinn(x, v0)
            loss_mse = F.mse_loss(v_hat, target)

            # Orthogonality regularization: two orthogonal inputs → orthogonal outputs.
            v1 = F.normalize(torch.randn(batch_size, self.d, device=device, dtype=dtype), dim=-1)
            # Gram-Schmidt: v2 ⊥ v1.
            v2 = torch.randn(batch_size, self.d, device=device, dtype=dtype)
            v2 = v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1
            v2 = F.normalize(v2, dim=-1)
            u1 = self._pinn(x.detach(), v1)
            u2 = self._pinn(x.detach(), v2)
            loss_ortho = ((u1 * u2).sum(dim=-1) ** 2).mean()

            loss = loss_mse + w_ortho * loss_ortho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                logger.info(
                    'PINN pretrain step %d / %d  mse=%.5f  ortho=%.5f',
                    step, steps, loss_mse.item(), loss_ortho.item(),
                )

        # Freeze PINN after pretraining; re-enable metric MLPs.
        for p in self._pinn.parameters():
            p.requires_grad_(False)
        for p in list(self._lam_mlp.parameters()) + list(self._omega_mlp.parameters()):
            p.requires_grad_(True)

        logger.info('PINN pretraining complete.')

    def _get_U(self, x: torch.Tensor) -> torch.Tensor:
        """Compute U(x) by evaluating the PINN for each basis vector.

        Assembles U column-by-column: U[:, :, j] = PINN(x, e_j).

        Args:
            x: (B, d)
        Returns:
            U: (B, d, d)
        """
        B, d = x.shape[0], self.d
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        cols = [self._pinn(x, eye[j].unsqueeze(0).expand(B, d)) for j in range(d)]
        return torch.stack(cols, dim=2)  # (B, d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A(x) = U_pinn(x) · Λ(x) for a batch.

        Args:
            x: (B, d)
        Returns:
            A: (B, d, d)
        """
        raw = self._lam_mlp(x)
        raw = raw - raw.mean(dim=1, keepdim=True)
        lam = torch.exp(raw)
        Lambda = torch.diag_embed(lam)  # (B, d, d)

        U = self._get_U(x)  # (B, d, d)
        return torch.bmm(U, Lambda)
