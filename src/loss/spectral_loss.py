from __future__ import annotations

import torch
import torch.nn as nn


class SpectralDirichletLoss(nn.Module):
    """Combined spectral loss: Gram orthogonality + Dirichlet energy + task CE.

    Gram loss:       L_gram = ||C_k - I||_F^2,  C_k = (1/B) Φ_{1..k}^T Φ_{1..k}
    Dirichlet loss:  L_dir  = mean(||A(x)∇φ_k||²)  (plain ||∇φ||² when A=None)
    Task loss:       L_task = BCE or CE from head.forward()

    Static mode (dynamic_weighting=False):
        total = w_gram*L_gram + w_dirichlet*L_dir + w_task*L_task

    Dynamic mode (dynamic_weighting=True, paper eq. 9-10):
        w_gram_eff   = w_gram   (always active)
        w_task_eff   = w_task  * exp(-gram_error / t_orth)
        w_mde_eff    = w_dirichlet * exp(-max(gram_error/t_orth, L_task/t_class))
        total = w_gram_eff*L_gram + w_task_eff*L_task + w_mde_eff*L_dir

        Weights are computed stop-gradient (paper eq. 9: sg(w)), so gradients
        flow only through the loss terms, not through the weight values.

    Args:
        w_gram: Base weight for gram orthogonality term.
        w_dirichlet: Base weight for Dirichlet energy term.
        w_task: Base weight for task (classification) term.
        dynamic_weighting: If True, use adaptive weighting from paper eq. 10.
        t_orth: Target gram_error = ||C-I||_F for dynamic weighting.
        t_class: Target task loss for dynamic weighting.
    """

    def __init__(
        self,
        w_gram: float = 1.0,
        w_dirichlet: float = 1.0,
        w_task: float = 1.0,
        dynamic_weighting: bool = False,
        t_orth: float = 0.1,
        t_class: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_gram = w_gram
        self.w_dirichlet = w_dirichlet
        self.w_task = w_task
        self.dynamic_weighting = dynamic_weighting
        self.t_orth = t_orth
        self.t_class = t_class

    def forward(
        self,
        phi_matrix: torch.Tensor,
        grad_phi_k: torch.Tensor,
        A: torch.Tensor | None,
        head_out: dict[str, torch.Tensor],
        k: int,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            phi_matrix: (B, k) basis outputs; columns 0..k-2 are detached.
            grad_phi_k: (B, d) gradient of the active φ_k w.r.t. x.
            A: (B, d, d) metric matrix, or None (identity metric).
            head_out: Output dict from BinaryHead / MulticlassHead.
            k: Current function index (1-based), used for off-diagonal tracking.
        Returns:
            Dict with keys: loss, loss_gram, loss_dirichlet, loss_task,
                            gram_error (||C-I||_F, scalar monitor),
                            off_diag_error_k (overlap of φ_k with predecessors),
                            w_task_eff, w_mde_eff (effective weights, for logging).
        """
        B = phi_matrix.shape[0]

        # Gram matrix and orthogonality loss.
        C_k = (phi_matrix.T @ phi_matrix) / B  # (k, k)
        I_k = torch.eye(k, device=phi_matrix.device, dtype=phi_matrix.dtype)
        diff = C_k - I_k
        gram_error = torch.norm(diff, p='fro')  # ||C-I||_F  (not squared, for monitoring)
        loss_gram = gram_error ** 2

        # Per-function overlap: how much φ_k overlaps with φ_1..k-1.
        if k > 1:
            off_diag_error_k = torch.norm(C_k[k - 1, : k - 1])
        else:
            off_diag_error_k = phi_matrix.new_zeros(())

        # Dirichlet energy: L_mde = mean ||A(x)∇φ||²  (paper eq. 5-6).
        if A is None:
            loss_dirichlet = (grad_phi_k ** 2).mean()
        else:
            Ag = torch.bmm(A, grad_phi_k.unsqueeze(-1)).squeeze(-1)  # (B, d)
            loss_dirichlet = (Ag ** 2).sum(dim=1).mean()  # ||A∇φ||²

        loss_task = head_out['loss']

        # Effective weights — static or dynamic (paper eq. 10).
        if self.dynamic_weighting:
            with torch.no_grad():
                # Ratios: how far each loss is from its target.
                ratio_gram = gram_error / self.t_orth          # scalar
                # When w_task=0 (unsupervised), ignore task ratio so MDE
                # activates based on gram convergence only (ratio_class=0).
                if self.w_task > 0:
                    ratio_class = loss_task / self.t_class
                else:
                    ratio_class = gram_error.new_zeros(())
                # stop-gradient weights (paper eq. 9: sg(w))
                w_task_eff = self.w_task * torch.exp(-ratio_gram)
                w_mde_eff = self.w_dirichlet * torch.exp(
                    -torch.max(ratio_gram, ratio_class)
                )
        else:
            w_task_eff = phi_matrix.new_tensor(self.w_task)
            w_mde_eff = phi_matrix.new_tensor(self.w_dirichlet)

        total = (
            self.w_gram * loss_gram
            + w_task_eff * loss_task
            + w_mde_eff * loss_dirichlet
        )

        return {
            'loss': total,
            'loss_gram': loss_gram,
            'loss_dirichlet': loss_dirichlet,
            'loss_task': loss_task,
            'gram_error': gram_error.detach(),
            'off_diag_error_k': off_diag_error_k.detach(),
            'w_task_eff': w_task_eff.detach(),
            'w_mde_eff': w_mde_eff.detach(),
        }
