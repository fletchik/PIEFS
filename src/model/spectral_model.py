from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis.basis_set import BasisSet


class BinaryHead(nn.Module):
    """Classification head for binary tasks: Linear(K→1) + BCEWithLogitsLoss.

    Args:
        K: Number of basis functions (input dimensionality).
    """

    def __init__(self, K: int) -> None:
        super().__init__()
        self.linear = nn.Linear(K, 1)

    def forward(self, phi: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            phi: (B, K) basis function outputs.
            y: (B,) binary labels in {0, 1}.
        Returns:
            dict with keys: loss (scalar), logits (B,), probs (B,).
        """
        logits = self.linear(phi).squeeze(-1)  # (B,)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        return {'loss': loss, 'logits': logits, 'probs': torch.sigmoid(logits)}


class MulticlassHead(nn.Module):
    """Classification head for multiclass tasks: Linear(K→C) + CrossEntropyLoss.

    Args:
        K: Number of basis functions (input dimensionality).
        num_classes: Number of target classes C.
    """

    def __init__(self, K: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(K, num_classes)
        self.num_classes = num_classes

    def forward(self, phi: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            phi: (B, K) basis function outputs.
            y: (B,) integer class labels in [0, C).
        Returns:
            dict with keys: loss (scalar), logits (B, C), probs (B, C).
        """
        logits = self.linear(phi)  # (B, C)
        loss = F.cross_entropy(logits, y.long())
        return {'loss': loss, 'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class SpectralModel(nn.Module):
    """Spectral Dirichlet model combining basis set, Riemannian metric, and task head.

    During sequential training the trainer calls basis_set.set_active(k) to
    unfreeze φ_k. This model's forward assembles Φ_{1..k} and routes the
    representation to the head.

    Args:
        basis_set: Manages K scalar eigenfunctions.
        metric: Riemannian metric module (None for OFF variant).
        head: BinaryHead or MulticlassHead, injected at construction.
    """

    def __init__(
        self,
        basis_set: BasisSet,
        metric: nn.Module | None,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.basis_set = basis_set
        self.metric = metric
        self.head = head
        self.K = basis_set.K
        self._active_k: int = 0

    def set_active_k(self, k: int) -> None:
        """Tell the model which function is currently being trained (1-indexed)."""
        self._active_k = k

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full forward pass for the current active function k.

        Args:
            x: (B, d) input features.
            y: (B,) labels.
        Returns:
            phi_matrix: (B, k) — first k-1 columns detached, k-th has gradient.
            grad_phi_k: (B, d) gradient of the active function w.r.t. x.
            A: (B, d, d) metric, or None for OFF.
            head_out: dict from head forward (loss, logits, probs).
        """
        k = self._active_k
        phi_matrix, grad_phi_k = self.basis_set.get_phi_matrix(x, k)  # (B, k), (B, d)

        # Zero-pad phi to (B, K) for the head so its linear layer has fixed shape.
        B = x.shape[0]
        phi_full = phi_matrix.new_zeros(B, self.K)
        phi_full[:, :k] = phi_matrix

        # Compute metric output. Three cases:
        #   None          → identity metric ('off')
        #   (B, d)        → diagonal metric (DiagMetric, fast element-wise)
        #   (B, d, d)     → full matrix (LambdaUSparse)
        #
        # For LambdaUPinn: use apply_to(x, grad) directly — one PINN call
        # instead of assembling the full U matrix (d calls). This gives O(d)
        # speedup. We store the result as (B, d) pre-applied vector 'Ag'.
        from src.model.metric.lambda_u_pinn import LambdaUPinn
        if self.metric is None:
            A = None
        elif isinstance(self.metric, LambdaUPinn) and grad_phi_k is not None:
            # Pre-apply: Ag = A(x)·∇φ_k  via single PINN call
            A = self.metric.apply_to(x, grad_phi_k)  # (B, d), already applied
        else:
            A = self.metric(x)  # (B, d) for diag or (B, d, d) for sparse

        head_out = self.head(phi_full, y)

        return {
            'phi_matrix': phi_matrix,
            'grad_phi_k': grad_phi_k,
            'A': A,
            'head_out': head_out,
        }
