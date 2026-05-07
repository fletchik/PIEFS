"""LambdaUTrotter: replacement for LambdaUPinn that fixes audit bugs §1.5, §1.6, §2.12.

Why this exists
---------------
LambdaUPinn used a frozen MLP to approximate expm(ω(x))·v.  The audit found
three problems:

  §1.5  apply_to() divides by ‖w‖, calls PINN on a unit vector, then multiplies
        by ‖w‖.  This assumes U·(αv) = α·(U·v), i.e. linearity in v.  But the
        PINN is MLP+Tanh — Tanh is NOT linear.  So ‖A∇φ‖² is biased.

  §1.6  After pretraining on iid Gaussian ω, the PINN is FROZEN forever.
        _omega_mlp(x) outputs values that drift outside the [−3, 3] range
        the PINN saw during pretraining → silent OOD failure.

  §2.12 _omega_mlp has unbounded Linear output.  After Tanh+Linear it can
        emit angles outside [-π, π] which makes no geometric sense and
        further amplifies OOD problems.

This class replaces the entire PINN machinery with a *direct* Trotter
product of Givens rotations.  The result is:

  •  Exact orthogonality:  U is a true rotation (product of 2D rotations).
  •  Exact 1-homogeneity in v:  R·(αv) = α·(R·v) by linearity of rotation.
  •  No distribution shift:  no learned approximator that can extrapolate.
  •  Same O(B·d) cost as the PINN's apply_to().
  •  Bounded angles:  ω(x) is squashed through tanh·π so geometry is sane.

Mathematical note
-----------------
The Trotter product is

   U_trotter(ω) = R_{d-2}(ω_{d-2}) · ... · R_1(ω_1) · R_0(ω_0)

where R_i is the 2D rotation in the (i, i+1) plane by angle ω_i.  This is
NOT identical to expm(skew_first_offdiag(ω)) but is itself a valid orthogonal
matrix lying in a (d−1)-parameter subgroup of SO(d).  For an unconstrained
SO(d) approximation pass two or three Trotter sweeps with shifted angles
(see RESEARCH_PLAN.md §1.3 Proposal C — multi-pass Trotter).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .diag_metric import _make_mlp


class LambdaUTrotter(nn.Module):
    """A(x) = U_trotter(ω(x)) · diag(λ(x)),  with bounded ω and exact orthogonality.

    Args:
        input_dim: Dimensionality d.
        hidden_dims: Hidden widths for both the λ-MLP and the ω-MLP.
        n_passes: Number of consecutive Trotter sweeps.  P=1 covers a
            (d−1)-parameter subgroup; P=2 or 3 covers progressively more
            of SO(d) at no extra asymptotic cost.  Default 1 for parity
            with LambdaUPinn / LambdaUSparse.
        bound_omega: If True, ω = π · tanh(MLP(x)) so angles ∈ [−π, π].
            Strongly recommended; the unbounded variant exists only for
            ablation studies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        n_passes: int = 1,
        bound_omega: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.d = input_dim
        self.n_passes = int(n_passes)
        self.bound_omega = bool(bound_omega)
        # λ(x): det(Λ) = 1 by  Σ_i log λ_i = 0  (mean-subtracted exp).
        self._lam_mlp = _make_mlp(input_dim, hidden_dims, input_dim)
        # ω(x): output dim = (d−1) per Trotter pass.  Multiple passes use
        # independent angle sets concatenated along the output axis.
        self._omega_mlp = _make_mlp(input_dim, hidden_dims,
                                     (input_dim - 1) * self.n_passes)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _omega(self, x: torch.Tensor) -> torch.Tensor:
        """Compute angles ω(x).  Shape (B, n_passes, d-1)."""
        raw = self._omega_mlp(x)
        if self.bound_omega:
            raw = math.pi * torch.tanh(raw)
        return raw.view(-1, self.n_passes, self.d - 1)

    def _trotter_rotate(self, omega: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply n_passes sweeps of (i,i+1) Givens rotations to v.

        Args:
            omega: (B, P, d-1)  angles per pass.
            v:     (B, d)
        Returns:
            (B, d) rotated vector.
        """
        result = v.clone()
        # Each pass shifts the starting plane by `pass_idx` so consecutive
        # sweeps cover different planes.  Pass 0: planes (0,1),(1,2),... ;
        # pass 1: planes (1,2),(2,3),...; etc.  This widens the reachable
        # subgroup of SO(d) without changing the per-pass cost.
        for p in range(self.n_passes):
            shift = p
            for i in range(self.d - 1):
                a = (i + shift) % self.d
                b = (i + 1 + shift) % self.d
                ang = omega[:, p, i]
                c = torch.cos(ang)
                s = torch.sin(ang)
                ra = result[:, a].clone()
                rb = result[:, b].clone()
                result[:, a] = ra * c - rb * s
                result[:, b] = ra * s + rb * c
        return result

    # ------------------------------------------------------------------
    # Public API (matches the LambdaU* contract used by SpectralModel)
    # ------------------------------------------------------------------

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A(x) · v = U_trotter(ω(x)) · (Λ(x) · v) directly.

        No norm-rescale trick.  Trotter is exactly 1-homogeneous in v, so
        the audit §1.5 issue does not arise.

        Args:
            x: (B, d)
            v: (B, d) gradient ∇φ
        Returns:
            (B, d) = U·Λ·v
        """
        # Λ part: det = 1 by mean-subtraction.
        raw = self._lam_mlp(x)
        raw = raw - raw.mean(dim=1, keepdim=True)
        lam = torch.exp(raw)            # (B, d) > 0,  Π λ_i = 1
        w = lam * v                     # element-wise scaling
        # U part: Trotter sweep(s).  Exact orthogonal map.
        omega = self._omega(x)          # (B, P, d-1)
        return self._trotter_rotate(omega, w)

    def get_lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Return diag(Λ(x)) as a (B, d) vector — for diagnostics/logging."""
        raw = self._lam_mlp(x)
        raw = raw - raw.mean(dim=1, keepdim=True)
        return torch.exp(raw)

    def get_omega(self, x: torch.Tensor) -> torch.Tensor:
        """Return rotation angles (B, P, d-1) — for diagnostics/logging."""
        return self._omega(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full A(x) as a (B, d, d) matrix.  Use apply_to() for training; this
        is for visualisation only.  Cost O(B·d²)."""
        B = x.shape[0]
        eye = torch.eye(self.d, device=x.device, dtype=x.dtype)
        # Apply A to each column of identity → assemble A column-by-column.
        cols = []
        for j in range(self.d):
            ej = eye[j].unsqueeze(0).expand(B, self.d).contiguous()
            cols.append(self.apply_to(x, ej))
        return torch.stack(cols, dim=2)  # (B, d, d)
