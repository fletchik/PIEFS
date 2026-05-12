"""LambdaUTrotter: Trotter-product rotation metric replacing the PINN-based variant.

Why this exists
---------------
The previous LambdaUPinn used a frozen MLP to approximate expm(ω(x))·v.
Three correctness issues motivated replacing it:

  (1) apply_to() divided by ‖w‖, called the PINN on a unit vector, then
      multiplied by ‖w‖.  This assumes linearity in v, but MLP+Tanh is
      NOT linear — so ‖A∇φ‖² was biased.

  (2) After pretraining on iid Gaussian ω, the PINN was frozen.  During
      training, _omega_mlp(x) drifted outside its pretraining range →
      silent out-of-distribution failure.

  (3) The output layer of _omega_mlp was unbounded, allowing angles
      outside [−π, π] — geometrically undefined and numerically unstable.

This class replaces the PINN with a *direct* Trotter product of Givens
rotations.  The result is:

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
(see RESEARCH_PLAN.md — multi-pass Trotter section).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .diag_metric import _make_mlp


class LambdaUTrotter(nn.Module):
    """A(x) = diag(λ(x)) · U_trotter(ω(x)),  with bounded ω and exact orthogonality.

    The rotation U is applied first (U·v), then the diagonal scaling Λ scales
    each component of the rotated vector.  This means
        ‖A(x)·v‖² = ‖Λ(x)·U(x)·v‖² = Σ_i λ_i(x)² · [U(x)·v]_i²
    which depends on both Λ AND U, so the ω-MLP receives gradients from the
    MDE loss.  (Reversing the order, U·Λ·v, would give ‖Λv‖² — independent of U.)

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
        """Apply n_passes of vectorised even-then-odd Givens rotations to v.

        Replaces the original sequential Python loop (O(d) iterations) with
        two vectorised half-sweeps per pass (O(1) Python iterations).

        Even half-sweep:  simultaneously applies independent pairs
            (0,1), (2,3), (4,5), …          using angles ω[0], ω[2], ω[4], …
        Odd half-sweep:   simultaneously applies independent pairs
            (1,2), (3,4), (5,6), …          using angles ω[1], ω[3], ω[5], …

        Together the two sweeps cover all d−1 Givens rotations.  The rotation
        ordering differs from the original sequential sweep, but:
          •  U is still a product of d−1 Givens rotations → still in SO(d).
          •  The network learns the optimal angles for this ordering.
          •  Autograd graph shrinks from O(d) to O(1) nodes → ~d/2× speedup.
          •  For d=784 this gives a ≈400× wall-clock speedup on CPU.

        Multi-pass (n_passes > 1) applies n_passes independent even-odd
        cycles with separate angle sets, covering a wider subgroup of SO(d).

        Args:
            omega: (B, P, d-1)  angles per pass.
            v:     (B, d)
        Returns:
            (B, d) rotated vector.
        """
        result = v.clone()
        d = self.d
        n_e = d // 2           # number of even pairs: (0,1),(2,3),…
        n_o = (d - 1) // 2    # number of odd  pairs: (1,2),(3,4),…

        for p in range(self.n_passes):
            ang = omega[:, p, :]           # (B, d-1)
            c = torch.cos(ang)             # (B, d-1)
            s = torch.sin(ang)             # (B, d-1)

            # ---- Even half-sweep ----------------------------------------
            # Pairs (0,1),(2,3),…,(2*(n_e-1), 2*(n_e-1)+1).
            # Angle for pair (2k, 2k+1) is ω[2k]  → stride-2 slice of c, s.
            c_e = c[:, 0::2]               # (B, n_e)
            s_e = s[:, 0::2]               # (B, n_e)
            a_e = result[:, 0::2].clone()  # (B, n_e)  positions 0,2,4,…
            b_e = result[:, 1::2].clone()  # (B, n_e)  positions 1,3,5,…
            result[:, 0::2] = a_e * c_e - b_e * s_e
            result[:, 1::2] = a_e * s_e + b_e * c_e

            # ---- Odd half-sweep ------------------------------------------
            # Pairs (1,2),(3,4),…
            # Angle for pair (2k+1, 2k+2) is ω[2k+1] → odd stride-2 slice.
            if n_o == 0:
                continue
            c_o = c[:, 1::2][:, :n_o]              # (B, n_o)
            s_o = s[:, 1::2][:, :n_o]              # (B, n_o)
            a_o = result[:, 1:2 * n_o + 1:2].clone()   # positions 1,3,…,2*n_o-1
            b_o = result[:, 2:2 * n_o + 2:2].clone()   # positions 2,4,…,2*n_o
            result[:, 1:2 * n_o + 1:2] = a_o * c_o - b_o * s_o
            result[:, 2:2 * n_o + 2:2] = a_o * s_o + b_o * c_o

        return result

    # ------------------------------------------------------------------
    # Public API (matches the LambdaU* contract used by SpectralModel)
    # ------------------------------------------------------------------

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A(x) · v = Λ(x) · (U_trotter(ω(x)) · v) directly.

        Order: first rotate v by U (exact orthogonal map), then scale by Λ.
        This ensures ‖Av‖² = Σ_i λ_i² · (Uv)_i² depends on ω → ω-MLP trains.

        Args:
            x: (B, d)
            v: (B, d) gradient ∇φ
        Returns:
            (B, d) = Λ·U·v
        """
        # U part: Trotter sweep(s).  Exact orthogonal map, applied first.
        omega = self._omega(x)          # (B, P, d-1)
        u_v = self._trotter_rotate(omega, v)  # U·v
        # Λ part: det = 1 by mean-subtraction, applied second.
        raw = self._lam_mlp(x)
        raw = raw - raw.mean(dim=1, keepdim=True)
        lam = torch.exp(raw)            # (B, d) > 0,  Π λ_i = 1
        return lam * u_v                # Λ·(U·v) = (ΛU)·v

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
