"""LocalLowRankMetric: A(x) = I + U(x)·Λ(x)·V(x)ᵀ — x-dependent low-rank metric.

Why this exists
---------------
GlobalLowRankMetric uses a single fixed A = I + UDVᵀ for all x. This is
powerful (no MLP bottleneck) but cannot adapt to local structure — near class
boundaries vs. class centres vs. outliers may benefit from different scalings.

LocalLowRankMetric keeps the low-rank structure but makes U, V, Λ functions of x:

    A(x) = I + U(x) · Λ(x) · V(x)ᵀ

where:
    U(x), V(x) ∈ ℝ^(d×r) are the columns of two MLPs: ℝ^d → ℝ^(d·r)
    Λ(x) = diag(λ(x)),  λ(x) ∈ ℝ^r,  from a third MLP head.

Advantages over Trotter
-----------------------
1. Full rank-r coverage: For rank r ≥ C-1, can represent any Fisher-optimal
   direction at each x. Trotter covers only (d-1) adjacent-Givens subgroup.
2. No subgroup restriction: Any rank-r perturbation of identity is reachable.

Advantages over GlobalLowRank
------------------------------
Local adaptivity: class boundaries (high σ(x)) can have stronger metric
perturbation than class interiors, potentially learning sharper eigenfunctions.

Computational cost
------------------
apply_to: O(B · d · r)    — same as GlobalLowRank
forward MLP: O(B · d · h) — same as DiagMetric

The shared backbone reduces cost vs. three separate MLPs.

Implementation
--------------
A shared MLP encoder maps x → h(x) ∈ ℝ^hidden.
Three linear heads decode:
    U_head: hidden → d·r  → reshape to (B, d, r)
    V_head: hidden → d·r  → reshape to (B, d, r)
    lam_head: hidden → r  → softplus → (B, r)  (positive scaling)

apply_to computes:
    Vtv  = einsum('bdr,bd->br', V(x), v)    # (B, r): V(x)ᵀ v
    scaled = Vtv * λ(x)                      # (B, r): Λ(x) V(x)ᵀ v
    Av = v + einsum('bdr,br->bd', U(x), scaled)  # (B, d): v + U(x) Λ(x) V(x)ᵀ v
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalLowRankMetric(nn.Module):
    """A(x) = I + U(x)·Λ(x)·V(x)ᵀ — x-dependent low-rank metric perturbation.

    A shared MLP backbone encodes x; three linear heads decode U, V, λ.

    Args:
        input_dim: Input dimensionality d.
        r: Rank of the perturbation. Recommended: r = num_classes - 1.
        hidden_dims: MLP hidden layer widths for the shared backbone.
        init_scale: Scale for the U/V head weight init (0.0 = exact identity at
            init; 0.01 = near-identity with nonzero gradient from step 1).
    """

    def __init__(
        self,
        input_dim: int,
        r: int = 8,
        hidden_dims: list[int] | None = None,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.d = input_dim
        self.r = r

        # Shared backbone: x → h ∈ ℝ^h_last
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        h_last = in_dim

        # U head: ℝ^h → ℝ^(d·r)
        self.U_head = nn.Linear(h_last, input_dim * r)
        # V head: ℝ^h → ℝ^(d·r)
        self.V_head = nn.Linear(h_last, input_dim * r)
        # λ head: ℝ^h → ℝ^r  (before softplus)
        self.lam_head = nn.Linear(h_last, r)

        # Near-identity init: small weights for U/V heads, zero bias for λ head.
        nn.init.normal_(self.U_head.weight, std=init_scale)
        nn.init.zeros_(self.U_head.bias)
        nn.init.normal_(self.V_head.weight, std=init_scale)
        nn.init.zeros_(self.V_head.bias)
        nn.init.zeros_(self.lam_head.weight)
        nn.init.zeros_(self.lam_head.bias)

    def _uvl(self, x: torch.Tensor):
        """Compute U(x), V(x), λ(x) from input x.

        Returns:
            U: (B, d, r)
            V: (B, d, r)
            lam: (B, r) — positive (after softplus)
        """
        h = self.backbone(x)           # (B, h_last)
        B = x.shape[0]
        U = self.U_head(h).view(B, self.d, self.r)     # (B, d, r)
        V = self.V_head(h).view(B, self.d, self.r)     # (B, d, r)
        lam = F.softplus(self.lam_head(h)) + 1e-6      # (B, r) positive
        return U, V, lam

    def apply_to(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute A(x)·v = v + U(x)·Λ(x)·V(x)ᵀ·v.

        Args:
            x: (B, d) input points.
            v: (B, d) gradient ∇φ_k.
        Returns:
            (B, d) = v + U(x) Λ(x) V(x)ᵀ v.
        """
        U, V, lam = self._uvl(x)                    # (B,d,r), (B,d,r), (B,r)
        Vtv = torch.einsum('bdr,bd->br', V, v)       # (B, r): V(x)ᵀ v
        scaled = Vtv * lam                            # (B, r): Λ(x) V(x)ᵀ v
        delta = torch.einsum('bdr,br->bd', U, scaled) # (B, d): U(x) Λ(x) V(x)ᵀ v
        return v + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full A(x) as a (B, d, d) matrix — for diagnostics/visualisation only.

        Cost O(B·d²·r) — do not call during training.
        """
        U, V, lam = self._uvl(x)        # (B,d,r), (B,d,r), (B,r)
        # A(x) = I + U Λ Vᵀ for each sample
        # Uλ = U * lam[:,None,:]  shape (B, d, r)
        Ulam = U * lam.unsqueeze(1)     # (B, d, r)
        perturbation = torch.bmm(Ulam, V.transpose(1, 2))  # (B, d, d)
        eye = torch.eye(self.d, device=x.device, dtype=x.dtype).unsqueeze(0)
        return eye + perturbation        # (B, d, d)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lambda_stats(self, x: torch.Tensor) -> dict[str, float]:
        """Statistics of λ(x) for logging effective rank."""
        with torch.no_grad():
            _, _, lam = self._uvl(x)    # (B, r)
        lam_mean = lam.mean(0)          # (r,)
        eff_rank = (lam_mean.sum() ** 2 / (lam_mean ** 2).sum()).item()
        return {
            'lam_mean': lam_mean.mean().item(),
            'lam_max': lam_mean.max().item(),
            'lam_min': lam_mean.min().item(),
            'effective_rank': eff_rank,
        }
