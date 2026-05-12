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
        w_task_eff   = w_task  * exp(-gram_error² / t_orth)
        w_mde_eff    = w_dirichlet * exp(-max(gram_error²/t_orth, L_task/t_class))
        total = w_gram_eff*L_gram + w_task_eff*L_task + w_mde_eff*L_dir

        Note: ratio_gram uses gram_error**2 (= L_gram = ||C-I||_F²), matching
        the original paper eq. 10 where w_class ∝ exp(-L_orth/T_orth) and
        L_orth = ||C-I||_F² (squared Frobenius norm). Using gram_error (not
        squared) suppresses the classifier ~10-12× more aggressively than
        intended, preventing the MDE term from ever activating.

        Weights are computed stop-gradient (paper eq. 9: sg(w)), so gradients
        flow only through the loss terms, not through the weight values.

    Args:
        w_gram: Base weight for gram orthogonality term.
        w_dirichlet: Base weight for Dirichlet energy term.
        w_task: Base weight for task (classification) term.
        dynamic_weighting: If True, use adaptive weighting from paper eq. 10.
        t_orth: Target gram_error² = ||C-I||_F² for dynamic weighting.
        t_class: Target task loss for dynamic weighting.
        phase1_end_step: Step at which Phase 1 ends and metric ramp begins.
            Phase 1 (0 → phase1_end_step): w_mde = 0 (basis trains without
            metric interference, equivalent to EFDO-off).
            Phase 2 (phase1_end_step → phase2_end_step): w_mde ramps linearly
            from 0 to the dynamic value.
            Phase 3 (phase2_end_step → end): full dynamic weighting.
            Set both to 0 to disable three-phase scheduling (legacy behaviour).
        phase2_end_step: Step at which Phase 2 ends and Phase 3 begins.
    """

    def __init__(
        self,
        w_gram: float = 1.0,
        w_dirichlet: float = 1.0,
        w_task: float = 1.0,
        dynamic_weighting: bool = False,
        t_orth: float = 0.1,
        t_class: float = 0.5,
        phase1_end_step: int = 0,
        phase2_end_step: int = 0,
    ) -> None:
        super().__init__()
        self.w_gram = w_gram
        self.w_dirichlet = w_dirichlet
        self.w_task = w_task
        self.dynamic_weighting = dynamic_weighting
        self.t_orth = t_orth
        self.t_class = t_class
        self.phase1_end_step = phase1_end_step
        self.phase2_end_step = phase2_end_step

    def forward(
        self,
        phi_matrix: torch.Tensor,
        grad_phi_k: torch.Tensor,
        A: torch.Tensor | None,
        head_out: dict[str, torch.Tensor],
        k: int,
        Ag_pinn: torch.Tensor | None = None,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            phi_matrix: (B, k) basis outputs; columns 0..k-2 are detached.
            grad_phi_k: (B, d) gradient of the active φ_k w.r.t. x.
            A: (B, d, d) metric matrix, or None (identity metric).
            head_out: Output dict from BinaryHead / MulticlassHead.
            k: Current function index (1-based), used for off-diagonal tracking.
            Ag_pinn: (B, d) pre-applied A(x)∇φ (PINN path, legacy).
            global_step: Current training step, used for three-phase scheduling.
        Returns:
            Dict with keys: loss, loss_gram, loss_dirichlet, loss_task,
                            gram_error (||C-I||_F, scalar monitor),
                            off_diag_error_k (overlap of φ_k with predecessors),
                            w_task_eff, w_mde_eff (effective weights, for logging),
                            phase (int 1/2/3, for logging).
        """
        B = phi_matrix.shape[0]

        # Gram matrix and orthogonality loss.
        # NOTE (audit §1.3, intentionally deferred):
        #   Code computes:  loss_gram = ‖Ê[φφᵀ] − I‖²_F  (bias of MC estimator, squared)
        #   Paper Eq. 7:    L_gram = Σ_{αβ} E[(φ_α φ_β − δ_αβ)²]  (variance-aware)
        # These differ when φ_α φ_β has non-zero variance across samples.
        # The code version is a valid loss (squared MC gram error) and has been empirically
        # stable. The paper formula should be updated to: L_gram = ‖(1/N)ΦᵀΦ − I‖²_F
        # to match this implementation. No code change needed here; paper text needs fix.
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
        # Ag_pinn: (B,d) pre-applied A(x)∇φ from PINN (already computed).
        # A=None:   identity → ||∇φ||²
        # A=(B,d):  diagonal → ||λ ⊙ ∇φ||²    (DiagMetric, element-wise)
        # A=(B,d,d):full mat → ||A∇φ||²         (LambdaUSparse, bmm)
        if Ag_pinn is not None:
            # PINN pre-applied: A(x)∇φ already computed via apply_to()
            loss_dirichlet = (Ag_pinn ** 2).sum(dim=1).mean()
        elif A is None:
            # ||∇φ||² per sample = sum over d, then mean over B.
            # BUG-FIX: was .mean() which divides by B×d instead of B,
            # making MDE d× weaker than diag/sparse cases.
            loss_dirichlet = (grad_phi_k ** 2).sum(dim=1).mean()
        elif A.dim() == 2:
            Ag = A * grad_phi_k  # λ ⊙ ∇φ, element-wise
            loss_dirichlet = (Ag ** 2).sum(dim=1).mean()
        else:
            Ag = torch.bmm(A, grad_phi_k.unsqueeze(-1)).squeeze(-1)
            loss_dirichlet = (Ag ** 2).sum(dim=1).mean()

        loss_task = head_out['loss']

        # Three-phase curriculum scheduling.
        # Phase 1 (0 → phase1_end_step):  basis trains freely, w_mde = 0.
        # Phase 2 (p1 → phase2_end_step): w_mde ramps linearly 0 → dynamic.
        # Phase 3 (p2 → end):             full dynamic weighting.
        # Disabled when both phase end-steps are 0 (legacy behaviour).
        p1 = self.phase1_end_step
        p2 = self.phase2_end_step
        if p2 > p1 > 0:
            if global_step < p1:
                current_phase = 1
                phase_mde_scale = 0.0
            elif global_step < p2:
                current_phase = 2
                phase_mde_scale = (global_step - p1) / float(p2 - p1)
            else:
                current_phase = 3
                phase_mde_scale = 1.0
        else:
            current_phase = 3
            phase_mde_scale = 1.0

        # Effective weights — static or dynamic (paper eq. 10).
        if self.dynamic_weighting:
            with torch.no_grad():
                # Ratios: how far each loss is from its target.
                # ratio > 1 → loss still far from target → weight suppressed.
                # ratio < 1 → loss near/below target → weight activates.
                # BUG-FIX: use gram_error**2 (= L_gram = ||C-I||_F²) to match
                # original paper eq. 10: w_class ∝ exp(-L_orth/T_orth) where
                # L_orth is the squared Frobenius norm, not the norm itself.
                # Using gram_error (not squared) over-suppresses w_task by
                # ~10-12×, preventing MDE from ever receiving gradient signal.
                ratio_gram = (gram_error ** 2) / self.t_orth   # scalar (fixed)
                # When w_task=0 (unsupervised), ignore task ratio so MDE
                # activates based on gram convergence only (ratio_class=0).
                if self.w_task > 0:
                    ratio_class = loss_task / self.t_class
                else:
                    ratio_class = gram_error.new_zeros(())
                # stop-gradient weights (paper eq. 9: sg(w))
                w_task_eff = self.w_task * torch.exp(-ratio_gram)
                w_mde_eff = self.w_dirichlet * phase_mde_scale * torch.exp(
                    -torch.max(ratio_gram, ratio_class)
                )
        else:
            ratio_gram = gram_error.new_tensor(float('nan'))
            ratio_class = gram_error.new_tensor(float('nan'))
            w_task_eff = phi_matrix.new_tensor(self.w_task)
            w_mde_eff = phi_matrix.new_tensor(self.w_dirichlet * phase_mde_scale)

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
            'phase': current_phase,             # 1/2/3 for three-phase logging
            'phase_mde_scale': phase_mde_scale, # 0.0 → 1.0 ramp value
            # Effective weights (same as base weights when dynamic_weighting=False)
            'w_task_eff': w_task_eff.detach(),
            'w_mde_eff': w_mde_eff.detach(),
            # Normalised ratios (NaN when static): gram_error/t_orth, task/t_class
            # ratio > 1 → weight near 0;  ratio = 0 → weight = base value
            'ratio_gram': ratio_gram.detach(),
            'ratio_class': ratio_class.detach(),
        }
