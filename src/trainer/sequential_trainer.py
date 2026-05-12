from __future__ import annotations

import json
import logging
import time
from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..loss.spectral_loss import SpectralDirichletLoss
from ..model.metric.lambda_u_pinn import LambdaUPinn
from ..model.spectral_model import SpectralModel

logger = logging.getLogger(__name__)


def _inf_loop(loader: DataLoader):
    """Yield batches from *loader* indefinitely."""
    for dl in repeat(loader):
        yield from dl


class SequentialTrainer:
    """Step-based sequential trainer for SpectralModel.

    Training loop:
        For k in 1..K:
            unfreeze φ_k only
            train for steps_per_function steps
            freeze φ_k

    Total steps = K × steps_per_function = total_steps (fixed at 60 000).
    Logs at every log_step; saves checkpoints at every save_period steps.

    Args:
        model: SpectralModel wrapping basis_set, metric, and head.
        criterion: SpectralDirichletLoss.
        optimizer_fn: Factory returning a fresh Optimizer for trainable params.
        train_loader: Training DataLoader (finite; will be looped infinitely).
        val_loader: Validation DataLoader.
        checkpoint_dir: Directory for saving .pt checkpoints.
        writer: WandBWriter (or any object with add_scalar / add_image).
        experiment_logger: ExperimentLogger instance.
        run_id: Unique identifier for this run (used in filenames).
        total_steps: Total gradient steps across all functions (default 60 000).
        log_step: Interval (global steps) at which to log metrics (default 15 000).
        save_period: Interval (global steps) for checkpoint saves (default 30 000).
        device: Device string.
        skip_oom: If True, skip batches that trigger CUDA OOM.
        config: Full config dict for saving in checkpoints.
    """

    def __init__(
        self,
        model: SpectralModel,
        criterion: SpectralDirichletLoss,
        optimizer_fn: Callable[[list[torch.nn.Parameter]], Optimizer],
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str | Path,
        writer,
        experiment_logger,
        run_id: str,
        total_steps: int = 60_000,
        log_step: int = 15_000,
        save_period: int = 30_000,
        device: str = 'cpu',
        skip_oom: bool = True,
        config: dict | None = None,
        noise_std: float = 0.0,
        wide_normal_fraction: float = 0.0,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.train_iter = _inf_loop(train_loader)
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.exp_logger = experiment_logger
        self.run_id = run_id
        self.config = config or {}

        self.K = model.K
        self.total_steps = total_steps
        self.steps_per_function = total_steps // self.K
        self.log_step = log_step
        self.save_period = save_period
        self.device = device
        self.skip_oom = skip_oom
        self.noise_std = noise_std
        self.wide_normal_fraction = wide_normal_fraction
        # FIX (P0, §1.7): max_grad_norm=None disables clipping by default.
        # The old hardcoded 1.0 was too aggressive — it collapses the gradient
        # norm to 1.0 regardless of which loss term is dominant, thereby
        # undoing the dynamic weighting schedule (w_task_eff, w_mde_eff).
        # Set to a positive float (e.g. 10.0) only if raw gradients diverge.
        self.max_grad_norm = max_grad_norm

        # Accumulated metrics for result logging (keyed by global step).
        self._metrics_history: dict[int, dict] = {}

        # JSONL file for per-step metrics (written at every log_step).
        # Captures weights, losses, gram_error, val_acc — readable without WandB.
        self._metrics_jsonl: Path = self.checkpoint_dir / 'metrics.jsonl'

        # Timing.
        self._start_time: float = 0.0
        self._wall_time_per_function: list[float] = []

        # Eigenvalue history (Dirichlet energy per frozen φ_k).
        self._eigenvalue_history: list[float] = []

        # Best val accuracy for checkpoint_best_val.pt
        self._best_val_acc: float = -1.0

        # Current optimizer (stored for checkpoint saving)
        self._optimizer: Optimizer | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full sequential training pipeline."""
        self._start_time = time.time()

        # Pretrain PINN rotation if needed.
        if isinstance(self.model.metric, LambdaUPinn):
            logger.info('Pretraining PINN rotation (5 000 steps)...')
            self.model.metric.pretrain(steps=5000)

        for k in range(1, self.K + 1):
            logger.info('=' * 60)
            logger.info('Training φ_%d / %d  (%d steps)', k, self.K, self.steps_per_function)
            logger.info('=' * 60)

            fn_start = time.time()
            self.model.basis_set.set_active(k)
            self.model.set_active_k(k)

            # Fresh optimizer over all currently-trainable parameters.
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = self.optimizer_fn(trainable)

            self._train_function(k, self._optimizer)

            fn_time = time.time() - fn_start
            self._wall_time_per_function.append(fn_time)

            # Freeze φ_k once done.
            self.model.basis_set.functions[k - 1].eval()
            for p in self.model.basis_set.functions[k - 1].parameters():
                p.requires_grad_(False)

            # Log eigenvalue (Dirichlet energy) for frozen φ_k.
            eig_k = self._compute_dirichlet_energy(k)
            self._eigenvalue_history.append(eig_k)
            ordering_ok = all(
                self._eigenvalue_history[i] <= self._eigenvalue_history[i + 1] * 1.05
                for i in range(len(self._eigenvalue_history) - 1)
            )
            logger.info(
                'φ_%d frozen  Dirichlet energy (eigenvalue proxy) = %.5f  '
                'ordering_satisfied=%s',
                k, eig_k, ordering_ok,
            )
            if self.writer is not None:
                self.writer.set_step((k) * self.steps_per_function)
                self.writer.add_scalar(f'eigenvalue/phi_{k}', eig_k)
                self.writer.add_scalar('eigenvalue/ordering_satisfied', float(ordering_ok))

        logger.info('Sequential training complete.')
        self._save_checkpoint(self.total_steps, is_final=True)

        # Full-dataset gram_error after training all K functions.
        gram_final = self._compute_gram_error_on_loader(self.val_loader, loader_name='val')
        logger.info(
            'gram_error_val (full val set) = %.5f  offdiag = %.5f',
            gram_final['gram_error_val'],
            gram_final['gram_error_offdiag_val'],
        )
        if self.writer is not None:
            self.writer.set_step(self.total_steps)
            for name, val in gram_final.items():
                self.writer.add_scalar(f'final/{name}', val)

        # Store in last history entry.
        if self._metrics_history:
            last_step = max(self._metrics_history)
            self._metrics_history[last_step].update(gram_final)
            self._metrics_history[last_step]['eigenvalue_history'] = self._eigenvalue_history
            self._metrics_history[last_step]['wall_time_per_function'] = self._wall_time_per_function
            self._metrics_history[last_step]['wall_time_total'] = time.time() - self._start_time
            self._metrics_history[last_step]['eigenvalue_ordering_satisfied'] = (
                all(
                    self._eigenvalue_history[i] <= self._eigenvalue_history[i + 1] * 1.05
                    for i in range(len(self._eigenvalue_history) - 1)
                )
                if len(self._eigenvalue_history) > 1 else True
            )

    # ------------------------------------------------------------------
    # Per-function training
    # ------------------------------------------------------------------

    def _train_function(self, k: int, optimizer: Optimizer) -> None:
        self.model.train()
        # Keep frozen functions in eval even during model.train() call.
        for i, fn in enumerate(self.model.basis_set.functions):
            if i != k - 1:
                fn.eval()

        for local_step in range(self.steps_per_function):
            global_step = (k - 1) * self.steps_per_function + local_step

            try:
                loss_dict = self._training_step(optimizer, global_step=global_step)
            except torch.cuda.OutOfMemoryError:
                if self.skip_oom:
                    logger.warning('OOM at step %d — skipping batch.', global_step)
                    torch.cuda.empty_cache()
                    continue
                raise

            # Log at step 0 and every log_step thereafter.
            if global_step == 0 or (global_step + 1) % self.log_step == 0:
                val_metrics = self._evaluate()
                wall = time.time() - self._start_time
                loss_dict['wall_time_seconds'] = wall
                self._log(global_step + 1, k, loss_dict, val_metrics)
                self._metrics_history[global_step + 1] = {**loss_dict, **val_metrics}

                # Track best val accuracy.
                va = val_metrics.get('val/accuracy', -1.0)
                if va > self._best_val_acc:
                    self._best_val_acc = va
                    self._save_checkpoint(global_step + 1, name='checkpoint_best_val.pt')

            if (global_step + 1) % self.save_period == 0:
                self._save_checkpoint(global_step + 1)

    def _augment_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation from the paper (Sections 3-4).

        Two augmentation techniques:
        1. Noise perturbation: x -> x + eps, eps ~ N(0, noise_std^2)
           Prevents overfitting where NN learns piecewise-constant functions
           with zero gradient at data points.
        2. Wide normal points: replace a fraction of the batch with samples
           from N(0, 3*std(x)), with RANDOM labels sampled from the batch.
           These points push eigenfunctions to be smooth far from the data manifold
           (Dirichlet term), while random labels avoid biasing the classification head.

           NOTE: using label=0 for all wide points would bias multiclass heads;
           random labels from the real batch distribution keep the head unbiased.
        """
        # 1. Noise perturbation
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Wide normal distribution points
        if self.wide_normal_fraction > 0:
            B = x.shape[0]
            n_wide = max(1, int(B * self.wide_normal_fraction))
            n_real = B - n_wide
            # Auto-compute sigma_wide as 3x data std
            data_std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            wide_points = torch.randn(n_wide, x.shape[1], device=x.device) * (3 * data_std)
            # BUG-FIX: was torch.zeros — labels=0 biases multiclass head.
            # Use random labels sampled (with replacement) from the real batch.
            rand_idx = torch.randint(0, n_real, (n_wide,), device=y.device)
            wide_labels = y[:n_real][rand_idx]
            # Replace last n_wide entries (don't expand batch — keep memory stable)
            x = torch.cat([x[:n_real], wide_points], dim=0)
            y = torch.cat([y[:n_real], wide_labels], dim=0)

        return x, y

    def _training_step(self, optimizer: Optimizer, global_step: int = 0) -> dict[str, float]:
        batch = next(self.train_iter)
        x = batch['x'].to(self.device)
        y = batch['labels'].to(self.device)

        # Apply augmentation (paper Sections 3-4)
        x, y = self._augment_batch(x, y)

        optimizer.zero_grad()
        out = self.model(x, y)

        loss_dict = self.criterion(
            phi_matrix=out['phi_matrix'],
            grad_phi_k=out['grad_phi_k'],
            A=out['A'],
            Ag_pinn=out.get('Ag_pinn'),
            head_out=out['head_out'],
            k=self.model._active_k,
            global_step=global_step,
        )
        loss_dict['loss'].backward()
        # Gradient clipping — disabled by default (max_grad_norm=None).
        # Enable via train.py config only if gradients diverge on a specific
        # dataset. A hardcoded 1.0 was previously used but it overrode the
        # dynamic weighting schedule (see CODE_AUDIT_REPORT §1.7).
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.max_grad_norm,
            )
            loss_dict['grad_norm'] = float(grad_norm)
        optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []
        # FIX (P1, CODE_AUDIT_REPORT §2.6):
        # Collect all φ_matrix rows across batches so the Gram matrix is
        # computed on the FULL validation set in one shot.
        # Old code averaged per-batch ‖C_batch − I‖_F, which is NOT the
        # same as ‖C_full − I‖_F — with small batches it is inflated by
        # high per-batch variance and can differ by 2–10× from the true value.
        all_phi_cols: list[torch.Tensor] = []

        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            y = batch['labels'].to(self.device)

            # grad must be enabled: BasisFunction.forward(return_grad=True) calls
            # torch.autograd.grad, which needs a live computation graph.
            out = self.model(x, y)
            phi = out['phi_matrix']           # (B, k)
            all_phi_cols.append(phi.detach().cpu())

            probs = out['head_out']['probs'].detach()
            if probs.dim() == 1:
                all_probs.extend(probs.cpu().tolist())
            else:
                all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().tolist())

        self.model.train()
        # Re-freeze non-active functions.
        for i, fn in enumerate(self.model.basis_set.functions):
            if i != self.model._active_k - 1:
                fn.eval()

        # Full-data Gram error — correct estimate of ‖E[φφᵀ] − I‖_F.
        phi_full = torch.cat(all_phi_cols, dim=0)   # (N_val, k)
        N_val = phi_full.shape[0]
        k = self.model._active_k
        C_full = (phi_full.T @ phi_full) / N_val
        eye_k = torch.eye(k, dtype=phi_full.dtype)
        gram_error_full = float(torch.norm(C_full - eye_k, p='fro').item())

        metrics: dict[str, float] = {'val/gram_error': gram_error_full}
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score

            labels_arr = np.array(all_labels)
            if isinstance(all_probs[0], list):
                probs_arr = np.array(all_probs)
                preds = probs_arr.argmax(axis=1)
                metrics['val/accuracy'] = float(accuracy_score(labels_arr, preds))
                try:
                    metrics['val/roc_auc'] = float(
                        roc_auc_score(labels_arr, probs_arr, multi_class='ovr', average='macro')
                    )
                except Exception:
                    pass
            else:
                probs_arr = np.array(all_probs)
                preds = (probs_arr >= 0.5).astype(int)
                metrics['val/accuracy'] = float(accuracy_score(labels_arr, preds))
                try:
                    metrics['val/roc_auc'] = float(roc_auc_score(labels_arr, probs_arr))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning('Could not compute val metrics: %s', exc)

        return metrics

    @torch.no_grad()
    def _compute_gram_error_on_loader(
        self,
        loader,
        loader_name: str = 'val',
    ) -> dict[str, float]:
        """Compute full-dataset orthogonality error on a given DataLoader.

        FIX (P2, CODE_AUDIT_REPORT §2.4):
          Renamed from _compute_gram_error_final (misleading — it ran on val,
          not the full dataset).  Now accepts a loader so callers can pass
          either self.val_loader or self.train_loader.

        Returns keys: gram_error_{loader_name}, gram_error_offdiag_{loader_name},
                       gram_error_diag_{loader_name}
        """
        self.model.eval()
        phi_cols = []
        for batch in loader:
            x = batch['x'].to(self.device)
            # Evaluate all K functions on this batch.
            fns = self.model.basis_set.functions
            batch_phi = torch.cat(
                [fn.predict(x) for fn in fns], dim=1
            )  # (B, K)
            phi_cols.append(batch_phi)

        Phi = torch.cat(phi_cols, dim=0)  # (N, K)
        N = Phi.shape[0]
        C = (Phi.T @ Phi) / N  # (K, K)
        K = C.shape[0]
        I_K = torch.eye(K, device=C.device, dtype=C.dtype)
        diff = C - I_K
        gram_final = torch.norm(diff, p='fro').item()
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        gram_offdiag = C[off_diag_mask].abs().max().item()
        gram_diag = (torch.diag(C) - 1).abs().norm().item()

        self.model.train()
        for i, fn in enumerate(self.model.basis_set.functions):
            if i != self.model._active_k - 1:
                fn.eval()

        return {
            f'gram_error_{loader_name}':        gram_final,
            f'gram_error_offdiag_{loader_name}': gram_offdiag,
            f'gram_error_diag_{loader_name}':    gram_diag,
        }

    def _compute_dirichlet_energy(self, k: int) -> float:
        """Estimate Dirichlet energy of φ_k on a batch from val loader."""
        self.model.eval()
        fn = self.model.basis_set.functions[k - 1]
        fn.eval()
        energies = []
        for i, batch in enumerate(self.val_loader):
            if i >= 4:  # a few batches suffice
                break
            x = batch['x'].to(self.device)
            _, grad = fn(x, return_grad=True)
            if grad is not None:
                if self.model.metric is None:
                    energy = (grad ** 2).mean().item()
                elif isinstance(self.model.metric, LambdaUPinn):
                    Ag = self.model.metric.apply_to(x, grad)  # one PINN call
                    energy = (Ag ** 2).sum(dim=1).mean().item()
                else:
                    A = self.model.metric(x)
                    if A.dim() == 2:
                        energy = ((A * grad) ** 2).sum(dim=1).mean().item()
                    else:
                        Ag = torch.bmm(A, grad.unsqueeze(-1)).squeeze(-1)
                        energy = (Ag ** 2).sum(dim=1).mean().item()
                energies.append(energy)
        self.model.train()
        for i, f in enumerate(self.model.basis_set.functions):
            if i != self.model._active_k - 1:
                f.eval()
        return float(np.mean(energies)) if energies else 0.0

    # ------------------------------------------------------------------
    # Logging and checkpoints
    # ------------------------------------------------------------------

    def _log(
        self,
        global_step: int,
        k: int,
        loss_dict: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        wall = loss_dict.get('wall_time_seconds', 0.0)
        w_task = loss_dict.get('w_task_eff', float('nan'))
        w_mde  = loss_dict.get('w_mde_eff',  float('nan'))
        r_gram = loss_dict.get('ratio_gram',  float('nan'))
        # Show effective weights in console so dynamic schedule is visible
        # without WandB.  For static mode w_task=w_mde=1.0 (nan ratios omitted).
        import math
        if not math.isnan(r_gram):
            logger.info(
                'step %6d  k=%d  loss=%.4f  gram_err=%.4f  '
                'w_task=%.3f  w_mde=%.3f  (r_g=%.2f)  val_acc=%.4f  wall=%.0fs',
                global_step, k,
                loss_dict.get('loss', float('nan')),
                loss_dict.get('gram_error', float('nan')),
                w_task, w_mde, r_gram,
                val_metrics.get('val/accuracy', float('nan')),
                wall,
            )
        else:
            logger.info(
                'step %6d  k=%d  loss=%.4f  gram_err=%.4f  '
                'w_task=%.3f(static)  w_mde=%.3f(static)  val_acc=%.4f  wall=%.0fs',
                global_step, k,
                loss_dict.get('loss', float('nan')),
                loss_dict.get('gram_error', float('nan')),
                w_task, w_mde,
                val_metrics.get('val/accuracy', float('nan')),
                wall,
            )

        # ── JSONL metrics file ──────────────────────────────────────────
        # Write one JSON line per log_step to <checkpoint_dir>/metrics.jsonl.
        # Readable offline without WandB:  pd.read_json('metrics.jsonl', lines=True)
        row: dict = {
            'step': global_step,
            'k': k,
            'wall': round(wall, 1),
            # losses
            'loss':           round(loss_dict.get('loss',           float('nan')), 6),
            'loss_gram':      round(loss_dict.get('loss_gram',      float('nan')), 6),
            'loss_task':      round(loss_dict.get('loss_task',      float('nan')), 6),
            'loss_dirichlet': round(loss_dict.get('loss_dirichlet', float('nan')), 6),
            # orthogonality
            'gram_error':     round(loss_dict.get('gram_error',     float('nan')), 6),
            'offdiag_k':      round(loss_dict.get('off_diag_error_k', float('nan')), 6),
            # dynamic weighting schedule
            'w_task_eff':     round(loss_dict.get('w_task_eff',    float('nan')), 6),
            'w_mde_eff':      round(loss_dict.get('w_mde_eff',     float('nan')), 6),
            'ratio_gram':     round(loss_dict.get('ratio_gram',    float('nan')), 6),
            'ratio_class':    round(loss_dict.get('ratio_class',   float('nan')), 6),
            # validation
            'val_acc':        round(val_metrics.get('val/accuracy', float('nan')), 6),
        }
        with open(self._metrics_jsonl, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        # ───────────────────────────────────────────────────────────────

        if self.writer is None:
            return

        self.writer.set_step(global_step)
        # Build flat dict for batch upload — one wandb.log call per step.
        wb: dict[str, float] = {'train/function_k': k}
        for name, value in {**loss_dict, **val_metrics}.items():
            if not isinstance(value, (int, float)):
                continue
            if '/' in name:
                wb[name] = value        # already prefixed (e.g. val/accuracy)
            else:
                wb[f'train/{name}'] = value
        self.writer.add_scalars(wb)

    def _save_checkpoint(
        self,
        global_step: int,
        name: str | None = None,
        is_final: bool = False,
    ) -> None:
        if name is None:
            name = f'checkpoint_{global_step // 1000}k.pt'

        path = self.checkpoint_dir / name
        # FIX (P3, §3.3 + §2.14): save _best_val_acc and criterion.t_class
        # so resume / analysis tools can read the model-selection threshold
        # and the GL-derived T_class without re-running GL pretraining.
        state = {
            'global_step': global_step,
            'current_function_k': self.model._active_k,
            'model_state_dict': self.model.state_dict(),
            'metrics_history': self._metrics_history,
            'gram_error_history': [
                m.get('val/gram_error') for m in self._metrics_history.values()
            ],
            'eigenvalue_history': self._eigenvalue_history,
            'wall_time_seconds': time.time() - self._start_time,
            'wall_time_per_function': self._wall_time_per_function,
            'config': self.config,
            # Trainer state for honest resume.
            'best_val_acc': self._best_val_acc,
            't_class': getattr(self.criterion, 't_class', None),
        }
        if self._optimizer is not None:
            state['optimizer_state_dict'] = self._optimizer.state_dict()

        torch.save(state, path)
        logger.info('Checkpoint saved: %s', path)

        if is_final:
            final_path = self.checkpoint_dir / 'checkpoint_final.pt'
            torch.save(state, final_path)
            logger.info('Final checkpoint: %s', final_path)

        if self.writer is not None:
            self.writer.add_checkpoint(str(path), str(self.checkpoint_dir.parent))

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path) -> dict:
        """Load checkpoint state dict."""
        return torch.load(checkpoint_path, map_location='cpu')
