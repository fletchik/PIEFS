from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Two-stage Markdown experiment logger.

    Stage 1 — log_experiment_start: writes <run_id>_config.md before training.
    Stage 2 — log_experiment_results: writes <run_id>_results.md after training.

    Both files are placed in *log_dir*.

    Args:
        log_dir: Directory for experiment log files.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def log_experiment_start(self, config: dict[str, Any], run_id: str) -> None:
        """Write experiment config as a Markdown file.

        Args:
            config: Resolved Hydra config (OmegaConf.to_container result).
            run_id: Unique run identifier used in filename and header.
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t = config.get('trainer', {})
        ds = config.get('dataset', {})
        model = config.get('model', {})
        loss = config.get('criterion', {})
        opt = config.get('optimizer', {})
        K = model.get('K', '?')
        total = t.get('total_steps', 60000)
        spf = total // K if isinstance(K, int) and K > 0 else '?'
        project = config.get('writer', {}).get('project_name', 'op-spectra-nn')

        lines = [
            '---',
            f'# Experiment: {run_id}',
            f'Date: {ts}',
            '---',
            '## Dataset',
            f'- name: {ds.get("name", "?")}',
            f'- n_samples: {ds.get("n_samples", "?")}',
            f'- input_dim: {ds.get("input_dim", "?")}',
            f'- num_classes: {ds.get("num_classes", "?")}',
            f'- train/val/test split: {ds.get("train_fraction", "?")} / auto / auto',
            f'- standardize: {ds.get("standardize", True)}',
            '',
            '## Model',
            f'- metric_type: {model.get("metric_type", "off")}',
            f'- num_functions K: {K}',
            f'- basis MLP hidden dims: {model.get("hidden_dims", [64, 64, 64])}',
            f'- metric MLP hidden dims: {model.get("metric_hidden_dims", [64, 64])}',
            f'- task: {model.get("task", "?")}',
            '',
            '## Training',
            f'- total_steps: {total}',
            f'- steps_per_function: {spf}',
            f'- log_step: {t.get("log_step", 15000)}  (checkpoints at 15k, 30k, 45k, 60k)',
            f'- save_period: {t.get("save_period", 30000)}',
            f'- batch_size: {t.get("batch_size", "?")}',
            f'- optimizer: Adam lr={opt.get("lr", "?")}',
            '',
            '## Loss weights',
            f'- w_gram: {loss.get("w_gram", "?")}',
            f'- w_dirichlet: {loss.get("w_dirichlet", "?")}',
            f'- w_task: {loss.get("w_task", "?")}',
            '',
            '## Pipeline',
        ]
        if model.get('metric_type') == 'lambda_u_pinn':
            lines.append('1. Pretrain PINN rotation for 5000 steps')
            step_start = 2
        else:
            step_start = 1
        lines.append(
            f'{step_start}. Sequential training: φ_1 (steps 0–{spf}) → freeze → φ_2 → ... → φ_{K}'
        )
        lines += [
            f'{step_start + 1}. At each log_step: compute val metrics + gram_error + visualizations',
            f'{step_start + 2}. Final evaluation on test set',
            f'{step_start + 3}. Save checkpoint',
            '',
            '## Expected output',
            f'- WandB run: {project}/{run_id}',
            f'- Checkpoint: PIEFS/logs/{run_id}/checkpoint_60k.pt',
            f'- Results MD: PIEFS/logs/{run_id}_results.md',
            '---',
        ]

        out_path = self.log_dir / f'{run_id}_config.md'
        out_path.write_text('\n'.join(lines))
        logger.info('Experiment config written to %s', out_path)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def log_experiment_results(
        self,
        metrics_history: dict[int, dict[str, float]],
        config: dict[str, Any],
        run_id: str,
    ) -> None:
        """Write experiment results as a Markdown file.

        Args:
            metrics_history: Dict mapping global step → metric dict. Should
                             contain entries at 15k, 30k, 45k, 60k.
            config: Resolved config.
            run_id: Unique run identifier.
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        steps = sorted(metrics_history.keys())

        def _gram_table() -> list[str]:
            rows = ['| Step | gram_error \\|\\|C-I\\|\\|_F |', '|------|----------------------|']
            for s in steps:
                val = metrics_history[s].get('gram_error', '—')
                cell = f'{val:.6f}' if isinstance(val, (int, float)) else str(val)
                rows.append(f'| {s} | {cell} |')
            return rows

        def _full_table() -> list[str]:
            header = '| Step | train_loss | gram_loss | dirichlet_loss | task_loss | val_acc | val_roc_auc |'
            sep = '|------|-----------|-----------|----------------|-----------|---------|-------------|'
            rows = [header, sep]
            for s in steps:
                m = metrics_history[s]
                rows.append(
                    '| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                        s,
                        m.get('loss', float('nan')),
                        m.get('loss_gram', float('nan')),
                        m.get('loss_dirichlet', float('nan')),
                        m.get('loss_task', float('nan')),
                        m.get('val/accuracy', float('nan')),
                        m.get('val/roc_auc', float('nan')),
                    )
                )
            return rows

        final = metrics_history.get(max(steps), {}) if steps else {}
        lines = [
            '---',
            f'# Results: {run_id}',
            f'Date: {ts}',
            '---',
            '## Orthogonality',
            *_gram_table(),
            '',
            '## Per-step metrics',
            *_full_table(),
            '',
            '## Final metrics (60k steps)',
            f'- val_accuracy: {final.get("val/accuracy", "—")}',
            f'- val_roc_auc: {final.get("val/roc_auc", "—")}',
            f'- test_accuracy: {final.get("test/accuracy", "—")}',
            f'- test_roc_auc: {final.get("test/roc_auc", "—")}',
            f'- final_gram_error: {final.get("gram_error", "—")}',
            '',
            '## Comparison baseline',
            '[fill if available: NeuralEF / raw LR / random]',
            '',
            '## Notes',
            '[any anomalies, warnings, convergence issues]',
            '---',
        ]

        out_path = self.log_dir / f'{run_id}_results.md'
        out_path.write_text('\n'.join(lines))
        logger.info('Experiment results written to %s', out_path)
