"""PIEFS training entry point.

Usage:
    python train.py run_id=exp01
    python train.py run_id=exp01 dataset=htru2 model.metric_type=diag model.K=6
    python train.py run_id=exp01 dataset=mnist_binary model.K=16 trainer.total_steps=60000
    python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """Build all components from config and run sequential training."""
    # Reproducibility — fix ALL sources of randomness.
    seed = cfg.trainer.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info('Config:\n%s', OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.trainer.device
    log.info('Device: %s', device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    ds_cfg = cfg.dataset

    train_ds, val_ds = _build_datasets(ds_cfg)
    from src.dataset.collate import collate_fn as CollateFn

    collate = CollateFn(use_label=True)  # all datasets carry labels (even dummy ones)

    from src.dataset.utils import make_loader

    train_loader = make_loader(
        train_ds, batch_size=cfg.trainer.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = make_loader(
        val_ds, batch_size=cfg.trainer.batch_size, shuffle=False, collate_fn=collate
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from src.model.basis.basis_set import BasisSet
    from src.model.metric.metric_net import build_metric
    from src.model.spectral_model import BinaryHead, MulticlassHead, SpectralModel

    K = cfg.model.K
    input_dim = ds_cfg.input_dim
    # output_bias=False: recommended for new experiments.
    # Default True keeps backward-compat with checkpoints trained before
    # this option was added.  Set model.output_bias: false in YAML for new runs.
    output_bias = bool(cfg.model.get('output_bias', True))
    basis_set = BasisSet(K=K, input_dim=input_dim,
                         hidden_dims=list(cfg.model.hidden_dims),
                         output_bias=output_bias)

    metric = build_metric(
        metric_type=cfg.model.metric_type,
        input_dim=input_dim,
        hidden_dims=list(cfg.model.metric_hidden_dims),
        pinn_hidden_dims=list(cfg.model.get('pinn_hidden_dims', [128, 128, 128])),
        low_rank_r=int(cfg.model.get('low_rank_r', 8)),
        low_rank_init_scale=float(cfg.model.get('low_rank_init_scale', 0.01)),
        normalize_det=bool(cfg.model.get('normalize_det', False)),
    )
    if metric is not None:
        metric = metric.to(device)

    num_classes = ds_cfg.num_classes
    if cfg.model.task == 'binary':
        head: torch.nn.Module = BinaryHead(K)
    else:
        head = MulticlassHead(K, num_classes)

    model = SpectralModel(basis_set, metric, head).to(device)
    log.info('Model:\n%s', model)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    from src.loss.spectral_loss import SpectralDirichletLoss

    # Three-phase curriculum end-steps (0 = disabled, legacy exponential only).
    _curriculum = cfg.get('curriculum', {})
    _p1 = int(_curriculum.get('phase1_end_step', 0))
    _p2 = int(_curriculum.get('phase2_end_step', 0))

    criterion = SpectralDirichletLoss(
        w_gram=cfg.criterion.w_gram,
        w_dirichlet=cfg.criterion.w_dirichlet,
        w_task=cfg.criterion.w_task,
        dynamic_weighting=cfg.criterion.get('dynamic_weighting', False),
        t_orth=cfg.criterion.get('t_orth', 0.1),
        t_class=cfg.criterion.get('t_class', 0.5),
        phase1_end_step=_p1,
        phase2_end_step=_p2,
    )

    # ------------------------------------------------------------------
    # Optimizer factory (fresh per function)
    # ------------------------------------------------------------------
    def optimizer_fn(params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )

    # ------------------------------------------------------------------
    # Logger / writer
    # ------------------------------------------------------------------
    run_id = cfg.run_id
    project_config = OmegaConf.to_container(cfg, resolve=True)
    # Path relative to this file so the project works from any cwd.
    log_dir = Path(__file__).parent / 'logs'

    from src.logger.experiment_logger import ExperimentLogger

    exp_logger = ExperimentLogger(log_dir)
    exp_logger.log_experiment_start(project_config, run_id)

    writer = None
    try:
        from src.logger.wandb_writer import WandBWriter

        # Auto-generate a human-readable run name and tags from config.
        # Override via cfg.writer.run_name / cfg.writer.tags if needed.
        auto_run_name = cfg.writer.get('run_name') or (
            f"{ds_cfg.name}_{cfg.model.metric_type}_K{K}_{run_id}"
        )
        auto_tags = list(cfg.writer.get('tags') or [
            ds_cfg.name,
            cfg.model.metric_type,
            f"K{K}",
            f"seed{cfg.trainer.seed}",
        ])

        writer = WandBWriter(
            project_config=project_config,
            project_name=cfg.writer.project_name,
            run_id=run_id,
            run_name=auto_run_name,
            entity=cfg.writer.get('entity') or None,
            mode=cfg.writer.mode,
            tags=auto_tags,
            notes=cfg.writer.get('notes') or None,
        )
        log.info('WandB run: %s  (mode=%s)', auto_run_name, cfg.writer.mode)
    except Exception as exc:
        log.warning('WandB init failed (%s); continuing without logging.', exc)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    from src.trainer.sequential_trainer import SequentialTrainer

    checkpoint_dir = log_dir / run_id
    # Augmentation config (paper Sections 3-4)
    aug_cfg = cfg.get('augmentation', {})
    noise_std = float(aug_cfg.get('noise_std', 0.0))
    wide_normal_fraction = float(aug_cfg.get('wide_normal_fraction', 0.0))

    # Gradient clipping: None = disabled (default).
    # Old hardcoded 1.0 overrode the dynamic weighting schedule.
    max_grad_norm_raw = cfg.trainer.get('max_grad_norm', None)
    max_grad_norm = float(max_grad_norm_raw) if max_grad_norm_raw is not None else None

    trainer = SequentialTrainer(
        model=model,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        experiment_logger=exp_logger,
        run_id=run_id,
        total_steps=cfg.trainer.total_steps,
        log_step=cfg.trainer.log_step,
        save_period=cfg.trainer.save_period,
        device=device,
        skip_oom=cfg.trainer.skip_oom,
        config=project_config,
        noise_std=noise_std,
        wide_normal_fraction=wide_normal_fraction,
        max_grad_norm=max_grad_norm,
    )

    # ------------------------------------------------------------------
    # Optional Graph Laplacian pretraining (paper Section 2.4)
    # ------------------------------------------------------------------
    pretrain_cfg = cfg.get('pretrain', None)
    if pretrain_cfg is not None and pretrain_cfg.get('graph_laplacian', False):
        from src.pretrain.graph_laplacian import GraphLaplacianPretrain

        log.info('=== Graph Laplacian pretraining enabled ===')
        gl = GraphLaplacianPretrain(
            K=K,
            n_points=pretrain_cfg.get('n_points', 1000),
            k_neighbors=pretrain_cfg.get('k_neighbors', 10),
            sigma=pretrain_cfg.get('sigma', None),
            distill_steps=pretrain_cfg.get('distill_steps', 2000),
            distill_lr=pretrain_cfg.get('distill_lr', 1e-3),
            device=device,
        )
        gl.compute_eigenfunctions(train_ds)

        if pretrain_cfg.get('update_t_class', True):
            t_class_new = gl.compute_t_class()
            criterion.t_class = t_class_new
            log.info('Updated criterion.t_class → %.4f (from GL logistic regression)', t_class_new)

        gl.pretrain_basis_set(model.basis_set)
        log.info('=== Graph Laplacian pretraining complete ===')

    # Optional resume from checkpoint.
    resume_path = cfg.get('resume', None)
    if resume_path:
        log.info('Resuming from checkpoint: %s', resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer._metrics_history = ckpt.get('metrics_history', {})
        trainer._eigenvalue_history = ckpt.get('eigenvalue_history', [])
        trainer._wall_time_per_function = ckpt.get('wall_time_per_function', [])
        # Restore best_val_acc and t_class on resume so the model-selection
        # threshold is not reset to -1.0.
        if 'best_val_acc' in ckpt:
            trainer._best_val_acc = ckpt['best_val_acc']
        if 'T_class' in ckpt and hasattr(trainer.criterion, 't_class'):
            trainer.criterion.t_class = ckpt['t_class']
        # Restore wall-clock start so resumed wall-time is cumulative.
        trainer._start_time = time.time() - ckpt.get('wall_time_seconds', 0.0)
        log.info('Loaded checkpoint at step %d', ckpt.get('global_step', 0))

    trainer.train()

    exp_logger.log_experiment_results(trainer._metrics_history, project_config, run_id)
    if writer is not None:
        writer.finish()


def _build_datasets(ds_cfg):
    """Instantiate train and val Dataset objects from the dataset config."""
    name = ds_cfg.name

    if name in ('two_moon', 'circles'):
        from src.dataset.sklearn_cls import SklearnDataset

        kwargs = dict(
            name=name,
            n_samples=ds_cfg.n_samples,
            noise=ds_cfg.noise,
            train_fraction=ds_cfg.train_fraction,
            standardize=ds_cfg.standardize,
        )
        return SklearnDataset(split='train', **kwargs), SklearnDataset(split='val', **kwargs)

    if name == 'htru2':
        from src.dataset.htru2 import HTRU2Dataset

        kwargs = dict(
            root=ds_cfg.root,
            train_fraction=ds_cfg.train_fraction,
            standardize=ds_cfg.standardize,
        )
        return HTRU2Dataset(split='train', **kwargs), HTRU2Dataset(split='val', **kwargs)

    if name in ('mnist', 'fashion_mnist', 'cifar10'):
        from src.dataset.torchvision_flat import TorchvisionFlatDataset

        kwargs = dict(
            name=name,
            root=ds_cfg.root,
            task=ds_cfg.task,
            binary_classes=tuple(ds_cfg.get('binary_classes', [0, 1])),
            val_fraction=ds_cfg.val_fraction,
            standardize=ds_cfg.standardize,
        )
        return (
            TorchvisionFlatDataset(split='train', **kwargs),
            TorchvisionFlatDataset(split='val', **kwargs),
        )

    if name == 'cifar10_features':
        from src.dataset.pretrained_features import PretrainedFeaturesDataset

        kwargs = dict(
            root=ds_cfg.root,
            val_fraction=ds_cfg.val_fraction,
            standardize=ds_cfg.standardize,
        )
        return (
            PretrainedFeaturesDataset(split='train', **kwargs),
            PretrainedFeaturesDataset(split='val', **kwargs),
        )

    if name == 'spotify':
        from src.dataset.spotify import SpotifyDataset

        kwargs = dict(
            root=ds_cfg.root,
            task=ds_cfg.task,
            train_fraction=ds_cfg.get('train_fraction', 0.7),
            val_fraction=ds_cfg.val_fraction,
            standardize=ds_cfg.standardize,
        )
        return (
            SpotifyDataset(split='train', **kwargs),
            SpotifyDataset(split='val', **kwargs),
        )

    raise ValueError(f"Unknown dataset name: '{name}'.")


if __name__ == '__main__':
    main()
