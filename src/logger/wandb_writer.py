from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class WandBWriter:
    """Experiment tracker backed by Weights & Biases.

    Wraps wandb.init and provides a consistent interface used by
    SequentialTrainer and ExperimentLogger.

    Args:
        project_config: Full Hydra config dict (logged to WandB run config).
        project_name: WandB project name.
        run_id: Unique run identifier (reused for resuming).
        run_name: Human-readable run name shown in the WandB UI.
        entity: WandB entity (username or team).
        mode: One of "online", "offline", "disabled".
        save_code: Whether to upload source code to WandB.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        run_id: str | None = None,
        run_name: str | None = None,
        entity: str | None = None,
        mode: Literal['online', 'offline', 'disabled'] = 'offline',
        save_code: bool = False,
    ) -> None:
        import wandb

        self.step: int = 0
        self._mode = mode
        self._timer = datetime.now()

        wandb.login()
        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            id=run_id,
            resume='allow',
            mode=mode,
            save_code=save_code,
        )
        self._wandb = wandb

    def set_step(self, step: int) -> None:
        """Advance the global step counter and log steps-per-second.

        Args:
            step: New global step value.
        """
        prev = self.step
        self.step = step
        if step > 0:
            elapsed = (datetime.now() - self._timer).total_seconds()
            if elapsed > 0:
                self.add_scalar('perf/steps_per_sec', (step - prev) / elapsed)
        self._timer = datetime.now()

    def add_scalar(self, name: str, value: float) -> None:
        """Log a scalar value at the current step.

        Args:
            name: Metric name (may include "/" for grouping in WandB).
            value: Scalar value.
        """
        self._wandb.log({name: value}, step=self.step)

    def add_scalars(self, values: dict[str, float]) -> None:
        """Log multiple scalars at the current step.

        Args:
            values: Dict mapping metric names to values.
        """
        self._wandb.log(values, step=self.step)

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        """Log an image at the current step.

        Args:
            name: Image name.
            image: numpy (H, W, C) array, or path to an image file.
        """
        self._wandb.log({name: self._wandb.Image(image)}, step=self.step)

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """Upload a checkpoint file to WandB.

        Args:
            checkpoint_path: Full path to the .pt file.
            save_dir: Base path used by wandb.save for relative path computation.
        """
        self._wandb.save(checkpoint_path, base_path=save_dir)

    def finish(self) -> None:
        """Finalise and close the WandB run."""
        self._wandb.finish()
