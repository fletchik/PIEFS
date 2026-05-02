from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DatasetName = Literal['two_moon', 'circles']


class SklearnDataset(Dataset):
    """2-D synthetic classification datasets from scikit-learn.

    Supports "two_moon" (sklearn.datasets.make_moons) and
    "circles" (sklearn.datasets.make_circles).

    Args:
        name: "two_moon" or "circles".
        split: "train", "val", or "test".
        n_samples: Total number of samples (before split).
        noise: Noise level for the generator.
        train_fraction: Fraction of data used for training.
        standardize: Z-score normalise features.
        seed: Random seed for reproducible generation and splitting.
    """

    num_classes: int = 2
    input_dim: int = 2

    def __init__(
        self,
        name: DatasetName,
        split: Literal['train', 'val', 'test'] = 'train',
        n_samples: int = 10_000,
        noise: float = 0.1,
        train_fraction: float = 0.7,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        X, y = self._generate(name, n_samples, noise, seed)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        X, y = X[perm], y[perm]

        n_train = int(n_samples * train_fraction)
        n_val = (n_samples - n_train) // 2
        splits = {
            'train': (0, n_train),
            'val': (n_train, n_train + n_val),
            'test': (n_train + n_val, n_samples),
        }
        lo, hi = splits[split]
        X_train = X[:n_train]

        if standardize:
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0).clip(1e-8)
            X = (X - mean) / std

        self.X = torch.tensor(X[lo:hi], dtype=torch.float32)
        self.y = torch.tensor(y[lo:hi], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}

    @staticmethod
    def _generate(
        name: DatasetName,
        n_samples: int,
        noise: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn import datasets

        if name == 'two_moon':
            return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        if name == 'circles':
            return datasets.make_circles(n_samples=n_samples, noise=noise, random_state=seed)
        raise ValueError(f"Unknown dataset name '{name}'. Expected 'two_moon' or 'circles'.")
