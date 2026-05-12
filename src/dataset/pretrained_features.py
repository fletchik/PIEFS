"""Dataset class for pre-extracted CNN feature vectors.

Loads numpy arrays produced by scripts/extract_cnn_features.py and
exposes them as a standard PyTorch Dataset compatible with the rest of
the PIEFS pipeline (returns dicts with 'x' and 'label' keys).

The train/val split is carved from the saved X_train.npy file so that
standardisation statistics are computed on the training slice only
(same leakage-free protocol as TorchvisionFlatDataset).

Args:
    root:          Directory containing X_train.npy, y_train.npy,
                   X_test.npy, y_test.npy (and optionally meta.json).
    split:         'train', 'val', or 'test'.
    val_fraction:  Fraction of the train set to use as validation.
    standardize:   Z-score normalise features (fit on train slice only).
    seed:          Random seed for the val/train carve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainedFeaturesDataset(Dataset):
    """Wraps pre-extracted CNN features for use in PIEFS training."""

    def __init__(
        self,
        root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        val_fraction: float = 0.1,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        root_path = Path(root)

        X_all_train = np.load(root_path / 'X_train.npy').astype(np.float32)
        y_all_train = np.load(root_path / 'y_train.npy').astype(np.int64)
        X_test      = np.load(root_path / 'X_test.npy').astype(np.float32)
        y_test      = np.load(root_path / 'y_test.npy').astype(np.int64)

        # Carve val before computing standardisation stats (no leakage).
        n_total = len(X_all_train)
        n_val   = int(n_total * val_fraction)
        rng     = torch.Generator().manual_seed(seed)
        perm    = torch.randperm(n_total, generator=rng).numpy()
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        if standardize:
            mean = X_all_train[train_idx].mean(axis=0)
            std  = X_all_train[train_idx].std(axis=0).clip(min=1e-8)
            X_all_train = (X_all_train - mean) / std
            X_test      = (X_test      - mean) / std

        if split == 'train':
            X, y = X_all_train[train_idx], y_all_train[train_idx]
        elif split == 'val':
            X, y = X_all_train[val_idx], y_all_train[val_idx]
        else:  # 'test'
            X, y = X_test, y_test

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

        self.input_dim  = self.X.shape[1]
        self.num_classes = int(np.unique(y_all_train).size)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}
