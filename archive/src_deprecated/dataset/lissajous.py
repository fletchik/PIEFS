from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class LissajousDataset(Dataset):
    """Unit-circle dataset for Lissajous / trigonometric eigenfunction tasks.

    Each sample is a 2-D point (cos θ, sin θ) drawn uniformly from [0, 2π).
    No class labels — used for unsupervised spectral learning where the target
    eigenfunctions are cos(kθ) and sin(kθ).

    Args:
        split: "train", "val", or "test".
        n_samples: Total samples before splitting.
        train_fraction: Fraction for training.
        seed: Random seed.
    """

    num_classes: int = 2
    input_dim: int = 2

    def __init__(
        self,
        split: Literal['train', 'val', 'test'] = 'train',
        n_samples: int = 50_000,
        train_fraction: float = 0.8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2 * np.pi, size=n_samples)
        X = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

        n_train = int(n_samples * train_fraction)
        n_val = (n_samples - n_train) // 2
        splits = {
            'train': (0, n_train),
            'val': (n_train, n_train + n_val),
            'test': (n_train + n_val, n_samples),
        }
        lo, hi = splits[split]
        self.X = torch.tensor(X[lo:hi])
        # Dummy labels (0) — spectral training ignores them.
        self.y = torch.zeros(len(self.X), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}
