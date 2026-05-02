from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_HTRU2_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip'


class HTRU2Dataset(Dataset):
    """HTRU2 pulsar detection dataset (binary classification, 8 features).

    Downloads and caches the CSV on first use. Applies optional z-score
    standardisation using training-split statistics.

    Dataset: 17 898 instances — 1 639 positive (pulsar), 16 259 negative.

    Args:
        root: Directory where HTRU_2.csv is stored (or will be downloaded to).
        split: One of "train", "val", "test".
        train_fraction: Fraction of data for training. Remainder is split
                        equally between val and test.
        standardize: Apply z-score normalisation.
        seed: Random seed for the train/val/test split.
    """

    num_classes: int = 2
    input_dim: int = 8

    def __init__(
        self,
        root: str | Path,
        split: Literal['train', 'val', 'test'] = 'train',
        train_fraction: float = 0.7,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        root = Path(root)
        csv_path = root / 'HTRU_2.csv'
        if not csv_path.exists():
            self._download(root)

        import numpy as np

        data = np.loadtxt(csv_path, delimiter=',')
        X = torch.tensor(data[:, :8], dtype=torch.float32)
        y = torch.tensor(data[:, 8], dtype=torch.long)

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator=rng)
        X, y = X[perm], y[perm]

        n = len(X)
        n_train = int(n * train_fraction)
        n_val = (n - n_train) // 2

        splits = {
            'train': (0, n_train),
            'val': (n_train, n_train + n_val),
            'test': (n_train + n_val, n),
        }
        lo, hi = splits[split]
        X_train = X[:n_train]

        if standardize:
            mean = X_train.mean(dim=0)
            std = X_train.std(dim=0, correction=0).clamp(min=1e-8)
            X = (X - mean) / std

        self.X = X[lo:hi]
        self.y = y[lo:hi]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}

    @staticmethod
    def _download(root: Path) -> None:
        import urllib.request
        import zipfile

        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / 'HTRU2.zip'
        logger.info('Downloading HTRU2 dataset to %s ...', zip_path)
        urllib.request.urlretrieve(_HTRU2_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        logger.info('HTRU2 dataset extracted.')
