from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DatasetName = Literal['mnist', 'cifar10']
TaskType = Literal['binary', 'multiclass']


class TorchvisionFlatDataset(Dataset):
    """Flat-feature wrapper around torchvision MNIST / CIFAR-10.

    Images are converted to float32, flattened to 1-D, and optionally
    z-score normalised.  For binary tasks, only two classes are kept and
    labels are remapped to {0, 1}.

    Args:
        name: "mnist" or "cifar10".
        split: "train", "val", or "test".
        root: Root directory for torchvision downloads.
        task: "binary" or "multiclass".
        binary_classes: For binary tasks, the two class indices to keep,
                        e.g. (0, 1) for MNIST 0-vs-1.
        val_fraction: Fraction of training data to use as validation set.
        standardize: Z-score normalise flattened features.
        seed: Random seed for val split.
    """

    def __init__(
        self,
        name: DatasetName,
        split: Literal['train', 'val', 'test'] = 'train',
        root: str = 'data',
        task: TaskType = 'multiclass',
        binary_classes: tuple[int, int] = (0, 1),
        val_fraction: float = 0.1,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        import torchvision.transforms as T

        to_tensor = T.ToTensor()
        raw_train = self._load(name, root, train=True, transform=to_tensor)
        raw_test = self._load(name, root, train=False, transform=to_tensor)

        X_train, y_train = self._flatten(raw_train)
        X_test, y_test = self._flatten(raw_test)

        if task == 'binary':
            X_train, y_train = self._filter_binary(X_train, y_train, binary_classes)
            X_test, y_test = self._filter_binary(X_test, y_test, binary_classes)

        if standardize:
            mean = X_train.mean(dim=0)
            std = X_train.std(dim=0, correction=0).clamp(min=1e-8)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        # Carve val from train.
        n_train = len(X_train)
        n_val = int(n_train * val_fraction)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_train, generator=rng)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        if split == 'train':
            self.X, self.y = X_train[train_idx], y_train[train_idx]
        elif split == 'val':
            self.X, self.y = X_train[val_idx], y_train[val_idx]
        else:
            self.X, self.y = X_test, y_test

        self.num_classes = len(torch.unique(self.y))
        self.input_dim = self.X.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}

    @staticmethod
    def _load(name: DatasetName, root: str, train: bool, transform) -> Dataset:
        import torchvision.datasets as tvd

        if name == 'mnist':
            return tvd.MNIST(root, train=train, download=True, transform=transform)
        if name == 'cifar10':
            return tvd.CIFAR10(root, train=train, download=True, transform=transform)
        raise ValueError(f"Unknown dataset '{name}'. Expected 'mnist' or 'cifar10'.")

    @staticmethod
    def _flatten(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        xs, ys = [], []
        for x_batch, y_batch in loader:
            xs.append(x_batch.flatten(start_dim=1).float())
            ys.append(y_batch.long())
        return torch.cat(xs), torch.cat(ys)

    @staticmethod
    def _filter_binary(
        X: torch.Tensor,
        y: torch.Tensor,
        classes: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (y == classes[0]) | (y == classes[1])
        X, y = X[mask], y[mask]
        # Remap to {0, 1}.
        y = (y == classes[1]).long()
        return X, y
