from __future__ import annotations

from itertools import repeat
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def inf_loop(loader: DataLoader) -> Any:
    """Yield batches from *loader* indefinitely.

    Args:
        loader: A finite DataLoader.
    Yields:
        Batches, cycling forever.
    """
    for dl in repeat(loader):
        yield from dl


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
) -> DataLoader:
    """Convenience wrapper around DataLoader construction.

    Args:
        dataset: The dataset to load.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle.
        num_workers: DataLoader worker processes.
        collate_fn: Optional collate function.
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def standardize(
    train_x: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Z-score normalise tensors using train statistics.

    Args:
        train_x: (N, d) training features. Statistics are computed here.
        *others: Additional tensors (e.g. val/test) to transform with the
                 same mean and std.
    Returns:
        Normalised tensors in the same order as inputs.
    """
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, correction=0).clamp(min=1e-8)
    results = [(t - mean) / std for t in (train_x, *others)]
    return tuple(results)  # type: ignore[return-value]
