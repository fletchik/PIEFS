from __future__ import annotations

from typing import Any

import torch


class collate_fn:
    """Collate individual dataset items into a batch dict.

    Args:
        use_label: Whether to collate the 'label' field.
    """

    def __init__(self, use_label: bool) -> None:
        self.use_label = use_label

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dicts from Dataset.__getitem__, each with key 'x'
                   and optionally 'label'.
        Returns:
            Dict with 'x' (B, d) tensor and optionally 'labels' (B,) long tensor.
        """
        x_batch = torch.stack([item['x'] for item in batch])
        result: dict[str, torch.Tensor] = {'x': x_batch}
        if self.use_label:
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            result['labels'] = labels
        return result
