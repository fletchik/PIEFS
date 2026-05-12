"""Spotify Songs Dataset loader.

Source:
    Kaggle: "Dataset of songs in Spotify" by Andrii Samoshyn (mrmorj)
    https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify

Download on the cluster:
    kaggle datasets download -d mrmorj/dataset-of-songs-in-spotify -p data/spotify
    unzip data/spotify/dataset-of-songs-in-spotify.zip -d data/spotify

The CSV contains 18 audio features (acousticness, danceability, duration_ms,
energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo,
time_signature, valence) plus categorical fields (artist_name, track_name,
track_id, genre, music_genre).

Two tasks are supported:
  binary      — classify by `mode` (0 = minor, 1 = major).  Balanced enough.
  multiclass  — classify by `music_genre` (10 genres, remapped 0..9).

Numeric features used (13 after dropping string columns and ID fields):
    acousticness, danceability, duration_ms, energy, instrumentalness,
    key, liveness, loudness, mode (excluded for multiclass task),
    speechiness, tempo, time_signature, valence
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Audio features to use as input regardless of task
_FEATURE_COLS = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness',
    'speechiness', 'tempo', 'time_signature', 'valence',
]
# For binary (mode) task, include 'mode' in features?  No — it IS the label.
# For multiclass, 'mode' can be an additional feature.
_MULTICLASS_EXTRA = ['mode']


class SpotifyDataset(Dataset):
    """Pytorch Dataset wrapper for the Spotify songs dataset.

    Args:
        root:           Directory containing 'genres_v2.csv'
                        (default output of the Kaggle download above).
        split:          'train', 'val', or 'test'.
        task:           'binary' (mode: 0=minor / 1=major) or
                        'multiclass' (music_genre, 10 classes).
        train_fraction: Fraction of data used for training.
        val_fraction:   Fraction of training data held out as validation.
        standardize:    Z-score normalise features (fit on train only).
        seed:           Random seed for the train/val/test split.
    """

    def __init__(
        self,
        root: str = 'data/spotify',
        split: Literal['train', 'val', 'test'] = 'train',
        task: Literal['binary', 'multiclass'] = 'multiclass',
        train_fraction: float = 0.7,
        val_fraction: float = 0.1,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        import pandas as pd

        root_path = Path(root)
        csv_path = root_path / 'genres_v2.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Spotify CSV not found at {csv_path}.\n"
                "Download with:\n"
                "  kaggle datasets download -d mrmorj/dataset-of-songs-in-spotify "
                "-p data/spotify\n"
                "  unzip data/spotify/dataset-of-songs-in-spotify.zip -d data/spotify"
            )

        df = pd.read_csv(csv_path, low_memory=False)
        logger.info('Loaded Spotify CSV: %d rows', len(df))

        # ── Feature matrix ────────────────────────────────────────────────
        if task == 'multiclass':
            feat_cols = _FEATURE_COLS + _MULTICLASS_EXTRA
        else:
            feat_cols = _FEATURE_COLS

        # Keep only rows where all needed columns are present and numeric
        df = df.dropna(subset=feat_cols)
        for col in feat_cols:
            df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        df[feat_cols] = df[feat_cols].apply(pd.to_numeric)

        # ── Label ──────────────────────────────────────────────────────────
        if task == 'binary':
            label_col = 'mode'
            df = df.dropna(subset=[label_col])
            labels = df[label_col].astype(int).values
        else:
            label_col = 'music_genre'
            df = df.dropna(subset=[label_col])
            genres = sorted(df[label_col].unique())
            genre_map = {g: i for i, g in enumerate(genres)}
            logger.info('Genres (%d): %s', len(genres), genres)
            labels = df[label_col].map(genre_map).astype(int).values

        X = df[feat_cols].values.astype(np.float32)
        logger.info('Feature matrix: %s  |  unique labels: %d',
                    X.shape, len(np.unique(labels)))

        # ── Train / val / test split ───────────────────────────────────────
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_test   = int(len(X) * (1 - train_fraction))
        n_val    = int((len(X) - n_test) * val_fraction)
        n_train  = len(X) - n_test - n_val

        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]

        # ── Standardise on train only ──────────────────────────────────────
        if standardize:
            mean = X[train_idx].mean(axis=0)
            std  = X[train_idx].std(axis=0).clip(min=1e-8)
            X = (X - mean) / std

        if split == 'train':
            sel = train_idx
        elif split == 'val':
            sel = val_idx
        else:
            sel = test_idx

        self.X = torch.from_numpy(X[sel])
        self.y = torch.from_numpy(labels[sel]).long()
        self.input_dim   = self.X.shape[1]
        self.num_classes = int(np.unique(labels).size)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'x': self.X[idx], 'label': self.y[idx]}
