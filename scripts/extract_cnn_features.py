"""Extract penultimate-layer features from a pretrained CNN for CIFAR-10.

Uses torchvision's pretrained ResNet-18 (ImageNet weights). The CIFAR-10
images (32×32) are upsampled to 224×224 and normalised per ImageNet stats
before passing through the network.  The output of the avgpool layer
(512-dim) is saved as a numpy array for later use by PIEFS.

Motivation (advisor suggestion):
    PIEFS operates on flat feature vectors; raw CIFAR-10 pixels (3072-dim
    with high spatial redundancy) are hard to cluster without any visual
    backbone.  Extracting 512-dim ResNet-18 features gives PIEFS access to
    semantically meaningful representations while keeping the method
    itself (eigenfunctions of MDE) unchanged.

Output files (in --out_dir):
    X_train.npy   float32, shape (50000, 512)
    y_train.npy   int64,   shape (50000,)
    X_test.npy    float32, shape (10000, 512)
    y_test.npy    int64,   shape (10000,)
    meta.json     {"input_dim": 512, "num_classes": 10, "model": "resnet18"}

Usage:
    .venv/bin/python3 scripts/extract_cnn_features.py
    .venv/bin/python3 scripts/extract_cnn_features.py --batch_size 512 --out_dir data/cifar10_features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tvd
import torchvision.transforms as T
from torch.utils.data import DataLoader


def build_extractor(device: str) -> nn.Module:
    """Return ResNet-18 with the final FC layer removed (→ avgpool features)."""
    import torchvision.models as tvm

    # Use the new 'weights' API; fall back to the deprecated pretrained flag
    # for older torchvision versions.
    try:
        from torchvision.models import ResNet18_Weights
        model = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except ImportError:
        model = tvm.resnet18(pretrained=True)  # type: ignore[arg-type]

    # Remove the classification head — keep everything up to avgpool.
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def extract_features(
    loader: DataLoader,
    model: nn.Module,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass over the whole loader; return (X, y) numpy arrays."""
    xs, ys = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model(imgs)              # (B, 512)
            xs.append(feats.cpu().float())
            ys.append(labels.cpu().long())
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='data/cifar10_features')
    parser.add_argument('--data_root',  default='data/cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device',     default='auto')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'Device: {device}')

    # ImageNet normalisation + upsample to 224×224 (ResNet expects ≥ 224).
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    print('Loading CIFAR-10 ...')
    train_ds = tvd.CIFAR10(args.data_root, train=True,  download=True, transform=transform)
    test_ds  = tvd.CIFAR10(args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('Building ResNet-18 extractor ...')
    model = build_extractor(device)

    print('Extracting train features ...')
    X_train, y_train = extract_features(train_loader, model, device)
    print(f'  X_train: {X_train.shape}  y_train: {y_train.shape}')

    print('Extracting test features  ...')
    X_test,  y_test  = extract_features(test_loader,  model, device)
    print(f'  X_test : {X_test.shape}   y_test : {y_test.shape}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'X_train.npy', X_train)
    np.save(out_dir / 'y_train.npy', y_train)
    np.save(out_dir / 'X_test.npy',  X_test)
    np.save(out_dir / 'y_test.npy',  y_test)

    meta = {
        'input_dim':   int(X_train.shape[1]),
        'num_classes': int(np.unique(y_train).size),
        'model':       'resnet18',
        'source':      'cifar10',
        'n_train':     int(len(X_train)),
        'n_test':      int(len(X_test)),
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to {out_dir}/')
    print(f'  meta: {meta}')


if __name__ == '__main__':
    main()
