from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from ..model.spectral_model import SpectralModel
    from .wandb_writer import WandBWriter

logger = logging.getLogger(__name__)


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib Figure to an (H, W, 4) RGBA numpy array."""
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    buf.close()
    return arr


# ------------------------------------------------------------------
# Lissajous / unit-circle (2-D spectral visualisation)
# ------------------------------------------------------------------


def plot_lissajous(
    model: 'SpectralModel',
    writer: 'WandBWriter',
    device: str = 'cpu',
) -> None:
    """Plot φ_a(cos θ) vs φ_b(sin θ) for all a, b pairs up to K//2.

    Args:
        model: Trained SpectralModel (basis_set frozen after training).
        writer: WandBWriter to log images.
        device: Torch device string.
    """
    K = model.K
    max_k = K // 2
    if max_k == 0:
        return

    dtype = next(model.basis_set.parameters()).dtype
    theta = torch.linspace(0, 2 * np.pi, 2000, device=device, dtype=dtype)
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    # Evaluate all K functions.
    with torch.no_grad():
        phis = [fn.predict(pts).squeeze(-1).cpu().numpy() for fn in model.basis_set.functions]

    fig, axes = plt.subplots(max_k, max_k, figsize=(3 * max_k, 3 * max_k))
    if max_k == 1:
        axes = np.array([[axes]])
    for a in range(max_k):
        for b in range(max_k):
            ax = axes[a, b]
            ax.plot(phis[2 * a], phis[2 * b + 1], linewidth=1.5)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'a={a + 1}, b={b + 1}', fontsize=9)
    plt.suptitle('Lissajous curves from learned basis', fontsize=12)
    plt.tight_layout()
    writer.add_image('viz/lissajous_family', fig_to_array(fig))
    plt.close(fig)


def plot_basis_heatmaps_2d(
    model: 'SpectralModel',
    writer: 'WandBWriter',
    grid_size: int = 200,
    x_range: tuple[float, float] = (-2.5, 2.5),
    device: str = 'cpu',
) -> None:
    """Plot each φ_k over a 2-D grid as a heatmap image.

    Args:
        model: SpectralModel.
        writer: WandBWriter.
        grid_size: Number of grid points per axis.
        x_range: Spatial range for the grid.
        device: Torch device string.
    """
    dtype = next(model.basis_set.parameters()).dtype
    xs = torch.linspace(*x_range, grid_size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
    pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    with torch.no_grad():
        for k, fn in enumerate(model.basis_set.functions):
            vals = fn.predict(pts).squeeze(-1).cpu().numpy().reshape(grid_size, grid_size)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(vals, origin='lower', extent=[*x_range, *x_range], cmap='RdBu_r')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'φ_{k + 1}(x)')
            writer.add_image(f'viz/basis_heatmap_phi_{k + 1}', fig_to_array(fig))
            plt.close(fig)


def plot_decision_boundary_2d(
    model: 'SpectralModel',
    writer: 'WandBWriter',
    grid_size: int = 200,
    x_range: tuple[float, float] = (-2.5, 2.5),
    device: str = 'cpu',
) -> None:
    """Plot the model decision boundary over a 2-D grid.

    Args:
        model: SpectralModel.
        writer: WandBWriter.
        grid_size: Grid resolution.
        x_range: Spatial extent.
        device: Device string.
    """
    dtype = next(model.basis_set.parameters()).dtype
    xs = torch.linspace(*x_range, grid_size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
    pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    dummy_y = torch.zeros(len(pts), dtype=torch.long, device=device)

    k = model._active_k
    with torch.no_grad():
        out = model(pts, dummy_y)
        probs = out['head_out']['probs'].cpu().numpy()
    if probs.ndim == 1:
        probs_map = probs.reshape(grid_size, grid_size)
    else:
        probs_map = probs[:, 1].reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(grid_x.cpu(), grid_y.cpu(), probs_map, levels=20, cmap='RdBu_r', alpha=0.8)
    ax.set_title(f'Decision boundary (k={k})')
    writer.add_image('viz/decision_boundary', fig_to_array(fig))
    plt.close(fig)


def plot_gram_heatmap(
    phi_matrix: torch.Tensor,
    writer: 'WandBWriter',
    step_label: str = '',
) -> None:
    """Plot the Gram matrix C_k as a heatmap.

    Args:
        phi_matrix: (B, k) basis function outputs.
        writer: WandBWriter.
        step_label: Added to the image name for disambiguation.
    """
    B, k = phi_matrix.shape
    C = (phi_matrix.T @ phi_matrix) / B
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(C.detach().cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Gram matrix C_{k}' + (f' ({step_label})' if step_label else ''))
    writer.add_image('viz/gram_matrix_heatmap', fig_to_array(fig))
    plt.close(fig)


# ------------------------------------------------------------------
# Dataset-specific dispatcher
# ------------------------------------------------------------------


def plot_for_dataset(
    dataset_name: str,
    model: 'SpectralModel',
    writer: 'WandBWriter',
    phi_matrix: torch.Tensor | None = None,
    device: str = 'cpu',
) -> None:
    """Dispatch to the right visualisation function based on dataset.

    Args:
        dataset_name: Name used in config (e.g. "two_moon", "circles", "lissajous").
        model: SpectralModel.
        writer: WandBWriter.
        phi_matrix: (B, k) batch phi values for Gram heatmap (optional).
        device: Torch device string.
    """
    try:
        if dataset_name in ('two_moon', 'circles'):
            plot_decision_boundary_2d(model, writer, device=device)
            plot_basis_heatmaps_2d(model, writer, device=device)
            if phi_matrix is not None:
                plot_gram_heatmap(phi_matrix, writer)
        elif dataset_name == 'lissajous':
            plot_lissajous(model, writer, device=device)
            if phi_matrix is not None:
                plot_gram_heatmap(phi_matrix, writer)
        else:
            if phi_matrix is not None:
                plot_gram_heatmap(phi_matrix, writer)
    except Exception as exc:
        logger.warning("Visualisation failed for dataset '%s': %s", dataset_name, exc)
