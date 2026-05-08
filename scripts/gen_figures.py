"""Generate all paper figures.

Produces three PNG files in paper_0/figures/:
  fig_training_curves.png  — val_acc over training steps (MNIST & CIFAR-10, 5 seeds)
  fig_eigenfunctions.png   — φ₁,φ₂,φ₃ contour maps for Two Moons and Circles
  fig_gram_matrix.png      — Gram matrix heatmap for MNIST (K eigenfunctions)

Run from project root:
  .venv/bin/python scripts/gen_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper_0" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})


def _build_model(ck_cfg: dict):
    """Reconstruct SpectralModel from checkpoint config dict."""
    import torch
    from src.model.basis.basis_set import BasisSet
    from src.model.metric.metric_net import build_metric
    from src.model.spectral_model import BinaryHead, MulticlassHead, SpectralModel

    mc = ck_cfg["model"]
    dc = ck_cfg["dataset"]
    basis = BasisSet(K=mc["K"], input_dim=dc["input_dim"],
                     hidden_dims=list(mc["hidden_dims"]),
                     output_bias=mc.get("output_bias", False))
    metric = build_metric(
        metric_type=mc["metric_type"],
        input_dim=dc["input_dim"],
        hidden_dims=list(mc["metric_hidden_dims"]),
        pinn_hidden_dims=list(mc.get("pinn_hidden_dims", [128, 128, 128])),
    )
    head = (BinaryHead(mc["K"]) if mc["task"] == "binary"
            else MulticlassHead(mc["K"], dc["num_classes"]))
    return SpectralModel(basis, metric, head)


def _load_run(ck_path: Path):
    import torch
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = _build_model(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["config"]


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Training curves
# ──────────────────────────────────────────────────────────────────────────────
def fig_training_curves():
    COLS = [
        ("MNIST",    "mnist_mc",          "EFDO_colab/logs"),
        ("CIFAR-10", "cifar10_features",  "EFDO_colab/logs"),
    ]
    METRICS = [("off", "#1f77b4", "EFDO-off"), ("diag", "#ff7f0e", "EFDO-diag")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Validation accuracy over training  (mean ± std, 5 seeds)",
                 fontsize=12, fontweight="bold")

    for col_idx, (ds_label, ds_key, base_rel) in enumerate(COLS):
        ax = axes[col_idx]
        base = ROOT / base_rel
        for metric, color, label in METRICS:
            all_val = []
            steps_ref = None
            for s in range(5):
                f = base / f"grid_{ds_key}_{metric}_s{s}" / "metrics.jsonl"
                if not f.exists():
                    continue
                rows = [json.loads(l) for l in f.read_text().strip().splitlines()]
                if steps_ref is None:
                    steps_ref = [r["step"] for r in rows]
                all_val.append([r["val_acc"] * 100 for r in rows])

            if not all_val:
                continue
            arr = np.array(all_val)
            m, s = arr.mean(0), arr.std(0)
            xs = np.array(steps_ref) / 1000
            ax.plot(xs, m, color=color, lw=2, label=label)
            ax.fill_between(xs, m - s, m + s, alpha=0.2, color=color)

        ax.set_title(ds_label)
        ax.set_xlabel("Training steps (×10³)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Start y-axis slightly below the min accuracy achieved
        ax.set_ylim(bottom=60)

    plt.tight_layout()
    out = OUT / "fig_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Eigenfunction visualisation (Two Moons + Circles)
# ──────────────────────────────────────────────────────────────────────────────
def fig_eigenfunctions():
    import torch
    from src.dataset.sklearn_cls import SklearnDataset

    RUNS = [
        ("Two Moons", "logs/grid_two_moon_off_s0"),
        ("Circles",   "logs/grid_circles_off_s0"),
    ]
    N_PHI = 3

    fig, axes = plt.subplots(len(RUNS), N_PHI,
                             figsize=(4 * N_PHI, 3.6 * len(RUNS)))
    fig.suptitle(r"Learned eigenfunctions $\phi_1,\,\phi_2,\,\phi_3$",
                 fontsize=13, fontweight="bold")

    for row, (ds_label, run_rel) in enumerate(RUNS):
        ck_path = ROOT / run_rel / "checkpoint_final.pt"
        if not ck_path.exists():
            print(f"  SKIP {run_rel}: no checkpoint")
            continue
        model, cfg = _load_run(ck_path)
        dc = cfg["dataset"]

        # Validation data for scatter overlay
        ds_name = dc["name"]  # 'two_moon' or 'circles'
        val_ds = SklearnDataset(
            split="val", name=ds_name,
            n_samples=dc["n_samples"], noise=dc["noise"],
            train_fraction=dc["train_fraction"], standardize=dc["standardize"],
        )
        X_val = np.array([val_ds[i]["x"] for i in range(len(val_ds))])
        y_val = np.array([int(val_ds[i]["label"]) for i in range(len(val_ds))])

        # 2-D evaluation grid
        lo, hi = X_val.min(0) - 0.4, X_val.max(0) + 0.4
        x1 = np.linspace(lo[0], hi[0], 220)
        x2 = np.linspace(lo[1], hi[1], 220)
        XX1, XX2 = np.meshgrid(x1, x2)
        grid_t = torch.tensor(
            np.stack([XX1.ravel(), XX2.ravel()], 1), dtype=torch.float32)

        K_model = cfg["model"]["K"]
        with torch.no_grad():
            phi_cols = [model.basis_set.functions[k].predict(grid_t)
                        for k in range(K_model)]
            phi = torch.cat(phi_cols, dim=1).numpy()  # (N_grid, K)

        cls_colors = ["#e41a1c", "#377eb8"]
        for col in range(N_PHI):
            ax = axes[row, col]
            Z = phi[:, col].reshape(220, 220)
            vmax = np.abs(Z).max()
            cf = ax.contourf(XX1, XX2, Z, levels=40,
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.82)
            plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
            for c_idx in range(2):
                mask = y_val == c_idx
                ax.scatter(X_val[mask, 0], X_val[mask, 1],
                           s=5, c=cls_colors[c_idx], alpha=0.55, linewidths=0)
            ax.set_title(f"{ds_label}  $\\phi_{col+1}$", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = OUT / "fig_eigenfunctions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Gram matrix heatmap (MNIST)
# ──────────────────────────────────────────────────────────────────────────────
def fig_gram_matrix():
    import torch
    from src.dataset.torchvision_flat import TorchvisionFlatDataset

    ck_path = ROOT / "EFDO_colab/logs/grid_mnist_mc_off_s0/checkpoint_final.pt"
    if not ck_path.exists():
        print(f"  SKIP gram matrix: checkpoint not found")
        return

    model, cfg = _load_run(ck_path)
    dc = cfg["dataset"]
    K  = cfg["model"]["K"]

    val_ds = TorchvisionFlatDataset(
        split="val", name=dc["name"],
        root=str(ROOT / dc.get("root", "data/mnist")),
        task=dc["task"],
        binary_classes=tuple(dc.get("binary_classes", [0, 1])),
        val_fraction=dc["val_fraction"],
        standardize=dc["standardize"],
    )
    N = min(4000, len(val_ds))
    X = torch.tensor(
        np.array([val_ds[i]["x"] for i in range(N)]), dtype=torch.float32)

    with torch.no_grad():
        phi_cols = []
        for k in range(K):
            chunks = [model.basis_set.functions[k].predict(X[i:i+512])
                      for i in range(0, N, 512)]
            phi_cols.append(torch.cat(chunks, dim=0))
        Phi = torch.cat(phi_cols, dim=1).numpy()  # (N, K)

    C = (Phi.T @ Phi) / N  # (K, K)
    D = C - np.eye(K)
    frob = float(np.linalg.norm(D, "fro"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Gram matrix $C_K = \\mathbb{{E}}[\\Phi\\Phi^\\top]$ "
        f"after training (MNIST, $K={K}$, EFDO-off, seed 0)",
        fontsize=11, fontweight="bold")

    # Left: heatmap
    vmax = max(np.abs(C).max(), 0.01)
    im = axes[0].imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title(f"$C_K$   ($\\|C-I\\|_F = {frob:.3f}$)")
    ticks = list(range(K))
    axes[0].set_xticks(ticks); axes[0].set_yticks(ticks)
    labels = [f"$\\phi_{k+1}$" for k in range(K)]
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_yticklabels(labels, fontsize=7)
    if K <= 10:
        for i in range(K):
            for j in range(K):
                col_txt = "black" if abs(C[i,j]) < 0.5*vmax else "white"
                axes[0].text(j, i, f"{C[i,j]:.2f}",
                             ha="center", va="center", fontsize=7, color=col_txt)

    # Right: diagonal bar chart
    axes[1].bar(ticks, np.diag(C), color="#1f77b4", alpha=0.85, label="$C_{ii}$ (diagonal)")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.5, label="Target = 1")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Value")
    axes[1].set_title("Diagonal elements of $C_K$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.6)

    plt.tight_layout()
    out = OUT / "fig_gram_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Figure 1: training curves...")
    fig_training_curves()

    print("Figure 2: eigenfunction visualisation...")
    try:
        fig_eigenfunctions()
    except Exception as e:
        import traceback; traceback.print_exc()

    print("Figure 3: Gram matrix heatmap...")
    try:
        fig_gram_matrix()
    except Exception as e:
        import traceback; traceback.print_exc()

    print("Done.")
