"""Generate additional figures: K-ablation and Gram error convergence."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "paper_0" / "figures"
LOGS = ROOT / "EFDO_colab" / "logs"

# Style
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})

# ============================================================================
# Figure A: Gram error convergence over training steps
# ============================================================================
def fig_gram_error_convergence():
    """Show gram_error decay during training for MNIST and CIFAR-10."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Gram Matrix Constraint Convergence (mean ± std, 5 seeds)",
                 fontsize=12, fontweight="bold")
    
    datasets = [
        ("MNIST", "grid_mnist_mc_off", 0),
        ("CIFAR-10", "grid_cifar10_features_off", 1)
    ]
    
    for ds_label, ds_key, ax_idx in datasets:
        ax = axes[ax_idx]
        
        all_gram = []
        steps_ref = None
        
        for s in range(5):
            log_dir = LOGS / f"{ds_key}_s{s}"
            metrics_file = log_dir / "metrics.jsonl"
            
            if not metrics_file.exists():
                continue
            
            rows = [json.loads(l) for l in metrics_file.read_text().strip().splitlines()]
            if steps_ref is None:
                steps_ref = [r["step"] for r in rows]
            
            gram_errors = []
            for row in rows:
                if "gram_error" in row:
                    gram_errors.append(row["gram_error"])
                else:
                    gram_errors.append(np.nan)
            
            all_gram.append(gram_errors)
        
        if not all_gram:
            print(f"  SKIP {ds_label}: no data")
            continue
        
        # Handle NaNs
        arr = np.array(all_gram)
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0)
        xs = np.array(steps_ref) / 1000
        
        ax.plot(xs, m, color="#1f77b4", lw=2.5, label="EFDO-off")
        ax.fill_between(xs, m - s, m + s, alpha=0.2, color="#1f77b4")
        
        ax.set_title(ds_label)
        ax.set_xlabel("Training steps (×10³)")
        ax.set_ylabel(r"Gram error $\|C_k - I_k\|_F$")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend()
    
    plt.tight_layout()
    out = OUT / "fig_gram_convergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ============================================================================
# Figure B: K-ablation (accuracy vs number of eigenfunctions)
# ============================================================================
def fig_k_ablation():
    """Show how accuracy depends on K (number of eigenfunctions)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # MNIST off variant: check if we have per-K data
    log_dir = LOGS / "grid_mnist_mc_off_s0"
    metrics_file = log_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        print("  SKIP K-ablation: no metrics.jsonl")
        return
    
    rows = [json.loads(l) for l in metrics_file.read_text().strip().splitlines()]
    
    # Group by k
    k_vals = {}
    for row in rows:
        k = row.get("k", 0)
        if k not in k_vals:
            k_vals[k] = []
        if "val_acc" in row:
            k_vals[k].append(row["val_acc"] * 100)
    
    if not k_vals:
        print("  SKIP K-ablation: no per-k data in metrics")
        return
    
    ks = sorted(k_vals.keys())
    means = [np.mean(k_vals[k]) for k in ks]
    stds = [np.std(k_vals[k]) for k in ks]
    
    ax.errorbar(ks, means, yerr=stds, fmt='o-', color="#2ca02c", 
                markersize=8, linewidth=2.5, capsize=5, capthick=2, label="EFDO-off")
    ax.fill_between(ks, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                    alpha=0.2, color="#2ca02c")
    
    ax.set_xlabel("Number of eigenfunctions K", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    ax.set_title("K-ablation: Spectral Expressivity on MNIST", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    out = OUT / "fig_k_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    print("Figure A: Gram error convergence...")
    try:
        fig_gram_error_convergence()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("Figure B: K-ablation...")
    try:
        fig_k_ablation()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("Done.")

