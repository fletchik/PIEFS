"""Generate EFDO_Trotter_Fixed_Colab.ipynb.

Reruns ALL lambda_u_trotter experiments (6 datasets × 5 seeds = 30 runs)
with the corrected A=ΛU metric order so that U actually receives gradients.

Usage:
    python3 scripts/gen_trotter_fixed_notebook.py
"""
import json
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n"),
    }


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n"),
    }


# ── cells ──────────────────────────────────────────────────────────────────

cells = []

# ── 0. Title ───────────────────────────────────────────────────────────────
cells.append(md("""# EFDO — Trotter Fixed: Full Re-run on A100
**Purpose:** Re-run all `lambda_u_trotter` experiments with the corrected
`A = Λ·U` metric order so that the rotation network actually receives gradients.

**What changed in the code:**
In `apply_to()` the rotation `U` is now applied *first* (`U·v`), then scaled by `Λ`.
Previously the order was reversed (`U·(Λv)`), which gave `‖UΛv‖²=‖Λv‖²` — independent of U.

**30 runs total:** 6 datasets × 5 seeds, all `lambda_u_trotter`.
Logs and checkpoints saved to Google Drive after every run.
Skip/resume logic: if `checkpoint_final.pt` exists on Drive → skip; if `checkpoint_best_val.pt` exists → resume.
"""))

# ── 1. Mount Drive ─────────────────────────────────────────────────────────
cells.append(md("## 1. Mount Google Drive"))
cells.append(code("""
from google.colab import drive
drive.mount('/content/drive')
"""))

# ── 2. Setup DRIVE_ROOT ────────────────────────────────────────────────────
cells.append(md("## 2. Configure paths\n\n"
"**Isolation note:** Each Colab runtime has its own VM so `/content/` never conflicts "
"with other notebooks. Drive writes go to `trotter_fixed_A/logs/` — a dedicated "
"subdirectory that no other EFDO notebook touches."))
cells.append(code("""
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/EFDO_colab")

# ── Dedicated output dir for this notebook ONLY ──────────────────────────
# Other notebooks write to:
#   EFDO_colab/logs/          (EFDO_GPU_Experiments)
#   EFDO_colab/EFDO_NeuralEF/ (NeuralEF_Colab_Benchmark)
#   EFDO_colab/baselines/     (EFDO_Baselines_Colab)
# This notebook writes ONLY to:
#   EFDO_colab/trotter_fixed_A/logs/   <-- isolated, safe to run in parallel
DRIVE_LOG = DRIVE_ROOT / "trotter_fixed_A" / "logs"
DRIVE_LOG.mkdir(parents=True, exist_ok=True)

# Data dir (shared read-only; CIFAR-10 features must already be here)
DATA_DIR = DRIVE_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("DRIVE_LOG :", DRIVE_LOG)
print("DATA_DIR  :", DATA_DIR)

assert (DRIVE_ROOT / "efdo_source.zip").exists(), (
    f"efdo_source.zip not found at {DRIVE_ROOT}/efdo_source.zip\\n"
    "On your Mac: bash scripts/package_for_colab.sh\\n"
    "Then upload efdo_source.zip to MyDrive/EFDO_colab/"
)
print("efdo_source.zip found ✓")
"""))

# ── 3. Install deps ────────────────────────────────────────────────────────
cells.append(md("## 3. Install dependencies"))
cells.append(code("""
%%capture
!pip install torch torchvision hydra-core omegaconf wandb scikit-learn numpy
"""))

# ── 4. Extract code and verify fix ─────────────────────────────────────────
cells.append(md("## 4. Extract code & verify fix"))
cells.append(code(r"""
import zipfile, os, shutil

CODE_DIR = Path("/content/efdo")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir()

with zipfile.ZipFile(DRIVE_ROOT / "efdo_source.zip") as zf:
    zf.extractall(CODE_DIR)
print("Code extracted to", CODE_DIR)

# Verify that the fix is present
trotter_file = CODE_DIR / "src/model/metric/lambda_u_trotter.py"
src_text = trotter_file.read_text()
assert "u_v = self._trotter_rotate(omega, v)" in src_text, (
    "FIX NOT FOUND in lambda_u_trotter.py!\n"
    "Rebuild efdo_source.zip from the fixed code:\n"
    "  bash scripts/package_for_colab.sh\n"
    "and re-upload to Drive."
)
print("Fix verified: U is applied first (U·v), then Λ ✓")
print("  apply_to order: Λ·(U·v)  →  ‖A∇φ‖² depends on both Λ and U")

os.chdir(CODE_DIR)
print("Working dir:", os.getcwd())
"""))

# ── 5. Symlink data ────────────────────────────────────────────────────────
cells.append(md("## 5. Symlink datasets from Drive"))
cells.append(code(r"""
import os

local_data = CODE_DIR / "data"
local_data.mkdir(exist_ok=True)

# CIFAR-10 features (pre-extracted ResNet-18 embeddings)
cifar_src = DATA_DIR / "cifar10_features"
cifar_dst = local_data / "cifar10_features"
if cifar_src.exists() and not cifar_dst.exists():
    cifar_dst.symlink_to(cifar_src)
    print("Symlinked cifar10_features ✓")
elif not cifar_src.exists():
    print(f"WARNING: {cifar_src} not found — CIFAR-10 runs will download/fail")
    print("  Run scripts/extract_cnn_features.py first and copy to Drive.")
else:
    print("cifar10_features already linked ✓")

print("data/ contents:", list(local_data.iterdir()))
"""))

# ── 6. Experiment config ───────────────────────────────────────────────────
cells.append(md("## 6. Experiment definitions"))
cells.append(code("""
# All 30 trotter runs: 6 datasets × 5 seeds
# Each entry: (display_name, dataset_config_key, task, K, total_steps)
EXPERIMENTS = [
    ("two_moon",          "two_moon",                    "binary",     6,  60_000),
    ("circles",           "circles",                     "binary",     6,  60_000),
    ("htru2",             "htru2",                       "binary",     6,  60_000),
    ("mnist_mc",          "mnist_multiclass",            "multiclass", 16, 60_000),
    ("cifar10_features",  "cifar10_features_multiclass", "multiclass", 16, 120_000),
    ("fashion_mnist",     "fashion_mnist_multiclass",    "multiclass", 16, 120_000),
]
SEEDS   = [0, 1, 2, 3, 4]
METRIC  = "lambda_u_trotter"

# Summary
total = len(EXPERIMENTS) * len(SEEDS)
print(f"Total runs planned: {total}  ({len(EXPERIMENTS)} datasets × {len(SEEDS)} seeds)")
for ds, cfg, task, K, steps in EXPERIMENTS:
    print(f"  {ds:<20}  K={K}  steps={steps:,}  task={task}")
"""))

# ── 7. Helper: run one experiment ─────────────────────────────────────────
cells.append(md("## 7. Training helper"))
_helper_lines = [
    "import subprocess, shutil, time, json\n",
    "\n",
    "def sync_to_drive(run_id: str):\n",
    "    # Copy local logs/run_id to DRIVE_LOG/run_id (full overwrite).\n",
    "    src = Path('logs') / run_id\n",
    "    dst = DRIVE_LOG / run_id\n",
    "    if dst.exists():\n",
    "        shutil.rmtree(dst)\n",
    "    shutil.copytree(src, dst)\n",
    "\n",
]
cells.append(code(r"""
import subprocess, shutil, time, json

def sync_to_drive(run_id: str):
    # Copy local logs/run_id to DRIVE_LOG/run_id (full overwrite).
    src = Path("logs") / run_id
    dst = DRIVE_LOG / run_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def run_experiment(ds_name, cfg_key, task, K, total_steps, seed):
    run_id = f"grid_{ds_name}_lambda_u_trotter_s{seed}"
    drive_final = DRIVE_LOG / run_id / "checkpoint_final.pt"
    drive_best  = DRIVE_LOG / run_id / "checkpoint_best_val.pt"

    # ── Skip if already done ──
    if drive_final.exists():
        print(f"  [SKIP]    {run_id}  (checkpoint_final on Drive)")
        return

    # ── Resume if partial ──
    resume_arg = ""
    if drive_best.exists():
        # Copy partial checkpoint from Drive to local
        local_run = Path("logs") / run_id
        local_run.mkdir(parents=True, exist_ok=True)
        shutil.copy(drive_best, local_run / "checkpoint_best_val.pt")
        resume_arg = f"+resume={local_run}/checkpoint_best_val.pt"
        print(f"  [RESUME]  {run_id}")
    else:
        print(f"  [START]   {run_id}")

    # ── Build command ──
    mc_extra = (
        f"model.task={task} model.K={K} "
        f"dataset={cfg_key}"
    ) if task == "multiclass" else (
        f"dataset={cfg_key}"
    )

    cmd = (
        f"python train.py "
        f"run_id={run_id} "
        f"{mc_extra} "
        f"model.metric_type={METRIC} "
        f"trainer.seed={seed} "
        f"trainer.total_steps={total_steps} "
        f"writer.mode=disabled "
        f"{resume_arg}"
    )

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = (time.time() - t0) / 60

    if result.returncode != 0:
        print(f"  [ERROR]   {run_id}  (exit {result.returncode})")
        sync_to_drive(run_id)   # save partial
        return

    # ── Sync to Drive ──
    sync_to_drive(run_id)
    print(f"  [DONE]    {run_id}  ({elapsed:.1f} min) → saved to Drive")

print("Helper defined ✓")
"""))

# ── 8. Run all ─────────────────────────────────────────────────────────────
cells.append(md("## 8. Run all experiments\n\n> **Runtime estimate on A100:** ~2–3 h total"))
cells.append(code(r"""
import time

t_total = time.time()
n_done, n_skip, n_err = 0, 0, 0

for ds_name, cfg_key, task, K, total_steps in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"=== {ds_name} (trotter, {total_steps:,} steps) ===")
    print(f"{'='*60}")
    for seed in SEEDS:
        run_id = f"grid_{ds_name}_lambda_u_trotter_s{seed}"
        drive_final = DRIVE_LOG / run_id / "checkpoint_final.pt"
        if drive_final.exists():
            n_skip += 1
        run_experiment(ds_name, cfg_key, task, K, total_steps, seed)
        n_done += 1

elapsed_total = (time.time() - t_total) / 60
print(f"\n{'='*60}")
print(f"All done!  total={n_done}  skipped={n_skip}  wall={elapsed_total:.1f} min")
print(f"Results saved to: {DRIVE_LOG}")
"""))

# ── 9. Collect results ─────────────────────────────────────────────────────
cells.append(md("## 9. Collect results"))
cells.append(code(r"""
import json, math, numpy as np

def read_best_val(run_dir: Path):
    jsonl = run_dir / "metrics.jsonl"
    if not jsonl.exists():
        return None
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return None
    return max(r.get("val_acc", 0.0) or 0.0 for r in rows)

results = {}   # ds_name → list of val_acc

for ds_name, *_ in EXPERIMENTS:
    accs = []
    for seed in SEEDS:
        run_id = f"grid_{ds_name}_lambda_u_trotter_s{seed}"
        run_dir = DRIVE_LOG / run_id
        acc = read_best_val(run_dir)
        if acc is not None:
            accs.append(acc * 100)   # → percent
    results[ds_name] = accs

print("\n=== Results: lambda_u_trotter (FIXED A=ΛU) ===\n")
for ds_name, accs in results.items():
    if not accs:
        print(f"  {ds_name:<22}  no results yet")
        continue
    mean = sum(accs) / len(accs)
    std  = math.sqrt(sum((x - mean)**2 for x in accs) / max(len(accs)-1, 1))
    seeds_str = "  ".join(f"{a:.2f}" for a in accs)
    print(f"  {ds_name:<22}  {mean:.2f}±{std:.2f}%  ({len(accs)}/5 seeds)  [{seeds_str}]")
"""))

# ── 10. LaTeX table row ────────────────────────────────────────────────────
cells.append(md("## 10. LaTeX row for paper Table 2"))
cells.append(code(r"""
import math

def fmt(accs):
    if not accs:
        return "---"
    mean = sum(accs) / len(accs)
    std  = math.sqrt(sum((x - mean)**2 for x in accs) / max(len(accs)-1, 1))
    mark = "" if len(accs) == 5 else f"^{{\\text{{{len(accs)}/5}}}}"
    return f"${mean:.2f}\\pm{std:.2f}{mark}$"

cols = ["two_moon", "circles", "htru2", "mnist_mc", "fashion_mnist", "cifar10_features"]
vals = "  &  ".join(fmt(results.get(c, [])) for c in cols)
row  = f"    EFDO (Trotter, fixed)  &  {vals}  \\\\"

print("Copy-paste into Table 2:\n")
print(row)

# Save to Drive
out = DRIVE_ROOT / "trotter_fixed_A" / "results_trotter_fixed_A.txt"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    f.write("=== lambda_u_trotter FIXED (A=ΛU) ===\n\n")
    for ds_name, accs in results.items():
        if accs:
            mean = sum(accs)/len(accs)
            std  = math.sqrt(sum((x-mean)**2 for x in accs)/max(len(accs)-1,1))
            per_seed = "  ".join(f"{a:.2f}" for a in accs)
            f.write(f"{ds_name}: {mean:.2f}±{std:.2f}%  seeds=[{per_seed}]\n")
    f.write("\nLaTeX row:\n")
    f.write(row + "\n")
print(f"\nSaved to {out}")
"""))

# ── 11. Download summary ───────────────────────────────────────────────────
cells.append(md("## 11. Download results summary"))
cells.append(code(r"""
from google.colab import files
results_file = DRIVE_ROOT / "trotter_fixed_A" / "results_trotter_fixed_A.txt"
if results_file.exists():
    files.download(str(results_file))
    print("Downloaded ✓")
else:
    print("No results file yet — run Cell 9 first")
"""))

# ── Assemble notebook ──────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "A100",
            "name": "EFDO_Trotter_Fixed_Colab.ipynb",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent.parent / "EFDO_Trotter_Fixed_Colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Generated: {out_path}")
print(f"  {len(cells)} cells")
