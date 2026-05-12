#!/usr/bin/env python3
"""Generates EFDO_GPU_Experiments.ipynb — a complete Colab notebook
that runs all EFDO experiments on GPU and saves results to Google Drive.

Run from project root:
    python3 scripts/gen_colab_notebook.py
"""
import json, textwrap

def code(src: str, *, collapsed: bool = False) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsed": collapsed},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip("\n"),
    }

def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip("\n"),
    }

# ── Cells ─────────────────────────────────────────────────────────────────────

cells = []

# --------------------------------------------------------------------------- #
cells.append(md("""
# EFDO GPU Experiments
**Purpose**: Run all EFDO grid experiments on Colab GPU (T4/A100), save every
result to Google Drive, then zip for download to your Mac.

## Setup checklist
1. Runtime → Change runtime type → **GPU (T4 or A100)**
2. Upload `efdo_source.zip` to `MyDrive/EFDO_Colab/` once
   *(created by `bash scripts/package_for_colab.sh` on your Mac)*
3. **Run All Cells** — experiments resume automatically if the session disconnects.

## What runs
| Block | Runs | Steps each | Est. time T4 |
|-------|------|-----------|-------------|
| MNIST mc — off | 5 seeds | 60 k | ~3 min |
| MNIST mc — diag | 5 seeds | 60 k | ~3 min |
| MNIST mc — trotter | 5 seeds | 60 k | ~4 min |
| CIFAR-10 — off | 5 seeds | 120 k | ~5 min |
| CIFAR-10 — diag | 5 seeds | 120 k | ~5 min |
| CIFAR-10 — trotter | 5 seeds | 120 k | ~6 min |
| Fashion-MNIST — trotter | 5 seeds | 120 k | ~6 min |
| Small datasets (Two Moons, Circles, HTRU2) | 45 | 60 k | ~20 min |
| **Total** | **80** | | **~2.5 h on T4** |

Results are synced to Drive after every run. Safe to disconnect & resume.
"""))

# --------------------------------------------------------------------------- #
cells.append(code("""
# ── Cell 1: GPU check + Mount Drive ─────────────────────────────────────────
import subprocess, os, shutil, json, time, zipfile
from pathlib import Path

# GPU check
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
     "--format=csv,noheader"],
    capture_output=True, text=True
)
print("GPU:", gpu.stdout.strip() if gpu.returncode == 0 else "NOT FOUND — switch to GPU runtime!")

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ── Paths
DRIVE_ROOT = Path("/content/drive/MyDrive/EFDO_Colab")
LOCAL_ROOT  = Path("/content/efdo")
LOG_DIR     = LOCAL_ROOT / "logs"
DRIVE_LOGS  = DRIVE_ROOT / "logs"

for d in [DRIVE_ROOT, DRIVE_LOGS, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Drive root  : {DRIVE_ROOT}")
print(f"Local root  : {LOCAL_ROOT}")
"""))

# --------------------------------------------------------------------------- #
cells.append(code("""
# ── Cell 2: Install dependencies + extract source code ───────────────────────
import sys

# Dependencies (torch & torchvision pre-installed on Colab)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "hydra-core>=1.3", "omegaconf>=2.3", "scikit-learn>=1.3"],
               check=True)
print("Dependencies installed ✓")

# Extract EFDO source from Drive
source_zip = DRIVE_ROOT / "efdo_source.zip"
if not source_zip.exists():
    raise FileNotFoundError(
        f"\\n\\n*** STOP ***\\n"
        f"Upload efdo_source.zip to your Google Drive at:\\n"
        f"  MyDrive/EFDO_Colab/efdo_source.zip\\n\\n"
        f"Create it on your Mac with:\\n"
        f"  bash scripts/package_for_colab.sh"
    )

LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(source_zip) as zf:
    zf.extractall(LOCAL_ROOT)
os.chdir(LOCAL_ROOT)

print(f"Source extracted to {LOCAL_ROOT}")
print("Files:", [p.name for p in LOCAL_ROOT.iterdir() if not p.name.startswith('.')])
"""))

# --------------------------------------------------------------------------- #
cells.append(code("""
# ── Cell 3: CIFAR-10 feature extraction ─────────────────────────────────────
# If you already ran this once, the features are cached in Drive (~250 MB).
CIFAR_LOCAL = LOCAL_ROOT / "data" / "cifar10_features"
CIFAR_DRIVE = DRIVE_ROOT / "cifar10_features"

if (CIFAR_DRIVE / "X_train.npy").exists():
    # Fast path: copy from Drive cache
    print("Loading CIFAR-10 features from Drive cache…")
    CIFAR_LOCAL.mkdir(parents=True, exist_ok=True)
    for f in CIFAR_DRIVE.iterdir():
        shutil.copy(f, CIFAR_LOCAL / f.name)
    print("Done ✓")
else:
    # Slow path: extract from raw CIFAR-10 with ResNet-18 (~5 min on GPU)
    print("Extracting CIFAR-10 features with ResNet-18 (first time, ~5 min)…")
    result = subprocess.run(
        [sys.executable, "scripts/extract_cnn_features.py",
         "--batch_size", "512", "--out_dir", str(CIFAR_LOCAL)],
        cwd=LOCAL_ROOT, capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError("Feature extraction failed.")
    # Save to Drive for future runs
    CIFAR_DRIVE.mkdir(parents=True, exist_ok=True)
    for f in CIFAR_LOCAL.iterdir():
        shutil.copy(f, CIFAR_DRIVE / f.name)
    print(f"Features saved to Drive at {CIFAR_DRIVE} ✓")
"""))

# --------------------------------------------------------------------------- #
cells.append(code("""
# ── Cell 4: Helper functions ─────────────────────────────────────────────────

def sync_run_to_drive(run_id: str):
    \"\"\"Copy local logs/<run_id>/ → Drive/logs/<run_id>/  (overwrite).\"\"\"
    src = LOG_DIR / run_id
    dst = DRIVE_LOGS / run_id
    if src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)


def restore_run_from_drive(run_id: str):
    \"\"\"Copy Drive/logs/<run_id>/ → local if not already present.\"\"\"
    src = DRIVE_LOGS / run_id
    dst = LOG_DIR / run_id
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)


def run_experiment(run_id: str, ds_args: str, metric: str,
                   seed: int, steps: int) -> bool:
    \"\"\"Run one EFDO experiment. Skip if done. Resume from best checkpoint.\"\"\"
    final_ckpt = LOG_DIR / run_id / "checkpoint_final.pt"
    best_ckpt  = LOG_DIR / run_id / "checkpoint_best_val.pt"

    # 1. Already done locally?
    if final_ckpt.exists():
        print(f"[SKIP]   {run_id}")
        return True

    # 2. Already done on Drive? Restore and skip.
    drive_final = DRIVE_LOGS / run_id / "checkpoint_final.pt"
    if drive_final.exists():
        restore_run_from_drive(run_id)
        print(f"[SKIP]   {run_id}  (restored from Drive)")
        return True

    # 3. Partial run on Drive? Restore for resume.
    drive_best = DRIVE_LOGS / run_id / "checkpoint_best_val.pt"
    if drive_best.exists() and not best_ckpt.exists():
        restore_run_from_drive(run_id)
        print(f"         {run_id}: restored partial from Drive for resume")

    # 4. Build command
    cmd = [
        sys.executable, "train.py",
        f"run_id={run_id}",
        *ds_args.split(),
        f"model.metric_type={metric}",
        f"trainer.seed={seed}",
        f"trainer.total_steps={steps}",
        "writer.mode=disabled",
    ]
    if best_ckpt.exists():
        cmd.append(f"+resume={best_ckpt}")
        print(f"[RESUME] {run_id}")
    else:
        print(f"[START]  {run_id}")

    # 5. Run
    t0 = time.time()
    result = subprocess.run(cmd, cwd=LOCAL_ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        # Get final val_acc
        metrics_file = LOG_DIR / run_id / "metrics.jsonl"
        val_acc = 0.0
        if metrics_file.exists():
            lines = metrics_file.read_text().strip().splitlines()
            if lines:
                val_acc = json.loads(lines[-1]).get("val_acc", 0) * 100
        sync_run_to_drive(run_id)
        print(f"[DONE]   {run_id}  val_acc={val_acc:.2f}%  time={elapsed/60:.1f} min")
        return True
    else:
        # Partial progress still useful: sync what we have for resume later
        sync_run_to_drive(run_id)
        print(f"[ERROR]  {run_id} (code {result.returncode}) — partial synced to Drive")
        return False

print("Helper functions defined ✓")
print(f"PyTorch device check:")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Block 1: MNIST multiclass  (15 runs × 60 k steps, ~25 min on T4)
"""))

cells.append(code("""
# ── Cell 5: MNIST multiclass ─────────────────────────────────────────────────
MC = "dataset=mnist_multiclass model.task=multiclass model.K=16"

print("=== MNIST mc: off ===")
for s in range(5):
    run_experiment(f"grid_mnist_mc_off_s{s}", MC, "off", s, 60_000)

print("\\n=== MNIST mc: diag ===")
for s in range(5):
    run_experiment(f"grid_mnist_mc_diag_s{s}", MC, "diag", s, 60_000)

print("\\n=== MNIST mc: trotter ===")
for s in range(5):
    run_experiment(f"grid_mnist_mc_lambda_u_trotter_s{s}", MC, "lambda_u_trotter", s, 60_000)

print("\\n✓ MNIST mc complete")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Block 2: CIFAR-10 features  (15 runs × 120 k steps, ~80 min on T4)
"""))

cells.append(code("""
# ── Cell 6: CIFAR-10 features ────────────────────────────────────────────────
CF = "dataset=cifar10_features_multiclass model.task=multiclass model.K=16"

print("=== CIFAR-10: off ===")
for s in range(5):
    run_experiment(f"grid_cifar10_features_off_s{s}", CF, "off", s, 120_000)

print("\\n=== CIFAR-10: diag ===")
for s in range(5):
    run_experiment(f"grid_cifar10_features_diag_s{s}", CF, "diag", s, 120_000)

print("\\n=== CIFAR-10: trotter ===")
for s in range(5):
    run_experiment(f"grid_cifar10_features_lambda_u_trotter_s{s}",
                   CF, "lambda_u_trotter", s, 120_000)

print("\\n✓ CIFAR-10 features complete")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Block 3: Fashion-MNIST trotter  (5 runs × 120 k steps, ~30 min on T4)
"""))

cells.append(code("""
# ── Cell 7: Fashion-MNIST trotter ────────────────────────────────────────────
FM = "dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"

print("=== Fashion-MNIST: trotter ===")
for s in range(5):
    run_experiment(f"grid_fashion_mnist_lambda_u_trotter_s{s}",
                   FM, "lambda_u_trotter", s, 120_000)

print("\\n✓ Fashion-MNIST trotter complete")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Block 4 (optional): Small datasets — Two Moons, Circles, HTRU2
Already done on Mac. Skip this cell if you have the results. Re-run takes ~20 min.
"""))

cells.append(code("""
# ── Cell 8: Small datasets (optional — skip if already done on Mac) ──────────
TM = "dataset=two_moon_multiclass model.task=binary model.K=6"
CI = "dataset=circles_multiclass model.task=binary model.K=6"
HT = "dataset=htru2 model.task=binary model.K=6"

SKIP_SMALL = True  # ← set False to re-run on Colab

if not SKIP_SMALL:
    for ds_id, ds_args in [("two_moon", TM), ("circles", CI), ("htru2", HT)]:
        for metric in ["off", "diag", "lambda_u_trotter"]:
            for s in range(5):
                run_experiment(
                    f"grid_{ds_id}_{metric}_s{s}",
                    ds_args, metric, s, 60_000
                )
    print("\\n✓ Small datasets complete")
else:
    print("SKIP_SMALL=True — skipping. Set to False to re-run.")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Results — collect metrics and generate paper table
"""))

cells.append(code("""
# ── Cell 9: Collect all results ──────────────────────────────────────────────
import statistics

def load_val_acc(run_id: str) -> float | None:
    \"\"\"Return final val_acc from metrics.jsonl, or None if not done.\"\"\"
    for base in [LOG_DIR, DRIVE_LOGS]:
        mf = base / run_id / "metrics.jsonl"
        if mf.exists():
            lines = mf.read_text().strip().splitlines()
            if lines:
                return json.loads(lines[-1]).get("val_acc", None)
    return None


def collect_block(prefix, metrics, seeds=range(5)):
    \"\"\"Return dict: metric -> list of val_acc (may contain None for missing).\"\"\"
    out = {}
    for m in metrics:
        tag = m.replace("lambda_u_trotter", "trotter")
        accs = []
        for s in seeds:
            rid = f"{prefix}_{m}_s{s}"
            acc = load_val_acc(rid)
            accs.append(round(acc * 100, 2) if acc else None)
        out[tag] = accs
    return out


results = {}

results["mnist_mc"]   = collect_block("grid_mnist_mc",
    ["off", "diag", "lambda_u_trotter"])
results["cifar10"]    = collect_block("grid_cifar10_features",
    ["off", "diag", "lambda_u_trotter"])
results["fashion_mnist_trotter"] = {
    "trotter": [load_val_acc(f"grid_fashion_mnist_lambda_u_trotter_s{s}")
                for s in range(5)]
}
# Load already-done FM off/diag
for m in ["off", "diag"]:
    accs = [load_val_acc(f"grid_fashion_mnist_{m}_s{s}") for s in range(5)]
    results.setdefault("fashion_mnist", {})[m] = [
        round(a*100,2) if a else None for a in accs]
results["fashion_mnist"]["trotter"] = [
    round(a*100,2) if a else None
    for a in results["fashion_mnist_trotter"]["trotter"]
]

# Pretty print
print("\\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for ds, block in results.items():
    if ds == "fashion_mnist_trotter":
        continue
    print(f"\\n{ds}:")
    for metric, accs in block.items():
        valid = [a for a in accs if a is not None]
        if len(valid) == 5:
            mean = statistics.mean(valid)
            std  = statistics.stdev(valid)
            print(f"  {metric:8s}: {mean:.2f} ± {std:.2f}%  {valid}")
        else:
            print(f"  {metric:8s}: {valid}  ({5-len(valid)} pending)")

# Save results JSON to Drive
results_path = DRIVE_ROOT / "results_summary.json"
results_path.write_text(json.dumps(results, indent=2))
print(f"\\nSaved to {results_path}")
"""))

# --------------------------------------------------------------------------- #
cells.append(code("""
# ── Cell 10: LaTeX table rows ────────────────────────────────────────────────

def fmt(accs):
    valid = [a for a in accs if a is not None]
    if not valid:
        return "---"
    if len(valid) < 5:
        return f"{statistics.mean(valid):.2f}$^*$"
    return f"${statistics.mean(valid):.2f}\\\\pm{statistics.stdev(valid):.2f}$"

print("% ── Table rows for paper ────────────────────────────────────")
print("% MNIST mc")
for m in ["off", "diag", "trotter"]:
    name = {"off": "off (identity)", "diag": "diag", "trotter": "Trotter"}[m]
    key  = {"trotter": "trotter"}.get(m, m)
    acc = fmt(results["mnist_mc"].get(key, [None]*5))
    print(f"EFDO ({name}) & {acc} \\\\\\\\")

print()
print("% CIFAR-10 features")
for m in ["off", "diag", "trotter"]:
    key = {"trotter": "trotter"}.get(m, m)
    acc = fmt(results["cifar10"].get(key, [None]*5))
    print(f"EFDO ({m}) & {acc} \\\\\\\\")

print()
print("% Fashion-MNIST")
for m in ["off", "diag", "trotter"]:
    acc = fmt(results.get("fashion_mnist", {}).get(m, [None]*5))
    print(f"EFDO FM ({m}) & {acc} \\\\\\\\")
"""))

# --------------------------------------------------------------------------- #
cells.append(md("""
## Download — package all results as zip for your Mac
"""))

cells.append(code("""
# ── Cell 11: Zip all results for download ────────────────────────────────────
import zipfile as zf
from datetime import datetime

ts  = datetime.now().strftime("%Y%m%d_%H%M")
out = DRIVE_ROOT / f"efdo_results_{ts}.zip"

print(f"Packaging results → {out} …")
with zf.ZipFile(out, "w", zf.ZIP_DEFLATED) as z:
    # Add every completed run directory
    for run_dir in sorted(DRIVE_LOGS.iterdir()):
        if not run_dir.is_dir():
            continue
        for f in run_dir.rglob("*"):
            if f.is_file() and f.suffix in {".pt", ".jsonl", ".json", ".md", ".log"}:
                z.write(f, f.relative_to(DRIVE_ROOT))
    # Add results summary
    if results_path.exists():
        z.write(results_path, results_path.relative_to(DRIVE_ROOT))

size_mb = out.stat().st_size / 1e6
print(f"Created: {out}  ({size_mb:.0f} MB)")
print()
print("To download to your Mac:")
print(f"  1. Open Google Drive → MyDrive/EFDO_Colab/")
print(f"  2. Download efdo_results_{ts}.zip")
print(f"  3. On Mac: cd /Users/varvaranazarenko/materials/EFDO")
print(f"     unzip ~/Downloads/efdo_results_{ts}.zip -d .")
print(f"     # Results will be merged into logs/")
"""))

# ── Notebook metadata ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out_path = "EFDO_GPU_Experiments.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
print(f"Generated: {out_path}  ({os.path.getsize(out_path)//1024} KB)")
print("Done ✓")
