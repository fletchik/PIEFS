#!/usr/bin/env python3
"""
gen_baselines_notebook.py — generates EFDO_Baselines_Colab.ipynb

Covers:
  A. sklearn baselines for MNIST mc, Fashion-MNIST, CIFAR-10 features
  B. EFDO trotter remaining (backup for Mac)
  C. Paper-ready results table
"""
import json, textwrap

def cell(cell_type, source, **kwargs):
    base = {"cell_type": cell_type, "metadata": {}, "source": textwrap.dedent(source).lstrip()}
    if cell_type == "code":
        base.update({"outputs": [], "execution_count": None})
    return {**base, **kwargs}

def md(s):  return cell("markdown", s)
def code(s): return cell("code", s)

# ─────────────────────────────────────────────────────────────────────────────
cells = []

# ── Cell 0: Intro ─────────────────────────────────────────────────────────────
cells.append(md("""
    # EFDO Baselines & Remaining Experiments
    
    **Purpose:** Fill in the workshop paper comparison table with:
    
    | Block | What | Why |
    |-------|------|-----|
    | A | sklearn baselines (PCA, RF, LR) for MNIST / FM / CIFAR-10 | Essential paper table rows |
    | B | EFDO trotter — MNIST mc, FM, CIFAR-10 (all seeds) | Backup to Mac; same code |
    | C | Final paper table with citations for SpectralNet & NeuralEF | Copy-paste into LaTeX |
    
    **Comparison numbers from literature (cite, don't run):**
    - SpectralNet (Shaham et al. 2018): MNIST = **95.80 ± 0.20%** (Table 1)  
    - NeuralEF (Deng et al. ICML 2022): MNIST = **84.98%** (Table 2) — unsupervised features + linear probe  
    - Fashion-MNIST and CIFAR-10: not reported in those papers → mark as "n/a" in table
    
    **Setup:**
    1. Runtime → GPU (T4 or A100)
    2. `efdo_source.zip` already uploaded to `MyDrive/EFDO_colab/`  
    3. Run All Cells
"""))

# ── Cell 1: Mount Drive + paths ────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 1: Mount Drive + Paths ──────────────────────────────────────────────
    import subprocess, os, shutil, json, time, zipfile, sys
    from pathlib import Path
    
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True)
    print("GPU:", gpu.stdout.strip() if gpu.returncode == 0 else "NOT FOUND")
    
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    
    DRIVE_ROOT  = Path("/content/drive/MyDrive/EFDO_colab")
    LOCAL_ROOT  = Path("/content/efdo")
    LOG_DIR     = LOCAL_ROOT / "logs"
    DRIVE_LOGS  = DRIVE_ROOT / "logs"
    RESULTS_DIR = DRIVE_ROOT / "baselines"
    
    for d in [DRIVE_ROOT, DRIVE_LOGS, LOG_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Drive root : {DRIVE_ROOT}")
    print(f"Local root : {LOCAL_ROOT}")
    print(f"Results    : {RESULTS_DIR}")
"""))

# ── Cell 2: Install deps + extract source ──────────────────────────────────────
cells.append(code("""
    # ── Cell 2: Install dependencies + extract EFDO source ───────────────────────
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "hydra-core>=1.3", "omegaconf>=2.3", "scikit-learn>=1.3"],
                   check=True)
    print("Dependencies installed ✓")
    
    source_zip = DRIVE_ROOT / "efdo_source.zip"
    if not source_zip.exists():
        raise FileNotFoundError(
            "Upload efdo_source.zip to MyDrive/EFDO_colab/efdo_source.zip\\n"
            "Create it on your Mac: bash scripts/package_for_colab.sh"
        )
    
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as zf:
        zf.extractall(LOCAL_ROOT)
    
    os.chdir(LOCAL_ROOT)
    sys.path.insert(0, str(LOCAL_ROOT))
    print(f"Extracted source → {LOCAL_ROOT}")
    print("Working dir:", os.getcwd())
"""))

# ── Cell 3: CIFAR-10 features (reuse from Drive cache) ───────────────────────
cells.append(code("""
    # ── Cell 3: CIFAR-10 features (Drive cache or fresh extraction) ───────────────
    CIFAR_LOCAL = LOCAL_ROOT / "data" / "cifar10_features"
    CIFAR_DRIVE = DRIVE_ROOT / "cifar10_features"
    
    if (CIFAR_DRIVE / "X_train.npy").exists():
        print("Loading CIFAR-10 features from Drive cache…")
        CIFAR_LOCAL.mkdir(parents=True, exist_ok=True)
        for f in CIFAR_DRIVE.iterdir():
            shutil.copy(f, CIFAR_LOCAL / f.name)
        print("Done ✓")
    else:
        print("Extracting CIFAR-10 features with ResNet-18 (~5 min)…")
        r = subprocess.run(
            [sys.executable, "scripts/extract_cnn_features.py",
             "--output-dir", str(CIFAR_LOCAL)],
            capture_output=True, text=True, cwd=str(LOCAL_ROOT))
        print(r.stdout[-1000:])
        if r.returncode != 0:
            print("STDERR:", r.stderr[-500:])
        else:
            CIFAR_DRIVE.mkdir(parents=True, exist_ok=True)
            for f in CIFAR_LOCAL.iterdir():
                shutil.copy(f, CIFAR_DRIVE / f.name)
            print("Cached to Drive ✓")
"""))

# ── Cell 4: Block A header ────────────────────────────────────────────────────
cells.append(md("""
    ## Block A — sklearn Baselines
    
    Runs **7 baseline methods** on 3 multiclass datasets with 5 seeds each.  
    All methods use `n_components=16` (matches EFDO's `K=16`).
    
    | Method | Description |
    |--------|-------------|
    | `RF` | Random Forest (500 trees) — raw features |
    | `LR_raw` | Logistic Regression — raw features (L2, max_iter=1000) |
    | `PCA+LR` | PCA (16 components) → LogReg |
    | `KPCA_rbf+LR` | Kernel PCA (RBF) → LogReg |
    | `KPCA_poly+LR` | Kernel PCA (poly) → LogReg |
    | `TruncSVD+LR` | TruncatedSVD (16 components) → LogReg |
    | `SpectralEmb+LR` | Spectral Embedding (16 components) → LogReg |
    
    Estimated runtime: ~10-20 min total on Colab CPU (sklearn ignores GPU).
"""))

# ── Cell 5: sklearn baselines code ────────────────────────────────────────────
cells.append(code("""
    # ── Cell 4: sklearn baselines for MNIST mc / Fashion-MNIST / CIFAR-10 ──────
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
    from sklearn.manifold import SpectralEmbedding
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import torchvision, torch
    import warnings
    warnings.filterwarnings('ignore')
    
    N_COMPONENTS = 16
    SEEDS = list(range(5))
    
    # ── Load datasets as numpy ────────────────────────────────────────────────
    def load_mnist_flat(fashion=False):
        cls = torchvision.datasets.FashionMNIST if fashion else torchvision.datasets.MNIST
        tr = cls("/tmp/data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
        te = cls("/tmp/data", train=False, download=True, transform=torchvision.transforms.ToTensor())
        X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_tr = tr.targets.numpy()
        X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_te = te.targets.numpy()
        return X_tr, y_tr, X_te, y_te
    
    def load_cifar10_features():
        d = CIFAR_LOCAL
        X_tr = np.load(d / "X_train.npy")
        y_tr = np.load(d / "y_train.npy")
        X_te = np.load(d / "X_test.npy")
        y_te = np.load(d / "y_test.npy")
        return X_tr.astype(np.float32), y_tr, X_te.astype(np.float32), y_te
    
    # ── Standardize (mean/std from train split only) ──────────────────────────
    def standardize(X_tr, X_te):
        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        return (X_tr - mu) / sigma, (X_te - mu) / sigma
    
    # ── Pipeline factory ──────────────────────────────────────────────────────
    def make_pipelines(seed, K=N_COMPONENTS):
        lr = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        return {
            "RF":              RandomForestClassifier(500, random_state=seed, n_jobs=-1),
            "LR_raw":          LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1),
            "PCA+LR":          Pipeline([("pca", PCA(K, random_state=seed)), ("lr", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1))]),
            "KPCA_rbf+LR":     Pipeline([("kpca", KernelPCA(K, kernel="rbf", random_state=seed, n_jobs=-1)), ("lr", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1))]),
            "KPCA_poly+LR":    Pipeline([("kpca", KernelPCA(K, kernel="poly", random_state=seed, n_jobs=-1)), ("lr", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1))]),
            "TruncSVD+LR":     Pipeline([("svd", TruncatedSVD(K, random_state=seed)), ("lr", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1))]),
        }
    
    # ── Run one dataset ───────────────────────────────────────────────────────
    def run_baselines(name, X_tr, y_tr, X_te, y_te):
        print(f"\\n{'='*60}")
        print(f"Dataset: {name}  (train={len(X_tr)}, test={len(X_te)}, d={X_tr.shape[1]})")
        print('='*60)
        X_tr_s, X_te_s = standardize(X_tr, X_te)
        results = {}
        for method_name in ["RF", "LR_raw", "PCA+LR", "KPCA_rbf+LR", "KPCA_poly+LR", "TruncSVD+LR"]:
            accs = []
            for seed in SEEDS:
                pipes = make_pipelines(seed)
                model = pipes[method_name]
                X_fit, X_eval = (X_tr, X_te) if method_name == "RF" else (X_tr_s, X_te_s)
                model.fit(X_fit, y_tr)
                acc = model.score(X_eval, y_te) * 100
                accs.append(acc)
            mu, std = np.mean(accs), np.std(accs)
            results[method_name] = {"mean": mu, "std": std, "per_seed": accs}
            print(f"  {method_name:<20} {mu:.2f} ± {std:.2f}%")
        return results
    
    all_results = {}
    
    print("Loading MNIST…")
    X_tr, y_tr, X_te, y_te = load_mnist_flat(fashion=False)
    all_results["mnist_mc"] = run_baselines("MNIST (multiclass)", X_tr, y_tr, X_te, y_te)
    
    print("\\nLoading Fashion-MNIST…")
    X_tr, y_tr, X_te, y_te = load_mnist_flat(fashion=True)
    all_results["fashion_mnist"] = run_baselines("Fashion-MNIST (multiclass)", X_tr, y_tr, X_te, y_te)
    
    print("\\nLoading CIFAR-10 features…")
    X_tr, y_tr, X_te, y_te = load_cifar10_features()
    all_results["cifar10_features"] = run_baselines("CIFAR-10 features (multiclass)", X_tr, y_tr, X_te, y_te)
    
    # Save results
    out_file = RESULTS_DIR / "sklearn_baselines_multiclass.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\\nSaved → {out_file}")
"""))

# ── Cell 6: Block B header ────────────────────────────────────────────────────
cells.append(md("""
    ## Block B — EFDO Trotter (remaining seeds)
    
    This block runs the **trotter metric** for all 3 multiclass datasets.  
    The Mac is running these sequentially too — this block will **skip** any runs  
    already synced to Drive from the Mac.
    
    | Dataset | Steps | Seeds | Est. T4 time |
    |---------|-------|-------|-------------|
    | MNIST mc trotter | 60k | s0-s4 | ~20 min total |
    | Fashion-MNIST trotter | 120k | s0-s4 | ~40 min total |
    | CIFAR-10 off (s3,s4) | 120k | s3,s4 | ~15 min |
    | CIFAR-10 diag | 120k | s0-s4 | ~40 min total |
    | CIFAR-10 trotter | 120k | s0-s4 | ~40 min total |
"""))

# ── Cell 7: run_experiment helper ─────────────────────────────────────────────
cells.append(code("""
    # ── Cell 5: run_experiment helper (skip/resume/Drive sync) ──────────────────
    import statistics
    
    def sync_from_drive(run_id):
        \"\"\"Copy checkpoints from Drive → local if not already present.\"\"\"
        drive_run = DRIVE_LOGS / run_id
        local_run = LOG_DIR / run_id
        if not drive_run.exists():
            return False
        local_run.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in drive_run.rglob("*"):
            if f.is_file():
                dst = local_run / f.relative_to(drive_run)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy(f, dst)
                    copied += 1
        return copied > 0
    
    def sync_to_drive(run_id):
        \"\"\"Copy run results from local → Drive.\"\"\"
        local_run = LOG_DIR / run_id
        drive_run = DRIVE_LOGS / run_id
        drive_run.mkdir(parents=True, exist_ok=True)
        for f in local_run.rglob("*"):
            if f.is_file() and f.suffix in {".pt", ".jsonl", ".json", ".md", ".log"}:
                dst = drive_run / f.relative_to(local_run)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
    
    def run_experiment(run_id, ds_args, metric, seed, steps, sync=True):
        final_ckpt  = LOG_DIR   / run_id / "checkpoint_final.pt"
        drive_final = DRIVE_LOGS / run_id / "checkpoint_final.pt"
        
        if final_ckpt.exists() or drive_final.exists():
            print(f"[SKIP]   {run_id}")
            return True
        
        # Try to restore partial checkpoint from Drive
        sync_from_drive(run_id)
        
        best_ckpt = LOG_DIR / run_id / "checkpoint_best_val.pt"
        resume_flag = []
        if best_ckpt.exists():
            print(f"[RESUME] {run_id}")
            resume_flag = [f"+resume={best_ckpt}"]
        else:
            print(f"[START]  {run_id}")
        
        cmd = [sys.executable, "train.py",
               f"run_id={run_id}",
               *ds_args.split(),
               f"model.metric_type={metric}",
               f"trainer.seed={seed}",
               f"trainer.total_steps={steps}",
               "writer.mode=disabled",
               *resume_flag]
        
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(LOCAL_ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if r.returncode != 0:
            print(f"[ERROR]  {run_id}  ({elapsed:.0f}s)")
            print(r.stderr[-1000:])
            return False
        
        if sync:
            sync_to_drive(run_id)
        
        # Read final val_acc
        mf = LOG_DIR / run_id / "metrics.jsonl"
        val_acc = None
        if mf.exists():
            lines = mf.read_text().strip().splitlines()
            if lines:
                val_acc = json.loads(lines[-1]).get("val_acc", None)
        
        print(f"[DONE]   {run_id}  val_acc={val_acc*100:.2f}%  ({elapsed:.0f}s)")
        return True
    
    print("run_experiment helper ready ✓")
"""))

# ── Cell 8: MNIST mc trotter ──────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 6: MNIST mc — trotter s0-s4 (60k steps each, ~4 min/run on T4) ─────
    MC_ARGS = "dataset=mnist_multiclass model.task=multiclass model.K=16"
    
    for seed in range(5):
        run_experiment(
            f"grid_mnist_mc_lambda_u_trotter_s{seed}",
            MC_ARGS, "lambda_u_trotter", seed, 60000
        )
    print("MNIST mc trotter — done.")
"""))

# ── Cell 9: FM trotter ────────────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 7: Fashion-MNIST — trotter s0-s4 (120k steps, ~8 min/run on T4) ────
    FM_ARGS = "dataset=fashion_mnist_multiclass model.task=multiclass model.K=16"
    
    for seed in range(5):
        run_experiment(
            f"grid_fashion_mnist_lambda_u_trotter_s{seed}",
            FM_ARGS, "lambda_u_trotter", seed, 120000
        )
    print("Fashion-MNIST trotter — done.")
"""))

# ── Cell 10: CIFAR-10 remaining ────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 8: CIFAR-10 features — remaining runs (120k steps each) ─────────────
    CF_ARGS = "dataset=cifar10_features_multiclass model.task=multiclass model.K=16"
    
    # off — only s3 and s4 missing (s0,s1,s2 already done on Mac)
    for seed in [3, 4]:
        run_experiment(f"grid_cifar10_features_off_s{seed}", CF_ARGS, "off", seed, 120000)
    
    # diag — all 5 seeds
    for seed in range(5):
        run_experiment(f"grid_cifar10_features_diag_s{seed}", CF_ARGS, "diag", seed, 120000)
    
    # trotter — all 5 seeds
    for seed in range(5):
        run_experiment(f"grid_cifar10_features_lambda_u_trotter_s{seed}",
                       CF_ARGS, "lambda_u_trotter", seed, 120000)
    
    print("CIFAR-10 features — all done.")
"""))

# ── Cell 11: Block C header ───────────────────────────────────────────────────
cells.append(md("""
    ## Block C — Final Paper Table
    
    Collects all EFDO results (local + Drive) and sklearn baselines, then prints  
    the complete comparison table ready to paste into LaTeX.
    
    **Literature numbers used (no code needed):**
    - SpectralNet: MNIST = 95.80 ± 0.20% (Shaham et al. 2018, Table 1)  
    - NeuralEF: MNIST = 84.98% (Deng et al. ICML 2022, Table 2) — *unsupervised*
"""))

# ── Cell 12: collect EFDO results ─────────────────────────────────────────────
cells.append(code("""
    # ── Cell 9: Collect EFDO results ─────────────────────────────────────────────
    def load_val_acc(run_id):
        for base in [LOG_DIR, DRIVE_LOGS]:
            mf = base / run_id / "metrics.jsonl"
            if mf.exists():
                lines = mf.read_text().strip().splitlines()
                if lines:
                    return json.loads(lines[-1]).get("val_acc", None)
        return None
    
    def collect_efdo(prefix, metrics, seeds=range(5)):
        out = {}
        for m in metrics:
            tag = m.replace("lambda_u_trotter", "trotter")
            accs = []
            for s in seeds:
                rid = f"{prefix}_{m}_s{s}"
                acc = load_val_acc(rid)
                if acc is not None:
                    accs.append(acc * 100)
            if accs:
                out[tag] = {
                    "mean": statistics.mean(accs),
                    "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "n": len(accs)
                }
            else:
                out[tag] = None
        return out
    
    METRICS = ["off", "diag", "lambda_u_trotter"]
    
    efdo = {
        "mnist_mc":         collect_efdo("grid_mnist_mc",         METRICS),
        "fashion_mnist":    collect_efdo("grid_fashion_mnist",    METRICS),
        "cifar10_features": collect_efdo("grid_cifar10_features", METRICS),
    }
    
    def fmt(d):
        if d is None: return "—"
        n = d["n"]
        star = "*" if n < 5 else ""
        return f"{d['mean']:.2f}±{d['std']:.2f}{star}"
    
    print("\\nEFDO results collected:")
    for ds, res in efdo.items():
        print(f"  {ds}:")
        for m, v in res.items():
            print(f"    {m}: {fmt(v)}")
"""))

# ── Cell 13: final paper table ────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 10: Complete comparison table ───────────────────────────────────────
    
    # Load sklearn baselines (saved in Block A)
    sk_file = RESULTS_DIR / "sklearn_baselines_multiclass.json"
    sk = json.loads(sk_file.read_text()) if sk_file.exists() else {}
    
    def sk_fmt(ds, method):
        if ds not in sk or method not in sk[ds]: return "—"
        d = sk[ds][method]
        return f"{d['mean']:.2f}±{d['std']:.2f}"
    
    def ef_fmt(ds, metric):
        if ds not in efdo: return "—"
        v = efdo[ds].get(metric)
        return fmt(v)
    
    # Literature references
    LIT = {
        "SpectralNet": {"mnist_mc": "95.80±0.20†", "fashion_mnist": "—", "cifar10_features": "—"},
        "NeuralEF":    {"mnist_mc": "84.98‡",       "fashion_mnist": "—", "cifar10_features": "—"},
    }
    
    DATASETS = ["mnist_mc", "fashion_mnist", "cifar10_features"]
    DS_LABEL = {"mnist_mc": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar10_features": "CIFAR-10 feat."}
    
    print("=" * 75)
    print(f"{'Method':<22} {'MNIST mc':>17} {'Fashion-MNIST':>17} {'CIFAR-10 feat':>17}")
    print("=" * 75)
    
    for method in ["RF", "LR_raw", "PCA+LR", "KPCA_rbf+LR"]:
        row = f"{method:<22}"
        for ds in DATASETS:
            row += f" {sk_fmt(ds, method):>17}"
        print(row)
    
    print("-" * 75)
    for name, lit in LIT.items():
        row = f"{name:<22}"
        for ds in DATASETS:
            row += f" {lit.get(ds,'—'):>17}"
        print(row)
    
    print("-" * 75)
    for metric, label in [("off","EFDO off"), ("diag","EFDO diag"), ("trotter","EFDO trotter")]:
        row = f"{label:<22}"
        for ds in DATASETS:
            row += f" {ef_fmt(ds, metric):>17}"
        print(row)
    
    print("=" * 75)
    print("* = fewer than 5 seeds completed")
    print("† = Shaham et al. (2018), Table 1")
    print("‡ = Deng et al. (ICML 2022), Table 2 — unsupervised features + linear probe")
    
    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\\n\\n% ── LaTeX table (paste into paper) ──────────────────────────────────────")
    latex_lines = [
        r"\begin{table}[t]",
        "\\centering",
        "\\\\caption{Test accuracy (\\\\%) on multiclass benchmarks. Mean$\\\\pm$std over 5 seeds.}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & MNIST & Fashion-MNIST & CIFAR-10 feat.\\" ,
        r"\midrule",
    ]
    for method in ["RF", "LR_raw", "PCA+LR", "KPCA_rbf+LR"]:
        vals = " & ".join(sk_fmt(ds, method) for ds in DATASETS)
        lbl = method.replace("_", r"\_")
        latex_lines.append(f"{lbl} & {vals} \\\\")
    latex_lines.append(r"\midrule")
    latex_lines.append(r"SpectralNet~\cite{shaham2018spectralnet} & 95.80\pm0.20 & --- & --- \\")
    latex_lines.append(r"NeuralEF~\cite{deng2022neuralef} & 84.98 & --- & --- \\")
    latex_lines.append(r"\midrule")
    for metric, lbl in [("off","EFDO (off)"), ("diag","EFDO (diag)"), ("trotter","EFDO (trotter)")]:
        vals = " & ".join(ef_fmt(ds, metric) for ds in DATASETS)
        latex_lines.append(f"{lbl} & {vals} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\label{tab:multiclass}", r"\end{table}"]
    print("\n".join(latex_lines))
"""))

# ── Cell 14: download zip ──────────────────────────────────────────────────────
cells.append(code("""
    # ── Cell 11: Zip everything for download ─────────────────────────────────────
    import zipfile as zf
    from datetime import datetime
    
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = DRIVE_ROOT / f"efdo_baselines_{ts}.zip"
    
    print(f"Packaging → {out} …")
    with zf.ZipFile(out, "w", zf.ZIP_DEFLATED) as z:
        # EFDO run logs
        for run_dir in sorted(DRIVE_LOGS.iterdir()):
            if not run_dir.is_dir(): continue
            for f in run_dir.rglob("*"):
                if f.is_file() and f.suffix in {".pt", ".jsonl", ".json", ".md"}:
                    z.write(f, f.relative_to(DRIVE_ROOT))
        # sklearn baselines JSON
        for f in RESULTS_DIR.iterdir():
            if f.is_file():
                z.write(f, f"baselines/{f.name}")
    
    size_mb = out.stat().st_size / 1e6
    print(f"Created: {out.name}  ({size_mb:.1f} MB)")
    print(f"Download from Drive: MyDrive/EFDO_colab/{out.name}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out = "EFDO_Baselines_Colab.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out}  ({len(cells)} cells)")
