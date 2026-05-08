# CIFAR-10 Binary Classification Results

**Date**: May 8, 2026  
**Dataset**: CIFAR-10 airplane vs automobile (subset)  
**Purpose**: Extend PIEFS evaluation to binary classification on raw pixels

---

## Baseline Results (Classical Methods)

### Multi-class CIFAR-10 (3072-dim raw pixels)

| Method  | Accuracy | Std Dev | Train Size | Test Size | Time/Run |
|---------|----------|---------|------------|-----------|----------|
| RF      | 45.31%   | ±0.28%  | 50k        | 10k       | 1.7 min  |
| LR_raw  | 38.31%   | ±0.01%  | 50k        | 10k       | 54.3 min |
| PCA+LR  | 33.03%   | ±0.03%  | 50k        | 10k       | 0.1 min  |

**Observations**:
- Random Forest strongest: 45.31% (random baseline ~10%)
- Linear Regression: 38.31% (still well above random)
- PCA+LR weakest: 33.03% (suggests PCA doesn't capture useful structure on raw pixels)
- **Clear signal**: Multi-class CIFAR-10 raw pixels contain learnable structure

---

### Binary Classification: Airplane vs Automobile (3072-dim raw pixels)

| Method  | Accuracy | Std Dev | Train Size | Test Size | Time/Run |
|---------|----------|---------|------------|-----------|----------|
| RF      | 88.43%   | ±0.46%  | 10k        | 2k        | 0.4 min  |
| LR_raw  | 80.50%   | ±0.00%  | 10k        | 2k        | 5.4 min  |
| PCA+LR  | 79.13%   | ±0.03%  | 10k        | 2k        | 0.0 min  |

**Key observations**:
- **Binary task is much easier** than multiclass (random baseline ~50%)
- Random Forest: **88.43%** - strong baseline
- Linear Regression: **80.50%** - surprisingly good for linear model
- PCA+LR: **79.13%** - almost as good as raw LR!
  - Suggests PCA captures airplane-automobile separation reasonably well
  - ✓ Different from multiclass where PCA fails

---

## PIEFS Performance Target (To Be Measured)

### CIFAR-10 Binary: Airplane vs Automobile

**Proposed experiment**:
- Train PIEFS variants: off, diag, trotter
- Same protocol: K=16, 60k steps, 5 seeds
- Binary cross-entropy loss (instead of multiclass)
- Report validation accuracy + std

**Hypothesis**:
- PIEFS-off: Should exceed RF baseline (88.43%)
- PIEFS-diag: Should be comparable or better
- PIEFS-trotter: Should be best variant

**Expected results** (speculative):
```
PIEFS-off:     92-95% (clear improvement over 88.43% RF)
PIEFS-diag:    93-96% (diagonal scaling helps with pixel correlations)
PIEFS-trotter: 94-97% (rotations capture pixel interactions)
```

**Rationale**:
- Binary separation is cleaner than 10-class
- Raw pixels have strong structure (texture, color, shape)
- PIEFS should learn meaningful features

---

## Comparison Strategy with NeuralEF

### Current Status

**NeuralEF CIFAR-10 multiclass (ResNet embeddings, K=16)**:
- Published: 84.98% (CNN-GP variant)
- Our rerun: 82.52% ± 0.29% (linear probe)
- PIEFS: 85.50% ± 0.53% (ResNet embeddings)

**NeuralEF CIFAR-10 binary**: Not yet available in literature

---

## What We Can Claim

### Tier 1: Direct Comparison (If PIEFS binary results available)

**Table: CIFAR-10 Binary Airplane vs Automobile**

| Method  | Accuracy | Note |
|---------|----------|------|
| RF      | 88.43%   | Classical baseline |
| LR_raw  | 80.50%   | Linear on raw pixels |
| PCA+LR  | 79.13%   | PCA (16D) + linear |
| NeuralEF| ?        | (Not published) |
| PIEFS-off | ?      | (To be measured) |
| PIEFS-diag | ?     | (To be measured) |
| PIEFS-trotter | ?  | (To be measured) |

---

### Tier 2: Indirect Comparison (Multiclass reasoning)

**Argument**:
1. NeuralEF designed for multiclass (10-way CIFAR-10)
2. Binary classification is simpler (cleaner separation)
3. PIEFS should perform relatively better on easier task
4. Therefore: PIEFS > 88.43% (RF baseline) expected

**Example wording for paper**:
> "On binary airplane-automobile classification, baselines reach 88.43% (RF) and 80.50% (LR). The simpler geometry suggests PIEFS should comfortably exceed these classical methods, though direct NeuralEF comparison on this task is unavailable."

---

## Recommended Next Steps

### Immediate (Today):
- [ ] Train PIEFS variants on CIFAR-10 binary (if GPU available)
  - Time: ~1.5 hours per variant (off/diag/trotter)
  - 5 seeds each → 7.5 hours total
  
### If Time Permits:
- [ ] Create Table comparing PIEFS vs baselines on binary task
- [ ] Add 1-2 figures: confusion matrices, loss curves
- [ ] Appendix: Eigenfunction visualizations (do φ₁, φ₂ separate airplane/car?)

### For Paper:
- [ ] Add CIFAR-10 binary row to results table
- [ ] Clarify: "Binary classification on raw 3072-dim pixels"
- [ ] Discuss why binary is easier (geometric intuition)

---

## Results Summary for Quick Reference

```
CIFAR-10 Multiclass (50k train, 10k test)
==========================================
RF:     45.31 ± 0.28%
LR:     38.31 ± 0.01%
PCA+LR: 33.03 ± 0.03%

CIFAR-10 Binary: Airplane vs Automobile (10k train, 2k test)
===========================================================
RF:     88.43 ± 0.46%   ← Strong baseline
LR:     80.50 ± 0.00%   ← Surprisingly strong
PCA+LR: 79.13 ± 0.03%   ← Captures separation

PIEFS Expected Range
====================
PIEFS-off:     92-95%
PIEFS-diag:    93-96%
PIEFS-trotter: 94-97%
```

---

## Files to Create/Update

- [ ] `results/CIFAR10_binary_baseline_results.csv` - Save the baseline results
- [ ] `results/CIFAR10_binary_piefs_results.json` - (When PIEFS runs complete)
- [ ] `paper_0/main.tex` - Add binary classification row to Table 1
- [ ] `APPENDIX_CIFAR10_BINARY.md` - Detailed binary classification analysis

