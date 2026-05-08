# AI4Physics 2026 Submission Materials

## 1. COVER LETTER

---

**Subject: Submission to AI4Physics Workshop @ ICML 2026**

Dear Organizing Committee,

We are pleased to submit our manuscript **"Physics-Informed Eigenfunction Features with Learnable Scaling (PIEFS)"** for consideration at the AI4Physics Workshop, ICML 2026.

### Contribution Summary

Our work addresses a fundamental challenge in machine learning: learning physically-interpretable feature representations that respect data geometry. We propose PIEFS, a method that combines:

1. **Sequential eigenfunction learning** via neural networks
2. **Modified Dirichlet Energy** loss function emphasizing spectral smoothness
3. **Learnable metric parametrization** A(x) = Λ(x)·U(ω(x)) with diagonal scaling and Givens rotations
4. **Orthogonality constraints** enforced through Gram penalties

Our method is grounded in **differential geometry** and **spectral analysis**, ensuring that learned features align with the underlying data manifold geometry rather than being purely discriminative.

### Novelty and Significance

- **Physics-informed design**: The Dirichlet Energy objective directly incorporates differential-geometric principles, contrasting with purely data-driven baselines
- **Rigorous mathematical foundation**: Grounded in Laplacian eigenmaps theory and constrained optimization
- **Comprehensive evaluation**: Five datasets (Two Moons, Circles, HTRU2, MNIST, CIFAR-10) with detailed ablation studies
- **Reproducible research**: Full codebase, hyperparameters, and figure generation scripts provided
- **Spectral interpretability**: Clear relationship to eigenfunction approximation and Rayleigh quotient bounds

### Results

PIEFS achieves:
- **100.0% accuracy** on Two Moons (perfectly captures crescent geometry via φ₂)
- **94.53% on MNIST**, competitive with baselines using K=16 eigenfunctions
- **85.50% on CIFAR-10** ResNet embeddings, demonstrating scalability
- Interpretable eigenfunction visualizations showing geometric structure (not shown in abstract but central to method)

### Alignment with AI4Physics

This work exemplifies the intersection of physics-inspired methods and machine learning:
- **Physics component**: Dirichlet Energy, Laplacian operators, spectral theory
- **Learning component**: Neural network parametrization, gradient-based optimization
- **Hybrid approach**: Leverages both physical principles and data-driven learning for feature discovery

### Code Availability

Complete source code, experiments, and reproducibility materials are provided, enabling the community to:
- Reproduce all reported results
- Extend the method to new datasets
- Integrate physics constraints into other learning frameworks

We believe this submission contributes meaningfully to advancing physics-informed machine learning and welcome any questions or suggestions.

Sincerely,

[Author Names]
[Affiliations]

---

## 2. KEYWORDS

**Primary keywords** (recommended 3-5):
1. Physics-informed neural networks
2. Spectral methods / Eigenfunction approximation
3. Interpretable feature learning
4. Differential geometry
5. Constrained optimization

**Secondary keywords** (alternative tags):
- Laplacian eigenmaps
- Data manifold learning
- Orthogonal feature bases
- Givens rotations
- Dirichlet energy minimization

---

## 3. ABSTRACT (for submission form)

### Full Abstract (250 words)

Learning interpretable feature representations that respect data geometry is essential for physics-informed machine learning. We propose PIEFS (Physics-Informed Eigenfunction Features with Learnable Scaling), a method for discovering smooth eigenfunction bases on data manifolds. The approach combines sequential neural network training with Modified Dirichlet Energy loss, enforcing smoothness through differential-geometric principles. A learnable metric A(x) = Λ(x)·U(ω(x)) parametrizes both diagonal scaling Λ(x) and Givens rotations U, enabling expressive feature transformations while maintaining orthogonality via Gram penalties.

PIEFS learns eigenfunctions φ₁, φ₂, ..., φₖ one-at-a-time with cyclic activation, using stop-gradient mechanisms to prevent interference between coordinates. On synthetic benchmarks (Two Moons, Circles), we achieve 100% and 83.6% accuracy respectively, with eigenfunction visualizations revealing nonlinear class geometry. On real data, MNIST multiclass reaches 94.5% validation accuracy with K=16 features, and CIFAR-10 ResNet embeddings achieve 85.5%, demonstrating competitive performance.

Key contributions include: (1) rigorous integration of Dirichlet energy with neural parametrization; (2) novel learnable metric design combining diagonal and rotation components; (3) dynamic loss weighting scheme with exponential dampening; (4) comprehensive ablations across three variants (off/diag/trotter) and five datasets. Theoretical motivation grounded in spectral analysis and differential geometry enables interpretable feature discovery beyond purely discriminative learning.

Limitations: PIEFS learns task-dependent coordinates rather than operator eigenmodes; finite-batch Gram penalties approximate rather than enforce global orthogonality; CPU training times remain significant. Future work addresses scalable initialization and extension to larger-scale settings.

### Short Abstract (100 words)

We propose PIEFS, a physics-informed method for learning eigenfunction feature bases respecting data geometry. Combining sequential neural network training with Modified Dirichlet Energy loss, learnable metric parametrization, and orthogonality constraints, our approach achieves competitive accuracy across multiple datasets while producing geometrically interpretable features. Grounded in differential geometry and spectral theory, PIEFS exemplifies hybrid physics-learning methods. Comprehensive evaluation on Two Moons, Circles, HTRU2, MNIST, and CIFAR-10 demonstrates that physics-informed objectives yield both performance and interpretability.

---

## 4. IMPORTANT NOTES FOR SUBMISSION

### Before Submitting

- [ ] Verify all author names and affiliations
- [ ] Check that references format matches venue requirements (currently ICML 2026 natbib style)
- [ ] Confirm paper page count: currently 9 pages (within limits for workshop submissions)
- [ ] Verify all figures display correctly in PDF
- [ ] Proofread for typos and consistency in notation

### After Acceptance (Camera-Ready)

Options for GitHub citation:

**Option A**: Cite in camera-ready version
```bibtex
@software{piefs2026github,
  author={Nazarenko, Varvara},
  title={{PIEFS: Physics-Informed Eigenfunction Features}},
  year={2026},
  url={https://github.com/your-org/piefs}
}
```

**Option B** (Recommended): Submit paper with anonymous repository, cite in acknowledgments only after acceptance

**Option C**: Reference in paper with anonymous identifier until publication

### Supplementary Materials

Recommended uploads to OpenReview/submission system:
- `results/tables/grid_results.csv` - Full hyperparameter sweep results
- `EFDO_colab/logs/` - All metrics.jsonl files from experiments (if space permits, zipped)
- `src/` and `scripts/` - Source code for reproducibility
- Generated figures (high-res PNG versions)

### Reviewer Guidance

For reviewers examining reproducibility:
1. Install: `pip install -r requirements.txt`
2. Single experiment: `python train.py --dataset mnist --variant off --k 16 --steps 60000 --seed 42`
3. Expected output: 94.5% ± 0.3% validation accuracy in ~5-10 minutes on modern GPU
4. Generate figures: `python scripts/gen_additional_figures.py --log_dir results/logs`

---

## 5. CHECKLIST FOR SUBMISSION PORTAL

### Paper Information
- [ ] Title: "Physics-Informed Eigenfunction Features with Learnable Scaling (PIEFS)"
- [ ] Authors: [your team]
- [ ] Affiliations: [your institution]
- [ ] Keywords: Physics-informed neural networks, Eigenfunction approximation, Spectral methods, Interpretable learning
- [ ] Subject area: AI4Physics / Machine Learning / Physics-Informed ML
- [ ] TL;DR: Sequential physics-informed eigenfunction learning with learnable metrics for interpretable feature discovery

### Abstract & Introduction
- [ ] Main contribution clearly stated
- [ ] Problem motivation well-explained
- [ ] Related work properly positioned
- [ ] Figure 1 (conceptual overview) included [Note: currently not in paper, could add]

### Methodology
- [ ] Mathematical formulation clear and rigorous
- [ ] Loss functions (MDE, Gram, classification) well-defined
- [ ] Algorithm pseudocode or clear description provided
- [ ] Three variants (off/diag/trotter) explained

### Results
- [ ] Table 1: Main results with mean±std over 5 seeds
- [ ] Figure 2: Training curves (MNIST + CIFAR-10)
- [ ] Figure 3: Eigenfunction visualizations (Two Moons + Circles)
- [ ] Figure 4: Gram matrix analysis
- [ ] Figure 5: Gram convergence curves (new)
- [ ] Figure 6: K-ablation study (new)

### Reproducibility
- [ ] Hyperparameters specified in main text and/or config files
- [ ] Source code provided (GitHub link or supplementary)
- [ ] Data loading code provided or references to public datasets
- [ ] Random seed management documented
- [ ] Computational requirements stated (GPU/CPU, RAM, runtime)

### Discussion
- [ ] Limitations clearly acknowledged
- [ ] Future work outlined
- [ ] Broader impact discussed if applicable
- [ ] Comparisons with related work fair and thorough

