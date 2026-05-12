# OpenReview Submission Format
## AI4Physics Workshop @ ICML 2026

---

## TL;DR

Physics-informed eigenfunction learning with learnable metrics achieves competitive accuracy across datasets while producing geometrically interpretable features grounded in Dirichlet energy minimization.

---

## Abstract

Learning interpretable feature representations that respect data geometry is essential for physics-informed machine learning. We propose PIEFS (Physics-Informed Eigenfunction Features with Learnable Scaling), a method for discovering smooth eigenfunction bases on data manifolds. The approach combines sequential neural network training with Modified Dirichlet Energy loss, enforcing smoothness through differential-geometric principles. A learnable metric parametrized as $$A(x) = \Lambda(x) \cdot U(\omega(x))$$ enables expressive feature transformations while maintaining orthogonality via Gram penalties, where $\Lambda(x)$ provides diagonal scaling and $U(\omega(x))$ represents Givens rotations with bounded angles $\omega_i = \pi \cdot \tanh(\text{MLP}(x)_i)$.

PIEFS learns eigenfunctions $\phi_1, \phi_2, \ldots, \phi_K$ sequentially with cyclic activation, using stop-gradient mechanisms to prevent interference between coordinates. The loss function combines three objectives with dynamic weighting:
$$L_{\text{total}} = \|\nabla \phi_k\|^2_{A(x)} + w_{\text{orth}} \cdot \left\|\frac{\Phi^T \Phi}{N} - I\right\|^2_F + w_{\text{class}} \cdot L_{\text{class}}$$

where $\|\cdot\|^2_{A(x)}$ denotes the Modified Dirichlet Energy, the Gram penalty enforces approximate $L^2$-orthogonality, and weights decay exponentially with time constants $T_{\text{orth}} = 0.1$, $T_{\text{class}} = 0.5$.

On synthetic benchmarks (Two Moons, Circles), PIEFS achieves 100% and 83.6% accuracy respectively, with eigenfunction visualizations revealing nonlinear class geometry aligned with the Laplacian spectrum. On real data, MNIST multiclass reaches $94.53\% \pm 0.33\%$ validation accuracy with $K=16$ features, and CIFAR-10 ResNet-18 embeddings achieve $85.50\% \pm 0.53\%$, demonstrating competitive performance against classical baselines (Random Forest, Logistic Regression) and learned baselines (NeuralEF).

Key contributions include: (1) rigorous integration of Dirichlet energy with neural parametrization; (2) novel learnable metric design combining diagonal and rotation components; (3) dynamic loss weighting scheme with exponential dampening; (4) comprehensive ablations across three variants (off/diag/trotter) and five datasets with interpretable eigenfunction analysis. Theoretical motivation grounded in spectral analysis and differential geometry enables feature discovery that is both geometrically meaningful and practically effective.

**Limitations**: PIEFS learns task-dependent coordinates rather than operator eigenmodes of fixed Laplacian; finite-batch Gram penalties approximate rather than enforce global $L^2$-orthogonality; CPU training times remain significant (∼2 hours per eigenfunction on commodity hardware). **Future work** addresses scalable initialization and extension to larger-scale settings.

---

## Keywords

Physics-informed neural networks; Spectral methods; Eigenfunction approximation; Interpretable feature learning; Differential geometry

