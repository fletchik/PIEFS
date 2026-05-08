# Abstract Variants for ICML AI4Physics

## VARIANT 1: Strong but Honest (RECOMMENDED)

Learning nonlinear representations that respect data geometry is central to unsupervised and semi-supervised learning. Classical spectral methods (Laplacian eigenmaps, diffusion maps) provide geometric insights but require expensive eigendecomposition at test time and are sensitive to kernel choice. We propose PIEFS (Physics-Informed Eigenfunction Features with Learnable Scaling), which learns spectral-like coordinates via neural networks trained with a modified Dirichlet energy loss. The key innovation is a learnable linear metric A(x) = Λ(x)·U(ω(x)) that adapts the spectral structure directly from data—diagonal scaling Λ(x) adjusts per-coordinate importance, while Givens rotations U parametrized by bounded angles ω_i = π·tanh(MLP(x)_i) capture coordinate coupling. Sequential optimization with stop-gradient mechanisms yields K orthonormal maps φ₁,...,φₖ under a composite loss balancing Dirichlet smoothness, approximate L²-orthogonality, and cross-entropy on linear logits.

Experiments on five benchmarks (Two Moons, Circles, HTRU2, MNIST, CIFAR-10) show competitive accuracy: perfect separation on Two Moons geometry, 94.53% on MNIST (K=16), and 85.50% on CIFAR-10 ResNet embeddings. Results do not uniformly dominate classical baselines—random forests remain strong on HTRU2, and Circles exhibits high variance (±14.9%)—but demonstrate that physics-informed Dirichlet penalties produce both geometric interpretability and practical performance. Limitations include task-dependent rather than operator-inherent learning, finite-batch orthogonality approximation, and significant CPU training time. Code and detailed logs are provided for reproducibility.

---

## VARIANT 2: Maximum Impact (Shorter, More Punchy)

Spectral methods offer geometric interpretability but require expensive eigendecomposition and are sensitive to kernel hyperparameters. We propose PIEFS, a learnable alternative grounded in modified Dirichlet energy: scalar maps φ₁,...,φₖ are trained sequentially with a learnable metric A(x) that adapts gradient scaling and rotation to the data. The metric parametrization A(x) = Λ(x)·U(ω(x)) combines diagonal scaling and Givens rotations, enabling expressive transformations while maintaining orthogonality by construction. 

PIEFS achieves 100% accuracy on crescent geometries (Two Moons), 94.53% on MNIST, and 85.50% on CIFAR-10 embeddings—comparable to classical baselines and demonstrating that physics-informed objectives yield interpretable representations. High variance on Circles (±14.9%) indicates room for optimization improvement. The method bridges spectral geometry and neural feature learning, providing cheap inference after training while remaining computationally expensive at train time. Reproducible code and ablations included.

---

## VARIANT 3: Physics-First Framing

Eigenfunction approximation underlies many scientific computing tasks, yet classical spectral operators (Laplacian, Dirichlet) are dataset-agnostic and computationally expensive. We introduce PIEFS (Physics-Informed Eigenfunction Features), which learns data-adaptive eigenfunction-like bases via neural networks trained with a modified Dirichlet energy penalty. A learnable metric A(x) = Λ(x)·U(ω(x)) parametrizes both diagonal scaling and Givens rotations, allowing the spectral structure to be jointly optimized with a classification objective under block-coordinate descent.

On canonical benchmarks (Two Moons: 100%, MNIST: 94.53%, CIFAR-10: 85.50%), PIEFS produces geometrically interpretable coordinates comparable to classical baselines. While results are competitive rather than uniformly superior—random forests remain strong on tabular data, and high-dimensional optimization presents challenges (Circles variance ±14.9%)—the method demonstrates feasibility of amortized spectral learning. We provide reproducible implementations and discuss limitations: task-dependency of learned modes, approximate rather than exact orthogonality, and training time on CPU. This work is relevant to AI4Physics as a bridge between physics-inspired principles and scalable neural feature discovery.

---

## VARIANT 4: Rigorous & Minimal (Most Academic)

We study learning of spectral-like coordinate maps via neural networks optimized with a modified Dirichlet energy penalty. Central to our approach is a learnable metric A(x) that parametrizes both per-coordinate scaling Λ(x) and structured rotations U(ω(x)), enabling data-dependent adaptation of the spectral objective. Sequential training with block-coordinate descent, Gram orthogonality constraints, and supervised cross-entropy yields K approximate eigenfunctions.

Empirical evaluation on five benchmarks shows competitive performance (Two Moons 100%, MNIST 94.53%, CIFAR-10 85.50%) with explicit geometric interpretability. Results do not exceed strong classical baselines uniformly, and training remains expensive. Limitations include task-dependent learning (not operator eigenmodes), finite-batch orthogonality, and high variance on some geometries. Reproducible code and detailed ablations provided.

---

## COMPARISON TABLE

| Aspect | Variant 1 | Variant 2 | Variant 3 | Variant 4 |
|--------|-----------|-----------|-----------|-----------|
| **Length** | Long (~330 words) | Medium (~200 words) | Long (~310 words) | Short (~140 words) |
| **Tone** | Balanced | Punchy | Physics-focused | Minimal/Rigorous |
| **Honest about limits?** | ✓ Very clear | ✓ Mentioned | ✓ Explicit | ✓ Stated |
| **Strong positioning?** | ✓ Good | ✓✓ Best | ✓ Good | ~ Neutral |
| **ICML appeal** | High | High | Medium (physics-heavy) | High |
| **AI4Physics appeal** | High | Medium | ✓✓ Highest | Medium |
| **Claim strength** | "Competitive accuracy" | "Comparable to baselines" | "Feasibility + bridging" | "Empirical evaluation" |

---

## RECOMMENDATION FOR ICML + AI4Physics

**Use VARIANT 1** because:
1. ✓ Clearly states the problem (expensive eigendecomposition, kernel sensitivity)
2. ✓ Highlights the core innovation (learnable metric A(x) with concrete parametrization)
3. ✓ Shows results honestly ("competitive," "not uniformly dominating," high variance acknowledged)
4. ✓ Explicit about limitations (task-dependent, finite-batch, CPU cost)
5. ✓ Balances physics-inspired motivation with practical learning
6. ✓ Length appropriate for ICML (~330 words is good for workshop)
7. ✓ Professional but not overselling

**Why not others:**
- Variant 2: Too punchy, loses credibility by omitting variance discussion
- Variant 3: Too physics-heavy, might not land well with core ML reviewers
- Variant 4: Too minimal, doesn't adequately explain the innovation

