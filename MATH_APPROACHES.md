# Eight Mathematical Approaches to A(x) in PIEFS

## Framing: What Are We Solving?

The Modified Dirichlet Energy (MDE) in PIEFS is:

$$\mathcal{D}_A[\varphi] = \int \|A(x)\nabla\varphi(x)\|^2 \, p(x) \, dx$$

This is equivalent to the eigenvalue problem for the weighted Laplace-Beltrami operator:

$$\mathcal{L}_A \varphi = -\frac{1}{\rho(x)} \nabla \cdot (A(x)^T A(x) \nabla\varphi)$$

where $M(x) = A(x)^T A(x)$ is the **learnable Riemannian metric tensor**. The eigenfunctions of this operator are the "natural" basis for the data manifold endowed with metric $M$.

**Key insight** (not used in ICML paper): Our method learns eigenfunctions of a *learned* Laplace-Beltrami operator. This connects to classical spectral geometry, diffusion maps, and information geometry.

---

## Approach 1: Off / Identity (Baseline)

$$A(x) = I$$

**What it is**: No metric learning. Plain squared gradient norm: $\mathcal{D}[\varphi] = \mathbb{E}[\|\nabla\varphi\|^2]$.

**Why it works**: Learns Euclidean eigenfunctions — smoothest functions in $L^2(p)$. Equivalent to NeuralEF.

**Pros**: Zero parameters, stable training, fast.  
**Cons**: Isotropic — treats all directions equally, misses class-relevant structure.

**Status**: ✅ Implemented (`metric_type: off`)

---

## Approach 2: Diagonal A(x) = diag(λ₁(x), ..., λ_d(x))

$$A(x) = \text{diag}(\lambda(x)), \quad \lambda(x) = \text{softplus}(\text{MLP}(x)) / \left(\prod_i \lambda_i\right)^{1/d}$$

**What it is**: Axis-aligned anisotropic scaling. The constraint $\det(A) = 1$ (volume-preserving) keeps the eigenvalue problem well-scaled.

**Why it works**: Can amplify directions where class boundaries are sharp, suppress noisy directions. Equivalent to learning a diagonal Mahalanobis distance.

**Pros**: $O(d)$ parameters per point, simple MLP.  
**Cons**: No rotation — cannot align with off-axis class boundaries.

**Status**: ✅ Implemented (`metric_type: diag`)

---

## Approach 3: Trotter-product A(x) = Λ(x) · U_Trotter(ω(x))

$$A(x) = \Lambda(x) \cdot \prod_{i=1}^{d-1} G_i(\omega_i(x))$$

where $G_i$ is a Givens rotation in the $(i, i+1)$ plane.

**What it is**: Diagonal scaling followed by a product of adjacent 2D rotations. $U$ is exactly orthogonal by construction.

**Critical limitation**: The $(d-1)$ adjacent Givens rotations span only a $(d-1)$-dimensional subgroup of $SO(d)$ (which has $d(d-1)/2$ dimensions). For $d=784$: coverage = $783/306936 \approx 0.25\%$ of all rotations. **Cannot represent most class-separating directions.**

**Pros**: Exact orthogonality, $O(Bd)$ cost.  
**Cons**: Severe subgroup restriction, MLP bottleneck at high $d$.

**Status**: ✅ Implemented (`metric_type: lambda_u_trotter`)

---

## Approach 4: Global Low-Rank A = I + U·D·Vᵀ  ⭐ TOP RECOMMENDATION

$$A = I + UDV^T, \quad U,V \in \mathbb{R}^{d \times r}, \quad D = \text{diag}(\exp(\mathbf{d}))$$

**What it is**: A rank-$r$ perturbation of identity. **Global** (not $x$-dependent). Direct parameters — no MLP.

**Why $r = C-1$ is optimal** (LDA connection): The optimal linear map $A^*$ for classification satisfies:
$$A^{*T}A^* = S_W^{-1/2} S_B S_W^{-1/2}$$
This is the **Fisher Linear Discriminant** matrix, which has rank exactly $C-1$. So $r = C-1$ is theoretically sufficient, and higher $r$ only adds redundancy.

**Identity recovery guarantee**: At initialization $U=V\approx 0$, $D\approx I$, so $A\approx I$ (PIEFS-off). The optimizer can only depart from $A=I$ if it reduces the loss — meaning GlobalLowRank is **never worse than PIEFS-off** at convergence.

**Pros**: No MLP bottleneck, covers full rank-$r$ space, $O(Bdr)$ cost, gradient flows directly.  
**Cons**: Not $x$-dependent — one metric for all of $\mathbb{R}^d$.

**Status**: ✅ Implemented (`metric_type: global_low_rank`)

---

## Approach 5: Conformal Metric A(x) = σ(x) · I  ⭐ SIMPLE BASELINE

$$A(x) = \sigma(x) \cdot I, \quad \sigma(x) = \text{softplus}(\text{MLP}(x)) > 0$$

**What it is**: Isotropic, $x$-dependent scaling. The Riemannian metric is $M(x) = \sigma^2(x) I$ — a conformal deformation of Euclidean space.

**What it learns**: Denser regions (higher $\sigma$) become "more important" for the Dirichlet energy. Equivalent to importance-weighting the gradient norm by spatial position.

**Why useful**: The simplest non-trivial $x$-dependent metric. If this outperforms `diag`, it means spatial position matters more than directional structure. A clean ablation.

**Mathematical note**: Conformal metrics preserve angles but change lengths — they correspond to Laplace-Beltrami with weight function $\rho(x) \to \rho(x)/\sigma^2(x)$.

**Pros**: Single output MLP, no det constraint needed, easy to interpret.  
**Cons**: No anisotropy — cannot separate correlated dimensions.

**Status**: 🟡 Planned (implement as `metric_type: conformal`)

---

## Approach 6: x-Dependent Low-Rank A(x) = I + U(x)Λ(x)V(x)ᵀ  ⭐ TOP RECOMMENDATION

$$A(x) = I + U(x)\Lambda(x)V(x)^T$$

where $U(x), V(x) \in \mathbb{R}^{d\times r}$ are MLP outputs, $\Lambda(x) = \text{diag}(\text{softplus}(\text{MLP}(x)))$.

**What it is**: A locally-adaptive rank-$r$ perturbation. Each point $x$ gets its own low-rank metric correction.

**Advantage over Trotter**: Covers the full rank-$r$ subspace (not just adjacent Givens rotations). For $r=C-1$ and $d=784$: coverage grows from $0.25\%$ to the full $C-1$ most discriminative directions.

**Advantage over GlobalLowRank**: Adapts to local structure — near class boundaries vs. class centers vs. sparse regions can have different metrics.

**Implementation note**: $U(x)$ and $V(x)$ are separate MLP heads sharing a backbone. Cost: $O(Bdr)$ for apply, $O(B \cdot d \cdot r \cdot h)$ for MLP.

**Pros**: Full rank-$r$ coverage, locally adaptive.  
**Cons**: MLP capacity must be split between basis and metric; gradient competition.

**Status**: 🟡 Planned (implement as `metric_type: local_low_rank`)

---

## Approach 7: Fisher Information Metric (FIM)

$$M(x) = F_\theta(x) = \mathbb{E}_{y \sim p(y|x,\theta)}\left[\nabla_x \log p(y|x,\theta) \, \nabla_x \log p(y|x,\theta)^T\right]$$

**What it is**: The natural Riemannian metric from information geometry. $F(x)$ measures how fast the predictive distribution changes as $x$ moves.

**Why it's theoretically optimal** (Connection Lemma): If $M = F_\theta$, the top-$K$ eigenfunctions achieve minimum Bayes error rate among all $K$-dimensional linear classifiers in $\varphi$-feature space (asymptotically). Proof: Information geometry (Amari 1985).

**Practical approximation**: Full FIM is $O(d^2)$ per point — impractical for $d=784$. Use:
- **Diagonal FIM**: $\hat{F}(x) = \text{diag}(\mathbb{E}[(\partial_j \log p)^2])$ — $O(d)$, implementable
- **K-FAC block diagonal**: approximates FIM in layer-block structure
- **Monte Carlo estimate**: sample $y \sim p(y|x,\theta)$, compute $\nabla_x \log p(y|x,\theta)$ once

**Implementation**: During each step, compute the Jacobian of `log_softmax(head(φ))` w.r.t. the input representation, use as $A(x)$.

**Pros**: Theoretically optimal for classification, principled from information geometry.  
**Cons**: Expensive to compute exactly; diagonal approx. loses off-diagonal structure.

**Status**: 🟡 Planned (implement as `metric_type: fisher_diag`)

---

## Approach 8: Neural Tangent Kernel (NTK) Metric

$$M(x) = J_\theta(x)^T J_\theta(x), \quad J_\theta(x) = \frac{\partial f_\theta(x)}{\partial x}$$

where $f_\theta$ is the basis network output.

**What it is**: The metric induced by the Jacobian of the basis network itself. In the infinite-width limit, this converges to the NTK, which determines gradient descent dynamics.

**Why interesting**: The NTK metric aligns the Dirichlet energy with the network's own learning dynamics — eigenfunctions computed under this metric are "natural" for the network.

**Practical issue**: Jacobian computation per sample is $O(K \cdot d)$ — feasible for small $K$ and $d$, expensive for MNIST ($K=16, d=784$: $16 \times 784 = 12544$ per sample).

**Relationship to FIM**: NTK and FIM coincide at infinite width when the network output approximates the log-likelihood (Jacot et al. 2018).

**Status**: ❌ Too expensive for current setup (deferred to future work)

---

## Summary Table

| Approach | $x$-dependent | Rank | Parameters | Cost | Status |
|----------|--------------|------|------------|------|--------|
| 1. Off (Identity) | ❌ | — | 0 | $O(Bd)$ | ✅ Implemented |
| 2. Diagonal | ✅ | $d$ (diag only) | $O(dh)$ MLP | $O(Bd)$ | ✅ Implemented |
| 3. Trotter | ✅ | $(d-1)$ subgroup | $O(dh)$ MLP | $O(Bd)$ | ✅ Implemented |
| 4. GlobalLowRank | ❌ | $r$ | $2dr + r$ | $O(Bdr)$ | ✅ Implemented |
| 5. Conformal | ✅ | 1 (scalar) | $O(h)$ MLP | $O(Bh)$ | 🟡 Planned |
| 6. LocalLowRank | ✅ | $r$ | $2O(drh)$ MLP | $O(Bdrh)$ | 🟡 Planned |
| 7. Fisher (diag) | ✅ | $d$ (diag) | reuses head | $O(BdK)$ | 🟡 Planned |
| 8. NTK | ✅ | $K$ | reuses basis | $O(BKd)$ | ❌ Deferred |

**Recommended experiment order**:
1. Off vs. GlobalLowRank (main comparison, tests g_k² fix)
2. Conformal (simple x-dep baseline)
3. LocalLowRank (full x-dep low-rank)
4. Fisher diagonal (theoretically motivated)

---

## Key Theoretical Connections

### Connection 1: Riemannian Geometry
MDE with metric $M = A^T A$ corresponds to the Dirichlet form of the operator:
$$\mathcal{L}_M \varphi = -\frac{1}{\rho}\nabla \cdot (M \nabla\varphi)$$
Eigenfunctions are the **harmonic basis** of the data manifold with metric $M$.

### Connection 2: LDA / Fisher Metric (GlobalLowRank)
The optimal $M^*$ for classification via spectral features satisfies:
$$M^* = S_W^{-1} S_B$$
where $S_W$ = within-class scatter, $S_B$ = between-class scatter.  
This has rank $C-1$, motivating $r = C-1$ for GlobalLowRank.

### Connection 3: Information Geometry (FIM)
If $M = F_\theta$ (Fisher Information), the $\varphi$-features are **sufficient statistics** in the information-geometric sense, achieving minimum Cramér-Rao bound for class estimation.

### Connection 4: Diffusion Maps
When $A(x) = I$ and $p(x)$ is approximated by kernel density, eigenfunctions correspond to **diffusion map coordinates** — the natural nonlinear embedding of the data manifold.
