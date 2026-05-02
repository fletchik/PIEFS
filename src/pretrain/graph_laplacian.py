"""Graph Laplacian pretraining (paper Section 2.4).

Pipeline:
  1. Sample N_gl points from the training dataset.
  2. Build a K-NN graph with Gaussian edge weights.
  3. Compute normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
  4. Find the K+1 smallest eigenvectors via scipy eigsh; skip the constant first one.
  5. Train logistic regression on those K features → cross-entropy → T_class.
  6. For k=1..K: distill φ_k to match eigenvector_k via MSE for `distill_steps` steps.

Returns updated T_class so dynamic weighting in train.py can use it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_knn_graph(X: np.ndarray, k_neighbors: int, sigma: float | None) -> np.ndarray:
    """Return symmetric weight matrix W (N, N) for a K-NN Gaussian graph."""
    from sklearn.neighbors import kneighbors_graph

    # Connectivity graph (unweighted)
    A = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
    # Make symmetric: keep edge if either direction is in K-NN
    A = (A + A.T).toarray()
    A = (A > 0).astype(np.float64)

    # Gaussian weights: W_ij = A_ij * exp(-||xi - xj||^2 / sigma^2)
    if sigma is None:
        # Median heuristic
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X, metric='euclidean')
        sigma = float(np.median(dists[dists > 0]))

    from sklearn.metrics import pairwise_distances
    sq_dists = pairwise_distances(X, metric='sqeuclidean')
    W = A * np.exp(-sq_dists / (sigma ** 2))
    return W


def _normalized_laplacian(W: np.ndarray):
    """L_sym = I - D^{-1/2} W D^{-1/2} as a scipy sparse matrix."""
    import scipy.sparse as sp

    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    W_sp = sp.csr_matrix(W)
    L_sym = sp.eye(W.shape[0]) - D_inv_sqrt @ W_sp @ D_inv_sqrt
    return L_sym


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GraphLaplacianPretrain:
    """Graph Laplacian pretraining helper.

    Args:
        K: Number of eigenfunctions to compute (matches model.K).
        n_points: Number of training points to subsample for graph construction.
        k_neighbors: Number of nearest neighbours per node.
        sigma: Gaussian bandwidth. If None, uses the median pairwise distance.
        distill_steps: SGD steps for MSE distillation of each φ_k.
        distill_lr: Learning rate for distillation optimizer.
        device: Torch device string.
    """

    def __init__(
        self,
        K: int,
        n_points: int = 1000,
        k_neighbors: int = 10,
        sigma: float | None = None,
        distill_steps: int = 2000,
        distill_lr: float = 1e-3,
        device: str = 'cpu',
    ) -> None:
        self.K = K
        self.n_points = n_points
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.distill_steps = distill_steps
        self.distill_lr = distill_lr
        self.device = device

        self._eigenvecs: np.ndarray | None = None   # (n_points, K)
        self._X_gl: np.ndarray | None = None         # (n_points, d)
        self.t_class: float = 0.5                    # updated after fit()

    # ------------------------------------------------------------------
    # Step 1-4: compute eigenfunctions
    # ------------------------------------------------------------------

    def compute_eigenfunctions(self, dataset: Dataset) -> None:
        """Subsample dataset, build K-NN graph, compute K non-trivial eigenvectors."""
        from scipy.sparse.linalg import eigsh

        # Subsample
        N = len(dataset)
        idx = np.random.choice(N, size=min(self.n_points, N), replace=False)
        X_list, y_list = [], []
        for i in idx:
            sample = dataset[i]
            x = sample['x'] if isinstance(sample, dict) else sample[0]
            y = sample['y'] if isinstance(sample, dict) else sample[1]
            X_list.append(x.numpy() if hasattr(x, 'numpy') else np.array(x))
            y_list.append(int(y) if hasattr(y, 'item') else int(y))

        X = np.stack(X_list, axis=0)   # (n_points, d)
        y = np.array(y_list)           # (n_points,)

        log.info('GL: building %d-NN graph on %d points (d=%d)', self.k_neighbors, len(X), X.shape[1])
        W = _build_knn_graph(X, k_neighbors=self.k_neighbors, sigma=self.sigma)
        L = _normalized_laplacian(W)

        # K+1 smallest eigenvalues; first is ~0 (constant)
        n_eigs = min(self.K + 1, len(X) - 1)
        log.info('GL: computing %d eigenvectors via eigsh...', n_eigs)
        eigenvalues, eigenvecs = eigsh(L, k=n_eigs, which='SM')

        # Sort by eigenvalue
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvecs = eigenvecs[:, order]   # (n_points, K+1)

        # Drop constant eigenfunction (eigenvalue ≈ 0)
        nontrivial = np.where(eigenvalues > 1e-6)[0]
        if len(nontrivial) == 0:
            log.warning('GL: all eigenvectors are constant — using raw eigenvecs')
            nontrivial = np.arange(1, n_eigs)

        eigenvecs = eigenvecs[:, nontrivial[:self.K]]   # (n_points, K)

        # Normalise columns to unit l2 norm over the point set
        norms = np.linalg.norm(eigenvecs, axis=0, keepdims=True) + 1e-8
        eigenvecs = eigenvecs / norms

        self._X_gl = X
        self._y_gl = y
        self._eigenvecs = eigenvecs

        log.info(
            'GL: eigenvalues (nontrivial) = %s',
            eigenvalues[nontrivial[:self.K]].round(4).tolist(),
        )

    # ------------------------------------------------------------------
    # Step 5: compute T_class via logistic regression
    # ------------------------------------------------------------------

    def compute_t_class(self) -> float:
        """Train logistic regression on eigenvectors → return cross-entropy T_class."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        if self._eigenvecs is None:
            raise RuntimeError('Call compute_eigenfunctions() first.')

        X_feat = self._eigenvecs
        y = self._y_gl

        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        clf.fit(X_feat, y)
        probs = clf.predict_proba(X_feat)
        t_class = float(log_loss(y, probs))
        self.t_class = t_class
        log.info('GL: T_class (LR cross-entropy on GL features) = %.4f', t_class)
        return t_class

    # ------------------------------------------------------------------
    # Step 6: distil basis functions to match eigenvectors
    # ------------------------------------------------------------------

    def pretrain_basis_set(self, basis_set: nn.Module) -> None:
        """MSE-distil each φ_k to match the k-th GL eigenvector.

        Uses basis_set.set_active(k) / freeze_all() to manage freeze state.
        After completion, all functions are frozen so the sequential trainer's
        set_active() calls work normally.
        """
        if self._eigenvecs is None:
            raise RuntimeError('Call compute_eigenfunctions() first.')

        X_torch = torch.tensor(self._X_gl, dtype=torch.float32, device=self.device)
        targets = torch.tensor(self._eigenvecs, dtype=torch.float32, device=self.device)

        loss = torch.tensor(0.0)  # fallback if distill_steps == 0
        for k_idx in range(self.K):
            # Unfreeze only function k_idx+1
            basis_set.set_active(k_idx + 1)
            basis_fn = basis_set.functions[k_idx]
            optimizer = torch.optim.Adam(basis_fn.parameters(), lr=self.distill_lr)
            target_k = targets[:, k_idx]   # (n_points,)

            log.info('GL: distilling φ_%d for %d steps...', k_idx + 1, self.distill_steps)
            for step in range(self.distill_steps):
                optimizer.zero_grad()
                # basis_fn.net(x) → (n_points, 1); squeeze to (n_points,)
                phi_k = basis_fn.net(X_torch).squeeze(-1)
                loss = nn.functional.mse_loss(phi_k, target_k)
                loss.backward()
                optimizer.step()
                if (step + 1) % 500 == 0:
                    log.info('  step %d/%d  mse=%.6f', step + 1, self.distill_steps, loss.item())

            log.info('GL: φ_%d distilled  mse=%.6f', k_idx + 1, loss.item())

        # Leave all functions frozen; sequential trainer's set_active() handles unfreezing
        basis_set.freeze_all()
        log.info('GL: pretraining complete — all functions frozen for sequential training')
