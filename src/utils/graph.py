"""
Graph Construction Module - Core Implementation of Innovation 1 & 2

Innovation 1: Density-adaptive graph construction using Quadratic OT
Innovation 2: Doubly stochastic Laplacian normalization (Sinkhorn)

Theoretical Foundation:
- Standard k-NN uses fixed neighborhood size, failing to adapt to local density
- Quadratic regularized OT produces sparse solutions that adapt to local geometry
- Doubly stochastic normalization removes degree bias and ensures spectral convergence
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Literal


class GraphConstructor:
    """
    Graph constructor with multiple methods for building affinity matrices.
    
    Supports:
    - Traditional k-NN with Gaussian kernel (baseline)
    - Quadratic regularized OT (Innovation 1)
    - Multiple Laplacian normalizations including doubly stochastic (Innovation 2)
    """
    
    def __init__(
        self,
        method: Literal["knn", "quadratic_ot"] = "quadratic_ot",
        n_neighbors: int = 30,
        scale_k: int = 15,
        is_local_scale: bool = False,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        symmetric: bool = True,
        laplacian_norm: Literal["unnormalized", "symmetric", "random_walk", "doubly_stochastic"] = "doubly_stochastic",
        sinkhorn_iter: int = 50,
        sinkhorn_tol: float = 1e-6,
        device: torch.device = None
    ):
        """
        Initialize graph constructor.
        
        Args:
            method: Graph construction method ('knn' or 'quadratic_ot')
            n_neighbors: Number of neighbors for k-NN
            scale_k: k for computing kernel scale
            is_local_scale: Whether to use local scale
            epsilon: Regularization coefficient for quadratic OT
            max_iter: Max iterations for OT optimization
            tol: Convergence tolerance
            symmetric: Whether to enforce symmetric graph
            laplacian_norm: Laplacian normalization method
            sinkhorn_iter: Max Sinkhorn iterations for doubly stochastic
            sinkhorn_tol: Sinkhorn convergence tolerance
            device: Torch device
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.scale_k = scale_k
        self.is_local_scale = is_local_scale
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.symmetric = symmetric
        self.laplacian_norm = laplacian_norm
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_tol = sinkhorn_tol
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Build affinity matrix from data.
        
        Args:
            X: Data tensor of shape (n_samples, n_features)
            
        Returns:
            W: Affinity matrix of shape (n_samples, n_samples)
        """
        X = X.to(self.device)
        
        if self.method == "knn":
            W = self._build_knn_affinity(X)
        elif self.method == "quadratic_ot":
            W = self._build_quadratic_ot_affinity(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return W
    
    def build_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """
        Build Laplacian matrix from affinity matrix.
        
        Args:
            W: Affinity matrix of shape (n_samples, n_samples)
            
        Returns:
            L: Laplacian matrix
        """
        if self.laplacian_norm == "unnormalized":
            L = self._unnormalized_laplacian(W)
        elif self.laplacian_norm == "symmetric":
            L = self._symmetric_laplacian(W)
        elif self.laplacian_norm == "random_walk":
            L = self._random_walk_laplacian(W)
        elif self.laplacian_norm == "doubly_stochastic":
            L = self._doubly_stochastic_laplacian(W)
        else:
            raise ValueError(f"Unknown normalization: {self.laplacian_norm}")
            
        return L
    
    def build_graph(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build both affinity and Laplacian matrices.
        
        Args:
            X: Data tensor
            
        Returns:
            W: Affinity matrix
            L: Laplacian matrix
        """
        W = self.build_affinity_matrix(X)
        L = self.build_laplacian(W)
        return W, L
    
    # ==================== k-NN Methods (Baseline) ====================
    
    def _build_knn_affinity(self, X: torch.Tensor) -> torch.Tensor:
        """Build k-NN affinity matrix with Gaussian kernel."""
        n = X.shape[0]
        
        # Compute pairwise distances
        D = torch.cdist(X, X)
        
        # Get k nearest neighbors
        Dis, Ids = self._get_nearest_neighbors(X, k=self.n_neighbors + 1)
        
        # Compute scale
        scale = self._compute_scale(Dis)
        
        # Build Gaussian kernel
        W = self._gaussian_kernel(D, scale, Ids)
        
        return W
    
    def _get_nearest_neighbors(self, X: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors using sklearn."""
        X_np = X.cpu().detach().numpy()
        k = min(k, len(X_np))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_np)
        Dis, Ids = nbrs.kneighbors(X_np)
        return Dis, Ids
    
    def _compute_scale(self, Dis: np.ndarray) -> np.ndarray:
        """Compute kernel scale from distances."""
        if self.is_local_scale:
            scale = np.median(Dis, axis=1)
        else:
            scale = np.median(Dis[:, min(self.scale_k, Dis.shape[1] - 1)])
        return scale
    
    def _gaussian_kernel(
        self, 
        D: torch.Tensor, 
        scale: np.ndarray, 
        Ids: np.ndarray
    ) -> torch.Tensor:
        """Build Gaussian kernel with k-NN sparsification."""
        n = D.shape[0]
        
        if self.is_local_scale:
            scale_tensor = torch.tensor(scale, device=self.device).float()
            W = torch.exp(-D.pow(2) / (scale_tensor[:, None].clamp_min(1e-7) ** 2))
        else:
            W = torch.exp(-D.pow(2) / (scale ** 2))
        
        # Apply k-NN mask
        mask = torch.zeros(n, n, device=self.device)
        mask[np.arange(n).reshape(-1, 1), Ids] = 1
        W = W * mask
        
        # Symmetrize
        W = (W + W.T) / 2.0
        
        # Remove self-loops
        W.fill_diagonal_(0)
        
        return W
    
    # ==================== Quadratic OT Methods (Innovation 1) ====================
    
    def _build_quadratic_ot_affinity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Build affinity matrix using quadratic regularized optimal transport.
        
        Innovation 1: Quadratic regularization produces sparse solutions that
        automatically adapt to local data density.
        
        The optimization problem:
            min_W <W, C> + (epsilon/2) ||W||_F^2
            s.t. W @ 1 = 1/n, W.T @ 1 = 1/n, W >= 0
            
        For graph construction, we use uniform marginals (1/n).
        """
        n = X.shape[0]
        
        # Compute cost matrix (squared Euclidean distance)
        C = torch.cdist(X, X).pow(2)
        
        # Uniform marginals
        mu = torch.ones(n, device=self.device) / n
        nu = torch.ones(n, device=self.device) / n
        
        # Solve quadratic OT
        W = self._solve_quadratic_ot(C, mu, nu)
        
        # Scale to make it an affinity matrix (larger values = more similar)
        # Convert transport plan to affinity: W_affinity = max(W) - W or exp(-W/scale)
        W = W * n  # Scale back from probability to weights
        
        if self.symmetric:
            W = (W + W.T) / 2.0
            
        # Remove self-loops
        W.fill_diagonal_(0)
        
        return W
    
    def _solve_quadratic_ot(
        self, 
        C: torch.Tensor, 
        mu: torch.Tensor, 
        nu: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve quadratic regularized optimal transport using projected gradient descent.
        
        Problem: min_W <W, C> + (epsilon/2) ||W||_F^2
                 s.t. W @ 1 = mu, W.T @ 1 = nu, W >= 0
                 
        We use the Dykstra-like algorithm for projection onto the intersection
        of simplex constraints.
        """
        n = C.shape[0]
        
        # Initialize with uniform coupling
        W = mu[:, None] * nu[None, :]
        
        # Step size for gradient descent (needs to be small enough for convergence)
        # For quadratic regularization, optimal step size is related to 1/epsilon
        step_size = 1.0 / (self.epsilon + C.max())
        
        # Inner iteration settings for marginal constraint projection
        inner_max_iter = 50
        inner_tol = 1e-8
        
        for iteration in range(self.max_iter):
            W_old = W.clone()
            
            # Gradient step: grad = C + epsilon * W
            grad = C + self.epsilon * W
            W = W - step_size * grad
            
            # Project onto non-negative orthant
            W = torch.clamp(W, min=0)
            
            # Project onto marginal constraints using Sinkhorn-like iterations
            # This is Bregman projection for the simplex constraints
            for inner_iter in range(inner_max_iter):
                W_inner_old = W.clone()
                
                # Row normalization (W @ 1 = mu)
                row_sum = W.sum(dim=1, keepdim=True).clamp_min(1e-10)
                W = W * (mu[:, None] / row_sum)
                
                # Column normalization (W.T @ 1 = nu)  
                col_sum = W.sum(dim=0, keepdim=True).clamp_min(1e-10)
                W = W * (nu[None, :] / col_sum)
                
                # Check inner convergence
                inner_diff = torch.norm(W - W_inner_old) / (torch.norm(W_inner_old) + 1e-10)
                if inner_diff < inner_tol:
                    break
            
            # Check outer convergence
            diff = torch.norm(W - W_old) / (torch.norm(W_old) + 1e-10)
            if diff < self.tol:
                break
                
        return W
    
    def _solve_quadratic_ot_closed_form(
        self,
        C: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative: Semi-closed form solution for quadratic OT with uniform marginals.
        
        For uniform marginals, the solution has a simpler structure.
        W* = max(0, (alpha @ 1.T + 1 @ beta.T - C) / epsilon)
        where alpha, beta are dual variables found by solving the dual problem.
        """
        n = C.shape[0]
        
        # For uniform marginals, use iterative scheme
        alpha = torch.zeros(n, device=self.device)
        beta = torch.zeros(n, device=self.device)
        
        for iteration in range(self.max_iter):
            alpha_old = alpha.clone()
            
            # Compute W from dual variables
            W = torch.clamp(
                (alpha[:, None] + beta[None, :] - C) / self.epsilon,
                min=0
            )
            
            # Update dual variables to satisfy marginal constraints
            # alpha update: sum_j W_ij = mu_i
            row_sum = W.sum(dim=1)
            alpha = alpha + self.epsilon * (mu - row_sum)
            
            # beta update: sum_i W_ij = nu_j
            col_sum = W.sum(dim=0)
            beta = beta + self.epsilon * (nu - col_sum)
            
            # Check convergence
            if torch.norm(alpha - alpha_old) < self.tol:
                break
        
        # Final W
        W = torch.clamp(
            (alpha[:, None] + beta[None, :] - C) / self.epsilon,
            min=0
        )
        
        return W
    
    # ==================== Laplacian Methods (Innovation 2) ====================
    
    def _unnormalized_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """Unnormalized Laplacian: L = D - W"""
        D = torch.diag(W.sum(dim=1))
        L = D - W
        return L
    
    def _symmetric_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}"""
        d = W.sum(dim=1).clamp_min(1e-10)
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
        L = torch.eye(W.shape[0], device=self.device) - d_inv_sqrt @ W @ d_inv_sqrt
        return L
    
    def _random_walk_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """Random walk Laplacian: L_rw = I - D^{-1} W"""
        d = W.sum(dim=1).clamp_min(1e-10)
        d_inv = torch.diag(1.0 / d)
        L = torch.eye(W.shape[0], device=self.device) - d_inv @ W
        return L
    
    def _doubly_stochastic_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """
        Doubly stochastic Laplacian using Sinkhorn normalization.
        
        Innovation 2: Sinkhorn normalization makes W doubly stochastic,
        eliminating degree bias and ensuring better spectral convergence
        to the Laplace-Beltrami operator.
        
        After Sinkhorn: W @ 1 = 1 and W.T @ 1 = 1
        Then: L = I - W
        """
        W_ds = self._sinkhorn_normalization(W)
        L = torch.eye(W.shape[0], device=self.device) - W_ds
        return L
    
    def _sinkhorn_normalization(self, W: torch.Tensor) -> torch.Tensor:
        """
        Apply Sinkhorn normalization to make W doubly stochastic.
        
        Iteratively normalizes rows and columns until convergence.
        This is equivalent to finding the optimal scaling D1 @ W @ D2
        where D1, D2 are diagonal matrices such that the result is
        doubly stochastic.
        """
        W = W.clone()
        n = W.shape[0]
        
        # Ensure non-negative
        W = torch.clamp(W, min=0)
        
        # Add small constant to avoid division by zero
        W = W + 1e-10
        
        for iteration in range(self.sinkhorn_iter):
            W_old = W.clone()
            
            # Row normalization
            row_sum = W.sum(dim=1, keepdim=True).clamp_min(1e-10)
            W = W / row_sum
            
            # Column normalization
            col_sum = W.sum(dim=0, keepdim=True).clamp_min(1e-10)
            W = W / col_sum
            
            # Check convergence
            diff = torch.abs(W - W_old).max()
            if diff < self.sinkhorn_tol:
                break
        
        # Final symmetrization (important for undirected graphs)
        W = (W + W.T) / 2.0
        
        return W


class AdaptiveGraphConstructor(GraphConstructor):
    """
    Extended graph constructor with additional adaptive features.
    
    Adds:
    - Automatic epsilon selection based on data statistics
    - Multi-scale graph construction
    - Spectral analysis utilities
    """
    
    def __init__(self, auto_epsilon: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.auto_epsilon = auto_epsilon
        
    def build_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Build affinity with optional automatic parameter selection."""
        if self.auto_epsilon and self.method == "quadratic_ot":
            self.epsilon = self._estimate_epsilon(X)
            
        return super().build_affinity_matrix(X)
    
    def _estimate_epsilon(self, X: torch.Tensor) -> float:
        """
        Estimate epsilon based on data statistics.
        
        Heuristic: epsilon should be proportional to the median pairwise distance
        to ensure the transport plan captures meaningful structure.
        """
        # Sample distances for efficiency
        n = X.shape[0]
        if n > 1000:
            idx = torch.randperm(n)[:1000]
            X_sample = X[idx]
        else:
            X_sample = X
            
        D = torch.cdist(X_sample, X_sample)
        
        # Use median distance as reference scale
        median_dist = torch.median(D[D > 0]).item()
        
        # Epsilon ~ 0.1 * median_dist^2 (for squared cost)
        epsilon = 0.1 * (median_dist ** 2)
        
        return epsilon
    
    def compute_spectral_gap(self, L: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, int]:
        """
        Compute eigenvalues and detect spectral gap.
        
        Returns:
            eigenvalues: First k eigenvalues
            gap_index: Index where largest relative gap occurs
        """
        # Compute eigenvalues
        L_np = L.cpu().numpy()
        eigenvalues, _ = np.linalg.eigh(L_np)
        eigenvalues = torch.tensor(eigenvalues[:k], device=self.device)
        
        # Find largest relative gap
        gaps = eigenvalues[1:] - eigenvalues[:-1]
        relative_gaps = gaps / (eigenvalues[:-1] + 1e-10)
        gap_index = torch.argmax(relative_gaps).item() + 1
        
        return eigenvalues, gap_index


def create_graph_constructor(config: dict, device: torch.device) -> GraphConstructor:
    """
    Factory function to create graph constructor from config.
    
    Args:
        config: Configuration dictionary with 'graph' section
        device: Torch device
        
    Returns:
        GraphConstructor instance
    """
    graph_config = config.get("graph", {})
    method = graph_config.get("method", "quadratic_ot")
    
    # Get method-specific parameters
    if method == "knn":
        knn_config = graph_config.get("knn", {})
        params = {
            "n_neighbors": knn_config.get("n_neighbors", 30),
            "scale_k": knn_config.get("scale_k", 15),
            "is_local_scale": knn_config.get("is_local_scale", False),
        }
    else:  # quadratic_ot
        ot_config = graph_config.get("quadratic_ot", {})
        params = {
            "epsilon": ot_config.get("epsilon", 0.1),
            "max_iter": ot_config.get("max_iter", 100),
            "tol": ot_config.get("tol", 1e-6),
            "symmetric": ot_config.get("symmetric", True),
        }
    
    # Get Laplacian parameters
    lap_config = graph_config.get("laplacian", {})
    params.update({
        "laplacian_norm": lap_config.get("normalization", "doubly_stochastic"),
        "sinkhorn_iter": lap_config.get("sinkhorn_iter", 50),
        "sinkhorn_tol": lap_config.get("sinkhorn_tol", 1e-6),
    })
    
    return AdaptiveGraphConstructor(
        method=method,
        device=device,
        **params
    )