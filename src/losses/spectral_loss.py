"""
Loss Functions for SUE Improved

Contains spectral losses and alignment losses.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpectralNetLoss(nn.Module):
    """
    Rayleigh quotient loss for spectral embedding.
    
    Minimizes: sum_{i,j} W_ij * ||y_i - y_j||^2
    
    This encourages connected points (high W_ij) to be close in embedding space.
    """
    
    def __init__(self, normalized: bool = False):
        """
        Args:
            normalized: Whether to use degree-normalized loss
        """
        super().__init__()
        self.normalized = normalized
        
    def forward(
        self,
        W: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            W: Affinity matrix (n, n)
            Y: Embeddings (n, k)
            
        Returns:
            loss: Scalar loss value
        """
        n = Y.shape[0]
        
        if self.normalized:
            # Degree-normalized version
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D.clamp_min(1e-10))[:, None]
        
        # Compute pairwise squared distances in embedding space
        Dy = torch.cdist(Y, Y).pow(2)
        
        # Weighted sum
        loss = torch.sum(W * Dy) / (2 * n)
        
        return loss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality constraint for spectral embeddings.
    
    Encourages Y^T Y ≈ I (orthonormal embeddings).
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality loss.
        
        Args:
            Y: Embeddings (n, k)
            
        Returns:
            loss: ||Y^T Y - I||_F^2
        """
        n, k = Y.shape
        
        # Normalize columns
        Y_norm = Y / torch.norm(Y, dim=0, keepdim=True).clamp_min(1e-10)
        
        # Compute Y^T Y
        gram = Y_norm.T @ Y_norm / n
        
        # Target is identity
        target = torch.eye(k, device=Y.device)
        
        # Frobenius norm squared
        loss = torch.norm(gram - target, p='fro').pow(2)
        
        return loss


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss with multi-scale RBF kernel.
    
    MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    """
    
    def __init__(
        self,
        n_scales: int = 5,
        mul_factor: float = 2.0,
        device: torch.device = None
    ):
        """
        Args:
            n_scales: Number of kernel scales
            mul_factor: Multiplicative factor between scales
            device: Torch device
        """
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create scale multipliers: [1/4, 1/2, 1, 2, 4] for n_scales=5
        exponent = torch.arange(n_scales) - n_scales // 2
        self.scale_multipliers = (mul_factor ** exponent).to(self.device)
        
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MMD loss between two distributions.
        
        Args:
            X: Samples from distribution P (n_x, d)
            Y: Samples from distribution Q (n_y, d)
            
        Returns:
            mmd: MMD^2 value
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute all pairwise distances
        XY = torch.cat([X, Y], dim=0)
        distances = torch.cdist(XY, XY).pow(2)
        
        # Compute adaptive scale based on median distance
        n = distances.shape[0]
        scale = distances.sum() / (n * n - n)
        
        # Compute multi-scale kernel
        kernel = torch.zeros_like(distances)
        for mult in self.scale_multipliers:
            kernel = kernel + torch.exp(-distances / (scale * mult))
        
        # Split kernel matrix
        n_x = X.shape[0]
        K_xx = kernel[:n_x, :n_x].mean()
        K_yy = kernel[n_x:, n_x:].mean()
        K_xy = kernel[:n_x, n_x:].mean()
        
        # MMD^2
        mmd = K_xx - 2 * K_xy + K_yy
        
        return mmd


class SinkhornDivergence(nn.Module):
    """
    Sinkhorn divergence - interpolates between MMD and Wasserstein distance.
    
    S_ε(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)
    
    This is a debiased version of entropic OT that:
    - Converges to Wasserstein as ε → 0
    - Converges to MMD as ε → ∞
    - Is always positive and metrizes weak convergence
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: torch.device = None
    ):
        """
        Args:
            epsilon: Entropic regularization coefficient
            max_iter: Maximum Sinkhorn iterations
            tol: Convergence tolerance
            device: Torch device
        """
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sinkhorn divergence.
        
        Args:
            X: Samples from P (n_x, d)
            Y: Samples from Q (n_y, d)
            
        Returns:
            divergence: Sinkhorn divergence value
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute costs
        C_xy = torch.cdist(X, Y).pow(2)
        C_xx = torch.cdist(X, X).pow(2)
        C_yy = torch.cdist(Y, Y).pow(2)
        
        # Compute OT costs
        ot_xy = self._sinkhorn_cost(C_xy)
        ot_xx = self._sinkhorn_cost(C_xx)
        ot_yy = self._sinkhorn_cost(C_yy)
        
        # Sinkhorn divergence (debiased)
        divergence = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
        
        return divergence
    
    def _sinkhorn_cost(self, C: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic OT cost using Sinkhorn algorithm.
        
        Args:
            C: Cost matrix (n, m)
            
        Returns:
            cost: Entropic OT cost
        """
        n, m = C.shape
        
        # Uniform marginals
        mu = torch.ones(n, device=self.device) / n
        nu = torch.ones(m, device=self.device) / m
        
        # Gibbs kernel
        K = torch.exp(-C / self.epsilon)
        
        # Initialize
        u = torch.ones(n, device=self.device)
        v = torch.ones(m, device=self.device)
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u_prev = u
            u = mu / (K @ v).clamp_min(1e-10)
            v = nu / (K.T @ u).clamp_min(1e-10)
            
            # Check convergence
            if torch.norm(u - u_prev) < self.tol:
                break
        
        # Transport plan
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # OT cost
        cost = torch.sum(P * C)
        
        return cost


class CombinedSpectralLoss(nn.Module):
    """
    Combined loss for spectral embedding training.
    
    L = L_spectral + alpha * L_orthogonality
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        normalized: bool = False
    ):
        """
        Args:
            alpha: Weight for orthogonality loss
            normalized: Whether to use normalized spectral loss
        """
        super().__init__()
        self.alpha = alpha
        self.spectral_loss = SpectralNetLoss(normalized=normalized)
        self.orthogonality_loss = OrthogonalityLoss()
        
    def forward(
        self,
        W: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            W: Affinity matrix
            Y: Embeddings
            
        Returns:
            loss: Combined loss value
        """
        l_spectral = self.spectral_loss(W, Y)
        l_orth = self.orthogonality_loss(Y)
        
        return l_spectral + self.alpha * l_orth