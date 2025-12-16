"""
Alignment Module for SUE Improved

Contains alignment methods:
- CCA (baseline)
- Future: Gromov-Wasserstein alignment (Innovation 3)
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.cross_decomposition import CCA



class CCAAligner:
    """
    Canonical Correlation Analysis for cross-modal alignment.
    
    This is the baseline method used in Stage 2 of SUE.
    """
    
    def __init__(self, n_components: int = 8):
        """
        Args:
            n_components: Number of CCA components
        """
        self.n_components = n_components
        self.cca = CCA(n_components=n_components)
        self.projection1 = None
        self.projection2 = None
        self.is_fitted = False
        
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):
        """
        Fit CCA on paired samples.
        
        Args:
            X: Modality 1 embeddings (n, d1)
            Y: Modality 2 embeddings (n, d2)
        """
        self.cca.fit(X, Y)
        self.projection1 = self.cca.x_rotations_
        self.projection2 = self.cca.y_rotations_
        self.is_fitted = True
        
        return self
    
    def transform(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform embeddings using fitted CCA projections.
        
        Args:
            X: Modality 1 embeddings
            Y: Modality 2 embeddings
            
        Returns:
            (projected_X, projected_Y)
        """
        if not self.is_fitted:
            raise RuntimeError("CCA not fitted. Call fit() first.")
            
        return X @ self.projection1, Y @ self.projection2
    
    def fit_transform(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(X, Y)
        return self.transform(X, Y)
    
    def get_canonical_correlations(self) -> np.ndarray:
        """Get canonical correlation coefficients."""
        if not self.is_fitted:
            raise RuntimeError("CCA not fitted.")
        return np.diag(self.cca.x_loadings_.T @ self.cca.y_loadings_)


class MMDAligner:
    """
    MMD-based alignment for distribution matching.
    
    This wraps the MMDNet for Stage 3 alignment.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        n_blocks: int = 1,
        device: torch.device = None
    ):
        from ..models.spectral_embedding import MMDNet
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MMDNet(input_dim, hidden_dim, n_blocks).to(self.device)
        
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        n_scales: int = 3
    ):
        """
        Train MMD alignment.
        
        Args:
            X: Source embeddings to transform
            Y: Target embeddings (reference distribution)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            n_scales: Number of kernel scales
        """
        from ..losses.spectral_loss import MMDLoss
        import torch.optim as optim
        from tqdm import tqdm
        
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float()
        
        criterion = MMDLoss(n_scales=n_scales, device=self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in tqdm(range(epochs), desc="MMD Alignment"):
            # Sample batches
            idx_x = np.random.permutation(len(X))[:batch_size]
            idx_y = np.random.permutation(len(Y))[:batch_size]
            
            batch_x = X_tensor[idx_x].to(self.device)
            batch_y = Y_tensor[idx_y].to(self.device)
            
            optimizer.zero_grad()
            x_out = self.model(batch_x)
            loss = criterion(x_out, batch_y)
            loss.backward()
            optimizer.step()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform embeddings."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            return self.model(X_tensor).cpu().numpy()


# Placeholder for future Innovation 3
class GromovWassersteinAligner:
    """
    Gromov-Wasserstein alignment for structure-preserving matching.
    
    This will be implemented as Innovation 3.
    
    GW directly matches distance structures without requiring paired samples,
    making it ideal for unsupervised cross-modal alignment.
    """
    
    def __init__(
        self,
        n_components: int = 8,
        epsilon: float = 0.1,
        max_iter: int = 100
    ):
        """
        Args:
            n_components: Output dimension
            epsilon: Entropic regularization
            max_iter: Maximum iterations
        """
        self.n_components = n_components
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        # To be implemented
        raise NotImplementedError(
            "GromovWassersteinAligner will be implemented as Innovation 3. "
            "Use CCAAligner for now."
        )
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit GW alignment."""
        pass
    
    def transform(self, X: np.ndarray, Y: np.ndarray):
        """Transform using GW alignment."""
        pass