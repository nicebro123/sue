"""
Spectral Embedding Models for SUE Improved

Contains SpectralNet architecture and related models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class SpectralNetModel(nn.Module):
    """
    Neural network for learning spectral embeddings.
    
    Maps input features to low-dimensional embeddings that approximate
    Laplacian eigenvectors.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Orthogonalization layer (Cholesky)
        self.orthogonalize = True
        
    def forward(
        self,
        x: torch.Tensor,
        should_update_orth_weights: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            should_update_orth_weights: Whether to orthogonalize output
            
        Returns:
            y: Embeddings (batch_size, output_dim)
        """
        y = self.network(x)
        
        if self.orthogonalize and should_update_orth_weights:
            y = self._orthogonalize_batch(y)
            
        return y
    
    def _orthogonalize_batch(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize embeddings using Cholesky decomposition.
        
        Ensures Y^T Y = I (up to scaling).
        """
        # Center the embeddings
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrix
        gram = Y.T @ Y / Y.shape[0]
        
        # Add small regularization for numerical stability
        gram = gram + 1e-6 * torch.eye(gram.shape[0], device=gram.device)
        
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(gram)
            # Y_orth = Y @ L^{-T}
            Y_orth = torch.linalg.solve_triangular(L.T, Y.T, upper=True).T
        except RuntimeError:
            # Fallback to SVD if Cholesky fails
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            Y_orth = U
            
        return Y_orth


class SpectralEmbedding(nn.Module):
    """
    Complete spectral embedding module with graph construction.
    
    This combines the graph construction (Innovation 1 & 2) with SpectralNet.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        graph_constructor,
        device: torch.device = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension  
            graph_constructor: GraphConstructor instance
            device: Torch device
        """
        super().__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.graph_constructor = graph_constructor
        
        self.spectral_net = SpectralNetModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        ).to(self.device)
        
        self.output_dim = output_dim
        
        # Sorting matrix for eigenvalue ordering
        self.Q = None
        
        # Cached graph for training
        self._cached_W = None
        self._cached_L = None
        
    def build_graph(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build and cache graph from data.
        
        Args:
            X: Input data
            
        Returns:
            W: Affinity matrix
            L: Laplacian matrix
        """
        W, L = self.graph_constructor.build_graph(X)
        self._cached_W = W
        self._cached_L = L
        return W, L
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral embeddings.
        
        Args:
            X: Input data (n, input_dim)
            
        Returns:
            embeddings: Spectral embeddings (n, output_dim)
        """
        X = X.to(self.device)
        embeddings = self.spectral_net(X)
        
        # Apply eigenvalue sorting if available
        if self.Q is not None:
            embeddings = embeddings @ self.Q
            
        return embeddings
    
    def compute_sorting_matrix(self, X: torch.Tensor):
        """
        Compute sorting matrix Q to order embeddings by eigenvalue.
        
        After training, the spectral embeddings may not be sorted by
        eigenvalue. This computes Q such that Y @ Q has columns
        ordered by increasing Rayleigh quotient.
        """
        X = X.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            Y = self.spectral_net(X, should_update_orth_weights=False)
        
        # Build Laplacian if not cached
        if self._cached_L is None:
            _, L = self.build_graph(X)
        else:
            L = self._cached_L
            
        L = L.to(self.device)
        
        # Compute Rayleigh quotients for each column
        # Lambda_ii = y_i^T L y_i / y_i^T y_i
        Y_np = Y.cpu().numpy()
        L_np = L.cpu().numpy()
        
        Lambda = Y_np.T @ L_np @ Y_np
        
        # SVD to get sorted eigenvectors
        Q, S, _ = np.linalg.svd(Lambda)
        
        # Sort by eigenvalue (ascending)
        indices = np.argsort(np.diag(Q.T @ Lambda @ Q))
        Q = Q[:, indices]
        
        self.Q = torch.tensor(Q, device=self.device, dtype=torch.float32)
        
    def get_affinity_matrix(self) -> Optional[torch.Tensor]:
        """Return cached affinity matrix."""
        return self._cached_W
    
    def get_laplacian(self) -> Optional[torch.Tensor]:
        """Return cached Laplacian matrix."""
        return self._cached_L


class MMDNet(nn.Module):
    """
    Residual network for MMD-based distribution alignment.
    
    Uses residual connections to ensure the transformation is close to identity,
    preserving the structure learned by spectral embedding.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        n_blocks: int = 1
    ):
        """
        Args:
            input_dim: Input/output dimension
            hidden_dim: Hidden layer dimension in residual blocks
            n_blocks: Number of residual blocks
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(input_dim, hidden_dim) for _ in range(n_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual blocks.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (same shape as input)
        """
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for MMDNet.
    
    f(x) = x + g(x) where g is a small MLP.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def create_spectral_embedding(
    config: dict,
    graph_constructor,
    modality: str,
    device: torch.device
) -> SpectralEmbedding:
    """
    Factory function to create SpectralEmbedding from config.
    
    Args:
        config: Configuration dictionary
        graph_constructor: GraphConstructor instance
        modality: 'modality1' or 'modality2'
        device: Torch device
        
    Returns:
        SpectralEmbedding instance
    """
    model_config = config.get("model", {})
    spectralnet_config = model_config.get("spectralnet", {}).get(modality, {})
    
    return SpectralEmbedding(
        input_dim=spectralnet_config.get("input_dim", 768),
        hidden_dims=spectralnet_config.get("hidden_dims", [4096, 4096, 1024]),
        output_dim=spectralnet_config.get("output_dim", 30),
        graph_constructor=graph_constructor,
        device=device
    )