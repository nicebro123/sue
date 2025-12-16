"""
Trainer Module for SUE Improved

Orchestrates the three-stage training process:
1. Spectral Embedding (with Innovation 1 & 2)
2. CCA Alignment
3. MMD Fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from sklearn.cross_decomposition import CCA

from .data.dataset import get_parallel_data, get_non_parallel_data, ModalityDataLoader
from .models.spectral_embedding import SpectralEmbedding, MMDNet, create_spectral_embedding
from .losses.spectral_loss import SpectralNetLoss, CombinedSpectralLoss, MMDLoss, OrthogonalityLoss
from .utils.graph import GraphConstructor, create_graph_constructor
from .utils.metrics import evaluate_alignment, print_metrics
from .utils.visualization import VisualizationManager


class SpectralNetTrainer:
    """
    Trainer for SpectralNet (Stage 1).
    
    Uses Innovation 1 (Quadratic OT graph) and Innovation 2 (Doubly stochastic Laplacian).
    """
    
    def __init__(
        self,
        model: SpectralEmbedding,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        train_config = config.get("training", {}).get("spectral", {})
        self.epochs = train_config.get("epochs", 100)
        self.batch_size = train_config.get("batch_size", 1024)
        self.lr = train_config.get("lr", 1e-4)
        self.lr_decay = train_config.get("lr_decay", 0.1)
        self.min_lr = train_config.get("min_lr", 1e-8)
        self.patience = train_config.get("patience", 10)
        
        self.loss_fn = CombinedSpectralLoss(alpha=1.0, normalized=False)
        self.history = {"loss": [], "spectral_loss": [], "orth_loss": []}
        
    def train(self, X: torch.Tensor) -> np.ndarray:
        """
        Train SpectralNet on given data.
        
        Args:
            X: Input data tensor (n, d)
            
        Returns:
            embeddings: Learned spectral embeddings (n, k)
        """
        X = X.to(self.device)
        n_samples = X.shape[0]
        
        print(f"Building graph with {self.model.graph_constructor.method} method...")
        print(f"  Laplacian normalization: {self.model.graph_constructor.laplacian_norm}")
        
        # Build graph (Innovation 1 & 2)
        W, L = self.model.build_graph(X)
        W = W.to(self.device)
        
        print(f"  Affinity matrix sparsity: {(W == 0).float().mean().item():.2%}")
        print(f"  Degree range: [{W.sum(dim=1).min().item():.4f}, {W.sum(dim=1).max().item():.4f}]")
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.spectral_net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_decay,
            patience=self.patience,
            min_lr=self.min_lr
        )
        
        # Training loop
        self.model.spectral_net.train()
        best_loss = float('inf')
        best_state = None
        
        # Determine if we use full batch or mini-batches
        use_full_batch = n_samples <= self.batch_size
        
        pbar = tqdm(range(1, self.epochs + 1), desc="SpectralNet Training")
        
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_spectral_loss = 0.0
            epoch_orth_loss = 0.0
            n_batches = 0
            
            if use_full_batch:
                # Use full batch for small datasets
                optimizer.zero_grad()
                Y = self.model.spectral_net(X)
                
                spectral_loss = SpectralNetLoss()(W, Y)
                orth_loss = OrthogonalityLoss()(Y)
                loss = spectral_loss + orth_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss = loss.item()
                epoch_spectral_loss = spectral_loss.item()
                epoch_orth_loss = orth_loss.item()
                n_batches = 1
            else:
                # Mini-batch training: iterate through multiple batches per epoch
                indices = torch.randperm(n_samples)
                
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_X = X[batch_indices]
                    batch_W = W[batch_indices][:, batch_indices]
                    
                    optimizer.zero_grad()
                    Y = self.model.spectral_net(batch_X)
                    
                    spectral_loss = SpectralNetLoss()(batch_W, Y)
                    orth_loss = OrthogonalityLoss()(Y)
                    loss = spectral_loss + orth_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_spectral_loss += spectral_loss.item()
                    epoch_orth_loss += orth_loss.item()
                    n_batches += 1
                
                # Average over batches
                epoch_loss /= n_batches
                epoch_spectral_loss /= n_batches
                epoch_orth_loss /= n_batches
            
            # Record history
            self.history["loss"].append(epoch_loss)
            self.history["spectral_loss"].append(epoch_spectral_loss)
            self.history["orth_loss"].append(epoch_orth_loss)
            
            # Update scheduler
            scheduler.step(epoch_loss)
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.model.spectral_net.state_dict())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{epoch_loss:.4f}",
                "spec": f"{epoch_spectral_loss:.4f}",
                "orth": f"{epoch_orth_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Load best model
        if best_state is not None:
            self.model.spectral_net.load_state_dict(best_state)
        
        # Compute final embeddings
        self.model.spectral_net.eval()
        with torch.no_grad():
            embeddings = self.model.spectral_net(X, should_update_orth_weights=False)
            embeddings = embeddings.cpu().numpy()
        
        # Compute sorting matrix (order by eigenvalue)
        self.model.compute_sorting_matrix(X)
        
        # Apply sorting
        embeddings = embeddings @ self.model.Q.cpu().numpy()
        
        return embeddings


class SUETrainer:
    """
    Main trainer for SUE Improved.
    
    Orchestrates the complete three-stage pipeline.
    """
    
    def __init__(self, config: Dict, device: torch.device = None):
        """
        Args:
            config: Configuration dictionary
            device: Torch device
        """
        self.config = config
        self.device = device if device else self._detect_device()
        
        print(f"Using device: {self.device}")
        
        # Get config values
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        
        self.n_parallel = data_config.get("n_parallel", 500)
        self.n_components = model_config.get("n_components", 8)
        self.n_components_actual = self.n_components  # May be updated during CCA
        self.n_spectral_dim = model_config.get("n_spectral_dim", 30)
        
        # Create graph constructors for each modality
        self.graph_constructor1 = create_graph_constructor(config, self.device)
        self.graph_constructor2 = create_graph_constructor(config, self.device)
        
        # Create spectral embedding models
        self.spectral_emb1 = create_spectral_embedding(
            config, self.graph_constructor1, "modality1", self.device
        )
        self.spectral_emb2 = create_spectral_embedding(
            config, self.graph_constructor2, "modality2", self.device
        )
        
        # MMD model
        self.mmd_model = MMDNet(self.n_components).to(self.device)
        
        # CCA projections
        self.projection1 = None
        self.projection2 = None
        
        # Training flags
        self.with_se = True
        self.with_cca = True
        self.with_mmd = True
        
        # Training history
        self.history = {
            "spectral1": {},
            "spectral2": {},
            "mmd": {"loss": []}
        }
        
        # Visualization manager
        vis_config = config.get("visualization", {})
        if vis_config.get("enabled", True):
            self.vis_manager = VisualizationManager(
                vis_config.get("save_path", "./outputs/visualizations/")
            )
        else:
            self.vis_manager = None
            
    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def fit(
        self,
        train_set: Tuple[torch.Tensor, torch.Tensor],
        with_se: bool = True,
        with_cca: bool = True,
        with_mmd: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the complete SUE pipeline.
        
        Args:
            train_set: (modality1_embeddings, modality2_embeddings)
            with_se: Whether to train spectral embedding
            with_cca: Whether to apply CCA
            with_mmd: Whether to train MMD
            
        Returns:
            (aligned_embeddings1, aligned_embeddings2)
        """
        self.with_se = with_se
        self.with_cca = with_cca
        self.with_mmd = with_mmd
        
        train_mod1, train_mod2 = train_set
        
        # Store original embeddings for visualization
        self.original_emb1 = train_mod1.clone()
        self.original_emb2 = train_mod2.clone()
        
        # Initialize embeddings
        embeddings1 = train_mod1.cpu().numpy()
        embeddings2 = train_mod2.cpu().numpy()
        
        # ==================== Stage 1: Spectral Embedding ====================
        if with_se:
            print("\n" + "="*60)
            print("Stage 1: Spectral Embedding (Innovation 1 & 2)")
            print("="*60)
            
            # Train modality 1
            print("\n[Modality 1]")
            trainer1 = SpectralNetTrainer(self.spectral_emb1, self.config, self.device)
            embeddings1 = trainer1.train(train_mod1)
            self.history["spectral1"] = trainer1.history
            
            # Train modality 2
            print("\n[Modality 2]")
            trainer2 = SpectralNetTrainer(self.spectral_emb2, self.config, self.device)
            embeddings2 = trainer2.train(train_mod2)
            self.history["spectral2"] = trainer2.history
            
            print(f"\nSpectral embeddings: {embeddings1.shape}, {embeddings2.shape}")
        
        # ==================== Stage 2: CCA Alignment ====================
        if with_cca:
            print("\n" + "="*60)
            print("Stage 2: CCA Alignment")
            print("="*60)
            
            # Get parallel samples
            parallel_emb1 = embeddings1[-self.n_parallel:]
            parallel_emb2 = embeddings2[-self.n_parallel:]
            
            print(f"Using {self.n_parallel} paired samples for CCA")
            
            # Determine valid number of CCA components
            max_components = min(
                self.n_components,
                parallel_emb1.shape[1],
                parallel_emb2.shape[1],
                self.n_parallel  # Cannot exceed number of samples
            )
            
            if max_components < self.n_components:
                print(f"Warning: Reducing CCA components from {self.n_components} to {max_components} "
                      f"due to dimension constraints")
            
            # Fit CCA
            cca = CCA(n_components=max_components)
            cca.fit(parallel_emb1, parallel_emb2)
            
            self.projection1 = cca.x_rotations_
            self.projection2 = cca.y_rotations_
            
            # Update n_components to actual value used
            self.n_components_actual = max_components
            
            # Project all embeddings
            embeddings1 = embeddings1 @ self.projection1
            embeddings2 = embeddings2 @ self.projection2
            
            print(f"CCA projections: {self.projection1.shape}, {self.projection2.shape}")
            print(f"Projected embeddings: {embeddings1.shape}, {embeddings2.shape}")
        
        # ==================== Stage 3: MMD Fine-tuning ====================
        if with_mmd:
            print("\n" + "="*60)
            print("Stage 3: MMD Fine-tuning")
            print("="*60)
            
            # Get actual dimension after CCA
            actual_dim = embeddings1.shape[1]
            
            # Reinitialize MMD model with correct dimension if needed
            if actual_dim != self.n_components:
                print(f"Reinitializing MMD model for dimension {actual_dim}")
                self.mmd_model = MMDNet(actual_dim).to(self.device)
            
            embeddings1, embeddings2 = self._train_mmd(embeddings1, embeddings2)
        
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        
        return embeddings1, embeddings2
    
    def _train_mmd(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train MMD network for distribution alignment.
        
        Args:
            embeddings1: Modality 1 embeddings
            embeddings2: Modality 2 embeddings
            
        Returns:
            (aligned_embeddings1, embeddings2)
        """
        mmd_config = self.config.get("training", {}).get("mmd", {})
        epochs = mmd_config.get("epochs", 100)
        batch_size = mmd_config.get("batch_size", 32)
        lr = mmd_config.get("lr", 1e-3)
        n_scales = mmd_config.get("n_scales", 3)
        val_split = mmd_config.get("val_split", 0.1)
        
        X = torch.from_numpy(embeddings1).float()
        Y = torch.from_numpy(embeddings2).float()
        
        # Train/val split
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Setup
        criterion = MMDLoss(n_scales=n_scales, device=self.device)
        optimizer = optim.AdamW(self.mmd_model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_model = None
        
        # Training loop
        pbar = tqdm(range(1, epochs + 1), desc="MMD Training")
        
        for epoch in pbar:
            # Training
            self.mmd_model.train()
            train_loss = 0.0
            n_batches = 0
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_x = X_train[start:end].to(self.device)
                batch_y = Y_train[start:end].to(self.device)
                
                optimizer.zero_grad()
                x_out = self.mmd_model(batch_x)
                loss = criterion(x_out, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation
            self.mmd_model.eval()
            with torch.no_grad():
                x_val_out = self.mmd_model(X_val.to(self.device))
                val_loss = criterion(x_val_out, Y_val.to(self.device)).item()
            
            self.history["mmd"]["loss"].append(train_loss)
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.mmd_model.state_dict())
            
            pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
        
        # Load best model
        if best_model is not None:
            self.mmd_model.load_state_dict(best_model)
        
        # Transform embeddings
        self.mmd_model.eval()
        with torch.no_grad():
            X_transformed = self.mmd_model(X.to(self.device)).cpu().numpy()
        
        return X_transformed, embeddings2
    
    def transform(
        self,
        data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using trained models.
        
        Args:
            data: (modality1_data, modality2_data)
            
        Returns:
            (transformed_emb1, transformed_emb2)
        """
        mod1, mod2 = data
        
        # Stage 1: Spectral embedding
        if self.with_se:
            emb1 = self.spectral_emb1(mod1.to(self.device)).cpu().detach().numpy()
            emb2 = self.spectral_emb2(mod2.to(self.device)).cpu().detach().numpy()
        else:
            emb1 = mod1.cpu().numpy()
            emb2 = mod2.cpu().numpy()
        
        # Stage 2: CCA projection
        if self.with_cca:
            emb1 = emb1 @ self.projection1
            emb2 = emb2 @ self.projection2
        
        # Stage 3: MMD transformation
        if self.with_mmd:
            self.mmd_model.eval()
            with torch.no_grad():
                emb1 = self.mmd_model(
                    torch.from_numpy(emb1).float().to(self.device)
                ).cpu().numpy()
        
        return emb1, emb2
    
    def evaluate(
        self,
        test_set: Tuple[torch.Tensor, torch.Tensor],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            test_set: (test_modality1, test_modality2)
            verbose: Whether to print results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Transform test data
        test_emb1, test_emb2 = self.transform(test_set)
        
        # Compute metrics
        eval_config = self.config.get("evaluation", {})
        k_values = eval_config.get("recall_k", [1, 5, 10])
        knn_k = eval_config.get("knn_preservation_k", 10)
        
        metrics = evaluate_alignment(
            torch.from_numpy(test_emb1),
            torch.from_numpy(test_emb2),
            original_emb1=test_set[0],
            original_emb2=test_set[1],
            k_values=k_values,
            knn_k=knn_k
        )
        
        if verbose:
            print_metrics(metrics, title="Test Set Evaluation")
        
        return metrics
    
    def visualize(
        self,
        test_set: Tuple[torch.Tensor, torch.Tensor],
        prefix: str = ""
    ):
        """
        Generate visualizations.
        
        Args:
            test_set: (test_modality1, test_modality2)
            prefix: Prefix for saved files
        """
        if self.vis_manager is None:
            print("Visualization is disabled")
            return
        
        # Get test embeddings
        test_emb1, test_emb2 = self.transform(test_set)
        
        # Get eigenvalues if available
        eigenvalues1 = None
        eigenvalues2 = None
        
        if self.spectral_emb1._cached_L is not None:
            L1 = self.spectral_emb1._cached_L.cpu().numpy()
            eigenvalues1, _ = np.linalg.eigh(L1)
        
        if self.spectral_emb2._cached_L is not None:
            L2 = self.spectral_emb2._cached_L.cpu().numpy()
            eigenvalues2, _ = np.linalg.eigh(L2)
        
        # Combine training histories
        combined_history = {}
        for key, values in self.history["spectral1"].items():
            combined_history[f"M1_{key}"] = values
        for key, values in self.history["spectral2"].items():
            combined_history[f"M2_{key}"] = values
        for key, values in self.history["mmd"].items():
            combined_history[f"MMD_{key}"] = values
        
        # Generate visualizations
        self.vis_manager.save_all(
            emb1=torch.from_numpy(test_emb1),
            emb2=torch.from_numpy(test_emb2),
            original_emb1=test_set[0],
            original_emb2=test_set[1],
            eigenvalues1=eigenvalues1,
            eigenvalues2=eigenvalues2,
            W1=self.spectral_emb1.get_affinity_matrix(),
            W2=self.spectral_emb2.get_affinity_matrix(),
            history=combined_history,
            prefix=prefix
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "spectral_emb1": self.spectral_emb1.spectral_net.state_dict(),
            "spectral_emb2": self.spectral_emb2.spectral_net.state_dict(),
            "mmd_model": self.mmd_model.state_dict(),
            "projection1": self.projection1,
            "projection2": self.projection2,
            "Q1": self.spectral_emb1.Q,
            "Q2": self.spectral_emb2.Q,
            "n_components_actual": self.n_components_actual,
            "config": self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.spectral_emb1.spectral_net.load_state_dict(checkpoint["spectral_emb1"])
        self.spectral_emb2.spectral_net.load_state_dict(checkpoint["spectral_emb2"])
        self.projection1 = checkpoint["projection1"]
        self.projection2 = checkpoint["projection2"]
        self.spectral_emb1.Q = checkpoint["Q1"]
        self.spectral_emb2.Q = checkpoint["Q2"]
        
        # Load n_components_actual if available
        if "n_components_actual" in checkpoint:
            self.n_components_actual = checkpoint["n_components_actual"]
            # Reinitialize MMD model with correct dimension
            self.mmd_model = MMDNet(self.n_components_actual).to(self.device)
        
        self.mmd_model.load_state_dict(checkpoint["mmd_model"])
        
        print(f"Checkpoint loaded from {path}")