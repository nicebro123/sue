"""
Visualization Module for SUE Improved

Provides visualization tools for:
- Embedding space visualization (t-SNE/UMAP)
- Similarity matrix heatmaps
- Eigenvalue spectrum comparison
- Training curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import os


def set_plot_style():
    """Set consistent plot style with fallback for compatibility."""
    # Try different style options for compatibility across matplotlib versions
    style_options = [
        'seaborn-v0_8-whitegrid',
        'seaborn-whitegrid', 
        'ggplot',
        'default'
    ]
    
    for style in style_options:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    
    # Set common parameters regardless of style
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def visualize_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    method: str = "tsne",
    n_samples: int = 1000,
    labels1: Optional[np.ndarray] = None,
    labels2: Optional[np.ndarray] = None,
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize embeddings from two modalities in 2D.
    
    Args:
        emb1: Modality 1 embeddings
        emb2: Modality 2 embeddings
        method: 'tsne' or 'umap'
        n_samples: Number of samples to visualize
        labels1: Optional labels for modality 1
        labels2: Optional labels for modality 2
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    # Convert to numpy
    emb1 = emb1.cpu().numpy() if isinstance(emb1, torch.Tensor) else emb1
    emb2 = emb2.cpu().numpy() if isinstance(emb2, torch.Tensor) else emb2
    
    # Sample if too many points
    n1, n2 = len(emb1), len(emb2)
    if n1 > n_samples:
        idx1 = np.random.choice(n1, n_samples, replace=False)
        emb1 = emb1[idx1]
        if labels1 is not None:
            labels1 = labels1[idx1]
    if n2 > n_samples:
        idx2 = np.random.choice(n2, n_samples, replace=False)
        emb2 = emb2[idx2]
        if labels2 is not None:
            labels2 = labels2[idx2]
    
    # Combine embeddings
    combined = np.vstack([emb1, emb2])
    modality_labels = np.array(["Modality 1"] * len(emb1) + ["Modality 2"] * len(emb2))
    
    # Dimensionality reduction
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    coords = reducer.fit_transform(combined)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by modality
    colors = {"Modality 1": "#1f77b4", "Modality 2": "#ff7f0e"}
    
    for modality in ["Modality 1", "Modality 2"]:
        mask = modality_labels == modality
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[modality],
            label=modality,
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_similarity_matrix(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    n_samples: int = 100,
    title: str = "Cross-Modal Similarity Matrix",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize similarity matrix as heatmap.
    
    Args:
        emb1: Modality 1 embeddings
        emb2: Modality 2 embeddings
        n_samples: Number of samples to show
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    # Convert and normalize
    emb1 = emb1.cpu() if isinstance(emb1, torch.Tensor) else torch.tensor(emb1)
    emb2 = emb2.cpu() if isinstance(emb2, torch.Tensor) else torch.tensor(emb2)
    
    emb1 = emb1 / emb1.norm(dim=1, keepdim=True).clamp_min(1e-10)
    emb2 = emb2 / emb2.norm(dim=1, keepdim=True).clamp_min(1e-10)
    
    # Sample
    n_samples = min(n_samples, len(emb1), len(emb2))
    emb1 = emb1[:n_samples]
    emb2 = emb2[:n_samples]
    
    # Compute similarity
    sim = (emb1 @ emb2.T).numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        sim,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine Similarity"}
    )
    
    ax.set_xlabel("Modality 2 Samples")
    ax.set_ylabel("Modality 1 Samples")
    ax.set_title(title)
    
    # Add diagonal reference line description
    ax.text(
        0.02, 0.98,
        "Diagonal = Ground Truth Matches",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_eigenvalue_spectrum(
    eigenvalues1: np.ndarray,
    eigenvalues2: np.ndarray,
    k: int = 30,
    title: str = "Eigenvalue Spectrum Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare eigenvalue spectra of two modalities.
    
    Used to verify the manifold isomorphism hypothesis.
    
    Args:
        eigenvalues1: Eigenvalues for modality 1
        eigenvalues2: Eigenvalues for modality 2
        k: Number of eigenvalues to show
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    k = min(k, len(eigenvalues1), len(eigenvalues2))
    ev1 = eigenvalues1[:k]
    ev2 = eigenvalues2[:k]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot eigenvalues
    ax = axes[0]
    x = np.arange(1, k + 1)
    ax.plot(x, ev1, 'o-', label="Modality 1", markersize=6)
    ax.plot(x, ev2, 's-', label="Modality 2", markersize=6)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot eigenvalue ratio (normalized)
    ax = axes[1]
    ev1_norm = ev1 / (ev1.max() + 1e-10)
    ev2_norm = ev2 / (ev2.max() + 1e-10)
    ax.scatter(ev1_norm, ev2_norm, alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="y=x (perfect match)")
    ax.set_xlabel("Modality 1 Eigenvalues (normalized)")
    ax.set_ylabel("Modality 2 Eigenvalues (normalized)")
    ax.set_title("Eigenvalue Correspondence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrum comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves.
    
    Args:
        history: Dictionary mapping metric names to lists of values
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    n_metrics = len(history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, values) in enumerate(history.items()):
        ax = axes[idx]
        ax.plot(values, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(history), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_affinity_matrix(
    W: torch.Tensor,
    n_samples: int = 500,
    title: str = "Affinity Matrix",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize affinity matrix structure.
    
    Args:
        W: Affinity matrix
        n_samples: Number of samples to show
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    W = W.cpu().numpy() if isinstance(W, torch.Tensor) else W
    
    n_samples = min(n_samples, W.shape[0])
    W = W[:n_samples, :n_samples]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax = axes[0]
    im = ax.imshow(W, cmap="viridis", aspect="auto")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    ax.set_title("Affinity Matrix")
    plt.colorbar(im, ax=ax, label="Affinity")
    
    # Degree distribution
    ax = axes[1]
    degrees = W.sum(axis=1)
    ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution")
    ax.axvline(degrees.mean(), color='r', linestyle='--', label=f"Mean: {degrees.mean():.2f}")
    ax.legend()
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved affinity matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_retrieval_results(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    query_indices: List[int],
    k: int = 5,
    title: str = "Retrieval Results",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize retrieval results for selected queries.
    
    Shows top-k retrieved items and whether correct match is in top-k.
    
    Args:
        emb1: Query embeddings (modality 1)
        emb2: Database embeddings (modality 2)
        query_indices: Indices of queries to visualize
        k: Number of top results to show
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    set_plot_style()
    
    # Compute similarity
    emb1 = emb1.cpu() if isinstance(emb1, torch.Tensor) else torch.tensor(emb1)
    emb2 = emb2.cpu() if isinstance(emb2, torch.Tensor) else torch.tensor(emb2)
    
    emb1 = emb1 / emb1.norm(dim=1, keepdim=True).clamp_min(1e-10)
    emb2 = emb2 / emb2.norm(dim=1, keepdim=True).clamp_min(1e-10)
    
    sim = emb1 @ emb2.T
    rankings = torch.argsort(sim, dim=1, descending=True)
    
    n_queries = len(query_indices)
    fig, axes = plt.subplots(n_queries, 1, figsize=(12, 3 * n_queries))
    
    if n_queries == 1:
        axes = [axes]
    
    for idx, query_idx in enumerate(query_indices):
        ax = axes[idx]
        
        top_k_indices = rankings[query_idx, :k].numpy()
        top_k_scores = sim[query_idx, top_k_indices].numpy()
        
        # Create bar chart
        colors = ['green' if i == query_idx else 'steelblue' for i in top_k_indices]
        bars = ax.bar(range(k), top_k_scores, color=colors, edgecolor='black')
        
        ax.set_xlabel("Rank")
        ax.set_ylabel("Similarity Score")
        ax.set_title(f"Query {query_idx}: Ground Truth Rank = {(rankings[query_idx] == query_idx).nonzero()[0].item() + 1}")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"#{i+1}\n(idx:{top_k_indices[i]})" for i in range(k)])
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Correct Match'),
            Patch(facecolor='steelblue', edgecolor='black', label='Other')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved retrieval results to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


class VisualizationManager:
    """
    Manager class for organizing all visualizations.
    """
    
    def __init__(self, save_dir: str = "./outputs/visualizations/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_all(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        original_emb1: Optional[torch.Tensor] = None,
        original_emb2: Optional[torch.Tensor] = None,
        eigenvalues1: Optional[np.ndarray] = None,
        eigenvalues2: Optional[np.ndarray] = None,
        W1: Optional[torch.Tensor] = None,
        W2: Optional[torch.Tensor] = None,
        history: Optional[Dict[str, List[float]]] = None,
        prefix: str = ""
    ):
        """
        Generate and save all visualizations.
        
        Args:
            emb1: Aligned modality 1 embeddings
            emb2: Aligned modality 2 embeddings
            original_emb1: Original modality 1 embeddings
            original_emb2: Original modality 2 embeddings
            eigenvalues1: Eigenvalues for modality 1
            eigenvalues2: Eigenvalues for modality 2
            W1: Affinity matrix for modality 1
            W2: Affinity matrix for modality 2
            history: Training history
            prefix: Prefix for filenames
        """
        # Embedding visualization
        visualize_embeddings(
            emb1, emb2,
            method="tsne",
            title="Aligned Embedding Space",
            save_path=str(self.save_dir / f"{prefix}embeddings_tsne.png"),
            show=False
        )
        
        # Similarity matrix
        visualize_similarity_matrix(
            emb1, emb2,
            title="Cross-Modal Similarity",
            save_path=str(self.save_dir / f"{prefix}similarity_matrix.png"),
            show=False
        )
        
        # Eigenvalue spectrum
        if eigenvalues1 is not None and eigenvalues2 is not None:
            visualize_eigenvalue_spectrum(
                eigenvalues1, eigenvalues2,
                title="Eigenvalue Spectrum Comparison",
                save_path=str(self.save_dir / f"{prefix}eigenvalue_spectrum.png"),
                show=False
            )
        
        # Affinity matrices
        if W1 is not None:
            visualize_affinity_matrix(
                W1,
                title="Modality 1 Affinity Matrix",
                save_path=str(self.save_dir / f"{prefix}affinity_m1.png"),
                show=False
            )
        if W2 is not None:
            visualize_affinity_matrix(
                W2,
                title="Modality 2 Affinity Matrix",
                save_path=str(self.save_dir / f"{prefix}affinity_m2.png"),
                show=False
            )
        
        # Training curves
        if history is not None:
            visualize_training_curves(
                history,
                title="Training Progress",
                save_path=str(self.save_dir / f"{prefix}training_curves.png"),
                show=False
            )
        
        # Retrieval results
        visualize_retrieval_results(
            emb1, emb2,
            query_indices=[0, 1, 2, 3, 4],
            k=10,
            title="Sample Retrieval Results",
            save_path=str(self.save_dir / f"{prefix}retrieval_results.png"),
            show=False
        )
        
        print(f"All visualizations saved to {self.save_dir}")