"""
Evaluation Metrics for SUE Improved

Contains:
- Retrieval metrics: Recall@k, MRR, mAP
- Alignment quality metrics: CKA, Procrustes distance
- Structure preservation metrics: k-NN preservation rate
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors


def compute_similarity_matrix(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two embedding sets.
    
    Args:
        emb1: Embeddings of shape (n, d)
        emb2: Embeddings of shape (m, d)
        normalize: Whether to L2 normalize embeddings
        
    Returns:
        sim: Similarity matrix of shape (n, m)
    """
    if normalize:
        emb1 = emb1 / emb1.norm(dim=1, keepdim=True).clamp_min(1e-10)
        emb2 = emb2 / emb2.norm(dim=1, keepdim=True).clamp_min(1e-10)
    
    sim = emb1 @ emb2.T
    return sim


# ==================== Retrieval Metrics ====================

def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    direction: str = "row"
) -> Dict[str, float]:
    """
    Compute Recall@k for retrieval.
    
    Assumes diagonal entries are ground truth matches.
    
    Args:
        similarity_matrix: (n, m) similarity matrix
        k_values: List of k values to compute
        direction: 'row' for row-to-column retrieval, 'col' for column-to-row
        
    Returns:
        Dictionary mapping 'R@k' to recall values
    """
    if direction == "col":
        similarity_matrix = similarity_matrix.T
        
    n = similarity_matrix.shape[0]
    
    # Get rankings
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Find position of correct match (diagonal)
    correct_indices = torch.arange(n, device=similarity_matrix.device)
    
    results = {}
    for k in k_values:
        # Check if correct index is in top-k
        top_k = rankings[:, :k]
        hits = (top_k == correct_indices[:, None]).any(dim=1)
        recall = hits.float().mean().item()
        results[f"R@{k}"] = recall
        
    return results


def compute_mrr(
    similarity_matrix: torch.Tensor,
    direction: str = "row"
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        similarity_matrix: (n, m) similarity matrix
        direction: 'row' or 'col'
        
    Returns:
        MRR value
    """
    if direction == "col":
        similarity_matrix = similarity_matrix.T
        
    n = similarity_matrix.shape[0]
    
    # Get rankings
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Find rank of correct match
    correct_indices = torch.arange(n, device=similarity_matrix.device)
    
    reciprocal_ranks = []
    for i in range(n):
        rank = (rankings[i] == correct_indices[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            reciprocal_ranks.append(1.0 / (rank[0].item() + 1))
        else:
            reciprocal_ranks.append(0.0)
            
    return np.mean(reciprocal_ranks)


def compute_map(
    similarity_matrix: torch.Tensor,
    direction: str = "row"
) -> float:
    """
    Compute Mean Average Precision.
    
    For single relevant item per query (diagonal), this equals MRR.
    
    Args:
        similarity_matrix: (n, m) similarity matrix
        direction: 'row' or 'col'
        
    Returns:
        mAP value
    """
    # For single relevant item per query, mAP = MRR
    return compute_mrr(similarity_matrix, direction)


def compute_all_retrieval_metrics(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for paired embeddings.
    
    Args:
        emb1: Modality 1 embeddings (n, d)
        emb2: Modality 2 embeddings (n, d)
        k_values: List of k values for Recall@k
        
    Returns:
        Dictionary of all metrics
    """
    sim = compute_similarity_matrix(emb1, emb2)
    
    results = {}
    
    # Forward direction (modality 1 -> modality 2)
    forward_recall = compute_recall_at_k(sim, k_values, direction="row")
    for k, v in forward_recall.items():
        results[f"M1->M2_{k}"] = v
    results["M1->M2_MRR"] = compute_mrr(sim, direction="row")
    
    # Backward direction (modality 2 -> modality 1)
    backward_recall = compute_recall_at_k(sim, k_values, direction="col")
    for k, v in backward_recall.items():
        results[f"M2->M1_{k}"] = v
    results["M2->M1_MRR"] = compute_mrr(sim, direction="col")
    
    # Average
    for k in k_values:
        results[f"Avg_R@{k}"] = (results[f"M1->M2_R@{k}"] + results[f"M2->M1_R@{k}"]) / 2
    results["Avg_MRR"] = (results["M1->M2_MRR"] + results["M2->M1_MRR"]) / 2
    
    return results


# ==================== Alignment Quality Metrics ====================

def compute_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: str = "linear"
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) similarity.
    
    CKA measures representation similarity between two sets of embeddings.
    
    Args:
        X: First embedding set (n, d1)
        Y: Second embedding set (n, d2)
        kernel: 'linear' or 'rbf'
        
    Returns:
        CKA similarity value in [0, 1]
    """
    X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    Y = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
    
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == "rbf":
        # RBF kernel with median heuristic
        K = _rbf_kernel(X)
        L = _rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Center the kernels
    K = _center_kernel(K)
    L = _center_kernel(L)
    
    # Compute CKA
    hsic = np.sum(K * L)
    norm_k = np.sqrt(np.sum(K * K))
    norm_l = np.sqrt(np.sum(L * L))
    
    cka = hsic / (norm_k * norm_l + 1e-10)
    
    return cka


def _rbf_kernel(X: np.ndarray) -> np.ndarray:
    """Compute RBF kernel with median heuristic."""
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    median_dist = np.median(sq_dists[sq_dists > 0])
    gamma = 1.0 / (median_dist + 1e-10)
    return np.exp(-gamma * sq_dists)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_procrustes_distance(
    X: torch.Tensor,
    Y: torch.Tensor
) -> float:
    """
    Compute Procrustes distance between two embedding sets.
    
    Finds optimal orthogonal transformation to align X to Y.
    
    Args:
        X: First embedding set (n, d)
        Y: Second embedding set (n, d)
        
    Returns:
        Procrustes distance (lower = more similar)
    """
    X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    Y = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
    
    # Center both
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Scale to unit norm
    X = X / np.linalg.norm(X, 'fro')
    Y = Y / np.linalg.norm(Y, 'fro')
    
    # Find optimal rotation: R = argmin ||X @ R - Y||
    U, S, Vh = np.linalg.svd(X.T @ Y)
    R = U @ Vh
    
    # Compute distance
    X_aligned = X @ R
    distance = np.linalg.norm(X_aligned - Y, 'fro')
    
    return distance


# ==================== Structure Preservation Metrics ====================

def compute_knn_preservation(
    original: torch.Tensor,
    transformed: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute k-NN preservation rate.
    
    Measures how well the transformation preserves local neighborhood structure.
    
    Args:
        original: Original embeddings (n, d1)
        transformed: Transformed embeddings (n, d2)
        k: Number of neighbors to consider
        
    Returns:
        Preservation rate in [0, 1]
    """
    original = original.cpu().numpy() if isinstance(original, torch.Tensor) else original
    transformed = transformed.cpu().numpy() if isinstance(transformed, torch.Tensor) else transformed
    
    n = original.shape[0]
    
    # Find k-NN in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(original)
    _, indices_orig = nbrs_orig.kneighbors(original)
    indices_orig = indices_orig[:, 1:]  # Exclude self
    
    # Find k-NN in transformed space
    nbrs_trans = NearestNeighbors(n_neighbors=k+1).fit(transformed)
    _, indices_trans = nbrs_trans.kneighbors(transformed)
    indices_trans = indices_trans[:, 1:]  # Exclude self
    
    # Compute overlap
    preservation_rates = []
    for i in range(n):
        orig_set = set(indices_orig[i])
        trans_set = set(indices_trans[i])
        overlap = len(orig_set & trans_set) / k
        preservation_rates.append(overlap)
    
    return np.mean(preservation_rates)


def compute_spectral_gap_similarity(
    eigenvalues1: np.ndarray,
    eigenvalues2: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute similarity between two spectra.
    
    Used to verify the manifold isomorphism hypothesis.
    
    Args:
        eigenvalues1: First set of eigenvalues
        eigenvalues2: Second set of eigenvalues
        k: Number of eigenvalues to compare
        
    Returns:
        Spectral similarity (cosine similarity of normalized eigenvalues)
    """
    ev1 = eigenvalues1[:k]
    ev2 = eigenvalues2[:k]
    
    # Normalize
    ev1 = ev1 / (np.linalg.norm(ev1) + 1e-10)
    ev2 = ev2 / (np.linalg.norm(ev2) + 1e-10)
    
    # Cosine similarity
    similarity = np.dot(ev1, ev2)
    
    return similarity


# ==================== Combined Evaluation ====================

def evaluate_alignment(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    original_emb1: Optional[torch.Tensor] = None,
    original_emb2: Optional[torch.Tensor] = None,
    k_values: List[int] = [1, 5, 10],
    knn_k: int = 10
) -> Dict[str, float]:
    """
    Comprehensive evaluation of alignment quality.
    
    Args:
        emb1: Aligned modality 1 embeddings
        emb2: Aligned modality 2 embeddings
        original_emb1: Original modality 1 embeddings (for structure preservation)
        original_emb2: Original modality 2 embeddings (for structure preservation)
        k_values: k values for Recall@k
        knn_k: k for k-NN preservation
        
    Returns:
        Dictionary of all metrics
    """
    results = {}
    
    # Retrieval metrics
    retrieval_metrics = compute_all_retrieval_metrics(emb1, emb2, k_values)
    results.update(retrieval_metrics)
    
    # Alignment quality
    results["CKA"] = compute_cka(emb1, emb2, kernel="linear")
    results["CKA_RBF"] = compute_cka(emb1, emb2, kernel="rbf")
    results["Procrustes"] = compute_procrustes_distance(emb1, emb2)
    
    # Structure preservation (if original embeddings provided)
    if original_emb1 is not None:
        results["KNN_Preservation_M1"] = compute_knn_preservation(
            original_emb1, emb1, k=knn_k
        )
    if original_emb2 is not None:
        results["KNN_Preservation_M2"] = compute_knn_preservation(
            original_emb2, emb2, k=knn_k
        )
    
    return results


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group metrics
    retrieval_metrics = {k: v for k, v in metrics.items() if "R@" in k or "MRR" in k}
    alignment_metrics = {k: v for k, v in metrics.items() if k in ["CKA", "CKA_RBF", "Procrustes"]}
    structure_metrics = {k: v for k, v in metrics.items() if "KNN" in k or "Spectral" in k}
    
    if retrieval_metrics:
        print("\n[Retrieval Metrics]")
        for k, v in sorted(retrieval_metrics.items()):
            print(f"  {k:20s}: {v:.4f}")
    
    if alignment_metrics:
        print("\n[Alignment Quality]")
        for k, v in sorted(alignment_metrics.items()):
            print(f"  {k:20s}: {v:.4f}")
    
    if structure_metrics:
        print("\n[Structure Preservation]")
        for k, v in sorted(structure_metrics.items()):
            print(f"  {k:20s}: {v:.4f}")
    
    print(f"\n{'='*60}\n")