"""Utility functions module."""

from .graph import (
    GraphConstructor,
    AdaptiveGraphConstructor,
    create_graph_constructor
)
from .metrics import (
    compute_similarity_matrix,
    compute_recall_at_k,
    compute_mrr,
    compute_map,
    compute_all_retrieval_metrics,
    compute_cka,
    compute_procrustes_distance,
    compute_knn_preservation,
    compute_spectral_gap_similarity,
    evaluate_alignment,
    print_metrics
)
from .visualization import (
    visualize_embeddings,
    visualize_similarity_matrix,
    visualize_eigenvalue_spectrum,
    visualize_training_curves,
    visualize_affinity_matrix,
    visualize_retrieval_results,
    VisualizationManager
)

__all__ = [
    # Graph
    'GraphConstructor',
    'AdaptiveGraphConstructor',
    'create_graph_constructor',
    # Metrics
    'compute_similarity_matrix',
    'compute_recall_at_k',
    'compute_mrr',
    'compute_map',
    'compute_all_retrieval_metrics',
    'compute_cka',
    'compute_procrustes_distance',
    'compute_knn_preservation',
    'compute_spectral_gap_similarity',
    'evaluate_alignment',
    'print_metrics',
    # Visualization
    'visualize_embeddings',
    'visualize_similarity_matrix',
    'visualize_eigenvalue_spectrum',
    'visualize_training_curves',
    'visualize_affinity_matrix',
    'visualize_retrieval_results',
    'VisualizationManager'
]