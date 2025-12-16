"""
Data Loading Module for SUE Improved

Handles loading pre-encoded features and creating weakly parallel data.
"""

import os
import torch
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal embeddings.
    
    Handles both paired and unpaired data scenarios.
    """
    
    def __init__(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        is_paired: bool = True
    ):
        """
        Args:
            embeddings1: Embeddings for modality 1 (n1, d1)
            embeddings2: Embeddings for modality 2 (n2, d2)
            is_paired: Whether the data is paired (n1 == n2 with correspondence)
        """
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.is_paired = is_paired
        
        if is_paired:
            assert len(embeddings1) == len(embeddings2), \
                "Paired data must have same length"
    
    def __len__(self) -> int:
        return len(self.embeddings1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_paired:
            return self.embeddings1[idx], self.embeddings2[idx]
        else:
            # For unpaired, return random sample from modality 2
            idx2 = np.random.randint(len(self.embeddings2))
            return self.embeddings1[idx], self.embeddings2[idx2]


def load_encoded_data(data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load pre-encoded features from disk.
    
    Args:
        data_path: Path to directory containing encoded1.pt and encoded2.pt
        
    Returns:
        encoded1: Tensor of modality 1 embeddings
        encoded2: Tensor of modality 2 embeddings
    """
    encoded1_path = os.path.join(data_path, "encoded1.pt")
    encoded2_path = os.path.join(data_path, "encoded2.pt")
    
    if not os.path.exists(encoded1_path):
        raise FileNotFoundError(f"Cannot find {encoded1_path}")
    if not os.path.exists(encoded2_path):
        raise FileNotFoundError(f"Cannot find {encoded2_path}")
    
    encoded1 = torch.load(encoded1_path, map_location='cpu')
    encoded2 = torch.load(encoded2_path, map_location='cpu')
    
    print(f"Loaded modality 1: {encoded1.shape}")
    print(f"Loaded modality 2: {encoded2.shape}")
    
    return encoded1, encoded2


def train_test_split(
    encoded1: torch.Tensor,
    encoded2: torch.Tensor,
    n_test: int = 400
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data into train and test sets.
    
    Test samples are taken from the beginning of the dataset.
    
    Args:
        encoded1: Modality 1 embeddings
        encoded2: Modality 2 embeddings
        n_test: Number of test samples
        
    Returns:
        train_set: (train_emb1, train_emb2)
        test_set: (test_emb1, test_emb2)
    """
    if encoded1.shape[0] != encoded2.shape[0]:
        raise ValueError("Both modalities must have the same number of samples")
    
    n_samples = encoded1.shape[0]
    
    test_set = (encoded1[:n_test], encoded2[:n_test])
    train_set = (encoded1[n_test:], encoded2[n_test:])
    
    print(f"Train set size: {train_set[0].shape[0]}")
    print(f"Test set size: {test_set[0].shape[0]}")
    
    return train_set, test_set


def create_weakly_parallel_data(
    train_set: Tuple[torch.Tensor, torch.Tensor],
    n_parallel: int,
    removal_percentage: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create weakly parallel data by keeping only n_parallel paired samples.
    
    The last n_parallel samples remain paired (for CCA).
    The rest are shuffled to remove correspondence.
    
    Args:
        train_set: (embeddings1, embeddings2)
        n_parallel: Number of paired samples to keep
        removal_percentage: Percentage of non-parallel samples to remove
        seed: Random seed
        
    Returns:
        (weakly_parallel_emb1, weakly_parallel_emb2)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    encoded1, encoded2 = train_set
    n_total = encoded1.shape[0]
    n_unparallel = n_total - n_parallel
    
    if n_parallel > n_total:
        raise ValueError(f"n_parallel ({n_parallel}) cannot exceed total samples ({n_total})")
    
    n_remove = int(removal_percentage * n_unparallel)
    
    # Remove random samples from the non-parallel portion
    indices_to_remove_1 = torch.randperm(n_unparallel)[:n_remove]
    indices_to_remove_2 = torch.randperm(n_unparallel)[:n_remove]
    
    mask1 = torch.ones(n_unparallel, dtype=torch.bool)
    mask1[indices_to_remove_1] = False
    encoded1_unparallel = encoded1[:n_unparallel][mask1]
    
    mask2 = torch.ones(n_unparallel, dtype=torch.bool)
    mask2[indices_to_remove_2] = False
    encoded2_unparallel = encoded2[:n_unparallel][mask2]
    
    # Shuffle the second modality to remove alignment
    shuffle_idx = torch.randperm(encoded2_unparallel.shape[0])
    encoded2_unparallel = encoded2_unparallel[shuffle_idx]
    
    # Keep the last n_parallel samples paired
    encoded1_parallel = encoded1[n_unparallel:]
    encoded2_parallel = encoded2[n_unparallel:]
    
    # Concatenate
    encoded1_weak = torch.cat([encoded1_unparallel, encoded1_parallel], dim=0)
    encoded2_weak = torch.cat([encoded2_unparallel, encoded2_parallel], dim=0)
    
    print(f"Created weakly parallel data:")
    print(f"  Non-parallel samples: {encoded1_unparallel.shape[0]} (mod1), {encoded2_unparallel.shape[0]} (mod2)")
    print(f"  Parallel samples: {n_parallel}")
    
    return encoded1_weak, encoded2_weak


def load_dataset(
    config: Dict
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load and prepare dataset according to config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_set: (train_emb1, train_emb2) with weak parallelism
        test_set: (test_emb1, test_emb2) fully paired for evaluation
    """
    data_config = config.get("data", {})
    data_path = data_config.get("data_path", "../data/flickr30/")
    n_test = data_config.get("n_test", 400)
    n_parallel = data_config.get("n_parallel", 500)
    
    # Load raw data
    encoded1, encoded2 = load_encoded_data(data_path)
    
    # Split into train/test
    train_set, test_set = train_test_split(encoded1, encoded2, n_test=n_test)
    
    # Create weakly parallel training data
    train_set = create_weakly_parallel_data(train_set, n_parallel=n_parallel)
    
    return train_set, test_set


def get_parallel_data(
    train_set: Tuple[torch.Tensor, torch.Tensor],
    n_parallel: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract parallel (paired) portion from training set.
    
    The parallel samples are at the end of the training set.
    
    Args:
        train_set: (embeddings1, embeddings2)
        n_parallel: Number of parallel samples
        
    Returns:
        (parallel_emb1, parallel_emb2)
    """
    emb1, emb2 = train_set
    return emb1[-n_parallel:], emb2[-n_parallel:]


def get_non_parallel_data(
    train_set: Tuple[torch.Tensor, torch.Tensor],
    n_parallel: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract non-parallel portion from training set.
    
    Args:
        train_set: (embeddings1, embeddings2)
        n_parallel: Number of parallel samples (to exclude)
        
    Returns:
        (non_parallel_emb1, non_parallel_emb2)
    """
    emb1, emb2 = train_set
    n1 = emb1.shape[0] - n_parallel
    n2 = emb2.shape[0] - n_parallel
    return emb1[:n1], emb2[:n2]


class ModalityDataLoader:
    """
    DataLoader wrapper for single modality training (e.g., SpectralNet).
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        n = len(self.embeddings)
        indices = torch.randperm(n) if self.shuffle else torch.arange(n)
        
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:min(end, n)]
            yield self.embeddings[batch_idx]
    
    def __len__(self):
        n = len(self.embeddings)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size