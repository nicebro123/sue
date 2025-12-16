#!/usr/bin/env python
"""
Training Script for SUE Improved

Usage:
    python train.py --config configs/flickr30_config.yaml [--train] [--test] [--visualize]
    
Examples:
    # Train and evaluate
    python train.py --config configs/flickr30_config.yaml --train --test
    
    # Only evaluate with checkpoint
    python train.py --config configs/flickr30_config.yaml --test --checkpoint checkpoints/best.pth
    
    # Train with baseline k-NN graph (for comparison)
    python train.py --config configs/flickr30_config.yaml --train --test --graph-method knn
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import load_dataset
from src.trainer import SUETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='SUE Improved Training')
    
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/flickr30_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Perform training'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Perform testing'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for loading'
    )
    parser.add_argument(
        '--save-checkpoint',
        type=str,
        default=None,
        help='Path to save checkpoint'
    )
    parser.add_argument(
        '--graph-method',
        type=str,
        choices=['knn', 'quadratic_ot'],
        default=None,
        help='Override graph construction method'
    )
    parser.add_argument(
        '--laplacian-norm',
        type=str,
        choices=['unnormalized', 'symmetric', 'random_walk', 'doubly_stochastic'],
        default=None,
        help='Override Laplacian normalization method'
    )
    parser.add_argument(
        '--no-se',
        action='store_true',
        help='Skip spectral embedding stage'
    )
    parser.add_argument(
        '--no-cca',
        action='store_true',
        help='Skip CCA stage'
    )
    parser.add_argument(
        '--no-mmd',
        action='store_true',
        help='Skip MMD stage'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.graph_method:
        config['graph']['method'] = args.graph_method
        print(f"Override graph method: {args.graph_method}")
    
    if args.laplacian_norm:
        config['graph']['laplacian']['normalization'] = args.laplacian_norm
        print(f"Override Laplacian normalization: {args.laplacian_norm}")
    
    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Load data
    print("\nLoading dataset...")
    train_set, test_set = load_dataset(config)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SUETrainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Training
    if args.train:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        start_time = datetime.now()
        
        trainer.fit(
            train_set,
            with_se=not args.no_se,
            with_cca=not args.no_cca,
            with_mmd=not args.no_mmd
        )
        
        elapsed = datetime.now() - start_time
        print(f"\nTraining completed in {elapsed}")
        
        # Save checkpoint
        if args.save_checkpoint:
            os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
            trainer.save_checkpoint(args.save_checkpoint)
        else:
            # Default save path
            checkpoint_dir = config.get('checkpoint', {}).get('save_path', './checkpoints/')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
            trainer.save_checkpoint(checkpoint_path)
    
    # Testing
    if args.test or not args.train:
        print("\n" + "="*60)
        print("Evaluation")
        print("="*60)
        
        metrics = trainer.evaluate(test_set, verbose=True)
    
    # Visualization
    if args.visualize:
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.visualize(test_set, prefix=f"{timestamp}_")
    
    print("\nDone!")


if __name__ == "__main__":
    main()