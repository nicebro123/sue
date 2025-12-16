# SUE Improved

Spectral Universal Embedding with theoretical improvements based on optimal transport and spectral graph theory.

## Innovations

### Innovation 1: Density-Adaptive Graph Construction (Quadratic OT)

Traditional k-NN graph construction uses a fixed number of neighbors for all points, which fails to adapt to varying local data density. Our approach uses **quadratic regularized optimal transport** to automatically determine adaptive neighborhoods:

$$W^* = \arg\min_W \langle W, C \rangle + \frac{\varepsilon}{2}\|W\|_F^2 \quad \text{s.t.} \quad W\mathbf{1} = \mu, W \geq 0$$

**Key benefits:**
- Automatically adapts to local data density
- Produces sparse solutions (unlike entropic OT)
- Better captures the intrinsic manifold structure

### Innovation 2: Doubly Stochastic Laplacian (Sinkhorn Normalization)

The standard unnormalized Laplacian $L = D - W$ is biased by degree distribution, which can distort the spectral embedding. We use **Sinkhorn normalization** to create a doubly stochastic affinity matrix:

$$\tilde{W} = \text{Sinkhorn}(W) \quad \text{s.t.} \quad \tilde{W}\mathbf{1} = \mathbf{1}, \tilde{W}^\top\mathbf{1} = \mathbf{1}$$

**Key benefits:**
- Eliminates degree bias
- Better convergence to the Laplace-Beltrami operator
- Theoretically grounded normalization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sue_improved

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The code expects pre-encoded features in the data directory:
```
data/flickr30/
├── encoded1.pt  # Modality 1 embeddings (e.g., DINO image features)
└── encoded2.pt  # Modality 2 embeddings (e.g., Sentence-BERT text features)
```

## Usage

### Training with Innovations (Recommended)

```bash
cd scripts
python train.py --config ../configs/flickr30_config.yaml --train --test --visualize
```

### Comparison with Baseline (k-NN + Unnormalized Laplacian)

```bash
python train.py --config ../configs/flickr30_config.yaml --train --test \
    --graph-method knn --laplacian-norm unnormalized
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--train` | Perform training |
| `--test` | Perform evaluation |
| `--visualize` | Generate visualizations |
| `--graph-method` | `knn` (baseline) or `quadratic_ot` (Innovation 1) |
| `--laplacian-norm` | `unnormalized`, `symmetric`, `random_walk`, or `doubly_stochastic` (Innovation 2) |
| `--checkpoint` | Path to load checkpoint |
| `--save-checkpoint` | Path to save checkpoint |

## Configuration

Edit `configs/flickr30_config.yaml` to customize:

```yaml
# Graph construction (Innovation 1)
graph:
  method: "quadratic_ot"  # or "knn"
  quadratic_ot:
    epsilon: 0.1
    max_iter: 100
  
  # Laplacian normalization (Innovation 2)
  laplacian:
    normalization: "doubly_stochastic"  # or "unnormalized"
    sinkhorn_iter: 50
```

## Project Structure

```
sue_improved/
├── configs/
│   └── flickr30_config.yaml     # Configuration
├── src/
│   ├── data/
│   │   └── dataset.py           # Data loading
│   ├── models/
│   │   ├── spectral_embedding.py # SpectralNet model
│   │   └── alignment.py         # Alignment methods
│   ├── losses/
│   │   └── spectral_loss.py     # Loss functions
│   ├── utils/
│   │   ├── graph.py             # Graph construction (Innovations 1 & 2)
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── visualization.py     # Visualization tools
│   └── trainer.py               # Training orchestration
├── scripts/
│   └── train.py                 # Training entry point
└── requirements.txt
```

## Evaluation Metrics

The code computes:

**Retrieval Metrics:**
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (mAP)

**Alignment Quality:**
- Centered Kernel Alignment (CKA)
- Procrustes distance

**Structure Preservation:**
- k-NN preservation rate

## Theoretical Foundation

Our improvements are based on the **Manifold Semantic Isomorphism Hypothesis**:

> Different modalities' data lie on low-dimensional manifolds that are isomorphic in their intrinsic geometry.

This hypothesis leads to three key requirements:
1. **Accurate spectral extraction** → Innovation 1 (better graph) + Innovation 2 (better Laplacian)
2. **Structure matching** → Future Innovation 3 (Gromov-Wasserstein)
3. **Distribution alignment** → MMD with structure preservation

## Citation

If you use this code, please cite:

```bibtex
@article{sue_improved,
  title={SUE Improved: Density-Adaptive Spectral Embedding for Cross-Modal Alignment},
  author={...},
  year={2024}
}
```

## License

MIT License