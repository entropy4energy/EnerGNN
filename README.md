# EnerGNN

A **Graph Neural Network (GNN)** model for predicting crystal structure properties including energy, forces, and stress tensors. Built on [PyTorch](https://pytorch.org) and [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/), with support for large-scale materials datasets including [Matbench Discovery](https://github.com/janosh/matbench-discovery) and Alexandria.

## Overview

EnerGNN implements a custom message-passing graph neural network designed to:
- **Predict total energy** of crystal structures
- **Calculate atomic forces** through automatic differentiation
- **Estimate stress tensors** for materials
- **Handle periodic boundary conditions** with lattice information

The model processes crystal structure graphs where:
- Nodes represent atoms (with atomic number and fractional coordinates)
- Edges represent atomic bonds (with learnable edge features)
- Graph-level properties are predicted (energy, stress)

## Project Structure

```
EnerGNN/
├── energnn/                    # Main package
│   ├── models.py              # GNN model architectures (EnerGDev, EnerGMP, custom layers)
│   ├── datasets.py            # Dataset loaders (Alexandria, WBM from matbench-discovery)
│   ├── utils.py               # Utility functions (path handling, etc.)
│   └── __init__.py
├── main.py                    # Entry point / example usage
├── main.ipynb                 # Interactive notebook (data loading, training, evaluation)
├── pyproject.toml             # Project dependencies and metadata
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python >= 3.12, < 3.13
- CUDA 12.6 (for GPU support)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /root/Projects/EnerGNN
   ```

2. **Install using uv package manager:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Install matbench-discovery manually:**
   ```bash
   git clone https://github.com/janosh/matbench-discovery --depth 1
   uv pip install -e ./matbench-discovery/
   ```

> The project uses **uv** as the package manager. See `pyproject.toml` for full dependency list.

## Usage

### Loading Data

```python
import energnn.datasets as datasets

# Load Alexandria dataset
train_data = datasets.alexandria(load_files=['000', '001', '002'], cutoff=8.0)

# Load WBM (Wyckoff Barometric Machine) dataset
test_data = datasets.wbm(cutoff=6.0)
```

### Creating and Training a Model

```python
import torch
import torch.nn as nn
from energnn.models import EnerGDev
from torch_geometric.data import DataLoader

# Initialize model
model = EnerGDev()
model = model.cuda()

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        batch = batch.cuda()
        pred = model(batch)
        loss = loss_fn(pred.squeeze(), batch.y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### Computing Forces

```python
# Forces can be computed via automatic differentiation
# Requires model output to enable gradients for atomic positions

test_batch = next(iter(DataLoader(test_data[:4], batch_size=1)))
test_batch.x.requires_grad_(True)

energy = model(test_batch)
energy.backward()

forces = test_batch.x.grad[:, 1:]  # Extract force components
print(f"Forces shape: {forces.shape}")
```

## Key Components

### Models (`energnn/models.py`)

- **EnerGDev**: Main energy prediction model with 5 graph convolution layers
- **EnerGMP**: Custom message-passing convolution layer
- **LeakySiLU**: Custom activation function
- **SoftAbs**: Differentiable absolute value activation

### Datasets (`energnn/datasets.py`)

- **`alexandria()`**: Load Alexandria database (50 JSON files with pre-computed structures)
  - Parameters: `load_files` (list of file indices), `cutoff` (bond cutoff distance)
  - Returns: List of PyG `Data` objects with node features, edge indices, lattice matrices

- **`wbm()`**: Load WBM dataset from matbench-discovery
  - Returns: Structures with computed energies from ab-initio methods

### Data Format

Each `Data` object contains:
- `x`: Node features `[n_atoms, 4]` — (atomic_number, frac_x, frac_y, frac_z)
- `edge_index`: Graph connectivity `[2, n_edges]`
- `edge_attr`: Edge features (Cartesian displacement vectors)
- `matrix`: Lattice matrix `[3, 3]`
- `y`: Target energy (eV)
- `stress`: Stress tensor (optional)
- `xyz`: Absolute coordinates `[n_atoms, 3]`

## Interactive Development

Open `main.ipynb` in Jupyter/JupyterLab for:
- Step-by-step data loading and exploration
- Model architecture visualization
- Training and evaluation workflows
- Force and stress tensor calculations
- Performance metrics and plotting

## Dependencies

Key packages:
- **torch** — Deep learning framework
- **torch-geometric (PyG)** — Graph neural network layers
- **matbench-discovery** — Materials datasets
- **pymatgen** — Crystal structure tools
- **pymatviz** — Visualization utilities
- **wandb** — Experiment tracking (optional)

Full dependency list available in `pyproject.toml`.

## Performance Targets

- **Energy prediction MAE**: < 0.05 eV/atom
- **Force prediction MAE**: < 0.10 eV/Å
- **Stress tensor RMSE**: < 0.01 GPa

(These targets depend on dataset, model hyperparameters, and training epochs)

## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Matbench Discovery](https://github.com/janosh/matbench-discovery)
- [Crystal Structure Representation Learning](https://arxiv.org/abs/2202.02044)

## License

See repository for license details.

## Development Notes

- GPU acceleration via CUDA 12.6 (configurable in `pyproject.toml`)
- Hydration and dehydration of graph data handled automatically
- Fractional-to-absolute coordinate conversion implemented in `EnerGDev.forward()`