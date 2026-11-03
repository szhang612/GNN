# Graph Convolutional Network on Cora

A two-layer GCN built from scratch with PyTorch Geometric's `MessagePassing`, trained for node classification on the Cora citation graph. **Test accuracy: 80.6%.**

## Model

Each layer updates node features as

$$
x_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}} \left( W \cdot x_j^{(k-1)} \right) + b
$$

with learnable $W$ and $b$. Architecture: `GCNConv(1433 → 32) → ReLU → GCNConv(32 → 7)`.

## Setup

- **Data:** Cora (2,708 nodes, 7 classes, 140 training nodes), downloaded automatically via `Planetoid`
- **Training:** Adam, lr 0.01, weight decay 5e-4, 2000 epochs, cross-entropy loss

## Files

- [`GNN.ipynb`](GNN.ipynb): main notebook

## References

- [PyTorch Geometric: Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
- MIT 6.7960 (Deep Learning), Problem Set 2
- Kipf & Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR 2017

## Use of AI

The model, training, and evaluation code were written by me. Claude (Anthropic) was used to organize the repository and draft this README; it did not write or modify the implementation.
