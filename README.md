# RicciGraphDTA

RicciGraphDTA is a graph neural network-based tool for drug-target affinity prediction. Unlike standard GNNs, it integrates **graph curvature information (Ricci curvature)** to enhance molecular graph representation.
---

## Dependencies
 
- **PyTorch**  
- **PyTorch Geometric** and its submodules:
  - `torch-scatter`
  - `torch-sparse`
  - `torch-cluster`
  - `torch-spline-conv`
  - `torch-geometric`
- **RDKit** (for molecular structure processing)
