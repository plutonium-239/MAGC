# Attributed Graph Clustering via Modularity Aided Coarsening

Submitted to TMLR

### Instructions to run

This repository consists of 4 main files: `q-fgc.py`, `q-gcn.py`, `q-vgae.py` and `q-gmm-vgae.py` corresponding to the Q-FGC, Q-GCN, Q-VGAE and Q-GMM-VGAE methods. These are intended to be run directly.

To specify the dataset, `set` it as an environment variable.
For eg.
In Linux: 

```bash
dataset=CiteSeer python q-vgae.py --alpha=1000 --beta=100
```

And in Windows:
```bash
set dataset=CiteSeer
python q-vgae.py --alpha=1000 --beta=100
```

(equivalently, you can use `export` in Linux).

For small datasets (Airports), since there are very less number of samples (~100-300), deep learning methods are more sensitive to random state, whereas Q-FGC is not.

Note that all the datasets are sensitive to some extent as the loss function used is non-convex.

Make sure to have [PyTorch 1.8+ installed, preferably 2.0+](https://pytorch.org/get-started/locally/), and [PyTorch Geometric 2.0+](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Then you can run 

```bash
pip install -r requirements.txt
```

