# SpatioCAD

## Overview
SpatioCAD (Spatially variable gene identification using Context-Aware graph Diffusion model) is a method that identifies spatially variable genes within highly heterogeneous tissues. It is a generalization of the classic graph diffusion model that deconfounds the influence of cellular heterogeneity, enabling the identification of meaningful spatial gene patterns.



## Installation

1. Clone the repository
```bash
git clone https://github.com/kotone-429/SpatioCAD.git
cd SpatioCAD

```

2. Create and activate the conda environment
```bash
conda create -n SpatioCAD python==3.9
conda activate SpatioCAD

```

3. Install dependencies
```bash
pip install scanpy[leiden] scipy numpy pandas anndata matplotlib scikit-learn

```



## Quick Start
```python
from SpatioCAD import SpatioCAD

# Initialize and load data
scad = SpatioCAD()
scad.read_h5ad(file = 'your_data.h5ad')
scad.preprocess()

# Construct the spatial graph
scad.compute_adjacency_matrix()

# Filter the noise genes
scad.filter_non_pattern_genes()

# Identify spatially variable genes (SVGs)
final_df = scad.compute_SVG(power_values = [1,1.5,2], num_eigen= 50, cutoff_error= 15.0)
final_df.to_csv('svg_result.csv')

# Cluster the SVGs
scad.cluster_gene(n_top_genes = 2000)
scad.compute_cluster_patterns()

```
