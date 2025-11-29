# Dendrogram Plot

A dendrogram is a tree-like diagram that records the sequences of merges or splits in a hierarchical clustering analysis. It's the primary way to visualize the output of this type of clustering, showing how clusters are related to one another and at what distance the merges occur.

## `plot_dendrogram`

This method generates the dendrogram from a fitted `HierarchicalClustering` model object.

### Usage
```python
from chemtools.plots.clustering import DendrogramPlotter
from chemtools.clustering import HierarchicalClustering
import numpy as np

# Sample Data
X = np.random.rand(10, 4)

# Fit model
model = HierarchicalClustering(X)
model.fit()

# Create Plot
plotter = DendrogramPlotter(theme='classic_professional_light')
fig = plotter.plot_dendrogram(model, title="Hierarchical Clustering Dendrogram")
fig.savefig("dendrogram.png")
```

### Parameters
- `model`: A fitted `HierarchicalClustering` object.
- `orientation` (str, optional): The direction to plot the dendrogram. Can be `'top'` (default), `'bottom'`, `'left'`, or `'right'`.
- `labels` (list, optional): A list of strings to use as labels for the leaves of the dendrogram.
- `color_threshold` (float, optional): The threshold to use for coloring clusters. All links below this value will have a unique color per cluster.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/clustering/dendrogram_classic_professional_dark.png">
  <img alt="Dendrogram" src="../../img/plots/clustering/dendrogram_classic_professional_light.png">
</picture>
