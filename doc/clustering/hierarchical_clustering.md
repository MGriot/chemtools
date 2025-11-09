# Hierarchical Clustering

Hierarchical clustering is an algorithm that builds a hierarchy of clusters. This method starts with each data point in its own cluster and then iteratively merges the closest pairs of clusters until only one cluster (or a specified number of clusters) is left. The result is a tree-based representation of the objects, called a dendrogram.

## Usage

```python
from chemtools.clustering import HierarchicalClustering
from chemtools.plots.clustering import DendrogramPlotter

# Create and fit the model
model = HierarchicalClustering(X, 
                             labels=labels,
                             linkage="average",
                             ordering_method="recursive")
model.fit()

# Create dendrogram plot
plotter = DendrogramPlotter(library="matplotlib", 
                           theme="light", 
                           style_preset="default")
plotter.plot_dendrogram(model, 
                       orientation="top",
                       color_threshold=0.5)
```

## Parameters

### HierarchicalClustering
- `data_or_processor`: Input data array or DistanceMatrix object
- `labels`: Optional list of labels for data points
- `linkage`: Linkage method ("single", "complete", "average")
- `ordering_method`: Method for ordering leaves ("original", "recursive", "olo", "tsp")
- `max_merges`: Maximum number of merge operations

## Key Features

1. **Multiple Linkage Methods**
   - Single linkage (nearest neighbor)
   - Complete linkage (furthest neighbor)
   - Average linkage (UPGMA)

2. **Flexible Ordering Options**
   - Original order preservation
   - Recursive leaf ordering
   - Optimal leaf ordering
   - Traveling salesman problem-based ordering

3. **Visualization Options**
   - Multiple orientations (top, bottom, left, right)
   - Color thresholds for cluster identification
   - Customizable styles and themes
   - Interactive plotting with Plotly support

## API Reference

```python
class HierarchicalClustering:
    def __init__(self, data_or_processor, labels=None, linkage="average",
                 ordering_method="recursive", max_merges=None)
    def fit(self) -> "HierarchicalClustering"
    
class DendrogramPlotter:
    def plot_dendrogram(self, model, orientation="top", color_threshold=None)
```

## Citing Sources
Further reading can be found on Wikipedia:
- [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)