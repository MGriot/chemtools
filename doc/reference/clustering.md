# Clustering Module Reference (`chemtools.clustering`)

The `chemtools.clustering` module provides implementations of various unsupervised clustering algorithms, including k-Means and Hierarchical Clustering, to discover inherent groupings within your data.

---

## `KMeans` Class

Implements the k-Means clustering algorithm.

### `KMeans(n_clusters=8, max_iter=300, random_state=None)`

*   **Parameters:**
    *   `n_clusters` (`int`, optional): The number of clusters (`k`) to form. Defaults to `8`.
    *   `max_iter` (`int`, optional): Maximum number of iterations for a single run. Defaults to `300`.
    *   `random_state` (`int`, optional): Seed for the random number generator for centroid initialization, ensuring reproducibility.

### Methods

*   **`fit(self, X, variables_names=None, objects_names=None)`**
    *   Computes k-Means clustering.
    *   **Parameters:**
        *   `X` (`np.ndarray`): Input data.
        *   `variables_names` (`list`, optional): Names for the features.
        *   `objects_names` (`list`, optional): Names for the observations.

*   **`predict(self, X_new) -> np.ndarray`**
    *   Predicts the cluster labels for new data points.
    *   **Parameters:** `X_new` (`np.ndarray`): New data points to assign to clusters.
    *   **Returns:** `np.ndarray`: Predicted cluster labels.

*   **`summary(self) -> str` (property)**
    *   Returns a summary of the fitted model.

### Attributes

*   **`cluster_centers_` (`np.ndarray`):** Coordinates of the cluster centers.
*   **`labels_` (`np.ndarray`):** Cluster label for each input data point.

### Usage Example (k-Means)

```python
from chemtools.clustering import KMeans
import numpy as np

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
print(f"Cluster labels: {kmeans.labels_}")
print(f"Cluster centers: {kmeans.cluster_centers_}")
```

---

<h2><code>HierarchicalClustering</code> Class</h2>

Performs hierarchical clustering analysis, building a tree-like dendrogram of clusters.

<h3><code>HierarchicalClustering(data_or_processor, labels=None, linkage="average", ordering_method="recursive", max_merges=None)</code></h3>

*   <b>Parameters:</b>
    *   `data_or_processor` (`np.ndarray` or `DistanceMatrix`): Input data array or a pre-calculated distance matrix.
    *   `labels` (`list`, optional): Labels for data points.
    *   `linkage` (`str`, optional): Linkage method (e.g., `"single"`, `"complete"`, `"average"`). Defaults to `"average"`.
    *   `ordering_method` (`str`, optional): Method for ordering leaves in the dendrogram (e.g., `"original"`, `"recursive"`, `"olo"`, `"tsp"`). Defaults to `"recursive"`.
    *   `max_merges` (`int`, optional): Maximum number of merge operations.

<h3>Methods</h3>

*   <b><code>fit(self) -> "HierarchicalClustering"</code></b>
    *   Performs the hierarchical clustering.
    *   <b>Returns:</b> `self` (fitted instance).

<h3>Key Features</h3>

*   <b>Multiple Linkage Methods:</b> Supports "single", "complete", "average".
*   <b>Flexible Ordering Options:</b> Allows different methods for ordering leaves in the dendrogram.

<h3>Usage Example (Hierarchical Clustering)</h3>

```python
from chemtools.clustering import HierarchicalClustering
import numpy as np

X = np.random.rand(10, 4) # Sample data
model = HierarchicalClustering(X, linkage="average", ordering_method="recursive")
model.fit()
# The clustering results (linkage matrix) are stored internally after fit
# These results can then be passed to a DendrogramPlotter.
```

---

<h2>Plotting Dendrograms</h2>

The `chemtools.plots.clustering.DendrogramPlotter` class is used to visualize the results of hierarchical clustering.

<h3><code>DendrogramPlotter()</code></h3>

*   <b>Parameters:</b> None (inherits styling from `BasePlotter`).

<h3>Methods</h3>

*   <b><code>plot_dendrogram(self, model, orientation="top", color_threshold=None, **kwargs)</code></b>
    *   Generates a dendrogram plot.
    *   <b>Parameters:</b>
        *   `model`: A fitted `HierarchicalClustering` object.
        *   `orientation` (`str`, optional): Orientation of the dendrogram (e.g., `"top"`, `"bottom"`, `"left"`, `"right"`). Defaults to `"top"`.
        *   `color_threshold` (`float`, optional): Distance threshold for coloring clusters.
        *   `**kwargs`: Additional plotting arguments.

<h3>Example Output (Dendrogram)</h3>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/plots/clustering/dendrogram_classic_professional_dark.png">
  <img alt="Dendrogram" src="../img/plots/clustering/dendrogram_classic_professional_light.png">
</picture>

---

This reference provides an overview of the Clustering module. Each class and function has more detailed documentation within the code itself, accessible through docstrings.
