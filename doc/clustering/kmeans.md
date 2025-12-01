# k-Means Clustering

k-Means clustering is one of the most popular unsupervised machine learning algorithms. It aims to partition a set of `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid). The `KMeans` class in `chemtools` provides a straightforward implementation of this algorithm.

## How k-Means Works

The algorithm works iteratively to assign each data point to one of `k` groups based on the features that are provided. The main steps are:
1.  **Initialization**: Randomly select `k` initial centroids from the data points.
2.  **Assignment**: Assign each data point to its nearest centroid, based on a distance metric (commonly Euclidean distance). This forms `k` clusters.
3.  **Update**: Recalculate the centroid of each cluster by taking the mean of all data points assigned to that cluster.
4.  **Iteration**: Repeat the assignment and update steps until the centroids no longer change significantly, or a maximum number of iterations is reached.

## Usage

Here is a basic example of how to use the `KMeans` class.

```python
from chemtools.clustering import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 1. Sample Data
np.random.seed(42)
# Create two distinct clusters
cluster1 = np.random.normal(loc=0, scale=1, size=(50, 2))
cluster2 = np.random.normal(loc=5, scale=1.5, size=(50, 2))
X = np.vstack([cluster1, cluster2])

# 2. Initialize and fit the KMeans model
# We expect 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# 3. Get the results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], c='blue', label='Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='red', label='Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('k-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# 5. Predict new data points
X_new = np.array([[0, 0], [6, 6]])
predicted_labels = kmeans.predict(X_new)
print(f"Prediction for [0, 0]: Cluster {predicted_labels[0]}")
print(f"Prediction for [6, 6]: Cluster {predicted_labels[1]}")
```

## API Reference

### `KMeans` Class

```python
class KMeans(BaseModel):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None)
    def fit(self, X, variables_names=None, objects_names=None)
    def predict(self, X_new)
    @property
    def summary(self) -> str
```

### Parameters & Attributes
-   **`n_clusters` (int)**: The number of clusters to form (`k`). Defaults to `8`.
-   **`max_iter` (int)**: Maximum number of iterations for a single run. Defaults to `300`.
-   **`random_state` (int, optional)**: Seed for the random number generator for centroid initialization, ensuring reproducibility.
-   **`cluster_centers_` (np.ndarray)**: After fitting, this attribute stores the coordinates of the cluster centers.
-   **`labels_` (np.ndarray)**: After fitting, this attribute contains the cluster label for each input data point.

---

## Further Reading

For a more detailed mathematical explanation, please refer to the Wikipedia article:
-   [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
