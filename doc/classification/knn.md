# k-Nearest Neighbors (k-NN)

The k-Nearest Neighbors (k-NN) algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. It is one of the simplest machine learning algorithms and can be used for both classification and regression. The `KNearestNeighbors` class in `chemtools` provides an implementation for classification.

## How k-NN Works

The core idea of k-NN is straightforward:
1.  **Training (Memorization)**: In the training phase, the algorithm simply stores the entire training dataset. No explicit model is built.
2.  **Prediction**: To classify a new, unseen data point, the algorithm does the following:
    *   It calculates the distance (commonly Euclidean distance) from the new point to every point in the training dataset.
    *   It identifies the `k` points in the training data that are nearest to the new point (its "neighbors").
    *   It assigns the class to the new point based on a majority vote among its `k` neighbors. The class that appears most frequently among the neighbors is chosen as the prediction.

## Usage

Here is a basic example of how to use the `KNearestNeighbors` class.

```python
from chemtools.classification import KNearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate Sample Data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                           random_state=42)

# 2. Initialize and fit the k-NN model
knn = KNearestNeighbors(n_neighbors=5)
knn.fit(X, y)

# 3. Create a new data point to classify
X_new = np.array([[0, 0]])

# 4. Predict the class of the new point
predicted_label = knn.predict(X_new)
print(f"Prediction for point [0, 0]: Class {predicted_label[0]}")

# 5. Visualize the data and prediction
plt.figure(figsize=(8, 6))
# Plot training data
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')
# Plot the new point
plt.scatter(X_new[:, 0], X_new[:, 1], c='green', marker='X', s=200, label=f'New Point (Predicted: Class {predicted_label[0]})')

plt.title('k-NN Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

## API Reference

### `KNearestNeighbors` Class

```python
class KNearestNeighbors(BaseModel):
    def __init__(self, n_neighbors=5)
    def fit(self, X, y, variables_names=None, objects_names=None)
    def predict(self, X_new)
    @property
    def summary(self) -> str
```

### Parameters & Attributes
-   **`n_neighbors` (int)**: The number of neighbors to use for classification (`k`). Defaults to `5`.
-   **`X_train` (np.ndarray)**: After fitting, this stores the training feature data.
-   **`y_train` (np.ndarray)**: After fitting, this stores the training class labels.

---

## Further Reading

For a more detailed explanation, please refer to the Wikipedia article:
-   [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
