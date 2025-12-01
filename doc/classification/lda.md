# Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a supervised machine learning algorithm used for both classification and dimensionality reduction. It is particularly effective for multi-class classification problems. The primary goal of LDA is to find a lower-dimensional space that maximizes the separability between classes.

The `LinearDiscriminantAnalysis` class in `chemtools` provides a robust implementation of this technique.

## How LDA Works

LDA finds a set of linear combinations of the input features, known as "discriminant functions," that best separate the different classes in the dataset. This is achieved by maximizing the ratio of between-class variance to the within-class variance. The main steps are:

1.  **Calculate Class Means**: Compute the mean vector for each class.
2.  **Compute Scatter Matrices**:
    *   **Within-Class Scatter Matrix (S_W)**: Measures the spread of data points within each class.
    *   **Between-Class Scatter Matrix (S_B)**: Measures the spread of data points between different classes.
3.  **Solve the Eigenvalue Problem**: Solve the generalized eigenvalue problem for the matrix `inv(S_W) * S_B`. The resulting eigenvectors represent the directions of maximum class separability (the discriminant functions), and the eigenvalues represent the amount of variance explained by each discriminant function.
4.  **Dimensionality Reduction**: The original data is projected onto the new space defined by the top `k` eigenvectors, where `k` is typically the number of classes minus one.

## Usage

Here is a basic example of how to use the `LinearDiscriminantAnalysis` class.

```python
from chemtools.classification import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate Sample Data
X, y = make_classification(n_samples=150, n_features=4, n_informative=2, 
                           n_redundant=0, n_classes=3, n_clusters_per_class=1, 
                           random_state=42)

# 2. Initialize and fit the LDA model
# We will reduce the dimensionality to 2 components (since we have 3 classes)
lda = LinearDiscriminantAnalysis(n_components=2)
X_transformed = lda.fit_transform(X, y)

# 3. Print a summary
print(lda.summary)

# 4. Visualize the transformed data
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, color in zip(np.unique(y), colors):
    plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], c=color, label=f'Class {i}')

plt.title('LDA Transformed Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True)
plt.show()

# 5. Predict the class of new data points
X_new = np.array([[0, 0, 0, 0], [5, 5, 5, 5]])
predicted_labels = lda.predict(X_new)
print(f"Prediction for [0, 0, 0, 0]: Class {predicted_labels[0]}")
print(f"Prediction for [5, 5, 5, 5]: Class {predicted_labels[1]}")
```

## API Reference

### `LinearDiscriminantAnalysis` Class

```python
class LinearDiscriminantAnalysis(BaseModel):
    def __init__(self, n_components=None)
    def fit(self, X, y, variables_names=None, objects_names=None)
    def transform(self, X)
    def fit_transform(self, X, y, **kwargs)
    def predict(self, X_new)
    @property
    def summary(self) -> str
```

### Parameters & Attributes
-   **`n_components` (int, optional)**: The number of discriminant components to retain. If `None`, it defaults to `min(n_features, n_classes - 1)`.
-   **`W` (np.ndarray)**: After fitting, this attribute stores the transformation matrix (eigenvectors) of shape `(n_features, n_components)`.
-   **`eigenvalues_` (np.ndarray)**: The eigenvalues corresponding to each discriminant component.

---

## Further Reading

For a more detailed mathematical explanation, please refer to the Wikipedia article:
-   [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
