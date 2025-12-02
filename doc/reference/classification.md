# Classification Module Reference (`chemtools.classification`)

The `chemtools.classification` module provides implementations of various supervised classification algorithms commonly used in chemometrics, including k-Nearest Neighbors (k-NN), Linear Discriminant Analysis (LDA), and Soft Independent Modeling of Class Analogies (SIMCA).

---

## `KNearestNeighbors` Class

Implements the k-Nearest Neighbors (k-NN) classification algorithm.

### `KNearestNeighbors(n_neighbors=5)`

*   **Parameters:**
    *   `n_neighbors` (`int`, optional): The number of neighbors (`k`) to use for classification. Defaults to `5`.

### Methods

*   **`fit(self, X, y, variables_names=None, objects_names=None)`**
    *   Stores the training data.
    *   **Parameters:**
        *   `X` (`np.ndarray`): Training feature data.
        *   `y` (`np.ndarray`): Training class labels.
        *   `variables_names` (`list`, optional): Names for the features.
        *   `objects_names` (`list`, optional): Names for the observations.

*   **`predict(self, X_new) -> np.ndarray`**
    *   Predicts the class labels for new data points.
    *   **Parameters:** `X_new` (`np.ndarray`): New data points to classify.
    *   **Returns:** `np.ndarray`: Predicted class labels.

*   **`summary(self) -> str` (property)**
    *   Returns a summary of the fitted model.

### Attributes

*   **`X_train` (`np.ndarray`):** Stores the training feature data after fitting.
*   **`y_train` (`np.ndarray`):** Stores the training class labels after fitting.

### Usage Example (k-NN)

```python
from chemtools.classification import KNearestNeighbors
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
knn = KNearestNeighbors(n_neighbors=5)
knn.fit(X, y)
X_new = np.array([[0, 0]])
predicted_label = knn.predict(X_new)
print(f"k-NN Prediction for [0, 0]: Class {predicted_label[0]}")
```

---

<h2><code>LinearDiscriminantAnalysis</code> Class</h2>

Implements Linear Discriminant Analysis (LDA) for classification and dimensionality reduction.

<h3><code>LinearDiscriminantAnalysis(n_components=None)</code></h3>

*   <b>Parameters:</b>
    *   <code>n_components</code> (<code>int</code>, optional): The number of discriminant components to retain. If <code>None</code>, defaults to <code>min(n_features, n_classes - 1)</code>.

<h3>Methods</h3>

*   <b><code>fit(self, X, y, variables_names=None, objects_names=None)</code></b>
    *   Fits the LDA model to the training data.
*   <b><code>transform(self, X) -> np.ndarray</code></b>
    *   Applies dimensionality reduction to X.
*   <b><code>fit_transform(self, X, y, **kwargs) -> np.ndarray</code></b>
    *   Fits LDA and then transforms X.
*   <b><code>predict(self, X_new) -> np.ndarray</code></b>
    *   Predicts class labels for new data points.
*   <b><code>summary(self) -> str</code> (property)</b>
    *   Returns a summary of the fitted model.

<h3>Attributes</h3>

*   <b><code>W</code> (<code>np.ndarray</code>):</b> The transformation matrix (eigenvectors) of shape <code>(n_features, n_components)</code>.
*   <b><code>eigenvalues_</code> (<code>np.ndarray</code>):</b> The eigenvalues corresponding to each discriminant component.

<h3>Usage Example (LDA)</h3>

```python
from chemtools.classification import LinearDiscriminantAnalysis
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=150, n_features=4, n_informative=2, n_redundant=0, n_classes=3, random_state=42)
lda = LinearDiscriminantAnalysis(n_components=2)
X_transformed = lda.fit_transform(X, y)
X_new = np.array([[0, 0, 0, 0]])
predicted_labels = lda.predict(X_new)
print(f"LDA Prediction for [0, 0, 0, 0]: Class {predicted_labels[0]}")
```

---

<h2><code>SIMCA</code> Class</h2>

Implements Soft Independent Modeling of Class Analogies (SIMCA) for "soft" classification and outlier detection.

<h3><code>SIMCA(n_components=2, alpha=0.05)</code></h3>

*   <b>Parameters:</b>
    *   <code>n_components</code> (<code>int</code>, optional): The number of principal components to use for each class model. Defaults to <code>2</code>.
    *   <code>alpha</code> (<code>float</code>, optional): The significance level for statistical distance thresholds. Defaults to <code>0.05</code> (95% confidence).

<h3>Methods</h3>

*   <b><code>fit(self, X, y, variables_names=None, objects_names=None)</code></b>
    *   Builds a separate PCA model for each class.
*   <b><code>predict(self, X_new) -> list[list[str]]</code></b>
    *   Predicts class membership for new samples. Returns a list of lists, as a sample can belong to multiple classes or none.
*   <b><code>summary(self) -> str</code> (property)</b>
    *   Returns a summary of the fitted SIMCA model.

<h3>Attributes</h3>

*   <b><code>class_models</code> (<code>dict</code>):</b> Dictionary mapping class labels to their fitted <code>PrincipalComponentAnalysis</code> models.

<h3>Usage Example (SIMCA)</h3>

```python
import numpy as np
from chemtools.classification import SIMCA

np.random.seed(42)
class_a_data = np.random.multivariate_normal([2, 5], [[1, 0.5], [0.5, 1]], 50)
class_b_data = np.random.multivariate_normal([8, 10], [[1.5, -0.7], [-0.7, 1.5]], 50)
X_train = np.vstack([class_a_data, class_b_data])
y_train = np.array(['Class A'] * 50 + ['Class B'] * 50)

simca = SIMCA(n_components=2, alpha=0.05)
simca.fit(X_train, y_train)
X_new = np.array([[5, 8]]) # A point that might be ambiguous
predictions = simca.predict(X_new)
print(f"SIMCA Prediction for [5, 8]: {predictions[0]}")
```
