# Soft Independent Modeling of Class Analogies (SIMCA)

Soft Independent Modeling of Class Analogies (SIMCA) is a supervised classification method particularly popular in chemometrics. Instead of finding boundaries to separate classes, SIMCA builds a separate Principal Component Analysis (PCA) model for each class. A new observation is then classified based on how well it fits into each of these class models.

The `SIMCA` class in `chemtools` provides a full implementation of this "soft" modeling technique, allowing for robust classification, outlier detection, and handling of class overlaps.

## How SIMCA Works

The core concept of SIMCA is to define a "class space" or "envelope" for each category using PCA.

1.  **Training Phase (Model Building)**:
    *   For each class in the training set (e.g., 'Class A', 'Class B'), a separate PCA model is built using only the data from that class.
    *   The number of principal components to retain for each class model is determined (e.g., to explain 95% of the variance).
    *   For each class model, statistical limits are calculated. These limits define the boundaries of the class in the PCA space and are typically based on:
        *   **Hotelling's T² statistic**: Measures the distance of a point from the center of the PCA model *within* the model's hyperplane.
        *   **Q-residuals (or SPE)**: Measures the distance of a point *to* the PCA model's hyperplane (i.e., the reconstruction error).

2.  **Prediction Phase (Classification)**:
    *   When a new, unknown sample is introduced, it is projected onto each of the trained class PCA models.
    *   For each projection, the sample's T² and Q-residual values are calculated.
    *   The sample is considered to belong to a class if both its T² and Q values are within the statistical limits defined for that class model.

This is a "soft" classification because a sample can be assigned to:
*   **One class**: It fits only one class model.
*   **Multiple classes**: It fits within the statistical boundaries of several class models (class overlap).
*   **No class (outlier)**: It does not fit within the boundaries of any class model.

## Usage

Here is a basic example of how to use the `SIMCA` class.

```python
import numpy as np
from chemtools.classification import SIMCA
from chemtools.plots.classification import SIMCAPlot
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data for two classes
np.random.seed(42)
class_a_data = np.random.multivariate_normal([2, 5], [[1, 0.5], [0.5, 1]], 50)
class_b_data = np.random.multivariate_normal([8, 10], [[1.5, -0.7], [-0.7, 1.5]], 50)
X_train = np.vstack([class_a_data, class_b_data])
y_train = np.array(['Class A'] * 50 + ['Class B'] * 50)

# 2. Initialize and fit the SIMCA model
# We'll use 2 principal components for each class model
simca = SIMCA(n_components=2, alpha=0.05)
simca.fit(X_train, y_train)

# 3. Print the model summary
print(simca.summary)

# 4. Create new samples for prediction
X_new = np.array([
    [2.5, 5.5],    # Should belong to Class A
    [8.2, 9.8],    # Should belong to Class B
    [5, 8],        # Might be ambiguous or an outlier
    [15, 15]       # Definitely an outlier
])

# 5. Predict the classes of the new samples
predictions = simca.predict(X_new)

# 6. Display prediction results
print("\n--- Prediction Results ---")
for i, p in enumerate(predictions):
    if not p:
        print(f"Sample {i+1} is an OUTLIER.")
    else:
        print(f"Sample {i+1} is classified as: {', '.join(p)}")
        
# 7. Visualize the class models using SIMCAPlot
print("\nGenerating SIMCA Scores Plot...")
simca_plotter = SIMCAPlot(simca)
fig = simca_plotter.plot_scores(title="SIMCA Class Models with 95% Confidence Ellipses")
# Add new points to the plot for visualization
for i, point in enumerate(X_new):
    label = "Outlier" if not predictions[i] else '/'.join(predictions[i])
    fig.gca().scatter(point[0], point[1], marker='X', s=100, label=f"New Sample {i+1}: {label}", zorder=30)
fig.gca().legend()
plt.show()
```

## API Reference

### `SIMCA` Class

```python
class SIMCA(BaseModel):
    def __init__(self, n_components=2, alpha=0.05)
    def fit(self, X, y, variables_names=None, objects_names=None)
    def predict(self, X_new)
    @property
    def summary(self) -> str
```

### Parameters & Attributes
-   **`n_components` (int)**: The number of principal components to use for each class model. Defaults to `2`.
-   **`alpha` (float)**: The significance level (e.g., 0.05 for 95% confidence) used for calculating the statistical distance thresholds. Defaults to `0.05`.
-   **`class_models` (dict)**: A dictionary where keys are the class labels and values are the fitted `PrincipalComponentAnalysis` models for each class.

---

## Further Reading

For a deeper dive into the methodology, you can refer to the original papers by Svante Wold and the following Wikipedia article:
-   [Soft independent modelling of class analogy](https://en.wikipedia.org/wiki/Soft_independent_modelling_of_class_analogy)
