# Classification Plots

This section details the specialized plotters designed to visualize the results of classification models from the `chemtools.classification` module.

## SIMCA Plot

The `SIMCAPlot` class is used to visualize the results of a Soft Independent Modeling of Class Analogies (`SIMCA`) model. Its primary purpose is to plot the scores of each class-specific PCA model, allowing for a visual assessment of how the classes are separated in the reduced dimensional space.

### `plot_scores`

This method plots the scores of each class model, typically with confidence ellipses that represent the "class space" or "envelope". This is crucial for understanding which classes are well-separated, which overlap, and how new data points relate to the established class models.

#### Usage
```python
from chemtools.classification import SIMCA
from chemtools.plots.classification import SIMCAPlot
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate and fit a SIMCA model (see SIMCA documentation for details)
np.random.seed(42)
X_train = np.vstack([
    np.random.multivariate_normal([2, 5], [[1, 0.5], [0.5, 1]], 50),
    np.random.multivariate_normal([8, 10], [[1.5, -0.7], [-0.7, 1.5]], 50)
])
y_train = np.array(['Class A'] * 50 + ['Class B'] * 50)
simca_model = SIMCA(n_components=2).fit(X_train, y_train)

# 2. Initialize the plotter with the fitted model
simca_plotter = SIMCAPlot(simca_model)

# 3. Generate the scores plot
fig = simca_plotter.plot_scores(
    title="SIMCA Class Models",
    confidence_level=0.95  # Draw a 95% confidence ellipse for each class
)
plt.show()
```

#### Parameters
-   `simca_model`: A fitted `SIMCA` object.
-   `components` (tuple): A tuple specifying the two principal components to plot on the x and y axes. Defaults to `(0, 1)`.
-   `confidence_level` (float): The confidence level for drawing the ellipses around each class. Defaults to `0.95`.
-   `show_legend` (bool): Whether to display the plot legend. Defaults to `True`.
-   `**kwargs`: Additional keyword arguments passed to the `BasePlotter` (e.g., `figsize`, `theme`).

#### Example Output

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/classification/simca_scores_plot_dark.png">
  <img alt="SIMCA Scores Plot" src="../../img/plots/classification/simca_scores_plot_light.png">
</picture>
