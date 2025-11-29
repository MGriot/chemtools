# Regression Plots

Visualizing the results of a regression analysis is crucial for understanding the model's performance and the relationship between variables. The `RegressionPlots` class provides a suite of methods to plot various aspects of a fitted regression model.

This plotter class takes a fitted regression object (like `OLSRegression`) during initialization.

```python
from chemtools.regression import OLSRegression
from chemtools.plots.regression import RegressionPlots
import numpy as np

# Sample Data and Model
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(50) * 2
model = OLSRegression()
model.fit(X, y)

# Initialize plotter with the model
# plotter = RegressionPlots(model)
```

## `plot_regression_results`

This is a comprehensive method that combines the data points, the fitted regression line, and the confidence and prediction bands into a single plot.

### Usage
```python
# plotter initialized as above
fig = plotter.plot_regression_results(title="Full Regression Results")
fig.savefig("regression_results.png")
```

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/regression/regression_results_classic_professional_dark.png">
  <img alt="Regression Results Plot" src="../../img/plots/regression/regression_results_classic_professional_light.png">
</picture>

---

## Other Plotting Methods

The `RegressionPlots` class also provides methods to generate individual components of the regression results plot.

- **`plot_data()`**: Plots only the raw input data points (scatter plot).
- **`plot_residuals()`**: Plots the model's residuals to help diagnose variance and bias.
- **`plot_regression_line()`**: Plots the data points with the regression line overlaid.
- **`plot_confidence_band()`**: Plots the confidence band for the mean of the response.
- **`plot_prediction_band()`**: Plots the prediction band for individual observations.
