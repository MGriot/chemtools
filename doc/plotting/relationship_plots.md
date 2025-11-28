# Relationship Plots

Relationship plots are used to understand the association between two or more variables. They are essential for exploring correlations and patterns in your data.

## 2D Density Charts

When you have a large number of data points, a standard scatter plot can become overcrowded and difficult to interpret due to overplotting. A 2D density chart is an excellent alternative, visualizing the distribution of data points across a two-dimensional space using a color gradient to represent the number of observations in a particular area.

The `DensityPlot` class, which is based on Matplotlib and Scipy, provides several ways to visualize this 2D distribution.

### Kernel Density Estimate (KDE) Plot

A 2D KDE plot smooths the observations with a Gaussian kernel to create a continuous density surface, which is often represented with contour lines and a filled area. This is useful for visualizing the shape and peaks of the data distribution without being confined to discrete bins.

**Usage:**
```python
from chemtools.plots.relationship import DensityPlot
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'x_var': np.random.normal(loc=10, scale=2, size=1000),
    'y_var': np.random.normal(loc=10, scale=3, size=1000)
})

# Create plot
plotter = DensityPlot(theme='classic_professional_light')
fig = plotter.plot(data, x='x_var', y='y_var', kind='kde')
fig.savefig("density_kde_example.png")
```

**Example Output:**

![KDE Plot](../img/plots/relationship/density_kde_light.png)

---

### 2D Histogram

A 2D histogram divides the plotting area into a grid of rectangular bins and uses color to represent the number of data points that fall into each bin. It is a straightforward way to see where data is most concentrated. You can control the number of bins with the `bins` argument.

**Usage:**
```python
# ... (imports and data from above)

# Create plot
plotter = DensityPlot(theme='classic_professional_light')
fig = plotter.plot(data, x='x_var', y='y_var', kind='hist2d', bins=(40, 40), cmap='viridis')
fig.savefig("density_hist2d_example.png")
```

**Example Output:**

![2D Histogram](../img/plots/relationship/density_hist2d_light.png)

---

### Hexbin Plot

A hexbin plot is similar to a 2D histogram but uses a grid of hexagonal bins. The hexagonal shape helps reduce sampling artifacts that can be present in square bins and provides a more natural representation of the density. The resolution of the grid can be adjusted with the `gridsize` argument.

**Usage:**
```python
# ... (imports and data from above)

# Create plot
plotter = DensityPlot(theme='classic_professional_light')
fig = plotter.plot(data, x='x_var', y='y_var', kind='hexbin', gridsize=25, cmap='viridis')
fig.savefig("density_hexbin_example.png")
```

**Example Output:**

![Hexbin Plot](../img/plots/relationship/density_hexbin_light.png)
