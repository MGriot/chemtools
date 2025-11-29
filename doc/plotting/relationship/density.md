# 2D Density Charts

When you have a large number of data points, a standard scatter plot can become overcrowded and difficult to interpret due to overplotting. A 2D density chart is an excellent alternative, visualizing the distribution of data points across a two-dimensional space using a color gradient to represent the number of observations in a particular area.

The `DensityPlot` class, which is based on Matplotlib and Scipy, provides several ways to visualize this 2D distribution.

## Kernel Density Estimate (KDE) Plot

A 2D KDE plot smooths the observations with a Gaussian kernel to create a continuous density surface, which is often represented with contour lines and a filled area. This is useful for visualizing the shape and peaks of the data distribution without being confined to discrete bins.

### Usage
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

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column name for the x-axis.
- `y` (str): The column name for the y-axis.
- `kind` (str): The type of density plot. Can be `'kde'`, `'hist2d'`, or `'hexbin'`.
- `**kwargs`: Additional arguments passed to the plotting function, such as `cmap`, `bins` (for hist2d), `gridsize` (for hexbin), `levels` (for kde), etc.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/density_kde_dark.png">
  <img alt="KDE Plot" src="../../img/plots/relationship/density_kde_light.png">
</picture>

---

## 2D Histogram

A 2D histogram divides the plotting area into a grid of rectangular bins and uses color to represent the number of data points that fall into each bin. It is a straightforward way to see where data is most concentrated. You can control the number of bins with the `bins` argument.

### Usage
```python
# ... (imports and data from above)

# Create plot
plotter = DensityPlot(theme='classic_professional_light')
fig = plotter.plot(data, x='x_var', y='y_var', kind='hist2d', bins=(40, 40))
fig.savefig("density_hist2d_example.png")
```

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/density_hist2d_dark.png">
  <img alt="2D Histogram" src="../../img/plots/relationship/density_hist2d_light.png">
</picture>


---

## Hexbin Plot

A hexbin plot is similar to a 2D histogram but uses a grid of hexagonal bins. The hexagonal shape helps reduce sampling artifacts that can be present in square bins and provides a more natural representation of the density. The resolution of the grid can be adjusted with the `gridsize` argument.

### Usage
```python
# ... (imports and data from above)

# Create plot
plotter = DensityPlot(theme='classic_professional_light')
fig = plotter.plot(data, x='x_var', y='y_var', kind='hexbin', gridsize=25)
fig.savefig("density_hexbin_example.png")
```

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/density_hexbin_dark.png">
  <img alt="Hexbin Plot" src="../../img/plots/relationship/density_hexbin_light.png">
</picture>
