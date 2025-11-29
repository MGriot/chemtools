# Pair Plot

A pair plot (also known as a scatter plot matrix) is a powerful tool for exploratory data analysis. It creates a matrix of plots to visualize the pairwise relationships between several variables in a dataset. 

For the `matplotlib` backend, the `chemtools` `PairPlot` provides an advanced implementation:
- **Lower Triangle**: Scatter plots to show the relationship between each pair of variables.
- **Diagonal**: Kernel Density Estimation (KDE) plots to show the distribution of each individual variable.
- **Upper Triangle**: Pearson correlation coefficients, calculated for the overall data and for each group if a `hue` is provided.

This makes it an excellent all-in-one tool for quickly assessing correlations and distributions.

## Usage

```python
from chemtools.plots.relationship import PairPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'sepal_length': np.random.normal(5.8, 0.8, 150),
    'sepal_width': np.random.normal(3.0, 0.4, 150),
    'petal_length': np.random.normal(3.7, 1.7, 150),
    'petal_width': np.random.normal(1.2, 0.7, 150),
    'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
})

# Create Plot
# The 'hue' parameter colors the data points by a categorical variable.
plotter = PairPlot(theme='classic_professional_light', figsize=(10, 10))
fig = plotter.plot(data, hue='species', title="Pair Plot of Fisher's Iris Data")
fig.savefig("pairplot.png", bbox_inches='tight')
```

## Parameters
- `data` (pd.DataFrame): The input DataFrame. The plot will be generated for the numerical columns.
- `hue` (str, optional): A categorical column in `data` to color the data points by. This also enables group-specific correlations in the upper triangle.
- `palette` (list, optional): A list of colors to use for the `hue` groups. If not provided, the theme's default `category_color_scale` is used.
- `showlegend` (bool): If `True` and `hue` is specified, a legend for the hue categories will be displayed (matplotlib only).
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

## Example Output

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/pairplot_with_legend_classic_professional_dark.png">
  <img alt="Pair Plot" src="../../img/plots/relationship/pairplot_with_legend_classic_professional_light.png">
</picture>
