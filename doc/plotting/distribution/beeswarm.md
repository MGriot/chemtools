# Beeswarm Plot

A beeswarm plot (or swarmplot) is a type of plot that displays individual data points for a numerical variable, typically grouped by a categorical variable. The points are arranged to avoid overlap, which gives a visual representation of the data's distribution and density. It is an excellent alternative to a box plot when you want to show all data points, especially for smaller datasets.

## `plot`

This method generates a Beeswarm plot.

### Usage
```python
from chemtools.plots.distribution import BeeswarmPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Value': np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(2, 1.5, 50),
        np.random.normal(-1, 0.8, 50)
    ]),
    'Category': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
})

# Create Plot
plotter = BeeswarmPlot(library='matplotlib', theme='classic_professional_light')
fig = plotter.plot(data, x='Category', y='Value', title="Beeswarm Plot")
fig.savefig("beeswarm_plot.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column name for the categorical variable.
- `y` (str): The column name for the numerical variable.
- `orientation` (str): `'vertical'` (default). 'horizontal' is not yet implemented.
- `point_size` (int): Size of the scatter points (area in points^2). Defaults to `50`.
- `spread_factor` (float): Controls the horizontal spread of the points. Tune for best appearance. Defaults to `0.05`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/beeswarm_plot_classic_professional_dark.png">
  <img alt="Beeswarm Plot" src="../../img/plots/distribution/beeswarm_plot_classic_professional_light.png">
</picture>
