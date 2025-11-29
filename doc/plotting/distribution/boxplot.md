# Box Plot

A box plot (or box-and-whisker plot) is a standardized way of displaying the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It is particularly effective for comparing distributions across multiple groups.

The `BoxPlot` class can also generate violin plots by setting the `mode` parameter.

## `plot`

This method creates a box plot.

### Usage
```python
from chemtools.plots.distribution import BoxPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'Category': np.repeat(['A', 'B', 'C'], 50),
    'Value': np.concatenate([
        np.random.normal(5, 1, 50),
        np.random.normal(8, 2, 50),
        np.random.normal(4, 1.5, 50)
    ])
})

plotter = BoxPlot()

# Grouped Box Plot
fig_box = plotter.plot(data, x='Category', y='Value', title="Box Plot by Category")
fig_box.savefig("boxplot.png")

# Violin Plot (using the same class)
fig_violin = plotter.plot(data, x='Category', y='Value', mode='violin', title="Violin Plot by Category")
fig_violin.savefig("violin_from_boxplot.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `y` (str): The numerical column for the y-axis.
- `x` (str, optional): The categorical column for the x-axis. If not provided, a single box plot for the `y` column is created.
- `mode` (str): `'box'` for a box plot or `'violin'` for a violin plot. Defaults to `'box'`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/boxplot_classic_professional_dark.png">
  <img alt="Box Plot" src="../../img/plots/distribution/boxplot_classic_professional_light.png">
</picture>
