# Parallel Coordinates Plot

A parallel coordinates plot is used for visualizing high-dimensional, multivariate data. Each vertical axis represents a different variable, and each colored line represents an individual observation, showing how it behaves across all variables. This type of plot is particularly useful for identifying clusters, patterns, and relationships between variables that might not be apparent in other plots.

## `plot`

This method generates the parallel coordinates plot.

### Usage
```python
from chemtools.plots.specialized import ParallelCoordinatesPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'var1': np.random.rand(90) * 1,
    'var2': np.random.rand(90) * 5,
    'var3': np.random.rand(90) * 10,
    'var4': np.random.rand(90) * 2,
    'class': np.repeat(['A', 'B', 'C'], 30)
})

# Create Plot
plotter = ParallelCoordinatesPlot()
fig = plotter.plot(data, class_column='class', title="Parallel Coordinates Plot")
fig.savefig("parallel_coordinates.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame. The plot will be generated for the numerical columns.
- `class_column` (str): The categorical column in `data` used to color the lines, representing different groups or classes.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/specialized/parallel_coordinates_plot_classic_professional_dark.png">
  <img alt="Parallel Coordinates Plot" src="../../img/plots/specialized/parallel_coordinates_plot_classic_professional_light.png">
</picture>
