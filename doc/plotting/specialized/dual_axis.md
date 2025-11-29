# Dual-Axis Chart

A dual-axis chart allows for the visualization of two different variables with different scales on the same plot. It uses two separate y-axes (a left and a right axis) that share a common x-axis. This is useful for comparing trends between two related metrics that have different units or magnitudes (e.g., Sales volume and Growth Percentage).

**Note:** This plot is currently only supported for the `matplotlib` library.

## `plot`

This method generates the dual-axis chart.

### Usage
```python
from chemtools.plots.specialized import DualAxisPlot
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [150, 200, 180, 220, 250, 210],
    'Growth (%)': [1.5, 1.8, 1.2, 2.0, 2.2, 1.9]
})

# Create Plot
plotter = DualAxisPlot(library='matplotlib')
fig = plotter.plot(data, 
                   x_column='Month', 
                   y1_column='Sales', 
                   y2_column='Growth (%)', 
                   plot1_kind='bar', 
                   plot2_kind='line',
                   title="Monthly Sales and Growth")
fig.savefig("dual_axis_chart.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x_column` (str): The column for the shared x-axis.
- `y1_column` (str): The column for the first y-axis (left).
- `y2_column` (str): The column for the second y-axis (right).
- `plot1_kind` (str, optional): The kind of plot for the first y-axis (`'bar'` or `'line'`). Defaults to `'bar'`.
- `plot2_kind` (str, optional): The kind of plot for the second y-axis (`'bar'` or `'line'`). Defaults to `'line'`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/specialized/dual_axis_plot_classic_professional_dark.png">
  <img alt="Dual-Axis Chart" src="../../img/plots/specialized/dual_axis_plot_classic_professional_light.png">
</picture>
