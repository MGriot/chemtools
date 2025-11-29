# Line Plot

Line plots are used to visualize data points connected by straight line segments. They are especially useful for showing trends in data over a continuous interval or time series. The `LinePlot` class also supports variations like dot plots and area charts.

## `plot`

This is the primary method for creating line, dot, and area plots.

### Usage
```python
from chemtools.plots.basic import LinePlot
import pandas as pd
import numpy as np

# Sample Data for trend
data = pd.DataFrame({
    'Time': np.arange(20),
    'Value': (np.arange(20) + np.random.randn(20) * 2).cumsum()
})

plotter = LinePlot()

# Standard Line Plot
fig_line = plotter.plot(data, x_column='Time', y_column='Value', mode='line', title="Line Plot")
fig_line.savefig("line_plot.png")

# Dot Plot
fig_dot = plotter.plot(data, x_column='Time', y_column='Value', mode='dot', title="Dot Plot")
fig_dot.savefig("dot_plot.png")

# Area Chart
fig_area = plotter.plot(data, x_column='Time', y_column='Value', mode='area', title="Area Chart")
fig_area.savefig("area_chart.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x_column` (str): The column for the x-axis.
- `y_column` (str): The column for the y-axis.
- `mode` (str): The type of plot to create. Can be `'line'`, `'dot'`, or `'area'`. Defaults to `'line'`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output (Line)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/line_plot_classic_professional_dark.png">
  <img alt="Line Plot" src="../../img/plots/basic/line_plot_classic_professional_light.png">
</picture>

### Example Output (Area)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/area_plot_classic_professional_dark.png">
  <img alt="Area Chart" src="../../img/plots/basic/area_plot_classic_professional_light.png">
</picture>
