# Pie Plot

Pie charts are circular statistical graphics, divided into slices to illustrate numerical proportion. The arc length of each slice is proportional to the quantity it represents. The `PiePlot` class also supports creating donut charts.

## `plot`

This method creates a pie or donut chart.

### Usage
```python
from chemtools.plots.basic import PiePlot
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [45, 25, 15, 15]
})

plotter = PiePlot()

# Standard Pie Chart
fig_pie = plotter.plot(data, names_column='Category', values_column='Value', title="Standard Pie Chart")
fig_pie.savefig("pie_chart.png")

# Donut Chart
fig_donut = plotter.plot(data, names_column='Category', values_column='Value', hole=0.4, title="Donut Chart")
fig_donut.savefig("donut_chart.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `names_column` (str): The column with the names of the slices.
- `values_column` (str): The column with the values of the slices.
- `hole` (float): The size of the hole for a donut chart (from 0 to 1). A value of 0 creates a standard pie chart. Defaults to `0`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output (Pie)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/pie_plot_classic_professional_dark.png">
  <img alt="Pie Plot" src="../../img/plots/basic/pie_plot_classic_professional_light.png">
</picture>

### Example Output (Donut)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/donut_plot_classic_professional_dark.png">
  <img alt="Donut Chart" src="../../img/plots/basic/donut_plot_classic_professional_light.png">
</picture>
