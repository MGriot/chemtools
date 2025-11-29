# Bar Plot

Bar plots are used to represent categorical data with rectangular bars, where the length of each bar is proportional to the value it represents. They are one of the most common plot types for comparing values across different groups. The `BarPlot` class supports simple bar charts, grouped bar charts, and stacked bar charts.

## `plot_counts`

This method is used to quickly visualize the frequency (count) of each category in a single column.

### Usage
```python
from chemtools.plots.basic import BarPlot
import pandas as pd

data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'A']})
plotter = BarPlot()
fig = plotter.plot_counts(data, column='Category', title="Frequency of Categories")
fig.savefig("bar_counts.png")
```

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/bar_plot_counts_classic_professional_dark.png">
  <img alt="Bar Plot Counts" src="../../img/plots/basic/bar_plot_counts_classic_professional_light.png">
</picture>

---

## `plot` (Grouped & Stacked)

This method creates grouped or stacked bar charts from long-format data to compare a numerical value (`y`) across different categories (`x`), often further subdivided by a `color` category.

### Usage
```python
from chemtools.plots.basic import BarPlot
import pandas as pd

data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Group': ['G1', 'G2', 'G1', 'G2', 'G1', 'G2'],
    'Value': [10, 12, 15, 18, 13, 9]
})
plotter = BarPlot()
# Grouped bar plot
fig_grouped = plotter.plot(data, x='Category', y='Value', color='Group', mode='group')
fig_grouped.savefig("bar_grouped.png")

# Stacked bar plot
fig_stacked = plotter.plot(data, x='Category', y='Value', color='Group', mode='stack')
fig_stacked.savefig("bar_stacked.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column for the main categories on the x-axis.
- `y` (str): The numerical column for the bar heights.
- `color` (str, optional): A column to group by and color the bars.
- `mode` (str): `'group'` for a grouped bar chart or `'stack'` for a stacked bar chart.

### Example Output (Grouped)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/basic/bar_plot_grouped_classic_professional_dark.png">
  <img alt="Grouped Bar Plot" src="../../img/plots/basic/bar_plot_grouped_classic_professional_light.png">
</picture>

---

## `plot_crosstab`

This method is designed to plot wide-format data, such as the output of `pd.crosstab`. It's particularly useful for creating 100% stacked bar charts to show proportions.

### Usage
```python
from chemtools.plots.basic import BarPlot
import pandas as pd

# Create a contingency table
data = pd.DataFrame({
    'Education': pd.Categorical(['High School', 'Bachelors', 'Bachelors', 'Masters', 'High School'], 
                                categories=['High School', 'Bachelors', 'Masters'], ordered=True),
    'Satisfaction': ['High', 'Medium', 'High', 'High', 'Low']
})
crosstab_df = pd.crosstab(data['Education'], data['Satisfaction'])

plotter = BarPlot()
# 100% stacked bar chart
fig = plotter.plot_crosstab(crosstab_df, stacked=True, normalize=True, title="Job Satisfaction by Education")
fig.savefig("bar_crosstab_normalized.png")
```

### Parameters
- `crosstab_data` (pd.DataFrame): A wide-format DataFrame where the index represents x-axis ticks and columns are categories.
- `stacked` (bool): If `True`, creates a stacked bar chart. If `False`, creates a grouped bar chart.
- `normalize` (bool): If `True` and `stacked` is `True`, creates a 100% stacked bar chart showing proportions.

### Example Output (100% Stacked)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/categorical/stacked_bar_chart_classic_professional_dark.png">
  <img alt="100% Stacked Bar Chart" src="../../img/plots/categorical/stacked_bar_chart_classic_professional_light.png">
</picture>
