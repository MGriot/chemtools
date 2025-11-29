# Violin Plot

The violin plot is a powerful visualization that combines aspects of a box plot with a kernel density estimate. It shows the distribution of quantitative data across several categories, providing a richer understanding of the data's shape, central tendency, and spread than a simple box plot. 

The `ViolinPlot` class in `chemtools` is enhanced with several features for detailed statistical visualization, such as adding jittered data points (a "raincloud" effect), mean markers, and automatic statistical annotations.

## `plot`

This is the primary method for creating enhanced violin plots.

### Usage
```python
from chemtools.plots import ViolinPlot
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

plotter = ViolinPlot()

# Violin plot with jittered points and mean markers
fig = plotter.plot(data, 
                   y='Value', 
                   x='Category', 
                   show_jitter=True, 
                   show_mean=True,
                   title="Enhanced Violin Plot")
fig.savefig("violin_enhanced.png")

# Violin plot with automatic statistical tests
fig_stats = plotter.plot(data, 
                         y='Value', 
                         x='Category', 
                         perform_stat_test=True,
                         title="Violin Plot with T-Tests")
fig_stats.savefig("violin_stats.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `y` (str): The numerical column to plot.
- `x` (str, optional): The categorical column for grouping.
- `show_jitter` (bool): If `True`, adds a "raincloud" of jittered data points.
- `show_mean` (bool): If `True`, adds a distinct marker and line for the mean of each category.
- `show_n` (bool): If `True`, shows the sample size (n) for each category on the x-axis labels.
- `h_lines` (list, optional): A list of y-values to draw horizontal grid lines.
- `perform_stat_test` (bool): If `True` and `x` is provided, automatically performs pairwise t-tests between categories and annotates significant results (p < `stat_test_alpha`). (Matplotlib only).
- `stat_test_alpha` (float): The significance level for automatic statistical tests.
- `stat_annotations` (list, optional): A list of dictionaries for manually plotting statistical comparisons, e.g., `[{'groups': ('A', 'B'), 'p_value': 'p < 0.001', 'y_pos': 15}]`. Overrides `perform_stat_test`. (Matplotlib only).
- `y_max_override` (float, optional): Manually sets the upper y-axis limit, useful for ensuring all stat annotations are visible.
- `violin_alpha` (float, optional): Transparency for the violin plot bodies.
- `jitter_alpha` (float, optional): Transparency for the jittered data points.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/violin_plot_classic_professional_dark.png">
  <img alt="Enhanced Violin Plot" src="../../img/plots/distribution/violin_plot_classic_professional_light.png">
</picture>

### Example with Statistical Annotations
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/violin_stats_classic_professional_dark.png">
  <img alt="Violin Plot with Stats" src="../../img/plots/distribution/violin_stats_classic_professional_light.png">
</picture>
