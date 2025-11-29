# Raincloud Plot

The Raincloud plot is a powerful visualization that provides a comprehensive view of data distributions by combining three different plot types:
1.  A **Violin Plot** (the "cloud") to show the shape of the distribution.
2.  A **Jittered Scatter Plot** (the "rain") to show individual data points.
3.  A **Box Plot** to show summary statistics (median, quartiles).

It is particularly useful for comparing distributions across different categories, showing the overall shape, individual data points, and key statistical summaries simultaneously.

## `plot`

This method generates a Raincloud plot.

### Usage
```python
from chemtools.plots.distribution import RaincloudPlot
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

plotter = RaincloudPlot(theme='classic_professional_light')

# Vertical Raincloud Plot
fig_v = plotter.plot(data, x='Category', y='Value', orientation='vertical', title="Vertical Raincloud Plot")
fig_v.savefig("raincloud_vertical.png")

# Horizontal Raincloud Plot
fig_h = plotter.plot(data, x='Value', y='Category', orientation='horizontal', title="Horizontal Raincloud Plot")
fig_h.savefig("raincloud_horizontal.png")
```

### Parameters
- `data` (pd.DataFrame): The input DataFrame.
- `x` (str): The column for the categorical variable (vertical) or numerical variable (horizontal).
- `y` (str): The column for the numerical variable (vertical) or categorical variable (horizontal).
- `orientation` (str): `'vertical'` (default) or `'horizontal'`.
- `violin_filled` (bool): If `True`, the violin plot is filled. Defaults to `True`.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output (Vertical)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/raincloud_vertical_classic_professional_dark.png">
  <img alt="Vertical Raincloud Plot" src="../../img/plots/distribution/raincloud_vertical_classic_professional_light.png">
</picture>

### Example Output (Horizontal)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/distribution/raincloud_horizontal_classic_professional_dark.png">
  <img alt="Horizontal Raincloud Plot" src="../../img/plots/distribution/raincloud_horizontal_classic_professional_light.png">
</picture>

### Citation
This implementation is inspired by the concepts presented in:
[https://python-graph-gallery.com/raincloud-plot-with-matplotlib-and-ptitprince/](https://python-graph-gallery.com/raincloud-plot-with-matplotlib-and-ptitprince/)
