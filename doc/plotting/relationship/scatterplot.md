# Scatter Plot

Scatter plots are a fundamental tool for visualizing the relationship between two or three numerical variables. They help in identifying correlations, trends, clusters, and outliers in the data. The `ScatterPlot` class in `chemtools` supports 2D, 3D, and bubble charts.

## 2D Scatter Plot

A 2D scatter plot displays individual data points on a two-dimensional graph. It is the standard way to visualize the relationship between two numerical variables.

### Usage
```python
from chemtools.plots.relationship import ScatterPlot
import pandas as pd
import numpy as np

# Sample Data
data = pd.DataFrame({
    'x_axis': np.random.rand(50) * 10,
    'y_axis': np.random.rand(50) * 10 + 5,
})

# Create Plot
plotter = ScatterPlot(theme='classic_professional_light')
fig = plotter.plot_2d(data, x_column='x_axis', y_column='y_axis', title="2D Scatter Plot")
fig.savefig("scatter_2d.png")
```

### Parameters for `plot_2d`
- `data` (pd.DataFrame): The input DataFrame.
- `x_column` (str): The column name for the x-axis.
- `y_column` (str): The column name for the y-axis.
- `size_column` (str, optional): The column for bubble size. If provided, a bubble chart is created.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/scatter_2d_classic_professional_dark.png">
  <img alt="2D Scatter Plot" src="../../img/plots/relationship/scatter_2d_classic_professional_light.png">
</picture>

---

## Bubble Chart

A bubble chart is a variation of the 2D scatter plot where a third dimension of data is represented by the size of the markers (bubbles). This is useful for comparing the relationship between three variables simultaneously.

### Usage
```python
# Sample Data
data = pd.DataFrame({
    'x_axis': np.random.rand(50) * 10,
    'y_axis': np.random.rand(50) * 10 + 5,
    'bubble_size': np.random.rand(50) * 1000 + 100
})

# Create Plot
plotter = ScatterPlot(theme='classic_professional_light')
fig = plotter.plot_2d(data, x_column='x_axis', y_column='y_axis', size_column='bubble_size', title="Bubble Chart")
fig.savefig("bubble_chart.png")
```

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/bubble_chart_classic_professional_dark.png">
  <img alt="Bubble Chart" src="../../img/plots/relationship/bubble_chart_classic_professional_light.png">
</picture>

---

## 3D Scatter Plot

A 3D scatter plot visualizes the relationship between three numerical variables by plotting points in a three-dimensional space.

### Usage
```python
# Sample Data
data = pd.DataFrame({
    'x_axis': np.random.rand(100),
    'y_axis': np.random.rand(100),
    'z_axis': np.random.rand(100)
})

# Create Plot
plotter = ScatterPlot(theme='classic_professional_light')
fig = plotter.plot_3d(data, x_column='x_axis', y_column='y_axis', z_column='z_axis', title="3D Scatter Plot")
fig.savefig("scatter_3d.png")
```
**Note:** The `plotly` backend is highly recommended for 3D plots as it provides interactivity (rotation, zoom).

### Parameters for `plot_3d`
- `data` (pd.DataFrame): The input DataFrame.
- `x_column` (str): The column for the x-axis.
- `y_column` (str): The column for the y-axis.
- `z_column` (str): The column for the z-axis.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter`.

### Example Output (matplotlib)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/relationship/scatter_3d_mpl_classic_professional_dark.png">
  <img alt="3D Scatter Plot" src="../../img/plots/relationship/scatter_3d_mpl_classic_professional_light.png">
</picture>
