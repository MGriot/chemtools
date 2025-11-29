# GridPlotter

The `GridPlotter` provides a flexible framework for arranging multiple `chemtools` plotters into a single, dynamic grid layout. This allows for the creation of complex, multi-panel figures with consistent theming and customizable subplot dimensions.

## Basic Usage

The `GridPlotter` takes `nrows` and `ncols` to define the grid dimensions. You can then add any `BasePlotter`-derived plot instance to a specific cell in this grid.

### Creating the Grid
When initializing `GridPlotter`, you can specify the number of rows and columns, as well as optional `width_ratios` and `height_ratios` to control the relative sizes of the subplots.

```python
from chemtools.plots.composite import GridPlotter
from chemtools.plots.basic import BarPlot, LinePlot
import pandas as pd
import numpy as np

# Initialize GridPlotter with 2 rows, 2 columns
grid_plotter = GridPlotter(nrows=2, ncols=2, figsize=(12, 8), 
                           subplot_titles=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
                           theme='classic_professional_light')

# Sample Data
data1 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [10, 15, 7]})
data2 = pd.DataFrame({'Time': np.arange(10), 'Value': np.random.randn(10).cumsum()})
data3 = pd.DataFrame({'X': np.random.rand(50), 'Y': np.random.rand(50)})
data4 = pd.DataFrame({'Task': ['T1', 'T2'], 'Progress': [80, 50]})

# Create plotter instances
bar_plotter = BarPlot()
line_plotter = LinePlot()
scatter_plotter = ScatterPlot() # Assuming ScatterPlot is imported
pie_plotter = PiePlot() # Assuming PiePlot is imported

# Add plots to specific grid cells
grid_plotter.add_plot(row=0, col=0, plotter_instance=bar_plotter, plot_method_name='plot_counts', 
                      data=data1, column='Category', subplot_title="Category Counts")

grid_plotter.add_plot(row=0, col=1, plotter_instance=line_plotter, plot_method_name='plot', 
                      data=data2, x_column='Time', y_column='Value', subplot_title="Time Series")

grid_plotter.add_plot(row=1, col=0, plotter_instance=scatter_plotter, plot_method_name='plot_2d', 
                      data=data3, x_column='X', y_column='Y', subplot_title="Scatter Plot")

grid_plotter.add_plot(row=1, col=1, plotter_instance=pie_plotter, plot_method_name='plot', 
                      data=data4, names_column='Task', values_column='Progress', subplot_title="Task Progress")

# Render the grid
fig = grid_plotter.render()
fig.savefig("grid_plot_example.png", bbox_inches='tight')
```

### Parameters for `GridPlotter.__init__`
- `nrows` (int): Number of rows in the grid.
- `ncols` (int): Number of columns in the grid.
- `width_ratios` (List[float], optional): Ratios of subplot widths. For example, `[1, 2]` would make the second column twice as wide as the first.
- `height_ratios` (List[float], optional): Ratios of subplot heights.
- `subplot_titles` (List[str], optional): Titles for each subplot, ordered row by row (e.g., `[title_0_0, title_0_1, title_1_0, ...]`). These are set by the GridPlotter directly.
- `**kwargs`: Additional keyword arguments passed to the `BasePlotter` (e.g., `figsize`, `title` for the overall figure).

### Parameters for `add_plot`
- `row` (int): The row index for the subplot (0-indexed).
- `col` (int): The column index for the subplot (0-indexed).
- `plotter_instance` (BasePlotter): An instance of a `chemtools` plotter class (e.g., `BarPlot()`, `LinePlot()`).
- `plot_method_name` (str): The name of the plotting method to call on the `plotter_instance` (e.g., `'plot'`, `'plot_counts'`).
- `data` (pd.DataFrame): The data to be passed to the plotting method.
- `**plot_kwargs`: Additional keyword arguments specific to the chosen `plot_method_name`. These will override any theme defaults from the `GridPlotter` for that specific subplot.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/composite/grid_plotter_example_classic_professional_dark.png">
  <img alt="Grid Plotter Example" src="../../img/plots/composite/grid_plotter_example_classic_professional_light.png">
</picture>
