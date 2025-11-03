# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in the data science workflow. It involves analyzing and investigating datasets to summarize their main characteristics, often using data visualization methods. The `ExploratoryDataAnalysis` class in `chemtools` provides a convenient way to perform EDA on your data.

## The `ExploratoryDataAnalysis` Class

The `ExploratoryDataAnalysis` class is designed to work with pandas DataFrames and provides a suite of methods for both non-graphical and graphical analysis.

### Initialization

To start, create an instance of the class with your DataFrame:

```python
import pandas as pd
from chemtools.exploration import ExploratoryDataAnalysis

df = pd.read_csv('your_data.csv')
eda = ExploratoryDataAnalysis(df)
```

### Univariate Analysis

#### Non-Graphical

Get a statistical summary of each variable:

```python
summary = eda.get_univariate_summary()
print(summary)
```

#### Graphical

Visualize the distribution of a numerical variable using a histogram or a box plot:

```python
# Histogram
hist_plotter = eda.histogram_plotter()
fig_hist = hist_plotter.plot(eda.data, 'your_numerical_column')
fig_hist.show()

# Box Plot
box_plotter = eda.boxplot_plotter()
fig_box = box_plotter.plot(eda.data, 'your_numerical_column')
fig_box.show()

# Bar Chart
bar_chart_plotter = eda.barchart_plotter()
fig_bar = bar_chart_plotter.plot(eda.data, 'your_categorical_column')
fig_bar.show()

# Stem-and-leaf Plot (prints to console)
eda.plot_stem_and_leaf('your_numerical_column')
```

### Multivariate Analysis

#### Non-Graphical

Calculate the correlation matrix for numerical variables:

```python
correlation = eda.get_correlation_matrix()
print(correlation)
```

#### Graphical

Visualize relationships between variables:

```python
# Heatmap of the correlation matrix
corr_matrix = eda.get_correlation_matrix()
heatmap_plotter = eda.heatmap_plotter()
fig_heatmap = heatmap_plotter.plot(corr_matrix)
fig_heatmap.show()

# 2D Scatter Plot
scatter_plotter = eda.scatter_plotter()
fig_scatter_2d = scatter_plotter.plot_2d(eda.data, 'column_x', 'column_y')
fig_scatter_2d.show()

# 3D Scatter Plot
fig_scatter_3d = scatter_plotter.plot_3d(eda.data, 'column_x', 'column_y', 'column_z')
fig_scatter_3d.show()

# Run Chart
run_chart_plotter = eda.run_chart_plotter()
fig_run = run_chart_plotter.plot(eda.data, 'time_column', 'value_column')
fig_run.show()

# Parallel Coordinates Plot
parallel_coordinates_plotter = eda.parallel_coordinates_plotter()
fig_parallel = parallel_coordinates_plotter.plot(eda.data, 'class_column')
fig_parallel.show()
```

## API Reference

### `ExploratoryDataAnalysis` Class

```python
class ExploratoryDataAnalysis:
    def __init__(self, data: pd.DataFrame, plotter_kwargs: dict = None)
    def get_univariate_summary(self) -> pd.DataFrame
    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame
    def histogram_plotter(self) -> HistogramPlotter
    def boxplot_plotter(self) -> BoxPlotter
    def barchart_plotter(self) -> BarChartPlotter
    def scatter_plotter(self) -> ScatterPlotter
    def heatmap_plotter(self) -> HeatmapPlotter
    def run_chart_plotter(self) -> RunChartPlotter
    def parallel_coordinates_plotter(self) -> ParallelCoordinatesPlotter
    def plot_stem_and_leaf(self, column: str)
```