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
fig_hist = eda.plot_histogram('your_numerical_column')
fig_hist.show()

fig_box = eda.plot_boxplot('your_numerical_column')
fig_box.show()
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
fig_heatmap = eda.plot_heatmap()
fig_heatmap.show()

# Scatter plot of two numerical variables
fig_scatter = eda.plot_scatter('column_x', 'column_y')
fig_scatter.show()
```

## API Reference

### `ExploratoryDataAnalysis` Class

```python
class ExploratoryDataAnalysis:
    def __init__(self, data: pd.DataFrame, plotter_kwargs: dict = None)
    def get_univariate_summary(self) -> pd.DataFrame
    def plot_histogram(self, column: str, **kwargs)
    def plot_boxplot(self, column: str, **kwargs)
    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame
    def plot_heatmap(self, **kwargs)
    def plot_scatter(self, x_column: str, y_column: str, **kwargs)
    def plot_barchart(self, column: str, **kwargs)
    def plot_run_chart(self, time_column: str, value_column: str, **kwargs)
    def plot_stem_and_leaf(self, column: str)
    def plot_scatter_3d(self, x_column: str, y_column: str, z_column: str, **kwargs)
    def plot_parallel_coordinates(self, class_column: str, **kwargs)
```
