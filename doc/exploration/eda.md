# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in the data science workflow. It involves analyzing and investigating datasets to summarize their main characteristics, often using data visualization methods. The `ExploratoryDataAnalysis` class in `chemtools` provides a comprehensive toolkit to perform EDA on your data, with special support for mixed-type (numerical and categorical) datasets.

## The `ExploratoryDataAnalysis` Class

The `ExploratoryDataAnalysis` class is designed to work with pandas DataFrames and provides a suite of methods for data inspection, non-graphical summaries, and graphical analysis.

### Initialization

To start, create an instance of the class with your DataFrame:

```python
import pandas as pd
from chemtools.exploration import ExploratoryDataAnalysis

df = pd.read_csv('your_data.csv')
# You can pass plotter_kwargs to set defaults for all plots
eda = ExploratoryDataAnalysis(df, plotter_kwargs={'theme': 'oceanic_slate_dark'})
```

---

## 1. Data Inspection

Before analysis, it's important to understand the structure of your data.

### Variable Classification

You can automatically classify columns into numerical and categorical types.

```python
numerical_cols, categorical_cols = eda.classify_variables()
print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)
```

### Missing Value Analysis

Quickly find which columns have missing data and visualize the pattern.

```python
# Get a summary of missing values
missing_summary = eda.get_missing_values_summary()
print(missing_summary)

# Plot a heatmap of missing values
fig = eda.plot_missing_values()
fig.show()
```

---

## 2. Univariate Analysis (One Variable)

Analyze each variable individually.

### Numerical Variables

Get a full statistical summary and visualize the distribution.

```python
# Get a statistical summary of all numerical variables
summary = eda.get_univariate_summary()
print(summary)

# Plot a [histogram](../plotting/plot_types.md) for a single numerical column
hist_plotter = eda.histogram_plotter()
fig_hist = hist_plotter.plot(eda.data, 'your_numerical_column', title='Distribution')
fig_hist.show()
```

### Categorical Variables

Get a summary of categories and visualize their frequencies.

```python
# Get a summary (cardinality, mode, missing) of all categorical variables
cat_summary = eda.get_categorical_summary()
print(cat_summary)

# Plot a [bar chart](../plotting/plot_types.md) of counts for a single categorical column
bar_plotter = eda.barchart_plotter()
fig_bar = bar_plotter.plot_counts(eda.data, 'your_categorical_column')
fig_bar.show()

# Plot a [pie chart](../plotting/plot_types.md)
pie_plotter = eda.pie_chart_plotter()
pie_data = eda.data['your_categorical_column'].value_counts().reset_index()
pie_data.columns = ['category', 'count']
fig_pie = pie_plotter.plot(pie_data, names_column='category', values_column='count')
fig_pie.show()
```

---

## 3. Bivariate & Mixed-Type Analysis

This is where the most valuable insights are often found, by analyzing the interactions between variables.

### Numerical vs. Numerical

Look for correlations between numerical variables.

```python
# Get the correlation matrix
corr_matrix = eda.get_correlation_matrix()

# Plot a [heatmap](../plotting/plot_types.md) of the correlation matrix
heatmap_plotter = eda.heatmap_plotter()
fig_heatmap = heatmap_plotter.plot(corr_matrix)
fig_heatmap.show()

# Create a 2D [scatter plot](../plotting/plot_types.md)
scatter_plotter = eda.scatter_plotter()
fig_scatter = scatter_plotter.plot_2d(eda.data, 'column_x', 'column_y')
fig_scatter.show()
```

### Categorical vs. Categorical

Analyze the relationship between two categorical variables.

```python
# Get a contingency table (crosstab)
crosstab = eda.get_crosstab('category_1', 'category_2')
print(crosstab)

# Visualize the crosstab as a stacked [bar chart](../plotting/plot_types.md)
bar_plotter = eda.barchart_plotter()
fig_crosstab = bar_plotter.plot_crosstab(crosstab, stacked=True)
fig_crosstab.show()
```

### Numerical vs. Categorical (Mixed-Type)

This is key for mixed-type datasets. See how a numerical variable's distribution changes across different categories.

```python
# Get summary statistics of a numerical var grouped by a categorical var
summary = eda.get_numerical_by_categorical_summary('numerical_col', 'categorical_col')
print(summary)

# Use the high-level plotting method to visualize this relationship
# Generate a [box plot](../plotting/plot_types.md)
fig_box = eda.plot_numerical_by_categorical('numerical_col', 'categorical_col', plot_type='box')
fig_box.show()

# Generate a [violin plot](../plotting/plot_types.md) (using plotly)
fig_violin = eda.plot_numerical_by_categorical(
    'numerical_col', 
    'categorical_col', 
    plot_type='violin',
    plotter_kwargs={'library': 'plotly'}
)
fig_violin.show()
```

---

## API Reference

### `ExploratoryDataAnalysis` Class

```python
class ExploratoryDataAnalysis:
    def __init__(self, data: pd.DataFrame, plotter_kwargs: dict = None)

    # --- Inspection & Summaries ---
    def classify_variables(self) -> tuple[list, list]
    def get_univariate_summary(self, alpha: float = 0.05) -> pd.DataFrame
    def get_categorical_summary(self) -> pd.DataFrame
    def get_crosstab(self, index_col: str, col_col: str, normalize: Union[bool, str] = False) -> pd.DataFrame
    def get_numerical_by_categorical_summary(self, numerical_col: str, categorical_col: str) -> pd.DataFrame
    def get_correlation_matrix(self, **kwargs) -> pd.DataFrame
    def get_missing_values_summary(self) -> pd.DataFrame
    def get_outliers(self, column: str, method: str = "iqr", threshold: float = None) -> pd.Series
    def get_vif(self) -> pd.DataFrame
    
    # --- Plotting ---
    def plot_missing_values(self, **kwargs)
    def plot_numerical_by_categorical(self, numerical_col: str, categorical_col: str, plot_type: str = "box", **kwargs)
    
    # --- Plotter Factories ---
    def histogram_plotter(self) -> HistogramPlot
    def boxplot_plotter(self) -> BoxPlot
    def barchart_plotter(self) -> BarPlot
    def pie_chart_plotter(self) -> PiePlot
    def violin_plotter(self) -> ViolinPlot
    def scatter_plotter(self) -> ScatterPlot
    def heatmap_plotter(self) -> HeatmapPlot
    # ... and others
```