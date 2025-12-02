# Exploration Module Reference (`chemtools.exploration`)

The `chemtools.exploration` module provides classes and functions for exploratory data analysis (EDA) and dimensionality reduction techniques, including Principal Component Analysis (PCA) and Factor Analysis of Mixed Data (FAMD).

---

## `ExploratoryDataAnalysis` Class

The `ExploratoryDataAnalysis` class is designed to work with pandas DataFrames and provides a suite of methods for data inspection, non-graphical summaries, and graphical analysis.

### `ExploratoryDataAnalysis(data: pd.DataFrame, plotter_kwargs: dict = None)`

*   **Parameters:**
    *   `data` (`pd.DataFrame`): The input DataFrame for analysis.
    *   `plotter_kwargs` (`dict`, optional): Keyword arguments to pass to plotters for setting default theme, etc.

### Methods

*   **`classify_variables(self) -> tuple[list, list]`**
    *   Classifies columns into numerical and categorical types.
    *   **Returns:** `tuple[list, list]`: Two lists, one for numerical column names and one for categorical.

*   **`get_univariate_summary(self, alpha: float = 0.05) -> pd.DataFrame`**
    *   Generates a statistical summary for all numerical variables.
    *   **Parameters:** `alpha` (`float`, optional): Significance level for confidence intervals.
    *   **Returns:** `pd.DataFrame`: DataFrame with univariate statistics.

*   **`get_categorical_summary(self) -> pd.DataFrame`**
    *   Generates a summary for all categorical variables (cardinality, mode, missing).
    *   **Returns:** `pd.DataFrame`: DataFrame with categorical summaries.

*   **`get_crosstab(self, index_col: str, col_col: str, normalize: Union[bool, str] = False) -> pd.DataFrame`**
    *   Computes a frequency table for two categorical variables.
    *   **Parameters:**
        *   `index_col` (`str`): Column for the DataFrame index.
        *   `col_col` (`str`): Column for the DataFrame columns.
        *   `normalize` (`Union[bool, str]`, optional): Normalization method.
    *   **Returns:** `pd.DataFrame`: Contingency table.

*   **`get_numerical_by_categorical_summary(self, numerical_col: str, categorical_col: str) -> pd.DataFrame`**
    *   Provides summary statistics of a numerical variable grouped by a categorical variable.
    *   **Returns:** `pd.DataFrame`: Grouped summary statistics.

*   **`get_correlation_matrix(self, **kwargs) -> pd.DataFrame`**
    *   Calculates the correlation matrix for numerical variables.
    *   **Returns:** `pd.DataFrame`: Correlation matrix.

*   **`get_missing_values_summary(self) -> pd.DataFrame`**
    *   Summarizes missing values per column.
    *   **Returns:** `pd.DataFrame`: Summary of missing values.

*   **`get_outliers(self, column: str, method: str = "iqr", threshold: float = None) -> pd.Series`**
    *   Identifies outliers in a specified column.
    *   **Returns:** `pd.Series`: Boolean series indicating outliers.

*   **`get_vif(self) -> pd.DataFrame`**
    *   Calculates Variance Inflation Factor (VIF) for multicollinearity assessment.
    *   **Returns:** `pd.DataFrame`: VIF values.

*   **`plot_missing_values(self, **kwargs)`**
    *   Plots a heatmap of missing values.

*   **`plot_numerical_by_categorical(self, numerical_col: str, categorical_col: str, plot_type: str = "box", **kwargs)`**
    *   Generates plots (box or violin) to visualize numerical distribution across categories.

*   **Plotter Factories:** Methods to get instances of specific plotters, pre-configured with EDA's data and settings.
    *   `histogram_plotter()` -> `HistogramPlot`
    *   `boxplot_plotter()` -> `BoxPlot`
    *   `barchart_plotter()` -> `BarPlot`
    *   `pie_chart_plotter()` -> `PiePlot`
    *   `violin_plotter()` -> `ViolinPlot`
    *   `scatter_plotter()` -> `ScatterPlot`
    *   `heatmap_plotter()` -> `HeatmapPlot`

### Usage Example (EDA)

```python
import pandas as pd
from chemtools.exploration import ExploratoryDataAnalysis

df = pd.read_csv('your_data.csv')
eda = ExploratoryDataAnalysis(df)
numerical_cols, categorical_cols = eda.classify_variables()
print("Numerical Columns:", numerical_cols)
```

---

## `PrincipalComponentAnalysis` Class

Performs Principal Component Analysis (PCA) for dimensionality reduction and data exploration.

### `PrincipalComponentAnalysis()`

*   **Parameters:** None (initialization without specific arguments).

### Methods

*   **`fit(self, X, variables_names=None, objects_names=None)`**
    *   Fits the PCA model to the data.
    *   **Parameters:**
        *   `X` (`np.ndarray` or `pd.DataFrame`): Input data.
        *   `variables_names` (`list`, optional): Names for the variables (columns).
        *   `objects_names` (`list`, optional): Names for the observations (rows).

*   **`reduction(self, n_components)`**
    *   Reduces the dimensionality of the fitted data.
    *   **Parameters:** `n_components` (`int`): Number of components to retain.

*   **`transform(self, X_new)`**
    *   Applies the learned PCA transformation to new data.
    *   **Returns:** `np.ndarray`: Transformed data.

*   **`statistics(self, alpha=0.05)`**
    *   Calculates various statistical metrics for the PCA model.

### Usage Example (PCA)

```python
from chemtools.exploration import PrincipalComponentAnalysis
import numpy as np
# Assuming X, variables, and objects are defined
X = np.random.rand(10, 5) # Sample data
variables = [f'Var{i}' for i in range(5)]
objects = [f'Obs{i}' for i in range(10)]

pca = PrincipalComponentAnalysis()
pca.fit(X, variables_names=variables, objects_names=objects)
print(pca.summary)
```

---

## `FactorAnalysisOfMixedData` Class

Performs Factor Analysis of Mixed Data (FAMD) for datasets containing both quantitative and qualitative variables.

### `FactorAnalysisOfMixedData(n_components: int = 2)`

*   **Parameters:**
    *   `n_components` (`int`, optional): The number of components to retain. Defaults to `2`.

### Methods

*   **`fit(self, X: pd.DataFrame, qualitative_variables: list)`**
    *   Fits the FAMD model to the mixed data.
    *   **Parameters:**
        *   `X` (`pd.DataFrame`): Input DataFrame with mixed data.
        *   `qualitative_variables` (`list`): List of column names identifying qualitative variables.

*   **`transform(self, X_new: pd.DataFrame) -> np.ndarray`**
    *   Applies the learned FAMD transformation to new data.
    *   **Returns:** `np.ndarray`: Transformed data.

### Usage Example (FAMD)

```python
import pandas as pd
from chemtools.dimensional_reduction import FactorAnalysisOfMixedData

data = pd.DataFrame({
    'quant1': [1.2, 2.3, 3.4, 4.5, 5.6],
    'quant2': [10.1, 9.0, 8.9, 7.8, 6.7],
    'qual1': ['A', 'A', 'B', 'B', 'A'],
    'qual2': ['X', 'Y', 'X', 'Y', 'X']
})
qualitative_vars = ['qual1', 'qual2']

famd = FactorAnalysisOfMixedData(n_components=2)
famd.fit(data, qualitative_variables=qualitative_vars)
print(famd.summary)
```
