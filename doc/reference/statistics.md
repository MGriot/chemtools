# Statistics Module Reference (`chemtools.stats`)

The `chemtools.stats` module provides a comprehensive set of functions for calculating descriptive statistics, performing Analysis of Variance (ANOVA), and essential regression statistics.

---

## Univariate Statistics (`chemtools.stats.univariate`)

This submodule offers functions for calculating descriptive statistics for a single variable, designed to work with NumPy arrays and Pandas Series.

### `descriptive_statistics(data: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> dict`

Calculates a comprehensive set of descriptive statistics for a given dataset.

*   **Parameters:**
    *   `data` (`Union[np.ndarray, pd.Series]`): The input data.
    *   `alpha` (`float`, optional): Significance level for confidence intervals. Defaults to `0.05`.
*   **Returns:** `dict`: A dictionary containing all calculated descriptive statistics.

### Usage Example (Univariate Statistics)

```python
import pandas as pd
from chemtools.stats.univariate import descriptive_statistics

data = pd.Series([10, 12, 12, 13, 15, 17, 18, 20, 22, 24, 25, 27, 28, 30, 32])
stats_report = descriptive_statistics(data, alpha=0.05)

for stat, value in stats_report.items():
    if isinstance(value, tuple):
        print(f"{stat}: ({value[0]:.3f}, {value[1]:.3f})")
    elif isinstance(value, (int, float)):
        print(f"{stat}: {value:.3f}")
    else:
        print(f"{stat}: {value}")
```

---

<h2>Analysis of Variance (ANOVA) Classes (`chemtools.stats.anova`)</h2>

This submodule provides classes for performing various types of Analysis of Variance.

<h3><code>OneWayANOVA</code> Class</h3>

Performs One-Way Analysis of Variance.

<h3><code>OneWayANOVA()</code></h3>

*   <b>Parameters:</b> None

<h3>Methods</h3>

*   <b><code>fit(self, data: pd.DataFrame, value_column: str, group_column: str)</code></b>
    *   Fits the One-Way ANOVA model.
    *   <b>Parameters:</b>
        *   `data` (`pd.DataFrame`): Input DataFrame.
        *   `value_column` (`str`): Name of the column containing the continuous dependent variable.
        *   `group_column` (`str`): Name of the column containing the categorical independent variable (groups).

*   <b><code>_get_summary_data(self) -> Dict[str, Any]</code></b>
    *   Internal method to collect summary data.

<h3>Usage Example (One-Way ANOVA)</h3>

```python
import pandas as pd
from chemtools.stats.anova import OneWayANOVA

data = pd.DataFrame({
    'Value': [10, 12, 11, 15, 14, 18, 17, 20, 19],
    'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
})
anova_model = OneWayANOVA()
anova_model.fit(data, value_column='Value', group_column='Group')
print(anova_model.summary)
```

<h3><code>TwoWayANOVA</code> Class</h3>

Performs Two-Way Analysis of Variance for a balanced design with repetitions.

<h3><code>TwoWayANOVA()</code></h3>

*   <b>Parameters:</b> None

<h3>Methods</h3>

*   <b><code>fit(self, data: pd.DataFrame, value_column: str, factor1_column: str, factor2_column: str)</code></b>
    *   Fits the Two-Way ANOVA model.
    *   <b>Parameters:</b>
        *   `data` (`pd.DataFrame`): Input DataFrame.
        *   `value_column` (`str`): Name of the continuous dependent variable column.
        *   `factor1_column` (`str`): Name of the first categorical independent variable column.
        *   `factor2_column` (`str`): Name of the second categorical independent variable column.

*   <b><code>_get_summary_data(self) -> Dict[str, Any]</code></b>
    *   Internal method to collect summary data.

<h3>Usage Example (Two-Way ANOVA)</h3>

```python
import pandas as pd
from chemtools.stats.anova import TwoWayANOVA

data_two_way = pd.DataFrame({
    'Value': [10, 11, 15, 16, 18, 19, 22, 23],
    'Factor1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2'],
    'Factor2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B1', 'B2', 'B2']
})
anova_model_two_way = TwoWayANOVA()
anova_model_two_way.fit(data_two_way, value_column='Value', factor1_column='Factor1', factor2_column='Factor2')
print(anova_model_two_way.summary)
```

<h3><code>MultiwayANOVA</code> Class (Placeholder)</h3>

Extends Two-Way ANOVA. (Full implementation not provided in this version).

<h3><code>MANOVA</code> Class (Placeholder)</h3>

Multivariate Analysis of Variance. (Full implementation not provided in this version).

---

<h2>Regression Statistics (`chemtools.stats.regression_stats`)</h2>

This submodule provides essential statistical functions primarily used in the context of regression analysis.

<h3><code>calculate_degrees_of_freedom(n_observations: int, n_parameters: int) -> int</code></h3>

Calculates the residual degrees of freedom for a regression model.

*   <b>Parameters:</b>
    *   `n_observations` (`int`): Number of observations.
    *   `n_parameters` (`int`): Number of parameters in the model.
*   <b>Returns:</b> `int`: Residual degrees of freedom.

<h3><code>centered_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float</code></h3>

Calculates the centered R-squared (coefficient of determination).

*   <b>Parameters:</b>
    *   `y` (`np.ndarray`): True values.
    *   `y_pred` (`np.ndarray`): Predicted values.
*   <b>Returns:</b> `float`: Centered R-squared value.

<h3><code>uncentered_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float</code></h3>

Calculates the uncentered R-squared.

*   <b>Parameters:</b>
    *   `y` (`np.ndarray`): True values.
    *   `y_pred` (`np.ndarray`): Predicted values.
*   <b>Returns:</b> `float`: Uncentered R-squared value.

<h3><code>uncentered_adjusted_r_squared(y: np.ndarray, y_pred: np.ndarray, k: int) -> float</code></h3>

Calculates the uncentered adjusted R-squared.

*   <b>Parameters:</b>
    *   `y` (`np.ndarray`): True values.
    *   `y_pred` (`np.ndarray`): Predicted values.
    *   `k` (`int`): Number of predictors.
*   <b>Returns:</b> `float`: Uncentered adjusted R-squared value.

<h3><code>t_students(alpha: float, d_f_: int) -> Tuple[float, float]`</h3>

Calculates the one-tailed and two-tailed critical values from Student's t-distribution.

*   <b>Parameters:</b>
    *   `alpha` (`float`): Significance level.
    *   `d_f_` (`int`): Degrees of freedom.
*   <b>Returns:</b> `Tuple[float, float]`: (One-tailed critical value, Two-tailed critical value).

<h3>Usage Example (Regression Statistics)</h3>

```python
import numpy as np
from chemtools.stats.regression_stats import (
    calculate_degrees_of_freedom,
    centered_r_squared,
    t_students,
)

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
n_obs = len(y_true)
n_params = 2 

df = calculate_degrees_of_freedom(n_obs, n_params)
print(f"Degrees of Freedom: {df}")

r2_centered = centered_r_squared(y_true, y_pred)
print(f"Centered R-squared: {r2_centered:.3f}")

alpha = 0.05
t_one, t_two = t_students(alpha, df)
print(f"One-tailed t-critical (alpha={alpha}): {t_one:.3f}")
print(f"Two-tailed t-critical (alpha={alpha}): {t_two:.3f}")
```
