# Regression Statistics

The `chemtools.stats.regression_stats` module provides essential statistical functions primarily used in the context of regression analysis. These functions are designed to support the calculation of various metrics such as degrees of freedom, R-squared values, and critical t-values.

## Available Functions

*   **`calculate_degrees_of_freedom(n_observations: int, n_parameters: int) -> int`**
    *   Calculates the residual degrees of freedom for a regression model.
*   **`centered_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float`**
    *   Calculates the centered R-squared (coefficient of determination).
*   **`uncentered_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float`**
    *   Calculates the uncentered R-squared.
*   **`uncentered_adjusted_r_squared(y: np.ndarray, y_pred: np.ndarray, k: int) -> float`**
    *   Calculates the uncentered adjusted R-squared, where `k` is the number of predictors.
*   **`t_students(alpha: float, d_f_: int) -> Tuple[float, float]`**
    *   Calculates the one-tailed and two-tailed critical values from Student's t-distribution.

## Usage Example

```python
import numpy as np
from chemtools.stats.regression_stats import (
    calculate_degrees_of_freedom,
    centered_r_squared,
    t_students,
)

# Sample data
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
n_obs = len(y_true)
n_params = 2 # e.g., for a simple linear regression with intercept and one slope

# Calculate degrees of freedom
df = calculate_degrees_of_freedom(n_obs, n_params)
print(f"Degrees of Freedom: {df}")

# Calculate centered R-squared
r2_centered = centered_r_squared(y_true, y_pred)
print(f"Centered R-squared: {r2_centered:.3f}")

# Calculate t-Student critical values
alpha = 0.05
t_one, t_two = t_students(alpha, df)
print(f"One-tailed t-critical (alpha={alpha}): {t_one:.3f}")
print(f"Two-tailed t-critical (alpha={alpha}): {t_two:.3f}")
```
