# Univariate Statistics

The `chemtools.stats.univariate` module provides a comprehensive set of functions for calculating descriptive statistics for a single variable. These functions are designed to work with both NumPy arrays and Pandas Series.

## Available Functions

The module includes functions for:

*   **Measures of Central Tendency:**
    *   `sample_number`: Number of observations.
    *   `arithmetic_mean`: The average of the data.
    *   `median`: The middle value of the data.
    *   `geometric_mean`: The nth root of the product of n numbers.
*   **Measures of Dispersion:**
    *   `variance`: Sample variance (ddof=1).
    *   `standard_deviation`: Sample standard deviation (ddof=1).
    *   `relative_standard_deviation`: Coefficient of variation (RSD) in percentage.
    *   `standard_error`: Standard error of the mean.
    *   `mean_absolute_difference`: Gini mean absolute difference.
    *   `median_absolute_deviation`: Median absolute deviation (MAD).
    *   `average_absolute_deviation`: Mean absolute deviation from the mean.
    *   `quartile_coefficient_of_dispersion`: Dimensionless measure of dispersion.
    *   `data_range`: Maximum value minus minimum value.
    *   `interquartile_range`: Q3 - Q1.
*   **Measures of Position:**
    *   `minimum_value`: The smallest value.
    *   `maximum_value`: The largest value.
    *   `lower_quartile`: The 25th percentile (Q1).
    *   `upper_quartile`: The 75th percentile (Q3).
*   **Measures of Shape:**
    *   `skewness`: Measure of the asymmetry of the probability distribution.
    *   `kurtosis`: Measure of the "tailedness" of the probability distribution.
*   **Confidence Intervals:**
    *   `confidence_interval_mean`: Confidence interval for the mean.

## Comprehensive Statistics

The `descriptive_statistics` function provides a single entry point to calculate all the above statistics for a given dataset.

### `descriptive_statistics` Function

```python
def descriptive_statistics(data: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> dict:
    """
    Calculates a comprehensive set of descriptive statistics for a given dataset.

    Args:
        data (Union[np.ndarray, pd.Series]): The input data.
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.

    Returns:
        dict: A dictionary containing all calculated descriptive statistics.
    """
```

## Usage Example

```python
import pandas as pd
from chemtools.stats.univariate import descriptive_statistics

# Sample data
data = pd.Series([10, 12, 12, 13, 15, 17, 18, 20, 22, 24, 25, 27, 28, 30, 32])

# Calculate descriptive statistics
stats_report = descriptive_statistics(data, alpha=0.05)

# Print the report
for stat, value in stats_report.items():
    if isinstance(value, tuple):
        print(f"{stat}: ({value[0]:.3f}, {value[1]:.3f})")
    elif isinstance(value, (int, float)):
        print(f"{stat}: {value:.3f}")
    else:
        print(f"{stat}: {value}")
```
