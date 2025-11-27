import numpy as np
import pandas as pd
from scipy.stats import gmean, skew, kurtosis as scipy_kurtosis, t
from typing import Union, Tuple, Optional

def sample_number(data: Union[np.ndarray, pd.Series]) -> int:
    """
    Calculates the number of samples (n).
    """
    if isinstance(data, pd.Series):
        return data.count() # Excludes NaN values
    return len(data)

def arithmetic_mean(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the arithmetic mean.
    """
    return np.mean(data)

def median(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the median.
    """
    return np.median(data)

def geometric_mean(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the geometric mean.
    Handles non-positive values by returning NaN.
    """
    if np.any(data <= 0):
        return np.nan
    return gmean(data)

def variance(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the sample variance (ddof=1).
    """
    return np.var(data, ddof=1)

def standard_deviation(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the sample standard deviation (ddof=1).
    """
    return np.std(data, ddof=1)

def relative_standard_deviation(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the relative standard deviation (RSD) in percentage.
    Returns NaN if mean is zero.
    """
    std_dev = standard_deviation(data)
    mean_val = arithmetic_mean(data)
    if mean_val == 0:
        return np.nan
    return (std_dev / mean_val) * 100

def standard_error(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the standard error of the mean.
    """
    n = sample_number(data)
    if n <= 1:
        return np.nan
    return standard_deviation(data) / np.sqrt(n)

def confidence_interval_mean(data: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculates the confidence interval for the mean.
    Returns (lower_bound, upper_bound).
    """
    n = sample_number(data)
    if n <= 1:
        return np.nan, np.nan
    
    mean_val = arithmetic_mean(data)
    std_err = standard_error(data)
    
    degrees_freedom = n - 1
    t_critical = t.ppf(1 - alpha / 2, degrees_freedom)
    
    margin_of_error = t_critical * std_err
    
    return mean_val - margin_of_error, mean_val + margin_of_error

def minimum_value(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the minimum value.
    """
    return np.min(data)

def maximum_value(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the maximum value.
    """
    return np.max(data)

def data_range(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the range (max - min).
    """
    return maximum_value(data) - minimum_value(data)

def lower_quartile(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the first quartile (Q1).
    """
    return np.percentile(data, 25)

def upper_quartile(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the third quartile (Q3).
    """
    return np.percentile(data, 75)

def interquartile_range(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the interquartile range (IQR = Q3 - Q1).
    """
    return upper_quartile(data) - lower_quartile(data)

def mean_absolute_difference(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the mean absolute difference (Gini mean absolute difference).
    """
    data_arr = np.asarray(data)
    n = len(data_arr)
    if n < 2:
        return 0.0
    diff_matrix = np.abs(data_arr[:, None] - data_arr[None, :])
    return np.sum(diff_matrix) / (n * (n - 1))

def median_absolute_deviation(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the median absolute deviation (MAD).
    """
    return np.median(np.abs(data - np.median(data)))

def average_absolute_deviation(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the average absolute deviation (mean absolute deviation from the mean).
    """
    return np.mean(np.abs(data - np.mean(data)))

def quartile_coefficient_of_dispersion(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the quartile coefficient of dispersion.
    Returns NaN if Q1 + Q3 is zero.
    """
    q1 = lower_quartile(data)
    q3 = upper_quartile(data)
    denominator = q3 + q1
    if denominator == 0:
        return np.nan
    return (q3 - q1) / denominator

def skewness(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the skewness.
    """
    return skew(data)

def kurtosis(data: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculates the Fisher (excess) kurtosis.
    """
    return scipy_kurtosis(data)

def descriptive_statistics(data: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> dict:
    """
    Calculates a comprehensive set of descriptive statistics for a given dataset.
    """
    if isinstance(data, pd.Series):
        data = data.dropna() # Ensure NaN values are handled consistently
    
    n = sample_number(data)
    if n == 0:
        return {
            "Sample Number": 0,
            "Arithmetic Mean": np.nan,
            "Median": np.nan,
            "Geometric Mean": np.nan,
            "Variance": np.nan,
            "Standard Deviation": np.nan,
            "Relative Standard Deviation (%)": np.nan,
            "Standard Error": np.nan,
            f"Confidence Interval (alpha={alpha})": (np.nan, np.nan),
            "Minimum Value": np.nan,
            "Maximum Value": np.nan,
            "Range": np.nan,
            "Lower Quartile (Q1)": np.nan,
            "Upper Quartile (Q3)": np.nan,
            "Interquartile Range (IQR)": np.nan,
            "Mean Absolute Difference": np.nan,
            "Median Absolute Deviation (MAD)": np.nan,
            "Average Absolute Deviation": np.nan,
            "Quartile Coefficient of Dispersion": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
        }

    mean_val = arithmetic_mean(data)
    std_dev = standard_deviation(data)
    
    ci_lower, ci_upper = confidence_interval_mean(data, alpha)

    stats = {
        "Sample Number": n,
        "Arithmetic Mean": mean_val,
        "Median": median(data),
        "Geometric Mean": geometric_mean(data),
        "Variance": variance(data),
        "Standard Deviation": std_dev,
        "Relative Standard Deviation (%)": relative_standard_deviation(data),
        "Standard Error": standard_error(data),
        f"Confidence Interval (alpha={alpha})": (ci_lower, ci_upper),
        "Minimum Value": minimum_value(data),
        "Maximum Value": maximum_value(data),
        "Range": data_range(data),
        "Lower Quartile (Q1)": lower_quartile(data),
        "Upper Quartile (Q3)": upper_quartile(data),
        "Interquartile Range (IQR)": interquartile_range(data),
        "Mean Absolute Difference": mean_absolute_difference(data),
        "Median Absolute Deviation (MAD)": median_absolute_deviation(data),
        "Average Absolute Deviation": average_absolute_deviation(data),
        "Quartile Coefficient of Dispersion": quartile_coefficient_of_dispersion(data),
        "Skewness": skewness(data),
        "Kurtosis": kurtosis(data),
    }
    return stats