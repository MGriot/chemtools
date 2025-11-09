import numpy as np
from typing import Union

def matrix_variance(x: np.ndarray, axis: int = 0) -> Union[np.ndarray, float]:
    """
    Calculates the variance of a matrix along a specified axis.

    This function computes the variance for each column (axis=0) or each row (axis=1)
    of the input matrix. The variance is a measure of the spread of the data.

    The formula for the variance of a vector `a` is:
        Var(a) = sum((a_i - mean(a))^2) / (n - 1)
    where n is the number of elements in `a`. This is the sample variance.

    Args:
        x (np.ndarray): The input data matrix (n_samples, n_features).
        axis (int): The axis along which to calculate the variance.
                    0 for column-wise variance (default).
                    1 for row-wise variance.

    Returns:
        Union[np.ndarray, float]: 
            - If axis is 0 or 1, returns an array of variances.
            - If axis is None, returns a single float value for the entire matrix.
            
    References:
        - https://en.wikipedia.org/wiki/Variance
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array.")
        
    # ddof=1 calculates the sample variance, which is standard practice.
    return np.var(x, axis=axis, ddof=1)