import numpy as np


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calculates the correlation matrix for a given data matrix.

    This function assumes the input matrix X is already autoscaled
    (i.e., each column has a mean of 0 and a standard deviation of 1).
    The correlation matrix is computed as (X.T @ X) / n, where n is the
    number of samples.

    Args:
        X (np.ndarray): The autoscaled input data matrix (n_samples, n_features).

    Returns:
        np.ndarray: The correlation matrix (n_features, n_features).
    """
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.empty((X.shape[1], X.shape[1]))
    return (X.T @ X) / n_samples