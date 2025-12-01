import numpy as np

def row_normalize_sum(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the rows of a matrix to a constant sum (typically 1).

    This is a common preprocessing step for spectroscopic data to account for
    variations in signal intensity due to factors like sample path length or
    measurement time. Also known as Total Sum Normalization.

    Args:
        x (np.ndarray): The input data matrix (n_samples, n_features).

    Returns:
        np.ndarray: The row-normalized data matrix.
    """
    row_sums = x.sum(axis=1)
    # Avoid division by zero for rows that sum to zero
    row_sums[row_sums == 0] = 1
    return x / row_sums[:, np.newaxis]

def pareto_scale(x: np.ndarray) -> np.ndarray:
    """
    Performs Pareto scaling on the input data.

    Pareto scaling involves mean-centering the data and then dividing each
    variable by the square root of its standard deviation. It is a popular
    scaling method in metabolomics and other fields as it down-weights large-fold
    changes, making it a good compromise between autoscaling and no scaling.

    Args:
        x (np.ndarray): The input data matrix (n_samples, n_features).

    Returns:
        np.ndarray: The Pareto-scaled data matrix.
    """
    mean_centered = x - np.mean(x, axis=0)
    sqrt_std = np.sqrt(np.std(mean_centered, axis=0))
    # Avoid division by zero
    sqrt_std[sqrt_std == 0] = 1.0
    return mean_centered / sqrt_std
