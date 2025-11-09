import numpy as np

def autoscaling(x: np.ndarray) -> np.ndarray:
    """
    Performs autoscaling (standardization) on the input data.

    Autoscaling transforms the data so that each column (variable) has a mean of 0
    and a standard deviation of 1. This is also known as Z-score normalization.

    The formula for autoscaling an element x_ij is:
        x_scaled_ij = (x_ij - mean_j) / std_dev_j
    where mean_j and std_dev_j are the mean and standard deviation of the j-th column.

    Args:
        x (np.ndarray): The input data matrix (n_samples, n_features).

    Returns:
        np.ndarray: The autoscaled data matrix.
    
    References:
        - https://en.wikipedia.org/wiki/Standard_score
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    # Avoid division by zero for columns with zero standard deviation
    std[std == 0] = 1.0
    
    return (x - mean) / std