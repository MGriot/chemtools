import numpy as np


def correlation_matrix(X):
    """_summary_

    Args:
        x (numpy matrix): autoscaled matrix

    Returns:
        (numpy matrix): correlation matrix
    """
    return (X.T @ X) / (X.shape[0])
