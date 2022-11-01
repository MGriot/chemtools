import numpy as np


def correlation_matrix(x):
    """_summary_

    Args:
        x (numpy matrix): autoscaled matrix

    Returns:
        (numpy matrix): correlation matrix
    """
    return (x.T @ x) / (x.shape[0])
