import numpy as np


def diagonalized_matrix(x):
    """_summary_

    Args:
        x (numpy arra, matrix): _description_

    Returns:
        V: eigenvalues
        L: eigenvectors
    """
    V, L = np.linalg.eig(x)
    return (V, L)
