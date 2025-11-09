import numpy as np
from typing import Tuple

def diagonalized_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs eigenvalue decomposition of a square matrix.

    This function computes the eigenvalues and right eigenvectors of a square matrix.
    The matrix is decomposed into the form X = L * diag(V) * L^-1, where V are the
    eigenvalues and L are the eigenvectors.

    Args:
        X (np.ndarray): A square matrix (n_features, n_features) to be diagonalized.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - V (np.ndarray): The eigenvalues of the matrix.
            - L (np.ndarray): The corresponding eigenvectors, where the column L[:, i]
                              is the eigenvector corresponding to the eigenvalue V[i].
    
    References:
        - https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
    """
    if X.shape[0] != X.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    V, L = np.linalg.eig(X)
    return V, L