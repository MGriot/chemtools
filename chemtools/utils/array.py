import numpy as np
from typing import Tuple


def array_to_column(array):
    """Converts a 1D array to a multi-line string."""
    return "".join(str(i) + "\n" for i in array)


def reorder_array(X):
    """
    Sorts an array from the highest value to the lowest and returns the
    sorted array and the original indices.

    Args:
        X (np.ndarray): Input array to be sorted.

    Returns:
        tuple: A tuple containing the sorted array and the original indices.
    """
    Y = np.arange(0, X.size, 1)
    sorted_indices = np.argsort(X)
    X = X[sorted_indices]
    Y = Y[sorted_indices]
    return np.flip(X), np.flip(Y)


def sort_arrays(x, y):
    """
    Sorts the rows of matrix `x` and vector `y` based on the
    ascending order of the non-constant (or non-one) column of `x`.

    Args:
        x (numpy.ndarray): A 2D array (matrix).
        y (numpy.ndarray): A 1D array (vector) with the same
                           number of rows as `x`.

    Returns:
        tuple: A tuple containing the sorted `x` and `y` arrays.
               If all columns of `x` are constant or one,
               returns the original arrays.

    Raises:
        ValueError: If the number of rows in `x` and `y` don't match.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays 'x' and 'y' must have the same number of rows.")

    # Find the non-constant column (skip if all elements are 1 or constant)
    for col_index in range(x.shape[1]):
        if len(set(x[:, col_index])) > 1:
            # Found a non-constant column
            sorted_indices = np.argsort(x[:, col_index])
            return x[sorted_indices], y[sorted_indices]

    # No non-constant columns found, return original arrays
    return x, y


def unfold_hypercube(cube: np.ndarray) -> np.ndarray:
    """
    Unfolds a 3D data cube (hypercube) into a 2D matrix.

    This is a common step for multivariate analysis of imaging data (e.g., MA-XRF),
    where each pixel becomes an object (row) and spectral channels are variables (columns).

    Args:
        cube (np.ndarray): A 3D numpy array with shape (height, width, depth/channels).

    Returns:
        np.ndarray: A 2D numpy array with shape (height * width, depth/channels).
    """
    if cube.ndim != 3:
        raise ValueError("Input must be a 3D numpy array (hypercube).")
    h, w, d = cube.shape
    return cube.reshape((h * w, d))


def refold_hypercube(matrix: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Refolds a 2D matrix back into a 3D data cube (hypercube).

    This is used to visualize results (e.g., PCA scores) as a map after analysis.

    Args:
        matrix (np.ndarray): A 2D numpy array with shape (h * w, depth/channels).
        h (int): The original height of the hypercube.
        w (int): The original width of the hypercube.

    Returns:
        np.ndarray: A 3D numpy array with shape (h, w, depth/channels).
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be a 2D numpy array.")
    if matrix.shape[0] != h * w:
        raise ValueError(f"Matrix rows ({matrix.shape[0]}) do not match target dimensions ({h}x{w}={h*w}).")
    
    d = matrix.shape[1]
    return matrix.reshape((h, w, d))