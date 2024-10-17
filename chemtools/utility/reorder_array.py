import numpy as np


def reorder_array(X):
    """
    Ordina un array dal valore più alto al più basso e restituisce l'array ordinato e gli indici originali.

    Args:
        X (np.ndarray): Array di input da ordinare.

    Returns:
        tuple: Una tupla contenente l'array ordinato e gli indici originali.
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
