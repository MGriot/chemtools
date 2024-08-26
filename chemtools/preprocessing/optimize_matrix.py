import numpy as np


def optimize_matrix(X, method="mean_column"):
    """
    Optimize the matrix by replacing or removing NaN values.

    Args:
        X (numpy.ndarray): Input matrix.
        method (str): Method to replace NaN values. Can be 'zero', 'mean_column', 'mean_row', 'median_column', 'median_row', 'remove'.

    Returns:
        numpy.ndarray: Optimized matrix.
    """
    X_optimized = X.copy()

    if method == "zero":
        # Replace NaNs with 0
        X_optimized = np.nan_to_num(X, nan=0.0)

    elif method in ["mean_column", "median_column"]:
        # Replace NaNs with column mean or median
        col_stat = (
            np.nanmean(X, axis=0)
            if method == "mean_column"
            else np.nanmedian(X, axis=0)
        )
        inds = np.where(np.isnan(X))
        X_optimized[inds] = np.take(col_stat, inds[1])

    elif method in ["mean_row", "median_row"]:
        # Replace NaNs with row mean or median
        row_stat = (
            np.nanmean(X, axis=1) if method == "mean_row" else np.nanmedian(X, axis=1)
        )
        inds = np.where(np.isnan(X))
        X_optimized[inds] = np.take(row_stat, inds[0])

    elif method == "remove":
        # Remove rows with NaNs
        X_optimized = X[~np.isnan(X).any(axis=1)]

    else:
        raise ValueError(
            "Invalid method. Choose from 'zero', 'mean_column', 'mean_row', 'median_column', 'median_row', 'remove'."
        )

    return X_optimized
