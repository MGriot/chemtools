import numpy as np


def matrix_mean(x, mode):
    """Function for generate the column and row mean values

    Args:
        x (numpy array or matrix): array or matrix
        mode (string): function working mode, "row" = row matrix mean, "column" = column matrix mean

    Returns:
        array: results
    """
    matrix_mean = np.array([])
    if mode == "column":
        for i in range(x.shape[1]):
            matrix_mean = np.append(matrix_mean, x[:, i].mean())
    elif mode == "row":
        for i in range(x.shape[0]):
            matrix_mean = np.append(matrix_mean, x[i, :].mean())
    return matrix_mean
