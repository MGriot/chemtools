import numpy as np


def matrix_standard_deviation(x, mode):
    matrix_std = np.array([])
    if mode == "column":
        for i in range(x.shape[1]):
            matrix_std = np.append(matrix_std, x[:, i].std())
    elif mode == "row":
        for i in range(x.shape[0]):
            matrix_std = np.append(matrix_std, x[i, :].std())
    return matrix_std
