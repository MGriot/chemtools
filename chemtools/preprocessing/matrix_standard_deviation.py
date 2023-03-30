import numpy as np


def matrix_standard_deviation(X, mode):
    """
    The function takes in a matrix X and a string mode. If the mode is "column", the function returns
    the standard deviation of each column of X. If the mode is "row", the function returns the standard
    deviation of each row of X
    
    :param X: The matrix you want to calculate the standard deviation of
    :param mode: "column" or "row"
    :return: The standard deviation of the matrix.
    """
    if mode == "column":
        matrix_std = np.std(X, axis=0)
    elif mode == "row":
        matrix_std = np.std(X, axis=1)
    return matrix_std