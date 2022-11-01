import numpy as np


def make_standards(x_min, x_max, n_standard=7, decimal=1):
    """
    Function that allows to obtain an array that contains the optimal concentrations to create a calibration
    line with equidistant points given a minimum and a maximum value.

    Args:
        x_min (_type_): minimum value of the merger domain in which to create the set of standards.
        x_max (_type_): maximum value of the merger domain in which to create the set of standards.
        n_standard (int, optional): number of standards you want to achieve. Defaults to 7.
        decimal (int, optional): number of decimal places to which you want to round the result. Defaults to 1.

    Returns:
        return: array of size n, number of standards, with the correct number of significant digits.
    """
    return np.around(np.linspace(x_min, x_max, n_standard), decimal)
