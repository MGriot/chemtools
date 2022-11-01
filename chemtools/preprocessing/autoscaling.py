import numpy as np

from .matrix_mean import matrix_mean
from .matrix_standard_deviation import matrix_standard_deviation


def autoscaling(x):
    X_a = np.zeros((x.shape[0], x.shape[1]))
    column_mean = matrix_mean(x, mode="column")
    column_std = matrix_standard_deviation(x, mode="column")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] - column_mean[j] == 0:
                a = 0
            else:
                a = (x[i, j] - column_mean[j]) / column_std[j]
            X_a[i, j] = a
    return X_a
