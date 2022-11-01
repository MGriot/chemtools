import numpy as np
from .autoscaling import autoscaling


def matrix_variance(x):
    return np.dot(autoscaling(x).transpose(), autoscaling(x)) / (
        x.shape[0] * x.shape[1]
    )
