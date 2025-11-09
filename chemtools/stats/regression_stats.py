import numpy as np
from scipy import stats
from typing import Union, Tuple, Optional

def calculate_degrees_of_freedom(n_observations: int, n_parameters: int) -> int:
    """
    Calculates the residual degrees of freedom for a regression model.

    Parameters:
    n_observations: int
        The total number of observations (samples).
    n_parameters: int
        The total number of parameters in the model (including intercept if fitted).

    Returns:
    int
        The residual degrees of freedom for the regression.
    """
    if n_observations <= n_parameters:
        raise ValueError(
            "Number of observations must be greater than the number of parameters "
            "to calculate degrees of freedom."
        )
    return n_observations - n_parameters


def centered_r_squared(y, y_pred):
    """
    Calculates the centered R^2.

    Parameters:
    y : array
        Observed values
    y_pred : array
        Predicted values from the model

    Returns:
    r_squared : float
        Centered R^2 determination coefficient
    """
    numerator = np.sum(np.square(y_pred - np.mean(y)))
    denominator = np.sum(np.square(y - np.mean(y)))
    return numerator / denominator


def uncentered_r_squared(y, y_pred):
    """
    Calculates the uncentered R^2.

    Parameters:
    y : array
        Observed values
    y_pred : array
        Predicted values from the model

    Returns:
    r_squared : float
        Uncentered R^2 determination coefficient
    """
    numerator = np.sum(np.square(y_pred))
    denominator = np.sum(np.square(y))
    return numerator / denominator


def uncentered_adjusted_r_squared(y, y_pred, k):
    """
    Calculates the uncentered adjusted R^2.

    Parameters:
    y : array
        Observed values
    y_pred : array
        Predicted values from the model
    k : int
        Number of predictors in the model

    Returns:
    adjusted_r_squared : float
        Uncentered adjusted R^2 determination coefficient
    """
    n = len(y)
    return 1 - (1 - uncentered_r_squared(y, y_pred)) * (n - 1) / (n - k)


def t_students(alpha, d_f_):
    """
    Calculates the one-tailed and two-tailed critical values from
    Student's t-distribution.

    Args:
        alpha (float): The significance level.
        d_f_ (int): The degrees of freedom.

    Returns:
        tuple: A tuple containing the one-tailed and two-tailed critical values.
    """
    t_one = stats.t.ppf(1 - alpha, d_f_)
    t_two = stats.t.ppf(1 - alpha / 2, d_f_)
    return t_one, t_two
