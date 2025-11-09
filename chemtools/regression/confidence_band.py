import numpy as np
from typing import Tuple

def confidence_band(number_data: int, X: np.ndarray, x_mean: np.ndarray, y_pred: np.ndarray, SSxx: np.ndarray, t_two: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the confidence band for a linear regression.

    The confidence band provides a range of values within which the true regression
    line is expected to lie with a certain level of confidence.

    Args:
        number_data (int): Total number of observations.
        X (np.ndarray): Array containing the x-values for which the confidence band is to be calculated.
        x_mean (np.ndarray): Mean of the x-values.
        y_pred (np.ndarray): Array containing the predicted y-values for the input x-values.
        SSxx (np.ndarray): Sum of squares of the deviations of X from its mean.
        t_two (float): The two-tailed critical value from Student's t-distribution.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - CI_Y_upper (np.ndarray): The upper bounds of the confidence band for each input x-value.
            - CI_Y_lower (np.ndarray): The lower bounds of the confidence band for each input x-value.

    Example:
        import numpy as np
        from scipy import stats

        # Define input parameters
        number_data = 100
        x = np.array([...])
        x_mean = np.mean(x)
        y_pred = np.array([...])
        SSxx = np.sum((x - x_mean) ** 2)
        t_two = stats.t.ppf(1 - 0.05 / 2, number_data - 2)

        # Call the confidence_band function
        CI_Y_upper, CI_Y_lower = confidence_band(number_data, x, x_mean, y_pred, SSxx, t_two)

        # Display the upper and lower bounds of the confidence band
        print(CI_Y_upper)
        print(CI_Y_lower)
    """
    # Calculate the upper and lower confidence band using vector operations
    diff = X - x_mean
    CI_term = t_two * np.sqrt(1 / number_data + np.sum((diff**2 / SSxx), axis=1))
    CI_Y_upper = y_pred + CI_term
    CI_Y_lower = y_pred - CI_term
    return CI_Y_upper, CI_Y_lower