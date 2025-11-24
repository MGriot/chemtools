import numpy as np
from typing import Tuple

def prediction_band(number_data: int, X: np.ndarray, x_mean: np.ndarray, y_pred: np.ndarray, SSxx: np.ndarray, t_two: float, fit_intercept: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the prediction band for a linear regression.

    The prediction band provides a range of values within which a future observation
    is expected to fall with a certain level of confidence. It is wider than the
    confidence band because it accounts for both the uncertainty in the estimated

    Args:
        number_data (int): Total number of observations.
        X (np.ndarray): Array containing the x-values for which the prediction band is to be calculated.
        x_mean (np.ndarray): Mean of the x-values.
        y_pred (np.ndarray): Array containing the predicted y-values for the input x-values.
        SSxx (np.ndarray): Sum of squares of the deviations of X from its mean.
        t_two (float): The two-tailed critical value from Student's t-distribution.
        fit_intercept (bool): Whether the model was fit with an intercept.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - PI_Y_upper (np.ndarray): The upper bounds of the prediction band for each input x-value.
            - PI_Y_lower (np.ndarray): The lower bounds of the prediction band for each input x-value.

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

        # Call the prediction_band function
        PI_Y_upper, PI_Y_lower = prediction_band(number_data, x, x_mean, y_pred, SSxx, t_two, True)

        # Display the upper and lower bounds of the prediction band
        print(PI_Y_upper)
        print(PI_Y_lower)
    """
    # Calculate the upper and lower prediction band using vector operations
    if fit_intercept:
        diff = X[:, 1:] - x_mean[1:]
    else:
        diff = X - x_mean

    y_pred = y_pred.flatten()

    PI_term = t_two * np.sqrt(1 + 1 / number_data + np.sum((diff**2 / SSxx), axis=1))
    PI_Y_upper = y_pred + PI_term
    PI_Y_lower = y_pred - PI_term
    return PI_Y_upper, PI_Y_lower