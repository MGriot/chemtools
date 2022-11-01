import math
import numpy as np


def prediction_band(number_data, x, x_mean, y_pred_orig, SSxx, t_two):
    PI_Y_upper = []
    PI_Y_lower = []
    for i in range(number_data):
        # UPPER
        # PI_y_upper=y_pred_orig[i]+t_two*SE*math.sqrt(1+(1/number_data)+pow(x[i]-x_mean,2)/SSxx) #vecchia
        PI_y_upper = y_pred_orig[i] + t_two * math.sqrt(
            1 + (1 / number_data) + pow(x[i] - x_mean, 2) / SSxx
        )
        PI_Y_upper = np.append(PI_Y_upper, PI_y_upper)
        # LOWER
        # PI_y_lower=y_pred_orig[i]-t_two*SE*math.sqrt(1+(1/number_data)+pow(x[i]-x_mean,2)/SSxx) #vecchia
        PI_y_lower = y_pred_orig[i] - t_two * math.sqrt(
            1 + (1 / number_data) + pow(x[i] - x_mean, 2) / SSxx
        )
        PI_Y_lower = np.append(PI_Y_lower, PI_y_lower)
    return PI_Y_upper, PI_Y_lower
