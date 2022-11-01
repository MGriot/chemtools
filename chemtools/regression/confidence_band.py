import math
import numpy as np


def confidence_band(number_data, x, x_mean, y_pred_orig, SSxx, t_two):
    CI_Y_upper = []
    CI_Y_lower = []
    for i in range(number_data):
        # UPPER
        # CI_y_upper=y_pred_orig[i]+t_two*SE*math.sqrt((1/number_data)+pow(x[i]-x_mean,2)/SSxx) #vecchia
        CI_y_upper = y_pred_orig[i] + t_two * math.sqrt(
            (1 / number_data) + pow(x[i] - x_mean, 2) / SSxx
        )
        CI_Y_upper = np.append(CI_Y_upper, CI_y_upper)
        # LOWER
        # CI_y_lower=y_pred_orig[i]-t_two*SE*math.sqrt((1/number_data)+pow(x[i]-x_mean,2)/SSxx) #vecchia
        CI_y_lower = y_pred_orig[i] - t_two * math.sqrt(
            (1 / number_data) + pow(x[i] - x_mean, 2) / SSxx
        )
        CI_Y_lower = np.append(CI_Y_lower, CI_y_lower)
    return CI_Y_upper, CI_Y_lower
