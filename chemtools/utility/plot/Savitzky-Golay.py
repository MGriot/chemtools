import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def smooth_plot(x, y, window_length=5, polyorder=2):
    y_smooth = savgol_filter(y, window_length, polyorder)
    plt.plot(x, y_smooth)
    plt.show()