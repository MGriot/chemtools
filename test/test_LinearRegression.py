import numpy as np
import pandas as pd
import sys
import os
import statsmodels.api as sm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chemtools.regression.LinearRegression import OLSRegression, WLSRegression
from chemtools.plots.regression.regression_plots import RegressionPlots


# Seed for reproducibility
np.random.seed(0)

# Define x as a sequence of integers
x = np.arange(100)

# Define y with a linear relationship and some noise
y = 2 * x + np.random.normal(scale=50, size=x.shape)
# x = np.array([0, 1, 2, 3, 4, 5, 6])
# y = np.array([0, 2, 3, 5, 6, 8, 10])
x_new = np.array([10])
weights = np.array([1, 1, 0.5, 1, 0, 1, 0.9])

# Example usage:
# Assuming X and y are your data
model = OLSRegression(fit_intercept=False)
model.fit(X=x, y=y)  # Fit the model with your data
print(model.summary)
print(model.predict(x_new))


# --- Plots ---
# regression_plots.plot_residuals(model)
# regression_plots.plot_data(model)
# plot = RegressionPlots(model, library="plotly")
plot = RegressionPlots(model)
plot.plot_regression_line()

# regression_plots.plot_confidence_band(model)
# regression_plots.plot_prediction_band(model)
# regression_plots.plot_regression_results(model)
# ... (The confidence and prediction bands won't work directly with LinearRegression
#     as it doesn't calculate them by default. You would need to implement those
#     calculations or use a different library like statsmodels)

# --- Plotly Examples ---
# ols_plots.plot_residuals(model, library="plotly")
# ols_plots.plot_data(model, library="plotly")
# ... (similarly for other plots)

# need to be implemented
# model_wls = WLSRegression(weights=weights)
# model_wls.fit(X=x, y=y)
# predictions_wls = model_wls.predict(X)
# model_wls.summary()

# x = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]])
mod = sm.OLS(y, x)
res = mod.fit()
print(res.summary())
