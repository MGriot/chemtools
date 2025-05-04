import numpy as np
import pandas as pd
import sys
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chemtools.regression.LinearRegression import OLSRegression, WLSRegression
from chemtools.plots.regression.regression_plots import RegressionPlots


# Seed for reproducibility
np.random.seed(0)

# Define x as a sequence of integers
x = np.arange(100)

# Define y with a linear relationship and some noise
y = 2 * x + np.random.normal(scale=50, size=x.shape)
x_new = np.array([10])
weights = np.array([1, 1, 0.5, 1, 0, 1, 0.9])

# Testing OLS Regression
model = OLSRegression(fit_intercept=False)
model.fit(X=x, y=y)  # Fit the model with your data
print("=== OLS Model Summary ===")
print(model.summary)
print("\nPrediction for x=10:", model.predict(x_new))

# Testing plots
print("\n=== Testing Plots ===")
plot = RegressionPlots(model)

# Test each plot type
print("1. Testing Residuals Plot")
fig_residuals = plot.plot_residuals()
plt.show()

print("2. Testing Data Plot")
fig_data = plot.plot_data()
plt.show()

print("3. Testing Regression Line Plot")
fig_regression = plot.plot_regression_line()
plt.show()

# Test with Plotly
print("\n=== Testing Plotly Plots ===")
plot_plotly = RegressionPlots(model, library="plotly")
fig_plotly = plot_plotly.plot_regression_line()
fig_plotly.show()

# Statsmodels comparison
print("\n=== Statsmodels Results ===")
mod = sm.OLS(y, x)
res = mod.fit()
print(res.summary())
