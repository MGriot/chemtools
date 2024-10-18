## chemtools: Linear Regression Module

This module provides classes and functions for performing linear regression analysis. It includes implementations for:

* **Ordinary Least Squares (OLS) Regression:** `OLSRegression`
* **Weighted Least Squares (WLS) Regression:** `WLSRegression`
* **Generalized Least Squares (GLS) Regression:** `GLSRegression`

The module also offers plotting utilities within `regression_plots.py` to visualize the regression results using either Matplotlib or Plotly.

### 1. Linear Regression Models (`LinearRegression.py`)

#### 1.1 Base Class: `LinearRegression`

The `LinearRegression` class serves as a base class for all linear regression models in this module. It defines the common structure and methods shared by different regression types, promoting code reusability and maintainability.

**Key Features:**

* **Fit Intercept Option:** Allows choosing whether to fit an intercept term (default: `True`).
* **Prediction:** Provides methods to predict target values for new data based on the fitted model.
* **Statistics Calculation:** Calculates and stores various regression statistics, including R-squared, adjusted R-squared, F-statistic, p-values, residuals, confidence intervals, and more.
* **Covariance Type:** Identifies the type of covariance used in the model (e.g., non-robust for OLS, HC0 for WLS).
* **Model Selection Criteria:** Computes Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) for model comparison.
* **Confidence and Prediction Bands:** Calculates and stores confidence and prediction bands for visualizing uncertainty in the regression estimates.

#### 1.2 Ordinary Least Squares (OLS) Regression: `OLSRegression`

Inherits from `LinearRegression` and implements the classic OLS regression. It estimates the model parameters by minimizing the sum of squared residuals between the observed and predicted values.

**Usage:**

```python
from chemtools.regression.LinearRegression import OLSRegression

model = OLSRegression()
model.fit(X, y)
predictions = model.predict(new_X)
```

**Assumptions:**

* **Linearity:** The relationship between the dependent and independent variables is linear.
* **Independence:** The residuals are independent of each other.
* **Homoscedasticity:** The residuals have constant variance across all levels of the independent variable.
* **Normality:** The residuals are normally distributed.

#### 1.3 Weighted Least Squares (WLS) Regression: `WLSRegression`

Extends `LinearRegression` and allows assigning weights to observations during the fitting process. This is particularly useful when dealing with heteroscedasticity, where the variance of residuals varies across different input values.

**Usage:**

```python
from chemtools.regression.LinearRegression import WLSRegression

model = WLSRegression(weights=weights_array)
model.fit(X, y)
predictions = model.predict(new_X)
```

**Note:** The `weights` argument should be an array of weights corresponding to each observation in the data.

#### 1.4 Generalized Least Squares (GLS) Regression: `GLSRegression`

Inherits from `LinearRegression` and provides a more general framework for handling correlated or heteroscedastic errors. It incorporates a covariance matrix (Omega) into the estimation process to account for these dependencies.

**Usage:**

```python
from chemtools.regression.LinearRegression import GLSRegression

model = GLSRegression(omega=covariance_matrix)
model.fit(X, y)
predictions = model.predict(new_X)
```

**Note:** The `omega` argument should be the covariance matrix representing the error structure.

### 2. Regression Plots (`regression_plots.py`)

This submodule offers functions to visualize regression results, including:

* `plot_residuals`: Plots residuals against observations to check for patterns and assess model assumptions.
  ![Residual Plot](/doc/img/regression/residuals.png) 

* `plot_data`: Creates a scatter plot of the input data.
  ![Data Plot](/doc/img/regression/data.png)
* `plot_regression_line`: Plots the regression line along with the data points.
  ![Regression Line Plot](/doc/img/regression/regression line.png)

* `plot_confidence_band`: Visualizes the confidence band around the regression line, indicating the uncertainty in the estimated relationship.
  ![Confidence Band Plot](/doc/img/regression/confidence band.png)
* `plot_prediction_band`: Displays the prediction band, which represents the range where future predictions are likely to fall.
  ![Prediction Band Plot](/doc/img/regression/prediction band.png)
* `plot_regression_results`: Provides a comprehensive visualization of regression results, allowing customization of what to include in the plot (data points, regression line, confidence band, prediction band).
  ![all Plot](/doc/img/regression/all.png)

These plotting functions support both Matplotlib and Plotly libraries, giving users flexibility in choosing their preferred visualization tool.

---

This documentation provides a brief overview of the Linear Regression module within `chemtools`. Each class and function has more detailed documentation within the code itself, accessible through docstrings.
