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

### 3. Summary Output

The `LinearRegression` models provide a well-formatted summary output containing key regression statistics and diagnostics. Here's an example of what the summary looks like:

```
==========================================================================================
                                Linear Regression Summary
------------------------------------------------------------------------------------------
Model:               Ordinary Least Squares    R-squared                             0.993
Date:                      Fri, 18 Oct 2024    Adjusted R-squared                    0.991
Time:                              09:36:27    F-statistic                       6.750e+02
Dep. Variable                            []    Prob (F-statistic)                1.578e-06
No. Observations                          7    Log-Likelihood                       -0.937
Df Residuals                              5    AIC                                   5.875
Df Model                                  1    BIC                                   5.767
Covariance Type                  non-robust
------------------------------------------------------------------------------------------

Coefficients:
------------------------------------------------------------------------------------------
              Coefficient   Std. Error    t             P>|t|         [0.025        0.975]
Intercept     0.036         0.223         0.160         0.879         -0.538        0.609
Beta 1        1.607         0.062         25.981        0.000         1.448         1.766
------------------------------------------------------------------------------------------
Omnibus:                                nan    Durbin-Watson:                        2.529
Prob(Omnibus):                          nan    Jarque-Bera (JB):                     0.418
Skew:                                -0.232    Prob(JB):                             0.811
Kurtosis:                             1.897    Cond. No.                             6.854
==========================================================================================
Notes:
[1] Normality tests are not valid with less than 8 observations; 7 samples were given.
    Results may be unreliable.
------------------------------------------------------------------------------------------
```

**Interpretation of the Summary Output:**

- **Model:** Indicates the type of linear regression model used (e.g., OLS, WLS, GLS).
- **R-squared:** Measures the proportion of variance in the dependent variable explained by the model.
- **Adjusted R-squared:** Similar to R-squared but considers the number of predictors in the model.
- **F-statistic:** Tests the overall significance of the model.
- **Prob (F-statistic):** The p-value associated with the F-statistic.
- **Coefficients:** Provides estimates, standard errors, t-statistics, p-values, and confidence intervals for each coefficient in the model.
- **Other Statistics:** Includes diagnostics such as Omnibus, Durbin-Watson, Jarque-Bera, Skewness, Kurtosis, and Condition Number.
- **Notes:** May contain additional information or warnings related to the analysis, such as insufficient data points for certain tests. 

---

This documentation provides a brief overview of the Linear Regression module within `chemtools`. Each class and function has more detailed documentation within the code itself, accessible through docstrings.