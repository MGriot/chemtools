# Regression Module Reference (`chemtools.regression`)

The `chemtools.regression` module provides classes for performing linear regression analysis, including Ordinary Least Squares (OLS), Weighted Least Squares (WLS), and Generalized Least Squares (GLS). It also integrates plotting utilities for visualizing regression results.

---

## `LinearRegression` Base Class

The `LinearRegression` class serves as a base for all linear regression models in `chemtools`. It provides common functionalities like fitting, prediction, and statistical calculations.

### Key Features (inherited by subclasses):

*   **Fit Intercept Option:** Configurable to include or exclude an intercept term (default: `True`).
*   **Prediction:** Methods to predict target values (`predict`).
*   **Statistics Calculation:** Calculates R-squared, adjusted R-squared, F-statistic, p-values, residuals, confidence intervals, AIC, BIC, and more.
*   **Covariance Type:** Identifies the type of covariance used (e.g., non-robust for OLS, HC0 for WLS).
*   **Confidence and Prediction Bands:** Stores calculated confidence and prediction bands.

---

<h2><code>OLSRegression</code> Class</h2>

Implements Ordinary Least Squares (OLS) regression, estimating parameters by minimizing the sum of squared residuals.

<h3><code>OLSRegression()</code></h3>

*   <b>Parameters:</b> None

<h3>Methods (inherits from <code>LinearRegression</code>)</h3>

*   <b><code>fit(self, X, y, variables_names=None, objects_names=None)</code></b>
    *   Fits the OLS model.
    *   <b>Parameters:</b>
        *   <code>X</code> (<code>array-like</code>): Independent variables.
        *   <code>y</code> (<code>array-like</code>): Dependent variable.
        *   <code>variables_names</code> (<code>list</code>, optional): Names for independent variables.
        *   <code>objects_names</code> (<code>list</code>, optional): Names for observations.
*   <b><code>predict(self, X_new) -> np.ndarray</code></b>
    *   Predicts <code>y</code> values for new <code>X_new</code> data.

<h3>Usage Example (OLS)</h3>

```python
from chemtools.regression import OLSRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = OLSRegression()
model.fit(X, y)
predictions = model.predict(np.array([[6]]))
print(f"OLS Prediction for X=6: {predictions[0]:.2f}")
print(model.summary)
```

---

<h2><code>WLSRegression</code> Class</h2>

Extends <code>LinearRegression</code> by allowing weights to be assigned to observations, useful for heteroscedastic data.

<h3><code>WLSRegression(weights: np.ndarray = None)</code></h3>

*   <b>Parameters:</b>
    *   <code>weights</code> (<code>np.ndarray</code>, optional): An array of weights for each observation.

<h3>Methods (inherits from <code>LinearRegression</code>)</h3>

*   <b><code>fit(self, X, y, variables_names=None, objects_names=None)</code></b>
*   <b><code>predict(self, X_new) -> np.ndarray</code></b>

<h3>Usage Example (WLS)</h3>

```python
from chemtools.regression import WLSRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
# Example weights: give more weight to observations with smaller X
weights = 1 / X.flatten() 

model = WLSRegression(weights=weights)
model.fit(X, y)
predictions = model.predict(np.array([[6]]))
print(f"WLS Prediction for X=6: {predictions[0]:.2f}")
print(model.summary)
```

---

<h2><code>GLSRegression</code> Class</h2>

Extends <code>LinearRegression</code> for handling correlated or heteroscedastic errors by incorporating a covariance matrix.

<h3><code>GLSRegression(omega: np.ndarray = None)</code></h3>

*   <b>Parameters:</b>
    *   <code>omega</code> (<code>np.ndarray</code>, optional): The covariance matrix representing the error structure.

<h3>Methods (inherits from <code>LinearRegression</code>)</h3>

*   <b><code>fit(self, X, y, variables_names=None, objects_names=None)</code></b>
*   <b><code>predict(self, X_new) -> np.ndarray</code></b>

<h3>Usage Example (GLS)</h3>

```python
from chemtools.regression import GLSRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
# Example covariance matrix (e.g., for correlated errors)
omega = np.identity(len(X)) + np.diag(np.ones(len(X)-1), k=1) * 0.2 + np.diag(np.ones(len(X)-1), k=-1) * 0.2

model = GLSRegression(omega=omega)
model.fit(X, y)
predictions = model.predict(np.array([[6]]))
print(f"GLS Prediction for X=6: {predictions[0]:.2f}")
print(model.summary)
```

---

<h2>Summary Output</h2>

All <code>LinearRegression</code> models provide a formatted summary output accessible via the <code>.summary</code> property, containing key regression statistics and diagnostics.

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

---

<h2>Visualizing Regression Results</h2>

The <code>chemtools.plots.regression.RegressionPlots</code> class is used to visualize regression results.

<h3><code>RegressionPlots(model, library="matplotlib")</code></h3>

*   <b>Parameters:</b>
    *   <code>model</code>: A fitted <code>LinearRegression</code> object (e.e.g., <code>OLSRegression</code>, <code>WLSRegression</code>, or <code>GLSRegression</code>).
    *   <code>library</code> (<code>str</code>, optional): Plotting backend (<code>"matplotlib"</code> or <code>"plotly"</code>). Defaults to <code>"matplotlib"</code>.

<h3>Methods</h3>

*   <b><code>plot_regression_results(self, show_data=True, show_regression_line=True, show_confidence_band=True, show_prediction_band=True, show_equation=True, **kwargs)</code></b>
    *   Provides a comprehensive visualization of regression results.
    *   <b>Parameters:</b>
        *   <code>show_data</code> (<code>bool</code>): Display original data points.
        *   <code>show_regression_line</code> (<code>bool</code>): Display the fitted regression line.
        *   <code>show_confidence_band</code> (<code>bool</code>): Display the confidence band for the mean response.
        *   <code>show_prediction_band</code> (<code>bool</code>): Display the prediction band for individual observations.
        *   <code>show_equation</code> (<code>bool</code>): Display the regression equation on the plot.
        *   <code>**kwargs</code>: Additional plotting arguments.
*   <b><code>plot_data(**kwargs)</code></b>: Plots only the raw input data points (scatter plot).
*   <b><code>plot_residuals(**kwargs)</code></b>: Plots the model's residuals.
*   <b><code>plot_regression_line(**kwargs)</code></b>: Plots the data points with the regression line overlaid.
*   <b><code>plot_confidence_band(**kwargs)</code></b>: Plots the confidence band.
*   <b><code>plot_prediction_band(**kwargs)</code></b>: Plots the prediction band.

<h3>Example Output</h3>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/plots/regression/regression_results_classic_professional_dark.png">
  <img alt="Regression Results Plot" src="../img/plots/regression/regression_results_classic_professional_light.png">
</picture>

---

This reference provides an overview of the Linear Regression module. Each class and function has more detailed documentation within the code itself, accessible through docstrings.
