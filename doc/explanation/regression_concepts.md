# Regression Concepts

Regression analysis is a set of statistical processes for estimating the relationships between a dependent variable (often called the 'response' or 'outcome' variable) and one or more independent variables (often called 'predictors', 'covariates', or 'features'). In chemometrics, regression is widely used to build predictive models that can quantify properties from instrumental data (e.g., predicting concentration from spectral data).

The `chemtools.regression` module provides various linear regression models, including Ordinary Least Squares (OLS), Weighted Least Squares (WLS), and Generalized Least Squares (GLS).

## Core Idea of Regression

At its heart, regression aims to model the expected value of a dependent variable `Y` given `X` (the independent variables). This typically involves finding a mathematical function that best fits the observed data, allowing for prediction of `Y` for new values of `X`.

## Types of Linear Regression

### Ordinary Least Squares (OLS) Regression

OLS is the most common and fundamental form of linear regression. It estimates the parameters of a linear regression model by minimizing the sum of the squares of the differences between the observed dependent variable (`Y`) and those predicted by the linear model (`Ŷ`).

*   **Assumptions of OLS:**
    *   **Linearity:** The relationship between `X` and `Y` is linear.
    *   **Independence:** Observations are independent of each other.
    *   **Homoscedasticity:** The variance of the residuals (errors) is constant across all levels of the independent variables.
    *   **Normality:** The residuals are normally distributed.
    *   **No Multicollinearity:** Independent variables are not highly correlated with each other.

### Weighted Least Squares (WLS) Regression

WLS is used when the assumption of homoscedasticity (constant variance of residuals) in OLS is violated, meaning the residuals have unequal variances. In WLS, different weights are assigned to each observation in the dataset. Observations with higher variance (less reliable) are given smaller weights, and observations with lower variance (more reliable) receive larger weights.

*   **Use Cases:** WLS is particularly useful when the precision of measurements varies across observations, or when certain data points are known to be more reliable than others.

### Generalized Least Squares (GLS) Regression

GLS is a more general and flexible regression technique than OLS or WLS. It is used when the residuals are correlated (autocorrelation) or have unequal variances (heteroscedasticity) in a way that is known or can be estimated. GLS accounts for these error structures by incorporating a covariance matrix into the estimation process.

*   **Use Cases:** GLS is suitable for time-series data where errors might be correlated over time, or panel data where observations within groups might be correlated.

## Model Evaluation and Diagnostics

After fitting a regression model, it's crucial to evaluate its performance and check its assumptions using various statistical measures and diagnostic plots:

*   **R-squared and Adjusted R-squared:** Measures the proportion of variance in the dependent variable that can be explained by the independent variables. Adjusted R-squared accounts for the number of predictors.
*   **F-statistic and p-value:** Tests the overall significance of the regression model.
*   **Coefficients and p-values:** Indicate the magnitude and statistical significance of each independent variable's effect on the dependent variable.
*   **Residual Analysis:** Plotting residuals against predicted values or independent variables can help detect violations of homoscedasticity, linearity, or normality assumptions.
*   **Confidence and Prediction Bands:**
    *   **Confidence Band:** Represents the uncertainty around the regression *line* (i.e., the mean response).
    *   **Prediction Band:** Represents the uncertainty around *individual predictions*. It is always wider than the confidence band because it accounts for both the uncertainty in the mean response and the irreducible error of individual observations.
*   **Information Criteria (AIC, BIC):** Used for model comparison, penalizing models with more parameters to avoid overfitting.

Understanding these concepts helps in selecting the appropriate regression model, interpreting its output, and ensuring the reliability of its predictions.

## Further Reading

*   [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) on Wikipedia
*   [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) on Wikipedia
*   [Weighted Least Squares](https://en.wikipedia.org/wiki/Weighted_least_squares) on Wikipedia
*   [Generalized Least Squares](https://en.wikipedia.org/wiki/Generalized_least_squares) on Wikipedia
