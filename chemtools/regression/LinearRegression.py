import numpy as np
from scipy import stats
from scipy.stats import t
from datetime import datetime
import warnings
from typing import Optional, Union, Tuple

from chemtools.base.base_models import BaseModel  # Assuming this is your base class
from .confidence_band import confidence_band
from .prediction_band import prediction_band
from chemtools.utility import t_students, degrees_of_freedom, sort_arrays
from chemtools.utility import (
    centered_r_squared,
    uncentered_r_squared,
    uncentered_adjusted_r_squared,
)
from chemtools.utility.get_var_name import get_variable_name


class LinearRegression(BaseModel):
    """
    Base class for linear regression models (OLS, WLS, GLS).

    Handles shared functionality like fitting, prediction, and basic statistics.
    """

    def __init__(self, fit_intercept=True):
        super().__init__()
        self.model_name = None
        self.method = "Linear Regression"
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        """Fit the model to the data."""
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    def predict(self, X: np.ndarray, new_data: bool = True) -> np.ndarray:
        """Predict using the fitted model."""
        if new_data is True and self.fit_intercept is True:
            # Make sure X is 2-dimensional before stacking
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.coefficients)

    def _calculate_statistics(self, alpha=0.05):
        """Calculate statistics after fitting the model."""
        self.objects_number = self.X.shape[0]
        self.object_order = np.arange(1, self.objects_number + 1)
        self.x_mean = np.mean(self.X, axis=0)
        self.y_mean = np.mean(self.y)
        self.alpha = alpha
        self.dof = self.objects_number - self.X.shape[1]
        self.y_pred = self.predict(self.X, new_data=False)
        self.residuals = self.y.reshape(-1, 1) - self.y_pred.reshape(-1, 1)
        self.SSxx = np.sum((self.X[:, 1:] - self.x_mean[1:]) ** 2, axis=0)
        self.SSyy = np.sum((self.y - self.y_mean) ** 2)
        self.SSxy = np.sum(
            (self.X[:, 1:] - self.x_mean[1:]) * (self.y - self.y_mean)[:, np.newaxis],
            axis=0,
        )
        self.SSres = np.sum(self.residuals**2)
        self.SSexp = np.sum((self.y_pred - self.y_mean) ** 2)
        self.s2 = self.SSres / self.dof
        self.rse = np.sqrt(self.s2)
        self.k = len(self.coefficients)
        self.dof_model = self.k - 1 if self.fit_intercept else self.k
        self.t_one, self.t_two = t_students(self.alpha, self.dof)
        # Covariance Type
        self.covariance_type = self._get_covariance_type()

        self.mse = self.SSres / self.objects_number
        self.rmse = np.sqrt(self.mse)

        if self.fit_intercept:
            self.R2 = self.SSexp / self.SSyy
            self.r2 = 1 - (self.SSres / self.SSyy)
            self.adjusted_r_squared = 1 - (
                (self.objects_number - 1) / self.dof * self.SSres / self.SSyy
            )
        else:
            self.r2 = uncentered_r_squared(self.y, self.y_pred)
            self.adjusted_r_squared = uncentered_adjusted_r_squared(
                self.y, self.y_pred, self.k
            )

        self.s2 = self.SSres / self.dof  # Correct calculation of s2
        self.cov_matrix = self.s2 * np.linalg.inv(self.X.T @ self.X)
        self.se_params = np.sqrt(np.diag(self.cov_matrix))
        self.t_params = self.coefficients / self.se_params
        self.f_statistic = (self.SSexp / self.dof_model) / (self.SSres / self.dof)
        # Handle F-statistic and p-value differently based on intercept:
        if self.fit_intercept:
            self.dof_model = self.k
            self.f_statistic = (self.SSexp / self.dof_model) / (self.SSres / self.dof)
            self.f_pvalue = 1 - stats.f.cdf(self.f_statistic, self.dof_model, self.dof)
        else:
            # Calculate F-statistic for model without intercept
            y_mean = np.mean(self.y)
            SS_total = np.sum((self.y - y_mean) ** 2)
            self.dof_model = self.k  # Degrees of freedom for the model
            self.f_statistic = (self.SSexp / self.dof_model) / (
                (SS_total - self.SSexp) / (self.objects_number - self.dof_model)
            )
            self.f_pvalue = 1 - stats.f.cdf(
                self.f_statistic, self.dof_model, self.objects_number - self.dof_model
            )
        self.p_params = 2 * (1 - stats.t.cdf(np.abs(self.t_params), self.dof))
        self.margin_of_error = self.t_two * self.se_params
        self.conf_int_lower = self.coefficients - self.margin_of_error
        self.conf_int_upper = self.coefficients + self.margin_of_error

        self.residuals_min = format(np.min(self.residuals), ".2e")
        self.residuals_1q = format(np.percentile(self.residuals, 25), ".2e")
        self.residuals_median = format(np.median(self.residuals), ".2e")
        self.residuals_3q = format(np.percentile(self.residuals, 75), ".2e")
        self.residuals_max = format(np.max(self.residuals), ".2e")

        if len(self.residuals) >= 8:
            self.omnibus, self.prob_omnibus = stats.normaltest(self.residuals)
        else:
            warning_msg = (
                f"Normality tests are not valid with less than 8 observations; "
                f"{len(self.residuals)} samples were given. Results may be unreliable."
            )
            warnings.warn(warning_msg)
            self.notes.append(warning_msg)  # Add warning to the notes list

            self.omnibus, self.prob_omnibus = np.nan, np.nan
        self.skewness = stats.skew(self.residuals)
        self.kurtosis = stats.kurtosis(self.residuals, fisher=False)
        self.dw = np.sum(np.diff(self.residuals, axis=0) ** 2) / np.sum(
            self.residuals**2
        )
        self.jb, self.prob_jb = stats.jarque_bera(self.residuals)
        self.cond_no = np.linalg.cond(self.X)

        self.log_likelihood = self._calculate_log_likelihood()
        self.aic = self._calculate_aic()
        self.bic = self._calculate_bic()

        # Confidence and prediction bands (updated for multiple coefficients)
        self.upper_confidence_band, self.lower_confidence_band = confidence_band(
            self.objects_number,
            self.X,  # Pass the entire X matrix
            self.x_mean,
            self.y_pred,
            self.SSxx,
            self.t_two,
        )

        self.upper_prediction_band, self.lower_prediction_band = prediction_band(
            self.objects_number,
            self.X,
            self.x_mean,
            self.y_pred,
            self.SSxx,  # Now an array for each feature
            self.t_two,
        )

    def _get_covariance_type(self):
        """Determine the type of covariance used in the model."""
        if isinstance(self, OLSRegression):
            return "non-robust"  # OLS uses the standard covariance
        elif isinstance(self, WLSRegression) or isinstance(self, GLSRegression):
            return "HC0"  # Placeholder, update with actual HC method if needed
        else:
            return "unknown"

    def _calculate_log_likelihood(self):
        """Calculate the log-likelihood of the model."""
        n = len(self.y)
        rss = self.SSres  # Residual sum of squares
        return -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)

    def _calculate_aic(self):
        """Calculate the Akaike Information Criterion (AIC)."""
        return 2 * self.k - 2 * self.log_likelihood

    def _calculate_bic(self):
        """Calculate the Bayesian Information Criterion (BIC)."""
        n = len(self.y)
        return self.k * np.log(n) - 2 * self._calculate_log_likelihood()

    def _get_summary_data(self):
        summary = {
            "general": {
                "Dep. Variable": f"{get_variable_name(self.y)}",
                "No. Observations": self.objects_number,
                "Df Residuals": self.dof,
                "Df Model": self.dof_model,
                "Covariance Type": self.covariance_type,
                "R-squared": f"{self.r2:.3f}",
                "Adjusted R-squared": f"{self.adjusted_r_squared:.3f}",  # f per float
                "F-statistic": f"{self.f_statistic:.3e}",
                "Prob (F-statistic)": f"{self.f_pvalue:.3e}",  # e per notazione scientifica
                "Log-Likelihood": f"{self.log_likelihood:.3f}",
                "AIC": f"{self.aic:.3f}",
                "BIC": f"{self.bic:.3f}",  # stop statsmodels
            },
            "coefficients": self._get_coefficient_table(),
        }
        # --- Add additional statistics to the summary dictionary ---
        summary["additional_stats"] = {
            "Omnibus:": (
                f"{float(self.omnibus):.3f}" if not np.isnan(self.omnibus) else "nan"
            ),
            "Prob(Omnibus):": (
                f"{float(self.prob_omnibus):.3f}"
                if not np.isnan(self.prob_omnibus)
                else "nan"
            ),
            "Skew:": (
                f"{self.skewness[0]:.3f}"
                if not np.isnan(self.skewness).any()
                else "nan"
            ),
            "Kurtosis:": f"{float(self.kurtosis):.3f}",
            "Durbin-Watson:": f"{self.dw:.3f}",
            "Jarque-Bera (JB):": f"{self.jb:.3f}",
            "Prob(JB):": f"{self.prob_jb:.3f}",
            "Cond. No.": f"{self.cond_no:.3f}",
        }
        return summary

    def _get_coefficient_table(self):
        table = []
        header = [
            " ",
            "Coefficient",
            "Std. Error",
            "t",
            "P>|t|",
            "[0.025",
            "0.975]",
        ]
        table.append(header)
        for i in range(len(self.coefficients)):
            if i == 0 and self.fit_intercept:
                coef_name = "Intercept"
            else:
                coef_name = f"Beta {i}"
            row = [
                coef_name,
                f"{float(self.coefficients[i]):.3f}",
                f"{self.se_params[i]:.3f}",
                np.array2string(
                    self.t_params[i], formatter={"float_kind": lambda x: "%.3f" % x}
                ),  # t-statistic
                np.array2string(
                    self.p_params[i], formatter={"float_kind": lambda x: "%.3f" % x}
                ),
                np.array2string(
                    self.conf_int_lower[i],
                    formatter={"float_kind": lambda x: "%.3f" % x},
                ),
                np.array2string(
                    self.conf_int_upper[i],
                    formatter={"float_kind": lambda x: "%.3f" % x},
                ),
            ]
            table.append(row)
        return table


class OLSRegression(LinearRegression):
    """
    Ordinary Least Squares Regression.
    """

    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept)
        self.model_name = "Ordinary Least Squares"

    def fit(self, X, y):
        """Fit the OLS model."""
        # Ensure X is always 2D (even for a single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_orig, self.y = sort_arrays(X, y.ravel())
        self.X_orig = np.array(self.X_orig)
        self.y = np.array([self.y]).T

        if self.fit_intercept:
            self.X = np.hstack((np.ones((X.shape[0], 1)), self.X_orig))
        else:
            self.X = self.X_orig

        self.coefficients = np.linalg.solve(
            self.X.T @ self.X, self.X.T @ self.y
        ).reshape(-1)
        self._calculate_statistics()


class WLSRegression(LinearRegression):
    """
    Weighted Least Squares Regression.
    """

    def __init__(self, weights, fit_intercept=True):
        super().__init__(fit_intercept)
        self.model_name = "Weighted Least Squares"
        self.weights = weights
        if self.weights is None:
            raise ValueError("Weights must be provided for WLS.")

    def fit(self, X, y):
        """Fit the WLS model."""
        W = np.diag(self.weights)
        self.coefficients = np.linalg.solve(X.T @ W @ X, X.T @ W @ y).reshape(
            -1
        )  # Use solve()
        self._calculate_statistics()


class GLSRegression(LinearRegression):
    """
    Generalized Least Squares Regression.
    """

    def __init__(self, omega, fit_intercept=True):
        super().__init__(fit_intercept)
        self.model_name = "Generalized Least Squares"
        self.omega = omega
        if self.omega is None:
            raise ValueError("Omega matrix must be provided for GLS.")

    def fit(self, X, y):
        """Fit the GLS model."""
        super().fit(X, y)
        Omega_inv = np.linalg.inv(self.omega)
        self.coefficients = np.linalg.solve(
            self.X.T @ Omega_inv @ self.X, self.X.T @ Omega_inv @ self.y
        ).reshape(-1)
        self._calculate_statistics()
