import numpy as np
from scipy import stats
from scipy.stats import t
from datetime import datetime

from chemtools.base.base_models import BaseModel  # Assuming this is your base class
from .confidence_band import confidence_band
from .prediction_band import prediction_band
from chemtools.utility import t_students, degrees_of_freedom, sort_arrays
from chemtools.utility import (
    centered_r_squared,
    uncentered_r_squared,
    uncentered_adjusted_r_squared,
)


class LinearRegression(BaseModel):
    """
    Base class for linear regression models (OLS, WLS, GLS).

    Handles shared functionality like fitting, prediction, and basic statistics.
    """

    def __init__(self, fit_intercept=True):
        super().__init__()
        self.model_name = "Linear Regression"
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        """Fit the model to the data."""
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    def predict(self, X, new_data=True):
        """Predict using the fitted model."""
        if new_data is True and self.fit_intercept is True:
            # Make sure X is 2-dimensional before stacking
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.coefficients)

    def _calculate_statistics(self, alpha=0.05):
        self.objects_number = self.X_orig.shape[0]
        self.object_order = np.arange(1, self.X_orig.shape[0] + 1)
        self.x_mean = np.mean(self.X_orig)
        self.y_mean = np.mean(self.y)
        self.alpha = alpha
        self.dof = degrees_of_freedom(self.X)  # degrees of freedom
        self.y_pred = self.predict(
            self.X, new_data=False
        )  # Calcola i valori predetti dal modello per la matrice X
        self.residuals = (
            self.y - self.y_pred
        )  # Calcola i residui come differenza tra i valori osservati e quelli predetti
        self.SSxx = np.sum(
            (self.X_orig - self.x_mean) ** 2
        )  # Calcola la somma dei quadrati delle deviazioni di X dalla sua media
        self.SSyy = np.sum(
            (self.y - self.y_mean) ** 2
        )  # Calcola la somma dei quadrati delle deviazioni di y dalla sua media
        self.SSxy = np.sum(
            (self.X_orig - self.x_mean) * (self.y - self.y_mean)
        )  # Calcola la somma dei prodotti delle deviazioni di X e y dalle loro medie
        self.S2x = np.sum(
            self.X_orig**2
        )  # Calcola la somma dei quadrati dei valori di X
        self.S2y = np.sum(self.y**2)  # Calcola la somma dei quadrati dei valori di y
        self.SSres = np.sum(
            (self.residuals) ** 2
        )  # Calcola la somma dei quadrati dei residui (somma dei quadrati degli errori)
        self.SSexp = (
            (self.y_pred - self.y_mean) ** 2
        ).sum()  # Calcola la somma dei quadrati spiegata dal modello
        self.s2 = self.SSres / self.dof  # varianza residua
        self.rse = np.sqrt(self.SSres / self.dof)  # root mean square error

        self.k = len(self.coefficients)  # number of coefficents that are calculated
        self.t_one, self.t_two = t_students(self.alpha, self.dof)  # t di students

        self.rse = np.sqrt(self.SSres / self.dof)  # root mean square error

        self.mse = self.SSres / len(self.y)  # usa la funzione predict che c'è già

        self.rmse = self.mse ** (1 / 2)
        if self.fit_intercept == True:
            self.R2 = self.SSexp / self.SSyy
            self.r2 = 1 - (self.SSres / self.SSyy)
            self.adjusted_r_squared = (
                1
                - ((self.objects_number - 1) / (self.objects_number - self.k - 1))
                * self.SSres
                / self.SSyy
            )
        else:
            self.r2 = uncentered_r_squared(self.y, self.y_pred)
            self.adjusted_r_squared = uncentered_adjusted_r_squared(
                self.y, self.y_pred, self.k
            )

        # Calcola la varianza residua s2
        s2 = self.residuals.T @ self.residuals / (self.X.shape[0] - self.X.shape[1])
        # Calcola la matrice di covarianza dei parametri del modello
        self.cov_matrix = s2 * np.linalg.inv(np.dot(self.X.T, self.X))
        # Calcola l'errore standard dei parametri del modello come radice quadrata della diagonale della matrice di covarianza ( Calcola la deviazione standard dei coefficienti)
        self.se_params = np.sqrt(np.diag(self.cov_matrix))
        # Calcola il valore t dei coefficienti come rapporto tra di questi ultimi e il loro errore standard
        self.t_params = self.coefficients / self.se_params
        # Calcola il p-value dei coefficienti utilizzando la distribuzione t di Student con gradi di libertà pari al numero di osservazioni meno il numero di parametri stimati
        self.p_params = 2 * (1 - stats.t.cdf(abs(self.t_params), self.dof))

        # Calcola i margini di errore per i coefficients
        self.margin_of_error = self.t_two * self.se_params
        # Calcola gli intervalli di confidenza al alpha%
        self.conf_int_lower = self.coefficients - self.margin_of_error
        self.conf_int_upper = self.coefficients + self.margin_of_error

        ## Statistiche sui residui ##
        self.residuals_min = format(np.min(self.residuals), ".2e")  # Minimo dei residui
        self.residuals_1q = format(
            np.percentile(self.residuals, 25), ".2e"
        )  # Primo quartile dei residui
        self.residuals_median = format(
            np.median(self.residuals), ".2e"
        )  # Mediana dei residui
        self.residuals_3q = format(
            np.percentile(self.residuals, 75), ".2e"
        )  # Terzo quartile dei residui
        self.residuals_max = format(np.max(self.residuals), ".2e")
        if len(self.residuals) >= 8:
            # Calcola il valore di Omnibus e Prob(Omnibus)
            self.omnibus, self.prob_omnibus = stats.normaltest(self.residuals)

        else:
            self.omnibus, self.prob_omnibus = np.nan, np.nan
        # Calcola il valore di Skewness e Kurtosis (minimum 8 numbers)
        if len(self.residuals) >= 8:
            self.skewness = stats.skew(self.residuals)
        else:
            self.skewness = np.nan  # Or some other indicator of insufficient data

        self.kurtosis = stats.kurtosis(self.residuals, fisher=False)
        # Calcola il valore di Durbin-Watson
        self.dw = np.sum(np.diff(self.residuals, axis=0) ** 2) / np.sum(
            self.residuals**2
        )
        # Calcola il valore di Jarque-Bera (JB) e Prob(JB)
        self.jb, self.prob_jb = stats.jarque_bera(self.residuals)

        # Calcola il valore di Cond. No.
        self.cond_no = np.linalg.cond(self.X)

        ## F-statistic
        # self.f_statistic=()

        ## confidence band
        self.upper_confidence_band, self.lower_confidence_band = confidence_band(
            self.objects_number,
            self.X_orig,
            self.x_mean,
            self.y_pred,
            self.SSxx,
            self.t_two,
        )
        ## Prediction band
        self.upper_prediction_band, self.lower_prediction_band = prediction_band(
            self.objects_number,
            self.X_orig,
            self.x_mean,
            self.y_pred,
            self.SSxx,
            self.t_two,
        )

    def _get_summary_data(self):
        """Returns a dictionary of data for the summary."""
        return {
            "R-squared": f"{self.r2:.3f}",
            "Adjusted R-squared": f"{self.adjusted_r_squared:.3f}",
            "No. Observations": self.objects_number,
            "Df Residuals": self.dof,
            # ... (Add other relevant statistics)
        }


class OLSRegression(LinearRegression):
    """
    Ordinary Least Squares Regression.
    """

    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept)
        self.model_name = "Ordinary Least Squares"

    def fit(self, X, y):
        """Fit the OLS model."""
        self.X_orig, self.y = sort_arrays(X.ravel(), y.ravel())
        self.X_orig = np.array([self.X_orig]).T
        self.y = np.array([self.y]).T
        if self.fit_intercept:
            self.X = np.hstack((np.ones((X.shape[0], 1)), self.X_orig))
        else:
            self.X = self.X_orig

        self.coefficients = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
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
        self.coefficients = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
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
        self.coefficients = (
            np.linalg.inv(self.X.T @ Omega_inv @ self.X) @ self.X.T @ Omega_inv @ self.y
        )
        self._calculate_statistics()
