import numpy as np
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt
import plotly.express as px
from tabulate import tabulate
from datetime import datetime
import pickle

from chemtools.regression import confidence_band
from chemtools.regression import prediction_band
from chemtools.utility import t_students
from chemtools.utility import degrees_of_freedom
from chemtools.utility import sort_arrays
from chemtools.utility import (
    centered_r_squared,
    uncentered_r_squared,
    uncentered_adjusted_r_squared,
)


class LinearRegression:
    def __init__(
        self, fit_intercept=True, model="OLS", weights=None, omega=None, X=None, y=None
    ):
        self.fit_intercept = fit_intercept
        self.X_orig, self.y = sort_arrays(X.ravel(), y.ravel())
        self.X_orig = np.array([self.X_orig]).T
        self.y = np.array([self.y]).T
        if (
            self.fit_intercept is True
        ):  # aggiunge alla matrice delle X una colonna di 1 per l'intercetta
            X = np.hstack((np.ones((X.shape[0], 1)), self.X_orig))
        self.X = X
        self.coefficients = None
        self.model = model
        self.weights = weights
        self.omega = omega
        self.today = datetime.now().strftime("%a, %d %b %Y")
        self.hour = datetime.now().strftime("%H:%M:%S")

    def fit(self):
        if self.model == "OLS":
            self.coefficients = (
                np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
            )
        elif self.model == "WLS":
            if self.weights is None:
                raise ValueError("Weights must be provided for WLS model")
            W = np.diag(self.weights)
            self.coefficients = (
                np.linalg.inv(self.X.T.dot(W).dot(self.X))
                .dot(self.X.T)
                .dot(W)
                .dot(self.y)
            )
        elif self.model == "GLS":
            if self.omega is None:
                raise ValueError("Omega matrix must be provided for GLS model")
            Omega_inv = np.linalg.inv(self.omega)
            self.coefficients = (
                np.linalg.inv(self.X.T.dot(Omega_inv).dot(self.X))
                .dot(self.X.T)
                .dot(Omega_inv)
                .dot(self.y)
            )

    def predict(self, X, new_data=True):
        if new_data is True and self.fit_intercept is True:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.coefficients)

    def statistics(self, alpha=0.05):
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
        # Calcola il valore di Omnibus e Prob(Omnibus)
        self.omnibus, self.prob_omnibus = stats.normaltest(self.residuals)
        # Calcola il valore di Skewness e Kurtosis (minimum 8 numbers)
        self.skewness = stats.skew(self.residuals)
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

    def plot_residuals(self, library="matplotlib"):
        if library == "matplotlib":
            plt.scatter(range(len(self.residuals)), self.residuals)
            plt.axhline(0, color="r", linestyle="--")
            plt.xlabel("Observations")
            plt.ylabel("Residuals")
        elif library == "plotly":
            fig = px.scatter(x=range(len(self.residuals)), y=self.residuals.ravel())
            fig.update_layout(
                shapes=[dict(type="line", x0=0, x1=len(self.residuals), y0=0, y1=0)],
                xaxis_title="Observations",
                yaxis_title="Residuals",
            )
            fig.show()

    def plot_data(self, library="matplotlib"):
        if library == "matplotlib":
            plt.scatter(self.X_orig, self.y, label="Data", c="#000000", alpha=0.25)
            plt.legend(loc="best")
        elif library == "plotly":
            fig = px.scatter(X.ravel(), y.ravel())
            fig.show()

    def plot_regression_line(self, library="matplotlib"):
        if library == "matplotlib":
            x_line = np.linspace(self.X_orig.min(), self.X_orig.max(), num=100)
            x_line = x_line.reshape(-1, 1)
            y_line = self.predict(x_line)
            plt.plot(x_line, y_line, "r", label="Regression Line")
            # Costruire la stringa dell'equazione di regressione in formato LaTeX
            equation_str = r"$y = "
            if self.fit_intercept:
                equation_str += f"{self.coefficients[0][0]:.2f} + "
                start_idx = 1
            else:
                start_idx = 0
            for i in range(start_idx, len(self.coefficients)):
                equation_str += (
                    f"{self.coefficients[i][0]:.2f} x_{{{i - start_idx + 1}}}"
                )
                if i != len(self.coefficients) - 1:
                    equation_str += " + "
            equation_str += r"$"
            # Aggiungere l'equazione alla legenda del grafico
            plt.plot([], [], " ", label=f"{equation_str}")
            plt.legend(loc="best")

    def plot_confidence_band(self, library="matplotlib"):
        if library == "matplotlib":
            plt.fill_between(
                self.X_orig.ravel(),
                self.lower_confidence_band.ravel(),
                self.upper_confidence_band.ravel(),
                color="b",
                alpha=0.3,
                label="Confidence Band",
            )
            plt.legend(loc="best")

    def plot_prediction_band(self, library="matplotlib"):
        if library == "matplotlib":
            plt.fill_between(
                self.X_orig.ravel(),
                self.lower_prediction_band.ravel(),
                self.upper_prediction_band.ravel(),
                color="g",
                alpha=0.3,
                label="Prediction Band",
            )
            plt.legend(loc="best")

    def plot_all(self):
        self.plot_data()
        self.plot_regression_line()
        self.plot_confidence_band()
        self.plot_prediction_band()

    def print_summary(self):
        headers1 = ["", "Regresssion", "result", ""]

        if self.fit_intercept == True:
            table1 = [
                ["Dep. Variable:", "Y", "R-squared:", f"{round(self.r2,3)}"],  # riga 1
                [
                    "Model:",
                    f"{self.model}",
                    "Adj. R-squared:",
                    f"{round(self.adjusted_r_squared,3)}",
                ],  # riga 2
                ["Method:", "Least Squares", "F-statistic:", "None"],  # riga 3
                ["Date:", f"{self.today}", "Prob (F-statistic):", "None"],
                ["Time:", f"{self.hour}", "Log-Likelihood:", "None"],
                ["No. Observations:", f"{self.objects_number}", "AIC:", "None"],
                ["Df Residuals:", f"{self.dof}", "BIC:", "None"],
                ["Df Model:", f"{self.k}", "", ""],
                ["Covariance Type:", "nonrobust", "", ""],
            ]
        else:
            table1 = [
                [
                    "Dep. Variable:",
                    "Y",
                    "R-squared (uncentered):",
                    f"{round(self.r2,3)}",
                ],  # riga 1
                [
                    "Model:",
                    f"{self.model}",
                    "Adj. R-squared (uncentered):",
                    f"{round(self.adjusted_r_squared,3)}",
                ],  # riga 2
                ["Method:", "Least Squares", "F-statistic:", "None"],  # riga 3
                ["Date:", f"{self.today}", "Prob (F-statistic):", "None"],
                ["Time:", f"{self.hour}", "Log-Likelihood:", "None"],
                ["No. Observations:", f"{self.objects_number}", "AIC:", "None"],
                ["Df Residuals:", f"{self.dof}", "BIC:", "None"],
                ["Df Model:", f"{self.k}", "", ""],
                ["Covariance Type:", "nonrobust", "", ""],
            ]

        print(
            tabulate(
                table1, headers=headers1, colalign=("left", "right", "left", "right")
            )
        )
        print("\n")
        if self.fit_intercept == False:
            print(
                "Notes:\n [1] R² is computed without centering (uncentered) since the model does not contain a constant."
            )
        print("\n")

        headers2 = ["", "Coef", "Std Err", "t", "P>|t|", "[0.025", "0.975]"]
        table2 = []
        for i in range(len(self.coefficients)):
            if self.fit_intercept == True:
                if i == 0:
                    row = [
                        f"Intercept",
                        f"{self.coefficients[i][0]}",
                        f"{self.se_params[i]}",
                        f"{self.t_params[i][0]}",
                        f"{self.p_params[i][0]}",
                        f"{self.conf_int_lower[i][0]}",
                        f"{self.conf_int_upper[i][0]}",
                    ]
                else:
                    row = [
                        f"Coefficient {i}",
                        f"{self.coefficients[i][0]}",
                        f"{self.se_params[i]}",
                        f"{self.t_params[i][0]}",
                        f"{self.p_params[i][0]}",
                        f"{self.conf_int_lower[i][0]}",
                        f"{self.conf_int_upper[i][0]}",
                    ]
            else:
                row = [
                    f"Coefficient {i+1}",
                    f"{self.coefficients[i][0]}",
                    f"{self.se_params[i]}",
                    f"{self.t_params[i][0]}",
                    f"{self.p_params[i][0]}",
                    f"{self.conf_int_lower[i][0]}",
                    f"{self.conf_int_upper[i][0]}",
                ]
            table2.append(row)
        print(
            tabulate(
                table2,
                headers=headers2,
                colalign=("left", "right", "right", "right", "right", "right", "right"),
            )
        )
        print("\n")

        headers3 = ["", "", "", ""]
        table3 = [
            [
                "Omnibus:",
                f"{self.omnibus[0]}",
                "Durbin-Watson:",
                f"{self.dw}",
            ],  # riga 1
            [
                "Prob(Omnibus):",
                f"{self.prob_omnibus[0]}",
                "Jarque-Bera (JB):",
                f"{self.jb}",
            ],  # riga 2
            ["Skew:", f"{self.skewness[0]}", "Prob(JB):", f"{self.prob_jb}"],  # riga 3
            ["Kurtosis:", f"{self.kurtosis[0]}", "Cond. No.", f"{self.cond_no}"],
        ]
        print(
            tabulate(
                table3, headers=headers3, colalign=("left", "right", "left", "right")
            )
        )

        print("\n")
        print(
            "Notes:[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
        )

        headers_residuals = ["Min", "1Q", "Median", "3Q", "Max"]
        table_residuals = [
            [
                f"{self.residuals_min}",
                f"{self.residuals_1q}",
                f"{self.residuals_median}",
                f"{self.residuals_3q}",
                f"{self.residuals_max}",
            ]
        ]
        print("\n")
        print("Residuals:")
        print(
            tabulate(
                table_residuals,
                headers=headers_residuals,
                colalign=("center", "center", "center", "center", "center"),
            )
        )
        print(
            f"Residual standard error: {round(self.rse, 2)} on {self.dof} degrees of freedom"
        )

    def save_model(self, filename="linear_regression"):
        # Salva l'istanza della classe su disco
        with open(f"{filename}.gri", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, filename="linear_regression"):
        # Carica l'istanza della classe da disco
        with open(f"{filename}.gri", "rb") as file:
            return pickle.load(file)
