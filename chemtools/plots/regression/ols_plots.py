import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


def plot_residuals(ols_object, library="matplotlib"):
    """Plots the residuals of the regression."""
    if library == "matplotlib":
        plt.scatter(range(len(ols_object.residuals)), ols_object.residuals)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Observations")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Observations")
        plt.show()
    elif library == "plotly":
        fig = px.scatter(
            x=range(len(ols_object.residuals)), y=ols_object.residuals.ravel()
        )
        fig.update_layout(
            shapes=[dict(type="line", x0=0, x1=len(ols_object.residuals), y0=0, y1=0)],
            xaxis_title="Observations",
            yaxis_title="Residuals",
        )
        fig.show()


def plot_data(ols_object, library="matplotlib"):
    """Plots the input data used for the regression."""
    if library == "matplotlib":
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c="#000000", alpha=0.25
        )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Data Points")
        plt.legend(loc="best")
        plt.show()
    elif library == "plotly":
        fig = px.scatter(x=ols_object.X_orig.ravel(), y=ols_object.y.ravel())
        fig.show()


def plot_regression_line(ols_object, library="matplotlib"):
    """Plots the regression line along with the input data."""
    if library == "matplotlib":
        x_line = np.linspace(ols_object.X_orig.min(), ols_object.X_orig.max(), num=100)
        x_line = x_line.reshape(-1, 1)
        y_line = ols_object.predict(x_line)
        plt.plot(x_line, y_line, "r", label="Regression Line")
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c="#000000", alpha=0.25
        )

        # Equation string
        equation_str = r"$y = "
        if ols_object.fit_intercept:
            equation_str += f"{ols_object.coefficients[0][0]:.2f} + "
            start_idx = 1
        else:
            start_idx = 0
        for i in range(start_idx, len(ols_object.coefficients)):
            equation_str += (
                f"{ols_object.coefficients[i][0]:.2f} x_{{{i - start_idx + 1}}}"
            )
            if i != len(ols_object.coefficients) - 1:
                equation_str += " + "
        equation_str += r"$"

        plt.plot([], [], " ", label=f"{equation_str}")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Regression Line")
        plt.legend(loc="best")
        plt.show()


def plot_confidence_band(ols_object, library="matplotlib"):
    """Plots the confidence band around the regression line."""
    if library == "matplotlib":
        plt.fill_between(
            ols_object.X_orig.ravel(),
            ols_object.lower_confidence_band.ravel(),
            ols_object.upper_confidence_band.ravel(),
            color="b",
            alpha=0.3,
            label="Confidence Band",
        )
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c="#000000", alpha=0.25
        )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Confidence Band")
        plt.legend(loc="best")
        plt.show()


def plot_prediction_band(ols_object, library="matplotlib"):
    """Plots the prediction band around the regression line."""
    if library == "matplotlib":
        plt.fill_between(
            ols_object.X_orig.ravel(),
            ols_object.lower_prediction_band.ravel(),
            ols_object.upper_prediction_band.ravel(),
            color="g",
            alpha=0.3,
            label="Prediction Band",
        )
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c="#000000", alpha=0.25
        )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Prediction Band")
        plt.legend(loc="best")
        plt.show()
