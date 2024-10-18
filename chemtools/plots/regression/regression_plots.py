import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Default theme colors and style for scientific reports
DEFAULT_THEME_COLOR = "#264653"  # Charcoal
DEFAULT_ACCENT_COLOR = "#e76f51"  # Salmon
DEFAULT_PREDICTION_BAND_COLOR = "#2a9d8f"  # Greenish blue
DEFAULT_CONFIDENCE_BAND_COLOR = "#f4a261"  # Orange

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    }
)


def plot_residuals(
    ols_object,
    library="matplotlib",
    theme_color=None,
    accent_color=None,
    xlabel="Observations",
    ylabel="Residuals",
):
    """Plots the residuals of the regression."""
    theme_color = theme_color or DEFAULT_THEME_COLOR
    accent_color = accent_color or DEFAULT_ACCENT_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))
        plt.scatter(
            range(len(ols_object.residuals)),
            ols_object.residuals,
            color=theme_color,
            alpha=0.7,
            label="Residuals",
        )
        plt.axhline(0, color=accent_color, linestyle="--", linewidth=1.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Residuals vs. Observations")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()
    elif library == "plotly":
        fig = px.scatter(
            x=range(len(ols_object.residuals)),
            y=ols_object.residuals.ravel(),
            color_discrete_sequence=[theme_color],
        )
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=0,
                    x1=len(ols_object.residuals),
                    y0=0,
                    y1=0,
                    line=dict(color=accent_color, dash="dash"),
                )
            ],
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
        fig.show()


def plot_data(
    ols_object, library="matplotlib", theme_color=None, xlabel="X", ylabel="y"
):
    """Plots the input data used for the regression."""
    theme_color = theme_color or DEFAULT_THEME_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))
        plt.scatter(
            ols_object.X_orig,
            ols_object.y,
            label="Data",
            c=theme_color,
            alpha=0.7,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Data Points")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
    elif library == "plotly":
        fig = px.scatter(
            x=ols_object.X_orig.ravel(),
            y=ols_object.y.ravel(),
            color_discrete_sequence=[theme_color],
        )
        fig.show()


def plot_regression_line(
    ols_object,
    library="matplotlib",
    theme_color=None,
    accent_color=None,
    xlabel="X",
    ylabel="y",
):
    """Plots the regression line along with the input data."""
    theme_color = theme_color or DEFAULT_THEME_COLOR
    accent_color = accent_color or DEFAULT_ACCENT_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))
        x_line = np.linspace(ols_object.X_orig.min(), ols_object.X_orig.max(), num=100)
        x_line = x_line.reshape(-1, 1)
        y_line = ols_object.predict(x_line)
        plt.plot(x_line, y_line, color=accent_color, label="Regression Line")
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c=theme_color, alpha=0.7
        )

        # Equation string
        equation_str = r"$y = "

        # Check if coefficients is a scalar or an array
        if hasattr(ols_object.coefficients, "__len__"):  # Check if is an array
            if ols_object.fit_intercept:
                equation_str += f"{ols_object.coefficients[0]:.2f} + "
                start_idx = 1
            else:
                start_idx = 0
            for i in range(start_idx, len(ols_object.coefficients)):
                equation_str += (
                    f"{ols_object.coefficients[i]:.2f} x_{{{i - start_idx + 1}}}"
                )
                if i != len(ols_object.coefficients) - 1:
                    equation_str += " + "
        else:  # if is a scalar just add it to the string
            equation_str += f"{ols_object.coefficients:.2f} x"

        equation_str += r"$"

        plt.plot([], [], " ", label=f"{equation_str}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Regression Line")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


def plot_confidence_band(
    ols_object,
    library="matplotlib",
    theme_color=None,
    confidence_band_color=None,
    xlabel="X",
    ylabel="y",
):
    """Plots the confidence band around the regression line."""
    theme_color = theme_color or DEFAULT_THEME_COLOR
    confidence_band_color = confidence_band_color or DEFAULT_CONFIDENCE_BAND_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))
        plt.fill_between(
            ols_object.X_orig.ravel(),
            ols_object.lower_confidence_band.ravel(),
            ols_object.upper_confidence_band.ravel(),
            color=confidence_band_color,
            alpha=0.3,
            label="Confidence Band",
        )
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c=theme_color, alpha=0.7
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Confidence Band")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


def plot_prediction_band(
    ols_object,
    library="matplotlib",
    theme_color=None,
    prediction_band_color=None,
    xlabel="X",
    ylabel="y",
):
    """Plots the prediction band around the regression line."""
    theme_color = theme_color or DEFAULT_THEME_COLOR
    prediction_band_color = prediction_band_color or DEFAULT_PREDICTION_BAND_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))
        plt.fill_between(
            ols_object.X_orig.ravel(),
            ols_object.lower_prediction_band.ravel(),
            ols_object.upper_prediction_band.ravel(),
            color=prediction_band_color,
            alpha=0.3,
            label="Prediction Band",
        )
        plt.scatter(
            ols_object.X_orig, ols_object.y, label="Data", c=theme_color, alpha=0.7
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Prediction Band")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


def plot_regression_results(
    ols_object,
    library="matplotlib",
    theme_color=None,
    accent_color=None,
    confidence_band_color=None,
    prediction_band_color=None,
    show_data=True,
    show_regression_line=True,
    show_confidence_band=True,
    show_prediction_band=True,
    xlabel="X",
    ylabel="y",
):
    """Plots the regression results with options to customize the plot.

    Args:
        ols_object: The fitted OLS object.
        library (str, optional): The plotting library to use.
                                  Options: 'matplotlib', 'plotly'. Defaults to 'matplotlib'.
        theme_color (str, optional): The color for the data points. Defaults to DEFAULT_THEME_COLOR.
        accent_color (str, optional): The color for the regression line. Defaults to DEFAULT_ACCENT_COLOR.
        confidence_band_color (str, optional): The color for the confidence band. Defaults to DEFAULT_CONFIDENCE_BAND_COLOR.
        prediction_band_color (str, optional): The color for the prediction band. Defaults to DEFAULT_PREDICTION_BAND_COLOR.
        show_data (bool, optional): Whether to show the data points. Defaults to True.
        show_regression_line (bool, optional): Whether to show the regression line. Defaults to True.
        show_confidence_band (bool, optional): Whether to show the confidence band. Defaults to True.
        show_prediction_band (bool, optional): Whether to show the prediction band. Defaults to True.
        xlabel: label of x axis.
        ylabel: label of y axis.
    """
    theme_color = theme_color or DEFAULT_THEME_COLOR
    accent_color = accent_color or DEFAULT_ACCENT_COLOR
    confidence_band_color = confidence_band_color or DEFAULT_CONFIDENCE_BAND_COLOR
    prediction_band_color = prediction_band_color or DEFAULT_PREDICTION_BAND_COLOR

    if library == "matplotlib":
        plt.figure(figsize=(8, 6))

        if show_data:
            plt.scatter(
                ols_object.X_orig,
                ols_object.y,
                label="Data",
                c=theme_color,
                alpha=0.7,
            )

        if show_regression_line:
            x_line = np.linspace(
                ols_object.X_orig.min(), ols_object.X_orig.max(), num=100
            )
            x_line = x_line.reshape(-1, 1)
            y_line = ols_object.predict(x_line)
            plt.plot(x_line, y_line, color=accent_color, label="Regression Line")

            # Equation string
            equation_str = r"$y = "

            # Check if coefficients is a scalar or an array
            if hasattr(ols_object.coefficients, "__len__"):  # Check if is an array
                if ols_object.fit_intercept:
                    equation_str += f"{ols_object.coefficients[0]:.2f} + "
                    start_idx = 1
                else:
                    start_idx = 0
                for i in range(start_idx, len(ols_object.coefficients)):
                    equation_str += (
                        f"{ols_object.coefficients[i]:.2f} x_{{{i - start_idx + 1}}}"
                    )
                    if i != len(ols_object.coefficients) - 1:
                        equation_str += " + "
            else:  # if is a scalar just add it to the string
                equation_str += f"{ols_object.coefficients:.2f} x"

            equation_str += r"$"

            plt.plot([], [], " ", label=f"{equation_str}")

        if show_confidence_band:
            plt.fill_between(
                ols_object.X_orig.ravel(),
                ols_object.lower_confidence_band.ravel(),
                ols_object.upper_confidence_band.ravel(),
                color=confidence_band_color,
                alpha=0.3,
                label="Confidence Band",
            )

        if show_prediction_band:
            plt.fill_between(
                ols_object.X_orig.ravel(),
                ols_object.lower_prediction_band.ravel(),
                ols_object.upper_prediction_band.ravel(),
                color=prediction_band_color,
                alpha=0.3,
                label="Prediction Band",
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Regression Results")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
