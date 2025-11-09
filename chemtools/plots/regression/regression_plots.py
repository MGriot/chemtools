import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ..base import BasePlotter


class RegressionPlots(BasePlotter):
    """Class for plotting regression results."""

    def __init__(self, regression_object, **kwargs):
        super().__init__(**kwargs)
        self.regression_object = regression_object

    def plot_residuals(self, xlabel="Observations", ylabel="Residuals", **kwargs):
        """Plots the residuals of the regression."""
        params = self._process_common_params(**kwargs)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.scatter(
                range(len(self.regression_object.residuals)),
                self.regression_object.residuals,
                color=self.colors["theme_color"],
                alpha=0.7,
                label="Residuals",
            )
            ax.axhline(
                0, color=self.colors["accent_color"], linestyle="--", linewidth=1.5
            )
            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Residuals vs. Observations"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            fig = px.scatter(
                x=range(len(self.regression_object.residuals)),
                y=self.regression_object.residuals.ravel(),
                labels={"x": xlabel, "y": ylabel},
                title=params.get("title", "Residuals vs. Observations"),
                color_discrete_sequence=[self.colors["theme_color"]],
            )
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors["accent_color"])
            self._apply_common_layout(fig, params)
            return fig

    def plot_data(self, xlabel="X", ylabel="y", **kwargs):
        """Plots the input data used for the regression.

        Args:
            xlabel (str, optional): Label for the x-axis. Defaults to "X".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            **kwargs: Additional keyword arguments passed to the plotting
                      functions.
        """
        params = self._process_common_params(**kwargs)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.scatter(
                self.regression_object.X_orig,
                self.regression_object.y,
                label="Data",
                c=self.colors["theme_color"],
                alpha=0.7,
            )
            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Data Points"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            fig = px.scatter(
                x=self.regression_object.X_orig.ravel(),
                y=self.regression_object.y.ravel(),
                labels={"x": xlabel, "y": ylabel},
                title=params.get("title", "Data Points"),
                color_discrete_sequence=[self.colors["theme_color"]],
            )
            self._apply_common_layout(fig, params)
            return fig

    def plot_regression_line(
        self, xlabel="X", ylabel="y", show_equation=True, **kwargs
    ):
        """Plots the regression line along with the input data.

        Args:
            xlabel (str, optional): Label for the x-axis. Defaults to "X".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            show_equation (bool, optional): Whether to display the
                          regression equation on the plot. Defaults to True.
            **kwargs: Additional keyword arguments passed to the plotting
                      functions.
        """
        params = self._process_common_params(**kwargs)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            x_line = np.linspace(
                self.regression_object.X_orig.min(),
                self.regression_object.X_orig.max(),
                num=100,
            ).reshape(-1, 1)
            y_line = self.regression_object.predict(x_line)
            ax.plot(
                x_line,
                y_line,
                color=self.colors["accent_color"],
                label="Regression Line",
            )
            ax.scatter(
                self.regression_object.X_orig,
                self.regression_object.y,
                label="Data",
                c=self.colors["theme_color"],
                alpha=0.7,
            )

            if show_equation:
                equation_str = self._generate_equation_string()
                ax.text(
                    0.05,
                    0.95,
                    equation_str,
                    transform=ax.transAxes,
                    verticalalignment="top",
                )

            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Regression Line"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            fig = go.Figure()
            x_line = np.linspace(
                self.regression_object.X_orig.min(),
                self.regression_object.X_orig.max(),
                num=100,
            ).reshape(-1, 1)
            y_line = self.regression_object.predict(x_line)

            fig.add_trace(
                go.Scatter(
                    x=x_line.flatten(),
                    y=y_line.flatten(),
                    mode="lines",
                    name="Regression Line",
                    line=dict(color=self.colors["accent_color"]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.regression_object.X_orig.flatten(),
                    y=self.regression_object.y.flatten(),
                    mode="markers",
                    name="Data",
                    marker=dict(color=self.colors["theme_color"]),
                )
            )

            if show_equation:
                equation_str = self._generate_equation_string()
                fig.add_annotation(
                    text=equation_str,
                    xref="paper",
                    yref="paper",
                    x=0.05,
                    y=0.95,
                    showarrow=False,
                )

            self._set_labels(fig, xlabel=xlabel, ylabel=ylabel, title=params.get("title", "Regression Line"))
            self._apply_common_layout(fig, params)
            return fig

    def _generate_equation_string(self):
        """Helper method to generate the equation string."""
        coefficients = self.regression_object.coefficients
        if not hasattr(coefficients, "__len__"):
            coefficients = np.array([coefficients])

        equation_str = r"$y = "

        # Handle single coefficient case
        if len(coefficients) == 1:
            equation_str += f"{coefficients[0]:.2f}x$"
            return equation_str

        # Handle multiple coefficients
        if self.regression_object.fit_intercept:
            equation_str += f"{coefficients[0]:.2f} + "
            for i in range(1, len(coefficients)):
                equation_str += f"{coefficients[i]:.2f}x_{{{i}}}"
                if i < len(coefficients) - 1:
                    equation_str += " + "
        else:
            for i in range(len(coefficients)):
                equation_str += f"{coefficients[i]:.2f}x_{{{i+1}}}"
                if i < len(coefficients) - 1:
                    equation_str += " + "

        equation_str += "$"
        return equation_str

    def plot_confidence_band(
        self, xlabel="X", ylabel="y", confidence_band_color=None, **kwargs
    ):
        """Plots the confidence band around the regression line.

        Args:
            xlabel (str, optional): Label for the x-axis. Defaults to "X".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            confidence_band_color (str, optional): Color of the confidence band.
                                   If None, uses theme's confidence_band color.
            **kwargs: Additional keyword arguments for plotting.
        """
        params = self._process_common_params(**kwargs)
        confidence_band_color = confidence_band_color or self.colors["confidence_band"]

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.fill_between(
                self.regression_object.X_orig.ravel(),
                self.regression_object.lower_confidence_band.ravel(),
                self.regression_object.upper_confidence_band.ravel(),
                color=confidence_band_color,
                alpha=0.3,
                label="Confidence Band",
            )
            ax.scatter(
                self.regression_object.X_orig,
                self.regression_object.y,
                label="Data",
                c=self.colors["theme_color"],
                alpha=0.7,
            )
            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Confidence Band"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # ... Implement Plotly version if needed ...
            pass

    def plot_prediction_band(
        self, xlabel="X", ylabel="y", prediction_band_color=None, **kwargs
    ):
        """Plots the prediction band around the regression line.

        Args:
            xlabel (str, optional): Label for the x-axis. Defaults to "X".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            prediction_band_color (str, optional): Color of the prediction band.
                                   If None, uses theme's prediction_band color.
            **kwargs: Additional keyword arguments for plotting.
        """
        params = self._process_common_params(**kwargs)
        prediction_band_color = prediction_band_color or self.colors["prediction_band"]

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            ax.fill_between(
                self.regression_object.X_orig.ravel(),
                self.regression_object.lower_prediction_band.ravel(),
                self.regression_object.upper_prediction_band.ravel(),
                color=prediction_band_color,
                alpha=0.3,
                label="Prediction Band",
            )
            ax.scatter(
                self.regression_object.X_orig,
                self.regression_object.y,
                label="Data",
                c=self.colors["theme_color"],
                alpha=0.7,
            )
            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Prediction Band"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # ... Implement Plotly version if needed ...
            pass

    def plot_regression_results(
        self,
        xlabel="X",
        ylabel="y",
        show_data=True,
        show_regression_line=True,
        show_confidence_band=True,
        show_prediction_band=True,
        show_equation=True,
        confidence_band_color=None,
        prediction_band_color=None,
        **kwargs,
    ):
        """Plots regression results with customization options.

        Args:
            xlabel (str, optional): Label for x-axis. Defaults to "X".
            ylabel (str, optional): Label for y-axis. Defaults to "y".
            show_data (bool, optional): Show data points. Defaults to True.
            show_regression_line (bool, optional): Show regression line.
                                                   Defaults to True.
            show_confidence_band (bool, optional): Show confidence band.
                                                    Defaults to True.
            show_prediction_band (bool, optional): Show prediction band.
                                                    Defaults to True.
            show_equation (bool, optional): Show regression equation.
                                             Defaults to True.
            confidence_band_color (str, optional): Confidence band color.
                                                   Defaults to default.
            prediction_band_color (str, optional): Prediction band color.
                                                    Defaults to default.
            **kwargs: Additional keyword arguments for plotting.
        """
        params = self._process_common_params(**kwargs)
        confidence_band_color = confidence_band_color or self.colors["confidence_band"]
        prediction_band_color = prediction_band_color or self.colors["prediction_band"]

        if self.library == "matplotlib":
            fig, ax = self._create_figure(figsize=params["figsize"])
            if show_data:
                ax.scatter(
                    self.regression_object.X_orig,
                    self.regression_object.y,
                    label="Data",
                    c=self.colors["theme_color"],
                    alpha=0.7,
                )

            if show_regression_line:
                x_line = np.linspace(
                    self.regression_object.X_orig.min(),
                    self.regression_object.X_orig.max(),
                    num=100,
                )
                x_line = x_line.reshape(-1, 1)
                y_line = self.regression_object.predict(x_line)
                ax.plot(
                    x_line,
                    y_line,
                    color=self.colors["accent_color"],
                    label="Regression Line",
                )

                if show_equation:
                    equation_str = self._generate_equation_string()
                    ax.plot([], [], " ", label=f"{equation_str}")

            if show_confidence_band:
                ax.fill_between(
                    self.regression_object.X_orig.ravel(),
                    self.regression_object.lower_confidence_band.ravel(),
                    self.regression_object.upper_confidence_band.ravel(),
                    color=confidence_band_color,
                    alpha=0.3,
                    label="Confidence Band",
                )

            if show_prediction_band:
                ax.fill_between(
                    self.regression_object.X_orig.ravel(),
                    self.regression_object.lower_prediction_band.ravel(),
                    self.regression_object.upper_prediction_band.ravel(),
                    color=prediction_band_color,
                    alpha=0.3,
                    label="Prediction Band",
                )

            self._set_labels(ax, xlabel, ylabel, params.get("subplot_title", "Regression Results"))
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # ... Implement Plotly version if needed ...
            pass