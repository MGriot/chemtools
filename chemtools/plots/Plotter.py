import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


# --- Master Plot Class ---


class Plotter:
    """Base class for plotting, providing shared settings and functionality."""

    DEFAULT_THEME_COLOR = "#264653"
    DEFAULT_ACCENT_COLOR = "#e76f51"
    DEFAULT_PREDICTION_BAND_COLOR = "#2a9d8f"
    DEFAULT_CONFIDENCE_BAND_COLOR = "#f4a261"

    def __init__(self, library="matplotlib", theme_color=None, **kwargs):
        """Initializes the Plotter with default settings."""
        self.library = library
        self.theme_color = theme_color or self.DEFAULT_THEME_COLOR
        self.accent_color = kwargs.get("accent_color", self.DEFAULT_ACCENT_COLOR)

        if self.library == "matplotlib":
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
        elif self.library == "plotly":
            import plotly.io as pio
            import plotly.express as px
            import plotly.graph_objects as go

            # Create a custom template based on the default "plotly" template
            pio.templates["chemtools"] = pio.templates["plotly"]

            # Update the template with your desired settings
            pio.templates["chemtools"].update(
                {
                    "layout": {
                        "font": {"family": "serif", "size": 12},
                        "xaxis": {
                            "title": {"font": {"size": 14}},
                            "tickfont": {"size": 10},
                        },
                        "yaxis": {
                            "title": {"font": {"size": 14}},
                            "tickfont": {"size": 10},
                        },
                        "title": {"font": {"size": 16}},
                        "showlegend": True,
                        "legend": {"font": {"size": 10}},
                        "margin": {"l": 60, "r": 10, "t": 70, "b": 50},
                    }
                }
            )

    def _create_figure(self, **kwargs):
        """Creates a figure and axes based on the chosen library."""
        if self.library == "matplotlib":
            fig, ax = plt.subplots(**kwargs)
            return fig, ax
        elif self.library == "plotly":
            fig = px.scatter(**{k: v for k, v in kwargs.items() if k != "figsize"})
            fig.update_layout(template="chemtools")
            return fig

    def _set_labels(self, ax, xlabel=None, ylabel=None, title=None):
        """Sets labels for the plot axes and title."""
        if self.library == "matplotlib":
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)
        elif self.library == "plotly":
            ax.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title)

    def _process_common_params(self, **kwargs):
        """Process common parameters for both matplotlib and plotly."""
        common_params = {
            "figsize": kwargs.get("figsize", (10, 7)),
            "title": kwargs.get("title", None),
            "xlabel": kwargs.get("xlabel", None),
            "ylabel": kwargs.get("ylabel", None),
            "orientation": kwargs.get("orientation", "top"),
            "color_threshold": kwargs.get("color_threshold", None),
            "labels": kwargs.get("labels", None),
            "height": kwargs.get("height", 600),
            "width": kwargs.get("width", 800),
        }
        return common_params

    def _apply_common_layout(self, fig, params):
        """Apply common layout parameters to both matplotlib and plotly figures."""
        if self.library == "matplotlib":
            if params["title"]:
                fig.suptitle(params["title"])
            fig.set_size_inches(params["figsize"])
        elif self.library == "plotly":
            layout_update = {
                "title": (
                    {"text": params["title"], "x": 0.5} if params["title"] else None
                ),
                "width": params["width"],
                "height": params["height"],
                "showlegend": False,
                "template": "chemtools",
            }
            fig.update_layout(
                **{k: v for k, v in layout_update.items() if v is not None}
            )
        return fig
