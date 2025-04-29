import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


# --- Master Plot Class ---


class Plotter:
    """Base class for plotting, providing shared settings and functionality."""

    # Color themes
    THEMES = {
        "light": {
            "bg_color": "#ffffff",
            "text_color": "#2b2b2b",
            "grid_color": "#e0e0e0",
            "theme_color": "#264653",
            "accent_color": "#e76f51",
            "prediction_band": "#2a9d8f",
            "confidence_band": "#f4a261",
        },
        "dark": {
            "bg_color": "#2b2b2b",
            "text_color": "#ffffff",
            "grid_color": "#404040",
            "theme_color": "#8ecae6",
            "accent_color": "#ff6b6b",
            "prediction_band": "#48cae4",
            "confidence_band": "#ffd60a",
        },
    }

    # Add style presets as class attribute
    STYLE_PRESETS = {
        "matplotlib": {
            "default": {
                "grid.alpha": 0.5,
                "axes.spines.top": True,
                "axes.spines.right": True,
            },
            "minimal": {
                "grid.alpha": 0,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
            },
            "grid": {"grid.alpha": 0.7, "axes.grid": True, "grid.linewidth": 0.5},
            "presentation": {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "grid.alpha": 0.5,
                "lines.linewidth": 3,
            },
        },
        "plotly": {
            "default": {
                "layout": {
                    "xaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "yaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "plot_bgcolor": None,  # Will use theme color
                    "paper_bgcolor": None,  # Will use theme color
                }
            },
            "minimal": {
                "layout": {
                    "xaxis": {
                        "showgrid": False,
                        "showline": False,
                        "zeroline": False,
                        "showticklabels": True,
                    },
                    "yaxis": {
                        "showgrid": False,
                        "showline": False,
                        "zeroline": False,
                        "showticklabels": True,
                    },
                }
            },
            "grid": {
                "layout": {
                    "xaxis": {
                        "showgrid": True,
                        "gridwidth": 1,
                        "showline": True,
                        "linewidth": 2,
                    },
                    "yaxis": {
                        "showgrid": True,
                        "gridwidth": 1,
                        "showline": True,
                        "linewidth": 2,
                    },
                }
            },
            "presentation": {
                "layout": {
                    "xaxis": {
                        "showgrid": True,
                        "gridwidth": 1.5,
                        "showline": True,
                        "linewidth": 2,
                        "tickfont": {"size": 14},
                        "title": {"font": {"size": 16}},
                    },
                    "yaxis": {
                        "showgrid": True,
                        "gridwidth": 1.5,
                        "showline": True,
                        "linewidth": 2,
                        "tickfont": {"size": 14},
                        "title": {"font": {"size": 16}},
                    },
                    "font": {"size": 14},
                    "title": {"font": {"size": 24}},
                }
            },
        },
    }

    def __init__(
        self, library="matplotlib", theme="light", style_preset="default", **kwargs
    ):
        """Initializes the Plotter with default settings."""
        self.library = library
        self.theme = theme
        self.style_preset = style_preset
        self.colors = self.THEMES[theme]
        self.watermark = kwargs.get("watermark", None)

        # Initialize based on library
        if self.library == "matplotlib":
            self._init_matplotlib_style()
        elif self.library == "plotly":
            self._init_plotly_style()

    def _init_matplotlib_style(self):
        """Initialize matplotlib style settings."""
        plt.style.use("default")  # Reset to default first

        # Base style
        base_style = {
            "figure.facecolor": self.colors["bg_color"],
            "axes.facecolor": self.colors["bg_color"],
            "axes.edgecolor": self.colors["text_color"],
            "axes.labelcolor": self.colors["text_color"],
            "text.color": self.colors["text_color"],
            "xtick.color": self.colors["text_color"],
            "ytick.color": self.colors["text_color"],
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "grid.color": self.colors["grid_color"],
            "grid.linestyle": "--",
            "lines.linewidth": 2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }

        # Apply preset-specific settings
        preset_style = self.STYLE_PRESETS["matplotlib"].get(self.style_preset, {})

        # Merge and update
        plt.rcParams.update({**base_style, **preset_style})

    def _init_plotly_style(self):
        """Initialize plotly style settings."""
        import plotly.io as pio

        template_name = f"chemtools_{self.theme}"
        pio.templates[template_name] = pio.templates["plotly"]

        # Get preset settings
        preset_settings = self.STYLE_PRESETS["plotly"].get(self.style_preset, {})

        # Update template with theme colors and settings
        pio.templates[template_name].update(
            {
                "layout": {
                    "paper_bgcolor": self.colors["bg_color"],
                    "plot_bgcolor": self.colors["bg_color"],
                    "font": {
                        "family": "serif",
                        "size": 12,
                        "color": self.colors["text_color"],
                    },
                    "xaxis": {
                        "gridcolor": self.colors["grid_color"],
                        "linecolor": self.colors["text_color"],
                        **preset_settings.get("xaxis", {}),
                    },
                    "yaxis": {
                        "gridcolor": self.colors["grid_color"],
                        "linecolor": self.colors["text_color"],
                        **preset_settings.get("yaxis", {}),
                    },
                    "title": {"font": {"size": 16}},
                    "showlegend": False,
                }
            }
        )

        self.plotly_template = template_name

    def add_watermark(self, fig, text="ChemTools", alpha=0.1):
        """Add watermark to the plot."""
        if self.library == "matplotlib":
            fig.text(
                0.5,
                0.5,
                text,
                fontsize=40,
                color=self.colors["text_color"],
                ha="center",
                va="center",
                alpha=alpha,
                transform=fig.transFigure,
            )
        elif self.library == "plotly":
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=40, color=self.colors["text_color"]),
                opacity=alpha,
                textangle=-30,
            )
        return fig

    def apply_style_preset(self, fig, preset=None):
        """Apply predefined style presets to an existing figure."""
        if preset is None:
            preset = self.style_preset

        if preset not in self.STYLE_PRESETS[self.library]:
            return fig

        if self.library == "matplotlib":
            ax = fig.gca()
            settings = self.STYLE_PRESETS["matplotlib"][preset]

            # Apply settings to current axes
            for key, value in settings.items():
                if key.startswith("axes."):
                    setter = key.replace("axes.", "set_")
                    if hasattr(ax, setter):
                        getattr(ax, setter)(value)

            fig.tight_layout()

        elif self.library == "plotly":
            settings = self.STYLE_PRESETS["plotly"][preset]

            # Apply layout settings
            if "layout" in settings:
                layout_settings = settings["layout"]
                # Add theme colors to layout settings
                if "plot_bgcolor" in layout_settings:
                    layout_settings["plot_bgcolor"] = self.colors["bg_color"]
                if "paper_bgcolor" in layout_settings:
                    layout_settings["paper_bgcolor"] = self.colors["bg_color"]
                if (
                    "xaxis" in layout_settings
                    and "gridcolor" in layout_settings["xaxis"]
                ):
                    layout_settings["xaxis"]["gridcolor"] = self.colors["grid_color"]
                if (
                    "yaxis" in layout_settings
                    and "gridcolor" in layout_settings["yaxis"]
                ):
                    layout_settings["yaxis"]["gridcolor"] = self.colors["grid_color"]

                fig.update_layout(**layout_settings)

            # Update trace colors based on theme
            fig.update_traces(
                line=dict(color=self.colors["theme_color"]),
                selector=dict(type="scatter"),
            )

        return fig

    def _create_figure(self, **kwargs):
        """Creates a figure and axes based on the chosen library."""
        if self.library == "matplotlib":
            fig, ax = plt.subplots(**kwargs)
            return fig, ax
        elif self.library == "plotly":
            fig = px.scatter(**{k: v for k, v in kwargs.items() if k != "figsize"})
            fig.update_layout(template=self.plotly_template)
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
                "template": self.plotly_template,
            }
            fig.update_layout(
                **{k: v for k, v in layout_update.items() if v is not None}
            )
        return fig
