import matplotlib.pyplot as plt
import plotly.express as px  # Plotly imports remain for Plotter's dual capability
import numpy as np
import pandas as pd
import scipy.stats as st
from typing import List, Dict, Optional, Union, Tuple, Any


# --- Master Plot Class (Modified) ---
class Plotter:
    """Base class for plotting, providing shared settings and functionality."""

    THEMES = {
        "light": {
            "bg_color": "#ffffff",
            "text_color": "#2b2b2b",
            "grid_color": "#e0e0e0",
            "detail_light_color": "#b4aea9",  # Formerly GREY_LIGHT
            "detail_medium_color": "#7F7F7F",  # Formerly GREY50
            "theme_color": "#264653",  # Primary color for plot elements
            "accent_color": "#e76f51",  # Accent color for highlights
            "category_color_scale": [
                "#1B9E77",
                "#D95F02",
                "#7570B3",
                "#E7298A",
                "#66A61E",
            ],
            "prediction_band": "#2a9d8f",
            "confidence_band": "#f4a261",
            "annotation_fontsize_small": 9,
            "annotation_fontsize_medium": 11,
        },
        "dark": {
            "bg_color": "#2b2b2b",
            "text_color": "#ffffff",
            "grid_color": "#404040",
            "detail_light_color": "#666666",
            "detail_medium_color": "#888888",  # Adjusted for better visibility on dark
            "theme_color": "#8ecae6",
            "accent_color": "#ff6b6b",
            "category_color_scale": [
                "#80b918",
                "#ff99c8",
                "#52b69a",
                "#fca311",
                "#adc178",
            ],  # Dark theme appropriate scale
            "prediction_band": "#48cae4",
            "confidence_band": "#ffd60a",
            "annotation_fontsize_small": 10,  # Slightly larger on dark themes for readability
            "annotation_fontsize_medium": 12,
        },
    }

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
                "axes.titlesize": 18,  # subplot title
                "figure.titlesize": 22,  # main figure title
                "grid.alpha": 0.5,
                "lines.linewidth": 3,
            },
        },
        "plotly": {  # Plotly presets remain for completeness
            "default": {
                "layout": {
                    "xaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "yaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "plot_bgcolor": None,
                    "paper_bgcolor": None,
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
        self.library = library
        self.theme = theme
        self.style_preset = style_preset
        self.colors = self.THEMES[theme]  # self.colors now contains the expanded theme
        self.watermark = kwargs.get("watermark", None)
        self.user_figsize = kwargs.get(
            "figsize", (10, 7)
        )  # Retain user_figsize and provide default

        if self.library == "matplotlib":
            self._init_matplotlib_style()
        elif self.library == "plotly":
            self._init_plotly_style()

    def _init_matplotlib_style(self):
        plt.style.use("default")
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
            "axes.titlesize": 14,  # For ax.set_title()
            "figure.titlesize": 16,
            "figure.titleweight": "bold",  # For fig.suptitle()
            "grid.color": self.colors["grid_color"],
            "grid.linestyle": "--",
            "lines.linewidth": 2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
        preset_style = self.STYLE_PRESETS["matplotlib"].get(self.style_preset, {})
        # Preset can override base, e.g. presentation font sizes
        plt.rcParams.update({**base_style, **preset_style})

    def _init_plotly_style(self):
        import plotly.io as pio

        template_name = f"chemtools_{self.theme}"
        # Start with a copy of plotly's default, then update
        base_template = pio.templates["plotly"]
        new_template = dict(base_template)
        preset_settings_layout = (
            self.STYLE_PRESETS["plotly"].get(self.style_preset, {}).get("layout", {})
        )

        new_template["layout"] = {
            **new_template.get(
                "layout", {}
            ),  # Keep existing layout settings from base_template
            "paper_bgcolor": self.colors["bg_color"],
            "plot_bgcolor": self.colors["bg_color"],
            "font": {
                "family": "serif",
                "size": 12,
                "color": self.colors["text_color"],
            },  # Base font for plotly
            "xaxis": {
                **new_template.get("layout", {}).get(
                    "xaxis", {}
                ),  # Keep existing xaxis settings
                "gridcolor": self.colors["grid_color"],
                "linecolor": self.colors["text_color"],
                **preset_settings_layout.get("xaxis", {}),  # Apply preset specifics
            },
            "yaxis": {
                **new_template.get("layout", {}).get(
                    "yaxis", {}
                ),  # Keep existing yaxis settings
                "gridcolor": self.colors["grid_color"],
                "linecolor": self.colors["text_color"],
                **preset_settings_layout.get("yaxis", {}),  # Apply preset specifics
            },
            # Apply other general preset layout settings, ensuring they don't overwrite critical theme items unless intended
            **{
                k: v
                for k, v in preset_settings_layout.items()
                if k not in ["xaxis", "yaxis", "font", "paper_bgcolor", "plot_bgcolor"]
            },
            "title": {  # Ensure title settings from preset are merged correctly
                **(
                    new_template.get("layout", {}).get("title", {})
                ),  # Base title settings
                **(preset_settings_layout.get("title", {})),  # Preset title settings
                # Ensure font color is from theme if not specified in preset
                "font": {
                    **(new_template.get("layout", {}).get("title", {}).get("font", {})),
                    **(preset_settings_layout.get("title", {}).get("font", {})),
                    "color": preset_settings_layout.get("title", {})
                    .get("font", {})
                    .get("color", self.colors["text_color"]),
                },
            },
            "showlegend": False,
        }
        # Ensure base font settings from preset are also applied if they exist
        if "font" in preset_settings_layout:
            new_template["layout"]["font"] = {
                **new_template["layout"]["font"],
                **preset_settings_layout["font"],
            }

        pio.templates[template_name] = new_template
        self.plotly_template = template_name

    def add_watermark(self, fig, text="ChemTools", alpha=0.1):
        # Same as user provided
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
        if preset is None:
            preset = self.style_preset
        if preset not in self.STYLE_PRESETS[self.library]:
            return fig
        if self.library == "matplotlib":
            # Update only specific axes properties instead of re-initializing style
            for ax_item in fig.get_axes():
                preset_style = self.STYLE_PRESETS["matplotlib"][preset]
                # Update grid settings
                if "axes.grid" in preset_style:
                    ax_item.grid(preset_style["axes.grid"])
                # Update spine visibility
                if "axes.spines.top" in preset_style:
                    ax_item.spines["top"].set_visible(preset_style["axes.spines.top"])
                if "axes.spines.right" in preset_style:
                    ax_item.spines["right"].set_visible(
                        preset_style["axes.spines.right"]
                    )
                # Update grid alpha if specified
                if "grid.alpha" in preset_style and ax_item.grid():
                    ax_item.grid(True, alpha=preset_style["grid.alpha"])

            fig.set_facecolor(plt.rcParams["figure.facecolor"])
            fig.tight_layout()
        elif self.library == "plotly":
            # Existing Plotly code remains the same
            settings = self.STYLE_PRESETS["plotly"][preset]
            if "layout" in settings:
                layout_settings = settings["layout"].copy()
                layout_settings["plot_bgcolor"] = self.colors["bg_color"]
                layout_settings["paper_bgcolor"] = self.colors["bg_color"]
                if "xaxis" in layout_settings:
                    layout_settings["xaxis"]["gridcolor"] = self.colors["grid_color"]
                    layout_settings["xaxis"]["linecolor"] = self.colors["text_color"]
                if "yaxis" in layout_settings:
                    layout_settings["yaxis"]["gridcolor"] = self.colors["grid_color"]
                    layout_settings["yaxis"]["linecolor"] = self.colors["text_color"]
                if "font" not in layout_settings:
                    layout_settings["font"] = {}
                layout_settings["font"]["color"] = self.colors["text_color"]
                fig.update_layout(**layout_settings)
            fig.update_traces(
                line=dict(color=self.colors["theme_color"]),
                selector=dict(type="scatter"),
            )
        return fig

    def _create_figure(self, **kwargs):
        # Uses self.user_figsize which is set in __init__
        figsize_to_use = kwargs.pop("figsize", self.user_figsize)
        if self.library == "matplotlib":
            fig_kwargs = {"figsize": figsize_to_use} if figsize_to_use else {}
            fig, ax = plt.subplots(**fig_kwargs, **kwargs)
            return fig, ax
        elif self.library == "plotly":
            # Plotly uses width/height. If figsize_to_use is a tuple, convert or expect width/height in kwargs.
            # For now, assuming Plotly figures are sized by _apply_common_layout or direct width/height.
            import plotly.graph_objects as go

            fig = go.Figure(**kwargs)  # kwargs for go.Figure
            fig.update_layout(template=self.plotly_template)
            return fig

    def _set_labels(
        self, ax_or_fig, xlabel=None, ylabel=None, title=None
    ):  # Renamed param
        if self.library == "matplotlib":  # ax_or_fig is ax
            if xlabel:
                ax_or_fig.set_xlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
            if ylabel:
                ax_or_fig.set_ylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])
            if title:
                ax_or_fig.set_title(
                    title, fontsize=plt.rcParams["axes.titlesize"]
                )  # Subplot title
        elif self.library == "plotly":  # ax_or_fig is fig
            ax_or_fig.update_layout(
                xaxis_title=xlabel, yaxis_title=ylabel, title=title
            )  # Main title for Plotly

    def _process_common_params(self, **kwargs):
        # Using self.user_figsize as a default for "figsize" if not in kwargs
        common_params = {
            "figsize": kwargs.get("figsize", self.user_figsize),
            "title": kwargs.get("title", None),
            "xlabel": kwargs.get("xlabel", None),
            "ylabel": kwargs.get("ylabel", None),
            "orientation": kwargs.get("orientation", "top"),
            "color_threshold": kwargs.get("color_threshold", None),
            "labels": kwargs.get("labels", None),
            "height": kwargs.get(
                "height", self.user_figsize[1] * 80 if self.user_figsize else 600
            ),  # Approx conversion
            "width": kwargs.get(
                "width", self.user_figsize[0] * 80 if self.user_figsize else 800
            ),  # Approx conversion
        }
        return common_params

    def _apply_common_layout(self, fig, params):
        if self.library == "matplotlib":
            if params["title"]:  # This is for fig.suptitle
                fig.suptitle(
                    params["title"],
                    fontsize=plt.rcParams["figure.titlesize"],
                    weight=plt.rcParams["figure.titleweight"],
                )
            if params["figsize"]:  # Ensure figure size is set
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
