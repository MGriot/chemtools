import matplotlib.pyplot as plt
import plotly.express as px  # Plotly imports remain for Plotter's dual capability
import numpy as np
import pandas as pd
import scipy.stats as st  # Although not directly used in Plotter, often useful with plots
from typing import List, Dict, Optional, Union, Tuple, Any
import matplotlib.patches as patches  # For DimensionalityReductionPlot
from matplotlib.patches import Ellipse  # For DimensionalityReductionPlot
from scipy.stats import chi2  # For DimensionalityReductionPlot's ellipse


# --- Master Plot Class (Plotter) ---
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
                "axes.grid": True,  # Default to grid on
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
                "axes.grid": True,
            },
        },
        "plotly": {  # Plotly presets remain for completeness
            "default": {
                "layout": {
                    "xaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "yaxis": {"showgrid": True, "showline": True, "gridwidth": 1},
                    "plot_bgcolor": None,  # Will be set by theme
                    "paper_bgcolor": None,  # Will be set by theme
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
                    "title": {"font": {"size": 24}},  # Plotly main title
                }
            },
        },
    }

    def __init__(
        self, library="matplotlib", theme="light", style_preset="default", **kwargs
    ):
        self.library = library
        if theme not in self.THEMES:
            print(f"Warning: Theme '{theme}' not found. Using 'light' theme.")
            self.theme = "light"
        else:
            self.theme = theme
        if style_preset not in self.STYLE_PRESETS[self.library]:
            print(
                f"Warning: Style preset '{style_preset}' not found for {self.library}. Using 'default'."
            )
            self.style_preset = "default"
        else:
            self.style_preset = style_preset

        self.colors = self.THEMES[
            self.theme
        ].copy()  # Use a copy to allow augmentation in subclasses
        self.watermark = kwargs.get("watermark", None)
        self.user_figsize = kwargs.get(
            "figsize", (10, 6)
        )  # Default figsize changed slightly

        if self.library == "matplotlib":
            self._init_matplotlib_style()
        elif self.library == "plotly":
            self._init_plotly_style()

    def _init_matplotlib_style(self):
        plt.style.use("default")  # Start with a known base
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
            "figure.titlesize": 16,  # For fig.suptitle()
            "figure.titleweight": "bold",
            "grid.color": self.colors["grid_color"],
            "grid.linestyle": "--",
            "lines.linewidth": 2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.labelcolor": self.colors["text_color"],  # Theme legend text
            "legend.facecolor": self.colors["bg_color"],  # Theme legend background
            "legend.edgecolor": self.colors[
                "detail_light_color"
            ],  # Theme legend border
        }
        preset_style = self.STYLE_PRESETS["matplotlib"].get(self.style_preset, {})
        # Preset can override base, e.g. presentation font sizes
        plt.rcParams.update({**base_style, **preset_style})

    def _init_plotly_style(self):
        import plotly.io as pio

        template_name = f"chemtools_{self.theme}_{self.style_preset}"  # Include preset in name for uniqueness
        # Start with a copy of plotly's default, then update
        base_template = (
            pio.templates["plotly_white"]
            if self.theme == "light"
            else pio.templates["plotly_dark"]
        )
        new_template = dict(
            base_template
        )  # Make a deep copy if modifying nested dicts heavily

        preset_layout_settings = (
            self.STYLE_PRESETS["plotly"].get(self.style_preset, {}).get("layout", {})
        )

        # Theme-specific layout settings
        theme_layout = {
            "paper_bgcolor": self.colors["bg_color"],
            "plot_bgcolor": self.colors["bg_color"],
            "font": {
                "family": "serif",  # Base font family
                "size": 12,  # Base font size
                "color": self.colors["text_color"],
            },
            "xaxis": {
                "gridcolor": self.colors["grid_color"],
                "linecolor": self.colors["text_color"],  # Axis line color
                "zerolinecolor": self.colors["grid_color"],
                "title": {"font": {"color": self.colors["text_color"]}},
                "tickfont": {"color": self.colors["text_color"]},
            },
            "yaxis": {
                "gridcolor": self.colors["grid_color"],
                "linecolor": self.colors["text_color"],  # Axis line color
                "zerolinecolor": self.colors["grid_color"],
                "title": {"font": {"color": self.colors["text_color"]}},
                "tickfont": {"color": self.colors["text_color"]},
            },
            "title": {  # For fig.update_layout(title_text="...")
                "font": {"color": self.colors["text_color"]},
                "x": 0.5,  # Center title
            },
            "legend": {
                "font": {"color": self.colors["text_color"]},
                "bgcolor": self.colors["bg_color"],
                "bordercolor": self.colors["detail_light_color"],
            },
        }

        # Merge: Start with new_template.layout, update with theme_layout, then with preset_layout_settings
        # This allows presets to override theme specifics if necessary, and theme to override base Plotly template.
        # Careful merging of nested dictionaries (like xaxis, yaxis, font)
        def merge_dicts(d1, d2):
            merged = d1.copy()
            for key, value in d2.items():
                if (
                    isinstance(value, dict)
                    and key in merged
                    and isinstance(merged[key], dict)
                ):
                    merged[key] = merge_dicts(merged[key], value)
                else:
                    merged[key] = value
            return merged

        new_template["layout"] = merge_dicts(
            new_template.get("layout", {}), theme_layout
        )
        new_template["layout"] = merge_dicts(
            new_template["layout"], preset_layout_settings
        )

        # Default trace colors for Plotly if not specified by user
        new_template["layout"]["colorway"] = self.colors["category_color_scale"]

        pio.templates[template_name] = new_template
        self.plotly_template = (
            template_name  # Store the name of the registered template
        )

    def add_watermark(self, fig, text=None, alpha=0.1):
        if text is None and self.watermark is None:
            return fig  # No watermark to add

        watermark_text = text if text is not None else self.watermark

        if self.library == "matplotlib":
            # Check if fig is a matplotlib Figure
            if not isinstance(fig, plt.Figure):
                print(
                    "Warning: add_watermark for matplotlib expects a matplotlib.figure.Figure instance."
                )
                return fig
            fig.text(
                0.5,
                0.5,
                watermark_text,
                fontsize=self.colors.get("annotation_fontsize_medium", 11)
                * 3,  # Scale font size
                color=self.colors["text_color"],
                ha="center",
                va="center",
                alpha=alpha,
                transform=fig.transFigure,
                zorder=-100,  # Ensure it's in background
            )
        elif self.library == "plotly":
            # Check if fig is a plotly Figure
            if not hasattr(fig, "add_annotation"):
                print(
                    "Warning: add_watermark for plotly expects a plotly.graph_objects.Figure instance."
                )
                return fig
            fig.add_annotation(
                text=watermark_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=self.colors.get("annotation_fontsize_medium", 12) * 3,
                    color=self.colors["text_color"],
                ),
                opacity=alpha,
                textangle=-30,
                layer="below",  # Ensure it's in background
            )
        return fig

    def apply_style_preset(self, fig, preset=None):
        # This method is more for Matplotlib when fig and ax are already created.
        # For Plotly, the template is applied at figure creation.
        # This can be used to tweak an existing Matplotlib figure.
        if self.library != "matplotlib":
            # For Plotly, re-applying a preset to an existing figure is more complex
            # and usually handled by updating layout with specific preset values.
            # The initial templating in _create_figure is the primary way.
            # However, we can attempt a layout update if a preset is given.
            if self.library == "plotly" and preset:
                _preset_settings = (
                    self.STYLE_PRESETS["plotly"].get(preset, {}).get("layout", {})
                )
                if _preset_settings and hasattr(fig, "update_layout"):
                    # Re-apply theme colors that might be overridden by a generic preset
                    _preset_settings["paper_bgcolor"] = self.colors["bg_color"]
                    _preset_settings["plot_bgcolor"] = self.colors["bg_color"]
                    if "font" not in _preset_settings:
                        _preset_settings["font"] = {}
                    _preset_settings["font"]["color"] = self.colors["text_color"]
                    # ... (add more specific theme reinstatements if needed)
                    fig.update_layout(**_preset_settings)
            return fig

        current_preset_name = preset if preset is not None else self.style_preset
        if current_preset_name not in self.STYLE_PRESETS["matplotlib"]:
            print(
                f"Warning: Matplotlib style preset '{current_preset_name}' not found."
            )
            return fig

        preset_settings = self.STYLE_PRESETS["matplotlib"][current_preset_name]

        # Re-apply the full rcParams for the chosen preset to ensure all settings take effect
        # This is more robust than trying to update individual ax properties for presets.
        original_rc = plt.rcParams.copy()
        try:
            base_style_for_apply = {  # Re-establish base theme colors
                "figure.facecolor": self.colors["bg_color"],
                "axes.facecolor": self.colors["bg_color"],
                "axes.edgecolor": self.colors["text_color"],
                "axes.labelcolor": self.colors["text_color"],
                "text.color": self.colors["text_color"],
                "xtick.color": self.colors["text_color"],
                "ytick.color": self.colors["text_color"],
                "grid.color": self.colors["grid_color"],
                "legend.labelcolor": self.colors["text_color"],
            }
            plt.rcParams.update(
                {**plt.rcParamsDefault, **base_style_for_apply, **preset_settings}
            )

            # Update existing figure and axes based on new rcParams
            fig.set_facecolor(plt.rcParams["figure.facecolor"])
            for ax_item in fig.get_axes():
                ax_item.set_facecolor(plt.rcParams["axes.facecolor"])
                ax_item.grid(
                    plt.rcParams["axes.grid"],
                    alpha=plt.rcParams["grid.alpha"],
                    color=plt.rcParams["grid.color"],
                )
                ax_item.spines["top"].set_visible(plt.rcParams["axes.spines.top"])
                ax_item.spines["right"].set_visible(plt.rcParams["axes.spines.right"])
                # Update other properties if needed, like label colors, tick colors etc.
                ax_item.xaxis.label.set_color(plt.rcParams["axes.labelcolor"])
                ax_item.yaxis.label.set_color(plt.rcParams["axes.labelcolor"])
                ax_item.title.set_color(plt.rcParams["text.color"])  # axes title
                for tick in ax_item.get_xticklabels() + ax_item.get_yticklabels():
                    tick.set_color(plt.rcParams["xtick.color"])

        finally:
            # plt.rcParams.update(original_rc) # Restore original rcParams if this func is only for temp changes
            # If this function permanently changes the style for the fig, don't restore.
            # For now, assume it's a permanent change for *this* figure's context.
            pass

        if hasattr(fig, "tight_layout"):
            fig.tight_layout()
        return fig

    def _create_figure(self, **kwargs):
        # Uses self.user_figsize which is set in __init__
        # kwargs can override self.user_figsize if 'figsize' is in kwargs
        figsize_to_use = kwargs.pop("figsize", self.user_figsize)

        if self.library == "matplotlib":
            # Matplotlib style is set in __init__ via rcParams.
            # So, subplots() will use those settings.
            fig_kwargs = {"figsize": figsize_to_use} if figsize_to_use else {}
            fig, ax = plt.subplots(**fig_kwargs, **kwargs)  # Pass other subplot kwargs
            return fig, ax
        elif self.library == "plotly":
            import plotly.graph_objects as go

            # For Plotly, width/height are typically passed to layout, not go.Figure() directly
            # but can be set. The template handles most styling.
            fig = go.Figure(**kwargs)  # Pass kwargs for go.Figure if any
            # Apply the pre-initialized plotly template
            fig.update_layout(template=self.plotly_template)
            return fig  # Plotly typically doesn't return ax separately in this way

    def _set_labels(
        self, ax_or_fig, xlabel=None, ylabel=None, title=None, subplot_title=None
    ):
        if self.library == "matplotlib":  # ax_or_fig is ax
            if xlabel:
                ax_or_fig.set_xlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
            if ylabel:
                ax_or_fig.set_ylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])
            if subplot_title:  # For individual subplot titles
                ax_or_fig.set_title(
                    subplot_title, fontsize=plt.rcParams["axes.titlesize"]
                )
            # Overall figure title (suptitle) is handled by _apply_common_layout using 'title' from params
        elif self.library == "plotly":  # ax_or_fig is fig
            layout_update = {}
            if xlabel:
                layout_update["xaxis_title"] = xlabel
            if ylabel:
                layout_update["yaxis_title"] = ylabel
            if title:  # This is the main figure title for Plotly
                layout_update["title_text"] = title
                layout_update["title_x"] = 0.5  # Center title
            ax_or_fig.update_layout(**layout_update)

    def _process_common_params(self, **kwargs):
        # Using self.user_figsize as a default for "figsize" if not in kwargs
        # For Plotly, convert figsize (inches) to pixels for width/height
        dpi = plt.rcParams.get("savefig.dpi", 100)  # Get DPI, default to 100 if not set
        default_width_px = self.user_figsize[0] * dpi
        default_height_px = self.user_figsize[1] * dpi

        common_params = {
            "figsize": kwargs.get("figsize", self.user_figsize),
            "title": kwargs.get("title", None),  # This is main figure title
            "subplot_title": kwargs.get(
                "subplot_title", None
            ),  # For Matplotlib ax.set_title
            "xlabel": kwargs.get("xlabel", None),
            "ylabel": kwargs.get("ylabel", None),
            # "orientation": kwargs.get("orientation", "top"), # Not used in base, maybe in specific plots
            # "color_threshold": kwargs.get("color_threshold", None), # Specific plot use
            "labels": kwargs.get(
                "labels", None
            ),  # E.g. for legend items or point labels
            "height": kwargs.get("height", default_height_px),  # For Plotly layout
            "width": kwargs.get("width", default_width_px),  # For Plotly layout
            "showlegend": kwargs.get(
                "showlegend", False
            ),  # Default to False, can be overridden
        }
        return common_params

    def _apply_common_layout(self, fig_or_ax, params):
        # fig_or_ax can be Matplotlib fig, or Plotly fig.
        # For Matplotlib, if only an ax is passed, suptitle won't apply well.
        # This method assumes 'fig_or_ax' is the main figure object.

        if self.library == "matplotlib":
            fig = fig_or_ax
            if not isinstance(fig, plt.Figure):  # If ax was passed, try to get fig
                if hasattr(fig_or_ax, "get_figure"):
                    fig = fig_or_ax.get_figure()
                else:
                    return fig_or_ax  # Cannot apply figure-level settings

            if params.get("title"):  # This is for fig.suptitle
                fig.suptitle(
                    params["title"],
                    fontsize=plt.rcParams["figure.titlesize"],
                    weight=plt.rcParams["figure.titleweight"],
                    color=self.colors["text_color"],
                )
            if params.get("figsize"):  # Ensure figure size is set or updated
                fig.set_size_inches(params["figsize"])

            # Basic legend handling for Matplotlib if showlegend is true
            # This assumes handles/labels were added to axes correctly before calling this.
            if params.get("showlegend", False):
                # Attempt to create a figure-level legend if multiple axes, or use axis legend
                # This is tricky to generalize perfectly.
                # Simple approach: if any axis has a legend, ensure it's styled.
                for ax_item in fig.get_axes():
                    if ax_item.get_legend_handles_labels()[0]:  # If handles exist
                        leg = ax_item.legend(labelcolor=self.colors["text_color"])
                        if leg:
                            leg.get_frame().set_facecolor(self.colors["bg_color"])
                            leg.get_frame().set_edgecolor(
                                self.colors["detail_light_color"]
                            )
                        break  # Assume one legend is enough or handled per axis

        elif self.library == "plotly":
            fig = fig_or_ax  # fig_or_ax is fig for Plotly
            layout_update = {
                # Title is handled by _set_labels or directly in plot methods for Plotly for more control
                # "title_text": params.get("title"), "title_x": 0.5,
                "width": params.get("width"),
                "height": params.get("height"),
                "showlegend": params.get("showlegend", False),
                # Template already applied at creation, but can update specific parts here
            }
            # Filter out None values before updating layout
            fig.update_layout(
                **{k: v for k, v in layout_update.items() if v is not None}
            )

        if self.watermark and fig:  # Add watermark if specified and fig is valid
            self.add_watermark(fig, text=self.watermark)

        if self.library == "matplotlib" and hasattr(fig, "tight_layout"):
            try:
                fig.tight_layout(
                    rect=[0, 0.03, 1, 0.95] if params.get("title") else None
                )  # Adjust for suptitle
            except ValueError:
                pass  # Sometimes tight_layout fails

        return fig_or_ax
