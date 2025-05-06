from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from Plotter import Plotter


class ViolinPlotter(Plotter):
    """
    Creates a highly customized Violin Plot. Inherits from Plotter.
    """

    # These are style attributes specific to the construction of a violin plot
    # Not general theme colors or primary font sizes, which are inherited from Plotter.
    VIOLIN_ELEMENT_STYLES = {
        "hlines_values": [40, 50, 60],  # Default values for horizontal lines
        "hlines_alpha": 0.8,
        "violin_width": 0.45,
        "jitter_df": 6,
        "jitter_scale": 0.04,
        "mean_dot_size": 250,
        "scatter_dot_size": 100,
        "scatter_alpha": 0.4,
        "median_linewidth": 4,
        "box_linewidth": 2,
        "violin_linewidth": 1.4,
        "mean_line_offset": 0.25,
        "mean_label_pad": 0.15,
        "p_value_tick_len": 0.25,
    }

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
        value_col: Optional[Union[str, int]] = None,
        category_col: Optional[Union[str, int]] = None,
        categories: Optional[List[str]] = None,
        theme: str = "light",
        style_preset: str = "default",
        watermark: Optional[str] = None,
        figsize: Tuple[float, float] = (14, 10),  # Default specific to ViolinPlotter
    ):
        super().__init__(
            library="matplotlib",
            theme=theme,
            style_preset=style_preset,
            watermark=watermark,
            figsize=figsize,
        )

        # Initialize violin-specific element styles by copying from class defaults
        self.element_styles = self.VIOLIN_ELEMENT_STYLES.copy()

        # Initialize specific font sizes for annotations using theme values from Plotter
        # These are not standard rcParams, so they are attributes of ViolinPlotter
        self.title_fontname = None  # Matplotlib default, can be overridden by user via set_fonts if needed
        self.annotation_fontsize = self.colors["annotation_fontsize_medium"]
        self.stats_fontsize = self.colors["annotation_fontsize_small"]
        self.p_value_fontsize = self.colors["annotation_fontsize_medium"]
        self.subtitle_fontsize_explicit = plt.rcParams[
            "axes.titlesize"
        ]  # for use with ax.set_title for subtitle

        self.y_data, self.x_data, self.x_jittered = [], [], []
        self.species, self.means, self.positions = [], [], []
        self._process_data(data, value_col, category_col, categories)

        self.p_value_labels, self.p_value_y_positions = None, None
        self.title_stats, self.footer_stats1, self.footer_stats2 = None, None, None

    def _process_data(self, data, value_col, category_col, categories_list):
        # (Same as your previously provided _process_data - no changes needed here based on prompt)
        if isinstance(data, pd.DataFrame):
            if value_col is None or category_col is None:
                raise ValueError(
                    "`value_col` and `category_col` must be provided for DataFrame input."
                )
            data_cleaned = data.dropna(subset=[value_col, category_col]).copy()
            data_cleaned[category_col] = data_cleaned[category_col].astype(str)
            self.species = sorted(data_cleaned[category_col].unique())
            self.y_data = [
                data_cleaned[data_cleaned[category_col] == specie][value_col].values
                for specie in self.species
            ]
        elif isinstance(data, np.ndarray):
            if value_col is None or category_col is None:
                raise ValueError(
                    "`value_col` and `category_col` (indices) must be provided for numpy array input."
                )
            if data.ndim != 2:
                raise ValueError("Numpy array must be 2-dimensional.")
            try:
                values = data[:, value_col].astype(float)
                raw_categories = data[:, category_col]
            except IndexError:
                raise ValueError(
                    "`value_col` or `category_col` index is out of bounds for the numpy array."
                )
            except ValueError:
                raise ValueError(
                    f"Could not convert data in value_col (index {value_col}) to numeric."
                )
            valid_indices = ~np.isnan(values)
            values_cleaned = values[valid_indices]
            categories_cleaned = raw_categories[valid_indices]
            unique_cats_cleaned = np.unique(categories_cleaned)
            self.species = sorted([str(cat) for cat in unique_cats_cleaned])
            self.y_data = []
            for specie_str in self.species:
                original_cat_for_specie = next(
                    (uc for uc in unique_cats_cleaned if str(uc) == specie_str), None
                )
                if original_cat_for_specie is not None:
                    self.y_data.append(
                        values_cleaned[categories_cleaned == original_cat_for_specie]
                    )
                else:
                    self.y_data.append(np.array([]))
        elif isinstance(data, list) and all(
            isinstance(arr, np.ndarray) for arr in data
        ):
            if categories_list is None or len(categories_list) != len(data):
                raise ValueError(
                    "`categories` list must be provided and match length of `data` list."
                )
            self.y_data = [arr[~np.isnan(arr)].astype(float) for arr in data]
            self.species = [str(c) for c in categories_list]
        else:
            raise TypeError(
                "Unsupported data type. Use pandas DataFrame, numpy ndarray, or List[np.ndarray]."
            )

        if not self.y_data or all(len(arr) == 0 for arr in self.y_data):
            print(
                "Warning: No valid data to plot after processing or all categories are empty."
            )
            self.positions, self.x_data, self.x_jittered, self.means = [], [], [], []
            return
        self.positions = list(range(len(self.species)))
        self.x_data = [np.array([i] * len(d)) for i, d in enumerate(self.y_data)]
        self.x_jittered = [
            (
                x
                + st.t(
                    df=self.element_styles["jitter_df"],
                    scale=self.element_styles["jitter_scale"],
                ).rvs(len(x))
                if len(x) > 0
                else np.array([])
            )
            for x in self.x_data
        ]
        self.means = [y.mean() if len(y) > 0 else np.nan for y in self.y_data]

    def set_colors(self, category_color_scale: Optional[List[str]] = None, **kwargs):
        """
        Set specific colors for the ViolinPlotter.
        Mainly used to override the category_color_scale from the theme.
        To change base theme colors (text, bg, accent), re-initialize the plotter with a new theme.
        """
        if category_color_scale is not None:
            self.colors["category_color_scale"] = category_color_scale
        if kwargs:
            print(
                f"Warning: For base theme color changes, re-initialize plotter. Unused color args: {kwargs}"
            )

    def set_fonts(
        self,
        annotation_fontsize: Optional[int] = None,
        stats_fontsize: Optional[int] = None,
        p_value_fontsize: Optional[int] = None,
        title_fontname: Optional[str] = None,
        subtitle_fontsize: Optional[int] = None,
    ):
        """
        Set specific font sizes for annotations, stats, and p-values.
        Base font sizes are controlled by Plotter's theme and presets.
        """
        if annotation_fontsize is not None:
            self.annotation_fontsize = annotation_fontsize
        if stats_fontsize is not None:
            self.stats_fontsize = stats_fontsize
        if p_value_fontsize is not None:
            self.p_value_fontsize = p_value_fontsize
        if title_fontname is not None:
            self.title_fontname = (
                title_fontname  # Allows overriding default for suptitle
            )
        if subtitle_fontsize is not None:
            self.subtitle_fontsize_explicit = subtitle_fontsize

    def set_layout(self, **kwargs):
        """
        Update layout parameters specific to ViolinPlotter elements
        (e.g., hlines_values, violin_width, jitter_scale).
        For figsize, re-initialize or use plotter's base figsize.
        """
        changed_jitter = False
        for key, value in kwargs.items():
            if key in self.element_styles:
                self.element_styles[key] = value
                if key in ["jitter_df", "jitter_scale"]:
                    changed_jitter = True
            elif key == "figsize":  # Allow changing figsize post-init
                self.user_figsize = value
                print(
                    f"Info: figsize changed to {value}. May need to replot or adjust figure."
                )
            else:
                print(
                    f"Warning: Layout key '{key}' not recognized in ViolinPlotter element styles."
                )

        if changed_jitter and hasattr(self, "x_data") and self.x_data:
            self.x_jittered = [
                (
                    x
                    + st.t(
                        df=self.element_styles["jitter_df"],
                        scale=self.element_styles["jitter_scale"],
                    ).rvs(len(x))
                    if len(x) > 0
                    else np.array([])
                )
                for x in self.x_data
            ]

    def set_statistical_annotations(
        self,
        p_values=None,
        p_value_y_pos=None,
        title_stats=None,
        footer_stats1=None,
        footer_stats2=None,
    ):
        # p_value_tick_len is now part of element_styles
        self.p_value_labels, self.p_value_y_positions = p_values, p_value_y_pos
        self.title_stats, self.footer_stats1, self.footer_stats2 = (
            title_stats,
            footer_stats1,
            footer_stats2,
        )

    def plot(
        self,
        title: str = "Distribution Comparison",
        y_label: str = "Value",
        x_label: str = "Category",
        violin_bw_method: Union[str, float, callable] = "silverman",
        show_means: bool = True,
        show_p_values: bool = True,
        show_stats_subtitle: bool = True,
        show_stats_footer: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:

        # Process common params like title, labels, figsize for _apply_common_layout
        common_plot_params = self._process_common_params(
            title=title, xlabel=x_label, ylabel=y_label, figsize=self.user_figsize
        )
        fig, ax = (
            self._create_figure()
        )  # figsize is handled by Plotter via self.user_figsize
        self._apply_common_layout(
            fig, common_plot_params
        )  # Applies suptitle and figsize

        if not self.y_data or not self.positions:
            ax.text(
                0.5,
                0.5,
                "No data to display.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=plt.rcParams["font.size"],
                color=self.colors["text_color"],
            )
            if self.watermark:
                self.add_watermark(fig, text=self.watermark)
            plt.tight_layout()
            if save_path:
                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=self.colors["bg_color"],
                )
            return fig, ax

        for h_val in self.element_styles["hlines_values"]:
            ax.axhline(
                h_val,
                color=self.colors["detail_medium_color"],
                ls=(0, (5, 5)),
                alpha=self.element_styles["hlines_alpha"],
                zorder=0,
            )

        valid_indices = [
            i for i, data_arr in enumerate(self.y_data) if len(data_arr) > 0
        ]
        if not valid_indices:
            ax.text(
                0.5,
                0.5,
                "All categories are empty.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=plt.rcParams["font.size"],
                color=self.colors["text_color"],
            )
            # ... (rest of empty plot handling)
            if self.watermark:
                self.add_watermark(fig, text=self.watermark)
            plt.tight_layout()
            if save_path:
                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=self.colors["bg_color"],
                )
            return fig, ax

        y_data_to_plot = [self.y_data[i] for i in valid_indices]
        positions_to_plot = [self.positions[i] for i in valid_indices]
        species_to_plot = [self.species[i] for i in valid_indices]
        x_jittered_to_plot = [self.x_jittered[i] for i in valid_indices]
        means_to_plot = [self.means[i] for i in valid_indices]

        violins = ax.violinplot(
            y_data_to_plot,
            positions=positions_to_plot,
            widths=self.element_styles["violin_width"],
            bw_method=violin_bw_method,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in violins["bodies"]:
            pc.set_facecolor("none")
            pc.set_edgecolor(self.colors["text_color"])
            pc.set_linewidth(self.element_styles["violin_linewidth"])
            pc.set_alpha(1)

        medianprops = dict(
            linewidth=self.element_styles["median_linewidth"],
            color=self.colors["detail_medium_color"],
            solid_capstyle="butt",
        )
        boxprops = dict(
            linewidth=self.element_styles["box_linewidth"],
            color=self.colors["detail_medium_color"],
        )
        ax.boxplot(
            y_data_to_plot,
            positions=positions_to_plot,
            showfliers=False,
            showcaps=False,
            medianprops=medianprops,
            whiskerprops=boxprops,
            boxprops=boxprops,
        )

        num_cat_colors = len(self.colors["category_color_scale"])
        for i_plot, original_idx in enumerate(valid_indices):
            cat_color = self.colors["category_color_scale"][
                original_idx % num_cat_colors
            ]
            ax.scatter(
                x_jittered_to_plot[i_plot],
                y_data_to_plot[i_plot],
                s=self.element_styles["scatter_dot_size"],
                color=cat_color,
                alpha=self.element_styles["scatter_alpha"],
                zorder=2,
            )

        if show_means:
            for i_plot, mean_val in enumerate(means_to_plot):
                if np.isnan(mean_val):
                    continue
                current_pos = positions_to_plot[i_plot]
                ax.scatter(
                    current_pos,
                    mean_val,
                    s=self.element_styles["mean_dot_size"],
                    color=self.colors["accent_color"],
                    zorder=3,
                )
                ax.plot(
                    [
                        current_pos,
                        current_pos + self.element_styles["mean_line_offset"],
                    ],
                    [mean_val, mean_val],
                    ls="dashdot",
                    color=self.colors["text_color"],
                    zorder=3,
                )
                ax.text(
                    current_pos + self.element_styles["mean_line_offset"],
                    mean_val,
                    f"$\\hat{{\\mu}} = {mean_val:.2f}$",
                    fontsize=self.annotation_fontsize,
                    va="center",
                    ha="left",
                    bbox=dict(
                        facecolor=self.colors["bg_color"],
                        edgecolor=self.colors["text_color"],
                        boxstyle="round",
                        pad=self.element_styles["mean_label_pad"],
                    ),
                    zorder=10,
                    color=self.colors["text_color"],
                )

        if show_p_values and self.p_value_labels and self.p_value_y_positions:
            pad = 0.2
            tick_len = self.element_styles["p_value_tick_len"]
            for (idx1, idx2), y_pos_val in self.p_value_y_positions.items():
                if idx1 in valid_indices and idx2 in valid_indices:
                    try:
                        plot_pos1 = positions_to_plot[valid_indices.index(idx1)]
                        plot_pos2 = positions_to_plot[valid_indices.index(idx2)]
                    except ValueError:
                        continue
                    if (idx1, idx2) in self.p_value_labels:
                        label_str = self.p_value_labels[(idx1, idx2)]
                        x_mid_val = (plot_pos1 + plot_pos2) / 2.0
                        ax.plot(
                            [plot_pos1, plot_pos1, plot_pos2, plot_pos2],
                            [
                                y_pos_val - tick_len,
                                y_pos_val,
                                y_pos_val,
                                y_pos_val - tick_len,
                            ],
                            c=self.colors["text_color"],
                        )
                        ax.text(
                            x_mid_val,
                            y_pos_val + pad,
                            label_str,
                            fontsize=self.p_value_fontsize,
                            va="bottom",
                            ha="center",
                            color=self.colors["text_color"],
                        )

        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color(self.colors["detail_light_color"])
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_color(self.colors["detail_light_color"])
        ax.spines["bottom"].set_linewidth(2)

        ax.tick_params(
            length=0, labelsize=plt.rcParams["xtick.labelsize"]
        )  # Use rcParams for tick label size
        ax.set_yticks(self.element_styles["hlines_values"])
        ax.set_yticklabels(
            [str(h) for h in self.element_styles["hlines_values"]]
        )  # Size already set by tick_params
        self._set_labels(
            ax, ylabel=common_plot_params["ylabel"], xlabel=common_plot_params["xlabel"]
        )  # Let Plotter handle axis labels

        xlabels_to_plot = [
            f"{species_to_plot[i_plot]}\n(n={len(self.y_data[original_idx])})"
            for i_plot, original_idx in enumerate(valid_indices)
        ]
        ax.set_xticks(positions_to_plot)
        ax.set_xticklabels(
            xlabels_to_plot, ha="center"
        )  # Size from tick_params, color from rcParams

        # Subtitle (using ax.set_title, font size from rcParams 'axes.titlesize')
        if show_stats_subtitle and self.title_stats:
            ax.set_title(
                ", ".join(self.title_stats),
                loc="left",
                fontsize=self.subtitle_fontsize_explicit,
                color=self.colors["text_color"],
                wrap=True,
            )

        # Main title (fig.suptitle) is handled by _apply_common_layout using figure.titlesize from rcParams
        # If specific fontname for suptitle is desired, it can be set via self.title_fontname
        # and used here, overriding _apply_common_layout's suptitle or by modifying _apply_common_layout
        # For now, _apply_common_layout sets the suptitle. We can override it here if specific fontname is set:
        if (
            self.title_fontname
        ):  # If user explicitly set a fontname for suptitle via set_fonts
            fig.suptitle(
                common_plot_params["title"],
                x=0.122,
                y=0.975,
                ha="left",
                fontsize=plt.rcParams["figure.titlesize"],
                fontname=self.title_fontname,  # Use the specific fontname
                color=self.colors[
                    "theme_color"
                ],  # Use a prominent theme color for title
                weight=plt.rcParams["figure.titleweight"],
            )

        if show_stats_footer:
            if self.footer_stats1:
                fig.text(
                    0.55,
                    0.03,
                    ", ".join(self.footer_stats1),
                    fontsize=self.stats_fontsize,
                    ha="left",
                    wrap=True,
                    color=self.colors["text_color"],
                )
            if self.footer_stats2:
                fig.text(
                    0.55,
                    0.005,
                    self.footer_stats2,
                    fontsize=self.stats_fontsize,
                    ha="left",
                    wrap=True,
                    color=self.colors["text_color"],
                )

        if self.watermark:
            self.add_watermark(fig, text=self.watermark)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Adjusted rect for suptitle
        if save_path:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=self.colors["bg_color"],
            )
            print(f"Plot saved to {save_path}")
        return fig, ax


# --- Example Usage (Same as before, should work with new structure) ---
if __name__ == "__main__":
    np.random.seed(42)

    print(
        "--- Example 1: Data as List[np.ndarray] (Dark Theme, Presentation Preset) ---"
    )
    data_group_A = np.random.normal(loc=50, scale=10, size=100)
    data_group_B = np.random.normal(loc=65, scale=12, size=120)
    data_group_C = np.random.normal(loc=55, scale=8, size=80)
    data_group_D = np.random.normal(loc=70, scale=5, size=5)
    data_group_E = np.array([])
    generic_data_list = [
        data_group_A,
        data_group_B,
        data_group_C,
        data_group_D,
        data_group_E,
    ]
    category_names = ["Group A", "Group B", "Group C", "Group D", "Group E"]

    plotter_list = ViolinPlotter(
        data=generic_data_list,
        categories=category_names,
        theme="dark",
        style_preset="presentation",  # Presentation preset
        watermark="Analysis Co.",
        figsize=(16, 10),
    )

    # Customize violin-specific layout elements
    plotter_list.set_layout(hlines_values=[40, 60, 80, 100], violin_width=0.6)
    # Override the category color scale provided by the dark theme
    plotter_list.set_colors(
        category_color_scale=["#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#073B4C"]
    )
    # Customize specific annotation font sizes if needed (Plotter's theme provides defaults)
    plotter_list.set_fonts(p_value_fontsize=13, annotation_fontsize=14)

    p_vals = {
        (0, 1): r"$p < 0.001$",
        (0, 2): r"$p = 0.045$",
        (1, 2): r"$p < 0.01$",
        (2, 3): r"$p=0.12$",
    }
    max_val = (
        np.max(np.concatenate([arr for arr in generic_data_list if len(arr) > 0]))
        if any(len(arr) > 0 for arr in generic_data_list)
        else 100
    )
    p_y_pos = {
        (0, 1): max_val + 7,
        (0, 2): max_val + 14,
        (1, 2): max_val + 21,
        (2, 3): max_val + 5,
    }
    title_s = [r"ANOVA: $F(3,291)=75.3, p<0.001, \eta^2_p=0.43$"]  # Subtitle
    footer_s1 = ["Post-hoc: Tukey HSD", "Data: Simulated observations"]

    plotter_list.set_statistical_annotations(
        p_values=p_vals,
        p_value_y_pos=p_y_pos,
        title_stats=title_s,
        footer_stats1=footer_s1,
    )

    fig1, ax1 = plotter_list.plot(
        title="Distribution Comparison by Group (Dark/Presentation)",  # Main title
        y_label="Measured Value (units)",
        x_label="Experimental Groups",
    )
    plt.show()

    print("\n--- Example 2: Data as Pandas DataFrame (Light Theme, Minimal Style) ---")
    all_values_df = np.concatenate([arr for arr in generic_data_list if len(arr) > 0])
    all_categories_df = []
    for i, name in enumerate(category_names):
        if len(generic_data_list[i]) > 0:
            all_categories_df.extend([name] * len(generic_data_list[i]))
    df = pd.DataFrame({"Measurement": all_values_df, "Category": all_categories_df})

    plotter_df = ViolinPlotter(
        data=df,
        value_col="Measurement",
        category_col="Category",
        theme="light",
        style_preset="minimal",
        figsize=(12, 7),
    )
    plotter_df.set_statistical_annotations(
        title_stats=[f"Overall Mean: {df['Measurement'].mean():.2f}"]
    )
    fig2, ax2 = plotter_df.plot(
        title="Widget Size Distribution (Light/Minimal)",
        y_label="Widget Size (cm)",
        x_label="Batch ID",
        show_p_values=False,
    )
    plt.show()

    print("\nAll generic examples finished.")
