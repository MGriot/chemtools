import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as st
from itertools import combinations
from .base import BasePlotter

class ViolinPlot(BasePlotter):
    """
    Creates a Violin Plot, with optional enhancements like jittered points (raincloud)
    and statistical annotations. Inherits from BasePlotter.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ViolinPlot class.

        Args:
            **kwargs: Keyword arguments for the BasePlotter.
        """
        super().__init__(**kwargs)

    def plot(self, 
             data: pd.DataFrame, 
             y: str, 
             x: str = None, 
             show_jitter: bool = False,
             show_mean: bool = False,
             show_n: bool = False,
             h_lines: list = None,
             stat_annotations: list = None,
             perform_stat_test: bool = False,
             stat_test_alpha: float = 0.05,
             y_max_override: float = None,
             violin_alpha: float = None, # New parameter
             jitter_alpha: float = None, # New parameter
             **kwargs):
        """
        Plots a violin plot with optional enhancements.

        Args:
            data (pd.DataFrame): The dataset to use.
            y (str): The numerical column for the violin plot.
            x (str, optional): The categorical column to group by. Defaults to None.
            show_jitter (bool): If True, adds a "raincloud" of jittered data points.
            show_mean (bool): If True, adds a marker for the mean of each category.
            show_n (bool): If True, shows the sample size (n) for each category on the x-axis labels.
            h_lines (list, optional): A list of y-values to draw horizontal grid lines.
            stat_annotations (list, optional): A list of dictionaries for manually plotting
                                              statistical comparisons. Takes precedence over
                                              `perform_stat_test`. Each dict should have:
                                              {'groups': ('Group1', 'Group2'), 'p_value': 'p < 0.01', 'y_pos': 65}.
            perform_stat_test (bool): If True, automatically performs pairwise t-tests and
                                      annotates significant results.
            stat_test_alpha (float): The significance level for automatic statistical tests.
            y_max_override (float, optional): Manually sets the upper limit of the y-axis to ensure
                                              all annotations are visible.
            violin_alpha (float, optional): Transparency (alpha) for the violin plot bodies.
                                            Overrides automatic adjustment if provided.
            jitter_alpha (float, optional): Transparency (alpha) for the jittered data points.
                                            Overrides automatic adjustment if provided.
            **kwargs: Additional keyword arguments passed to the plotter.
        
        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if y not in data.columns:
            raise ValueError(f"Value column '{y}' not found in the data.")
        if not pd.api.types.is_numeric_dtype(data[y]):
            raise TypeError(f"Value column '{y}' must be numeric for a violin plot.")
        if x and x not in data.columns:
            raise ValueError(f"Category column '{x}' not found in the data.")

        if self.library == "matplotlib":
            return self._plot_matplotlib(data, y, x, show_jitter, show_mean, show_n, h_lines, 
                                         stat_annotations, perform_stat_test, stat_test_alpha, 
                                         y_max_override, violin_alpha, jitter_alpha, params)
        elif self.library == "plotly":
            return self._plot_plotly(data, y, x, params, **kwargs)
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def _draw_stat_annotations(self, ax, annotations, labels):
        """Helper function to draw statistical annotation lines and text."""
        if not annotations:
            return

        tick_height_fraction = 0.02
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        tick_height = tick_height_fraction * y_range

        for annot in annotations:
            try:
                group1, group2 = annot['groups']
                p_val_text = annot['p_value']
                y_pos = annot['y_pos']
                
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                
                ax.plot([idx1, idx1, idx2, idx2], [y_pos, y_pos + tick_height, y_pos + tick_height, y_pos], c=self.colors['text_color'], zorder=30)
                ax.text((idx1 + idx2) / 2, y_pos + tick_height, p_val_text, ha='center', va='bottom', color=self.colors['text_color'], zorder=30)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not plot stat_annotation {annot}. Reason: {e}")

    def _plot_matplotlib(self, data, y, x, show_jitter, show_mean, show_n, h_lines, 
                         stat_annotations, perform_stat_test, stat_test_alpha, 
                         y_max_override, violin_alpha_param, jitter_alpha_param, params):
        fig, ax = self._create_figure(figsize=params["figsize"])
        
        labels = sorted(data[x].dropna().unique()) if x else [y]
        positions = np.arange(len(labels))
        plot_data = [data[y][data[x] == cat].dropna() for cat in labels] if x else [data[y].dropna()]

        # --- Aesthetic adjustments based on jitter or user input ---
        final_violin_alpha = violin_alpha_param if violin_alpha_param is not None else (0.4 if show_jitter else 0.8)
        final_jitter_alpha = jitter_alpha_param if jitter_alpha_param is not None else (0.5 if show_jitter else 0.3)

        if h_lines:
            for h_line in h_lines:
                ax.axhline(h_line, color=self.colors['grid_color'], linestyle='--', linewidth=1, zorder=1)

        parts = ax.violinplot(plot_data, positions=positions, showmeans=False, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            color = self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])]
            pc.set_facecolor(color)
            pc.set_edgecolor(self.colors['text_color'])
            pc.set_alpha(final_violin_alpha)
            pc.set_zorder(10)
        
        for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if part_name in parts:
                vp = parts[part_name]
                vp.set_edgecolor(self.colors['text_color'])
                vp.set_linewidth(1.5)
                vp.set_zorder(15)

        if show_jitter:
            jitter_width = 0.08
            for i, d in enumerate(plot_data):
                x_jittered = positions[i] + st.t(df=6, scale=jitter_width).rvs(len(d))
                ax.scatter(x_jittered, d, alpha=final_jitter_alpha, s=20, color=self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])], zorder=5)

        if show_mean:
            means = [d.mean() for d in plot_data]
            ax.scatter(positions, means, s=120, marker='o', edgecolors=self.colors['bg_color'], color=self.colors['accent_color'], zorder=20, label="Mean")

            for i, mean_val in enumerate(means):
                x_offset = 0.25
                ha = 'left'
                if len(positions) > 1 and i == len(positions) - 1:
                    x_offset = -x_offset
                    ha = 'right'
                
                ax.plot([positions[i], positions[i] + x_offset], [mean_val, mean_val], ls="dashdot", color=self.colors['text_color'], zorder=25)
                ax.text(
                    positions[i] + x_offset, mean_val, f" $\hat{{\mu}} = {mean_val:.2f}$",
                    fontsize=self.colors['annotation_fontsize_small'], va="center", ha=ha,
                    bbox=dict(facecolor=self.colors['bg_color'], edgecolor=self.colors['text_color'], boxstyle="round,pad=0.15", alpha=0.8),
                    zorder=25
                )

        annotations_to_draw = []
        if stat_annotations:
            annotations_to_draw = stat_annotations
        elif perform_stat_test and x and len(labels) > 1:
            y_max = data[y].max()
            y_range = y_max - data[y].min()
            y_step = y_range * 0.15 if y_range > 0 else 1
            y_pos = y_max + y_step
            
            sorted_combinations = sorted(list(combinations(labels, 2)), key=lambda c: labels.index(c[1]) - labels.index(c[0]))

            for group1, group2 in sorted_combinations:
                data1 = data[y][data[x] == group1].dropna()
                data2 = data[y][data[x] == group2].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    _, p_value = st.ttest_ind(data1, data2, equal_var=False)
                    if p_value < stat_test_alpha:
                        p_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
                        annotations_to_draw.append({'groups': (group1, group2), 'p_value': p_text, 'y_pos': y_pos})
                        y_pos += y_step
        
        if annotations_to_draw:
            if y_max_override:
                ax.set_ylim(top=y_max_override)
            else:
                max_annot_y = max((annot.get('y_pos', -np.inf) for annot in annotations_to_draw), default=-np.inf)
                current_ylim_top = ax.get_ylim()[1]
                if max_annot_y > current_ylim_top:
                    padding = 0.30 * (max_annot_y - current_ylim_top)
                    ax.set_ylim(top=max_annot_y + padding)
            self._draw_stat_annotations(ax, annotations_to_draw, labels)

        x_labels = labels
        if show_n:
            x_labels = [f"{label}\n(n={len(d)})" for label, d in zip(labels, plot_data)]
        
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels)
        self._set_labels(ax, subplot_title=params.get("subplot_title"), xlabel=x, ylabel=y)
        self._apply_common_layout(fig, params)
        return fig

    def _plot_plotly(self, data, y, x, params, **kwargs):
        kwargs.pop('title', None)
        unique_categories = data[x].dropna().unique()
        color_map = {
            cat: self.colors['category_color_scale'][i % len(self.colors['category_color_scale'])]
            for i, cat in enumerate(unique_categories)
        }
        fig = px.violin(data, x=x, y=y, title=params.get('title'), color=x, 
                        color_discrete_map=color_map, **kwargs)
        self._apply_common_layout(fig, params)
        return fig
