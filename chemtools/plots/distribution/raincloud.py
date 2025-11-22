import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from ..base import BasePlotter


class RaincloudPlot(BasePlotter):
    """
    A plotter for creating Raincloud plots, combining violin plots, jittered scatter plots,
    and box plots to visualize data distributions.
    """

    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        orientation: str = "vertical",
        violin_filled: bool = True,
        **kwargs,
    ):
        """
        Generates a Raincloud plot.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column name for the categorical variable.
            y (str): The column name for the numerical variable.
            orientation (str): 'vertical' (default) or 'horizontal'.
                                If 'vertical', x is categorical, y is numerical.
                                If 'horizontal', x is numerical, y is categorical.
            violin_filled (bool): If True, the violin plot is filled. Otherwise, only the contour is drawn.
            **kwargs: Additional keyword arguments passed to the plotter.
                      Includes 'jitter_amount', 'box_width', 'violin_width', 'plot_offset', 'show_legend'.
        """
        params = self._process_common_params(**kwargs)

        if orientation == "vertical":
            categorical_var = x
            numerical_var = y
        elif orientation == "horizontal":
            categorical_var = y
            numerical_var = x
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'.")

        if categorical_var not in data.columns or numerical_var not in data.columns:
            raise ValueError(
                f"Columns '{categorical_var}' and '{numerical_var}' must be in the data."
            )
        if not pd.api.types.is_numeric_dtype(data[numerical_var]):
            raise TypeError(f"Column '{numerical_var}' must be numerical.")
        if not (
            pd.api.types.is_string_dtype(data[categorical_var])
            or pd.api.types.is_categorical_dtype(data[categorical_var])
        ):
            raise TypeError(f"Column '{categorical_var}' must be categorical.")

        categories = data[categorical_var].dropna().unique()
        categories = sorted(categories)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(
                figsize=params.get("figsize", (10, 2 + len(categories) * 1.5))
            )

            jitter_amount = kwargs.get("jitter_amount", 0.04)
            point_size = kwargs.get("point_size", 20)
            plot_offset = kwargs.get("plot_offset", 0.25)

            n_categories = len(categories)
            colors = self.colors["category_color_scale"][:n_categories]
            color_map = dict(zip(categories, colors))

            for i, cat in enumerate(categories):
                subset = data[data[categorical_var] == cat][numerical_var].dropna()
                if subset.empty:
                    continue

                center = i
                cloud_position = center + plot_offset
                data_viz_position = center - plot_offset
                
                kde = gaussian_kde(subset, bw_method="scott")
                
                data_range = subset.max() - subset.min() if len(subset) > 1 else 1
                kde_min = subset.min() - data_range * 0.1
                kde_max = subset.max() + data_range * 0.1
                
                if orientation == "vertical":
                    kde_vals = np.linspace(kde_min, kde_max, 200)
                    kde_dist = kde(kde_vals)
                    scaled_kde = kde_dist / kde_dist.max() * 0.4

                    ax.plot(cloud_position + scaled_kde, kde_vals, color=color_map[cat], zorder=10)

                    if violin_filled:
                        ax.fill_betweenx(
                            kde_vals, cloud_position, cloud_position + scaled_kde,
                            facecolor=color_map[cat], alpha=0.3, zorder=9
                        )

                    jittered_x = np.random.uniform(
                        data_viz_position - jitter_amount,
                        data_viz_position + jitter_amount,
                        size=len(subset),
                    )
                    ax.scatter(jittered_x, subset, color=color_map[cat], alpha=0.5, s=point_size, zorder=5)

                    bp = ax.boxplot(
                        subset, positions=[data_viz_position], widths=[0.15],
                        showfliers=False, patch_artist=True, vert=True,
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor('none')
                        patch.set_edgecolor(self.colors['text_color'])
                        patch.set_zorder(20)
                    for element in ['whiskers', 'caps', 'medians']:
                        for line in bp[element]:
                            line.set_color(self.colors['text_color'])
                            line.set_zorder(20)

                else:  # horizontal
                    kde_vals = np.linspace(kde_min, kde_max, 200)
                    kde_dist = kde(kde_vals)
                    scaled_kde = kde_dist / kde_dist.max() * 0.4

                    ax.plot(kde_vals, cloud_position + scaled_kde, color=color_map[cat], zorder=10)

                    if violin_filled:
                        ax.fill_between(
                            kde_vals, cloud_position, cloud_position + scaled_kde,
                            facecolor=color_map[cat], alpha=0.3, zorder=9
                        )

                    jittered_y = np.random.uniform(
                        data_viz_position - jitter_amount,
                        data_viz_position + jitter_amount,
                        size=len(subset),
                    )
                    ax.scatter(subset, jittered_y, color=color_map[cat], alpha=0.5, s=point_size, zorder=5)

                    bp = ax.boxplot(
                        subset, positions=[data_viz_position], widths=[0.15],
                        showfliers=False, patch_artist=True, vert=False,
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor('none')
                        patch.set_edgecolor(self.colors['text_color'])
                        patch.set_zorder(20)
                    for element in ['whiskers', 'caps', 'medians']:
                        for line in bp[element]:
                            line.set_color(self.colors['text_color'])
                            line.set_zorder(20)

            if orientation == "vertical":
                ax.set_xticks(np.arange(n_categories))
                ax.set_xticklabels(categories)
                self._set_labels(
                    ax,
                    subplot_title=params.get("subplot_title", f"{numerical_var} by {categorical_var}"),
                    xlabel=categorical_var,
                    ylabel=numerical_var,
                )
            else:
                ax.set_yticks(np.arange(n_categories))
                ax.set_yticklabels(categories)
                self._set_labels(
                    ax,
                    subplot_title=params.get("subplot_title", f"{numerical_var} by {categorical_var}"),
                    xlabel=numerical_var,
                    ylabel=categorical_var,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            if orientation == "vertical":
                fig = px.violin(
                    data, x=categorical_var, y=numerical_var, color=categorical_var,
                    points="all", box=True, title=params.get("title", f"Raincloud Plot: {numerical_var} by {categorical_var}"),
                    color_discrete_sequence=self.colors["category_color_scale"], **kwargs,
                )
            else:  # horizontal
                fig = px.violin(
                    data, x=numerical_var, y=categorical_var, color=categorical_var,
                    points="all", box=True, title=params.get("title", f"Raincloud Plot: {numerical_var} by {categorical_var}"),
                    orientation="h", color_discrete_sequence=self.colors["category_color_scale"], **kwargs,
                )

            fig.update_traces(side="positive", width=0.7, points='all', jitter=1, pointpos=-1.2)
            if not violin_filled:
                fig.update_traces(fillcolor='rgba(0,0,0,0)')
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")
