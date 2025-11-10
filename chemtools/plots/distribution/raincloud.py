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
        # It's better to sort to ensure a consistent plotting order
        categories = sorted(categories)

        if self.library == "matplotlib":
            fig, ax = self._create_figure(
                figsize=params.get("figsize", (10, 2 + len(categories) * 1.5))
            )

            # ## UPDATED: More descriptive parameter names
            jitter_amount = kwargs.get("jitter_amount", 0.1)
            point_size = kwargs.get("point_size", 20)
            plot_offset = kwargs.get("plot_offset", 0.2)  # Controls separation

            n_categories = len(categories)
            colors = self.colors["category_color_scale"][:n_categories]
            color_map = dict(zip(categories, colors))

            for i, cat in enumerate(categories):
                subset = data[data[categorical_var] == cat][numerical_var].dropna()
                if subset.empty:
                    continue

                # ## UPDATED: Centralize position and use offset for clarity
                center = i
                rain_position = center - plot_offset
                cloud_position = center + plot_offset

                # 1. Half-Violin Plot (Cloud)
                kde = gaussian_kde(
                    subset, bw_method="scott"
                )  # Using Scott's rule for bandwidth

                if orientation == "vertical":
                    # KDE plot (the "cloud")
                    kde_x = np.linspace(subset.min(), subset.max(), 100)
                    kde_y = kde(kde_x)
                    scaled_kde = kde_y / kde_y.max() * 0.4  # Scale width of the cloud
                    ax.fill_betweenx(
                        kde_x,
                        cloud_position,
                        cloud_position + scaled_kde,
                        facecolor=color_map[cat],
                        alpha=0.6,
                        label=f"KDE {cat}",
                    )

                    # Jittered Scatter Plot (the "rain")
                    jittered_x = np.random.uniform(
                        rain_position - jitter_amount,
                        rain_position + jitter_amount,
                        size=len(subset),
                    )
                    ax.scatter(
                        jittered_x,
                        subset,
                        color=color_map[cat],
                        alpha=0.4,
                        s=point_size,
                        label=f"Data {cat}",
                    )

                    # Box Plot (aligned with the rain)
                    ax.boxplot(
                        subset,
                        positions=[rain_position],
                        widths=[0.15],
                        showfliers=False,
                        patch_artist=True,
                        vert=True,
                        boxprops=dict(facecolor="white", edgecolor="black", alpha=0.8),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"),
                    )
                else:  # horizontal
                    # KDE plot (the "cloud")
                    kde_y = np.linspace(subset.min(), subset.max(), 100)
                    kde_x = kde(kde_y)
                    scaled_kde = kde_x / kde_x.max() * 0.4  # Scale height of the cloud
                    ax.fill_between(
                        kde_y,
                        cloud_position,
                        cloud_position + scaled_kde,
                        facecolor=color_map[cat],
                        alpha=0.6,
                        label=f"KDE {cat}",
                    )

                    # Jittered Scatter Plot (the "rain")
                    jittered_y = np.random.uniform(
                        rain_position - jitter_amount,
                        rain_position + jitter_amount,
                        size=len(subset),
                    )
                    ax.scatter(
                        subset,
                        jittered_y,
                        color=color_map[cat],
                        alpha=0.4,
                        s=point_size,
                        label=f"Data {cat}",
                    )

                    # Box Plot (aligned with the rain)
                    ax.boxplot(
                        subset,
                        positions=[rain_position],
                        widths=[0.15],
                        showfliers=False,
                        patch_artist=True,
                        vert=False,
                        boxprops=dict(facecolor="white", edgecolor="black", alpha=0.8),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"),
                    )

            # Configure axes ticks and labels
            if orientation == "vertical":
                ax.set_xticks(np.arange(n_categories))
                ax.set_xticklabels(categories)
                self._set_labels(
                    ax,
                    subplot_title=params.get(
                        "subplot_title", f"{numerical_var} by {categorical_var}"
                    ),
                    xlabel=categorical_var,
                    ylabel=numerical_var,
                )
            else:  # horizontal
                ax.set_yticks(np.arange(n_categories))
                ax.set_yticklabels(categories)
                self._set_labels(
                    ax,
                    subplot_title=params.get(
                        "subplot_title", f"{numerical_var} by {categorical_var}"
                    ),
                    xlabel=numerical_var,
                    ylabel=categorical_var,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            self._apply_common_layout(fig, params)
            return fig

        elif self.library == "plotly":
            # Plotly logic remains untouched as it was correct
            if orientation == "vertical":
                fig = px.violin(
                    data,
                    x=categorical_var,
                    y=numerical_var,
                    color=categorical_var,
                    points="all",
                    box=True,
                    title=params.get(
                        "title", f"Raincloud Plot: {numerical_var} by {categorical_var}"
                    ),
                    color_discrete_sequence=self.colors["category_color_scale"],
                    **kwargs,
                )
            else:  # horizontal
                fig = px.violin(
                    data,
                    x=numerical_var,
                    y=categorical_var,
                    color=categorical_var,
                    points="all",
                    box=True,
                    title=params.get(
                        "title", f"Raincloud Plot: {numerical_var} by {categorical_var}"
                    ),
                    orientation="h",
                    color_discrete_sequence=self.colors["category_color_scale"],
                    **kwargs,
                )

            # This part is a nice-to-have to better emulate the half-violin plot
            fig.update_traces(
                side="positive", width=0.7
            )  # Show only one side of the violin
            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")
