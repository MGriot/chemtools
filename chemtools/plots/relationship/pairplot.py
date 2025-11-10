import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ..base import BasePlotter
from scipy.stats import pearsonr
import numpy as np

class PairPlot(BasePlotter):
    """
    A plotter for creating advanced pair plots.

    This class generates a pair plot matrix. For the 'matplotlib' library,
    it replicates a complex style with scatter plots in the lower triangle,
    kernel density estimates on the diagonal, and correlation coefficients
    in the upper triangle.
    """

    def _plot_correlations(
        self,
        x_var: str,
        y_var: str,
        data: pd.DataFrame,
        hue: str,
        ax: plt.Axes,
        palette: list, # Added palette argument
        **kwargs,
    ):
        """
        Helper method to calculate and plot correlation coefficients on an Axes object.

        Args:
            x_var (str): The name of the variable for the x-axis.
            y_var (str): The name of the variable for the y-axis.
            data (pd.DataFrame): The dataset containing the data.
            hue (str): The name of the grouping variable.
            ax (plt.Axes): The matplotlib axes to plot on.
            palette (list): The color palette to use for hue groups.
            **kwargs: Additional keyword arguments.
        """
        # Select non-null data for correlation calculation
        subset_data = data[[x_var, y_var, hue]].dropna() if hue else data[[x_var, y_var]].dropna()
        x = subset_data[x_var]
        y = subset_data[y_var]

        # Calculate overall correlation and p-value
        r, p_value = pearsonr(x, y)

        # Determine significance level stars
        stars = ""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"

        # Clear the axes and turn it off for a clean text display
        ax.cla()
        ax.set_axis_off()

        # Display overall correlation
        ax.text(
            0.5,
            0.8,
            f"Corr: {r:.3f}{stars}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            weight="bold",
        )

        # Display group-specific correlations
        if hue:
            # Use the passed palette directly
            hue_groups = subset_data[hue].unique()
            colors = sns.color_palette(palette, len(hue_groups)) # Use the passed palette
            color_map = dict(zip(hue_groups, colors))

            y_pos = 0.6
            for i, group in enumerate(hue_groups):
                group_data = subset_data[subset_data[hue] == group]
                group_r, _ = pearsonr(group_data[x_var], group_data[y_var])
                ax.text(
                    0.5,
                    y_pos,
                    f"{group}: {group_r:.3f}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color=color_map.get(group),
                )
                y_pos -= 0.2

    def plot(self, data: pd.DataFrame, **kwargs):
        """
        Plots an advanced pair plot of the given DataFrame.

        For the 'matplotlib' library, this method creates a grid with:
        - Lower triangle: Scatter plots.
        - Diagonal: Kernel Density Estimation plots.
        - Upper triangle: Pearson correlation coefficients (overall and per hue-group).

        Args:
            data (pd.DataFrame): The dataset to use.
            **kwargs: Additional keyword arguments passed to the plotter.
                      Key arguments for matplotlib: 'hue', 'palette', 'title'.

        Returns:
            A matplotlib or plotly figure object.
        """
        params = self._process_common_params(**kwargs)
        if self.library == "matplotlib":
            title = params.get("title", "Pair Plot")
            hue = kwargs.pop("hue", None)
            
            # Extract palette from kwargs or use theme's category_color_scale
            plot_palette = kwargs.pop("palette", self.colors["category_color_scale"])
            kwargs.pop('title', None) # Remove title from kwargs to prevent error

            # Slice the palette if hue is used to avoid seaborn warning
            if hue:
                n_colors = data[hue].nunique()
                plot_palette = plot_palette[:n_colors]

            # Generate the base pairplot. Seaborn automatically handles mixed
            # categorical/numeric data types for the plots outside the main numeric matrix.
            g = sns.pairplot(data, diag_kind="kde", hue=hue, palette=plot_palette, **kwargs)

            # Systematically update the upper triangle plots
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                ax = g.axes[i, j]
                x_var = g.x_vars[j]
                y_var = g.y_vars[i]

                # Ensure we only calculate correlation for numeric columns
                if (
                    data[x_var].dtype.kind in "ifcm"
                    and data[y_var].dtype.kind in "ifcm"
                ):
                    self._plot_correlations(
                        x_var=x_var, y_var=y_var, data=data, hue=hue, ax=ax, palette=plot_palette, **kwargs
                    )

            g.fig.suptitle(title, y=1.02)
            self._apply_common_layout(g.fig, params)
            return g.fig

        elif self.library == "plotly":
            title = params.get("title", "Pair Plot")
            # Extract 'color' from kwargs to handle it explicitly for Plotly
            color_col = kwargs.pop('color', None)
            
            if color_col:
                fig = px.scatter_matrix(data, title=title, color=color_col, 
                                        color_discrete_sequence=self.colors['category_color_scale'], **kwargs)
            else:
                fig = px.scatter_matrix(data, title=title, **kwargs)

            self._apply_common_layout(fig, params)
            return fig
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")