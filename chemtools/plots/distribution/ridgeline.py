import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ..base import BasePlotter
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class RidgelinePlot(BasePlotter):
    """
    A plotter for creating ridgeline charts (joyplots) using Matplotlib.
    This plot type is excellent for visualizing the distribution of a numeric
    variable for several groups.
    """

    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        overlap: float = 0.5,
        **kwargs,
    ):
        """
        Generates a ridgeline plot using a manual matplotlib approach.

        Each row in the plot corresponds to a category in the 'y' column,
        showing the distribution of the 'x' variable.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column name for the numerical variable.
            y (str): The column name for the categorical variable that defines the rows.
            overlap (float): The amount of vertical overlap between plots. A value of 0
                             means no overlap, 1 means plots are fully on top of each other.
            **kwargs: Additional keyword arguments passed to the plotter.
        """
        if self.library != 'matplotlib':
            raise NotImplementedError("Ridgeline plot is only implemented for matplotlib.")

        params = self._process_common_params(**kwargs)
        
        if y not in data.columns:
            raise ValueError(f"Categorical column '{y}' not found in data.")
        if x not in data.columns:
            raise ValueError(f"Numerical column '{x}' not found in data.")

        cats = sorted(data[y].dropna().unique(), reverse=True)
        n_cats = len(cats)
        
        # Using a slightly taller figure for better spacing
        fig_height = n_cats * (1 - overlap) * 1.5
        figsize = params.get("figsize", (10, fig_height))
        fig, axes = plt.subplots(n_cats, 1, figsize=figsize, sharex=True)
        if n_cats == 1: # Ensure axes is always a list-like object
            axes = [axes]

        # Use theme colors
        theme_colors = self.colors['category_color_scale']

        for i, (ax, cat) in enumerate(zip(axes, cats)):
            subset = data[data[y] == cat][x].dropna()
            if subset.empty:
                continue
            
            try:
                kde = gaussian_kde(subset)
                kde_x = np.linspace(subset.min(), subset.max(), 200)
                kde_y = kde(kde_x)
            except (np.linalg.LinAlgError, ValueError): # Handle cases with too few points
                ax.hist(subset, bins=10)
                continue

            line_color = self.colors['text_color']
            fill_color = theme_colors[i % len(theme_colors)]

            # Plot the density
            ax.plot(kde_x, kde_y, color=line_color, lw=1.5, zorder=11)
            ax.fill_between(kde_x, kde_y, color=fill_color, alpha=0.7, zorder=10)

            # Style the axes
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 10)) # Keep bottom spine
            ax.set_yticks([]) # Remove y-ticks

            # Add category label
            ax.text(0.0, 0.1, cat, transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='bottom')

        # Adjust layout for overlap
        fig.subplots_adjust(hspace=-overlap)

        self._set_labels(axes[-1], xlabel=params.get('xlabel', x))
        if params.get("title"):
             fig.suptitle(params["title"], y=1.0)
        self._apply_common_layout(fig, params)
        
        return fig

    def plot_annotated(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        bandwidth: float = 1.0,
        show_mean_line: bool = True,
        show_quantiles: bool = True,
        annotations: dict = None,
        show_legend: bool = True,
        **kwargs
    ):
        """
        Creates a highly annotated and detailed ridgeline plot using Matplotlib.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column for the numerical variable.
            y (str): The column for the categorical variable.
            bandwidth (float): Controls the smoothness of the KDE plot.
            show_mean_line (bool): If True, shows a global mean line and individual mean points.
            show_quantiles (bool): If True, shades quantile regions on each distribution.
            annotations (dict): A dictionary for custom text annotations on the figure.
            show_legend (bool): If True, displays the detailed inset legend.
            **kwargs: Additional keyword arguments.
        """
        if self.library != 'matplotlib':
            raise NotImplementedError("Annotated ridgeline plot is only implemented for matplotlib.")
        
        params = self._process_common_params(**kwargs)
        
        sorted_cats = data.groupby(y)[x].mean().sort_values().index.tolist()
        n_groups = len(sorted_cats)

        fig, axs = plt.subplots(nrows=n_groups, ncols=1, figsize=params.get('figsize', (8, 10)))
        if n_groups == 1:
            axs = [axs]
        axs = axs.flatten()

        quantile_colors = self.colors.get('category_color_scale', ['#E7E5CB', '#C2D6A4', '#9BC184', '#C2D6A4', '#E7E5CB'])
        if len(quantile_colors) < 5:
            quantile_colors = (quantile_colors * 5)[:5]

        for i, cat in enumerate(sorted_cats):
            ax = axs[i]
            subset = data[data[y] == cat]
            x_values = subset[x].dropna()

            if x_values.empty:
                ax.set_axis_off()
                continue
                
            # Manually plot KDE
            try:
                kde = gaussian_kde(x_values, bw_method=bandwidth)
                kde_x_vals = np.linspace(x_values.min(), x_values.max(), 200)
                kde_y_vals = kde(kde_x_vals)
                ax.plot(kde_x_vals, kde_y_vals, color='grey', lw=1.0)
                ax.fill_between(kde_x_vals, kde_y_vals, color='lightgrey', alpha=0.5)
            except (np.linalg.LinAlgError, ValueError):
                ax.hist(x_values, bins=10, density=True, color='lightgrey')


            if show_mean_line:
                global_mean = data[x].mean()
                ax.axvline(global_mean, color=self.colors.get('detail_medium_color', '#525252'), linestyle='--')

            if show_quantiles:
                quantiles = np.percentile(x_values, [2.5, 10, 25, 75, 90, 97.5])
                for j in range(len(quantiles) - 1):
                    ax.fill_between([quantiles[j], quantiles[j+1]], 0, 0.0002, color=quantile_colors[j])

            if show_mean_line:
                mean = x_values.mean()
                ax.scatter([mean], [0.0001], color='black', s=10)

            ax.text(-0.1, 0, cat.upper(), transform=ax.transAxes, ha='right', fontsize=10, fontweight='bold', color=self.colors['text_color'])
            ax.set_xlim(data[x].min(), data[x].max())
            ax.set_ylim(0, 0.001)
            ax.set_axis_off()

        if annotations:
            if 'title' in annotations:
                fig.text(0, 1.01, annotations['title'], ha='left', fontsize=18, fontweight='bold', transform=fig.transFigure)
            if 'description' in annotations:
                fig.text(0, 0.9, annotations['description'], ha='left', fontsize=12, transform=fig.transFigure)
            if 'credit' in annotations:
                fig.text(0, -0.05, annotations['credit'], ha='left', fontsize=8, transform=fig.transFigure)
            if 'xlabel' in annotations:
                 fig.text(0.5, 0.06, annotations['xlabel'], ha='center', fontsize=14, transform=fig.transFigure)

        if show_legend and n_groups > 0:
            subax = inset_axes(parent_axes=axs[0], width="40%", height="350%", loc='upper right')
            self._create_inset_legend(subax, data, x, quantile_colors)

        self._apply_common_layout(fig, params)
        return fig

    def _create_inset_legend(self, subax, data, x, colors):
        """Helper to create the detailed inset legend."""
        subax.set_xticks([])
        subax.set_yticks([])
        subax.set_facecolor(self.colors['bg_color'])

        legend_subset = data[x].dropna().sample(n=min(100, len(data[x].dropna())))
        if legend_subset.empty: return

        try:
            kde = gaussian_kde(legend_subset)
            kde_x = np.linspace(legend_subset.min(), legend_subset.max(), 100)
            kde_y = kde(kde_x)
            subax.plot(kde_x, kde_y, color='grey', lw=0.5)
            subax.fill_between(kde_x, kde_y, color='lightgrey', alpha=0.5)
        except (np.linalg.LinAlgError, ValueError):
             pass

        quantiles = np.percentile(legend_subset, [2.5, 10, 25, 75, 90, 97.5])
        for j in range(len(quantiles) - 1):
            subax.fill_between([quantiles[j], quantiles[j+1]], 0, 0.00004, color=colors[j])

        subax.set_xlim(legend_subset.min() - (legend_subset.max()-legend_subset.min())*0.1, legend_subset.max()*1.1)
        subax.set_ylim(-0.0002, 0.0006)
        
        mean = legend_subset.mean()
        subax.scatter([mean], [0.00002], color='black', s=10)

        subax.text(0, 1.05, 'Legend', transform=subax.transAxes, ha='left', fontsize=12, fontweight='bold')
        subax.text(1, 0.6, 'Distribution', transform=subax.transAxes, ha='center', fontsize=7)
        subax.text(mean, 0.00015, 'Mean', ha='center', fontsize=7)

        subax.text(np.percentile(legend_subset, 50), -0.00018, "50% of data\nfall in this range", ha='center', fontsize=6)
        subax.text(np.percentile(legend_subset, 85), -0.00015, "80% of data", ha='center', fontsize=6)
        subax.text(np.percentile(legend_subset, 95), -0.00015, "95% of data", ha='center', fontsize=6)
        
        def add_arrow(head_pos, tail_pos, ax):
            style = "Simple, tail_width=0.01, head_width=1, head_length=2"
            kw = dict(arrowstyle=style, color="k", linewidth=0.2)
            arrow = patches.FancyArrowPatch(tail_pos, head_pos, connectionstyle="arc3,rad=.5", **kw)
            ax.add_patch(arrow)

        add_arrow((mean, 0.00005), (mean, 0.00013), subax)
        add_arrow((np.percentile(legend_subset, 85), 0), (np.percentile(legend_subset, 85), -0.00011), subax)
        add_arrow((np.percentile(legend_subset, 95), 0), (np.percentile(legend_subset, 95), -0.00011), subax)
        add_arrow((np.percentile(legend_subset, 50), 0), (np.percentile(legend_subset, 50), -0.0001), subax)
