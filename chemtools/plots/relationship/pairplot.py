import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ..base import BasePlotter
from scipy.stats import pearsonr, gaussian_kde
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
            colors = palette # Use the passed palette directly
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
            hue_col = kwargs.pop("hue", None)
            plot_palette = kwargs.pop("palette", self.colors["category_color_scale"])

            # Filter out non-numeric columns for the pair plot matrix
            numerical_data = data.select_dtypes(include=np.number)
            if numerical_data.empty:
                raise ValueError("No numerical columns found in the data for PairPlot.")

            numerical_cols = numerical_data.columns.tolist()
            n_vars = len(numerical_cols)

            # Create figure and axes grid
            fig, axes = plt.subplots(n_vars, n_vars, figsize=params["figsize"])
            if n_vars == 1: # Handle single variable case
                axes = np.array([[axes]]) # Make it 2D for consistent indexing
            
            # Prepare data for hue if present
            if hue_col:
                if hue_col not in data.columns:
                    raise ValueError(f"Hue column '{hue_col}' not found in the data.")
                hue_groups = data[hue_col].unique()
                n_hue_groups = len(hue_groups)
                # Ensure plot_palette has enough colors
                if len(plot_palette) < n_hue_groups:
                    plot_palette = (plot_palette * (n_hue_groups // len(plot_palette) + 1))[:n_hue_groups]
                color_map = dict(zip(hue_groups, plot_palette))
            else:
                color_map = {None: self.colors['theme_color']} # Default color if no hue

            # Iterate through the grid
            for i in range(n_vars):
                for j in range(n_vars):
                    ax = axes[i, j]
                    x_var = numerical_cols[j]
                    y_var = numerical_cols[i]

                    # Set common properties for all subplots
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.tick_params(axis='y', labelrotation=0)
                    ax.set_facecolor(self.colors['bg_color']) # Apply theme background
                    ax.set_xlabel("") # Clear default labels
                    ax.set_ylabel("") # Clear default labels

                    # Diagonal plots (KDE)
                    if i == j:
                        ax.set_yticklabels([]) # No y-ticks on diagonal
                        ax.set_xticklabels([]) # No x-ticks on diagonal
                        ax.set_xticks([]) # No x-ticks on diagonal
                        ax.set_yticks([]) # No y-ticks on diagonal
                        ax.set_frame_on(False) # Remove frame

                        # Plot KDE
                        if hue_col:
                            for group in hue_groups:
                                subset = data[data[hue_col] == group][x_var].dropna()
                                if not subset.empty:
                                    # Use twinx to avoid scaling issues with scatter plots
                                    ax_kde = ax.twinx()
                                    ax_kde.set_ylabel("")
                                    ax_kde.set_yticklabels([])
                                    ax_kde.set_xticks([])
                                    ax_kde.set_yticks([])
                                    ax_kde.set_frame_on(False)
                                    
                                    # Calculate KDE
                                    kde = gaussian_kde(subset)
                                    x_vals = np.linspace(subset.min(), subset.max(), 100)
                                    ax_kde.plot(x_vals, kde(x_vals), color=color_map[group], label=group)
                                    ax_kde.fill_between(x_vals, kde(x_vals), color=color_map[group], alpha=0.2)
                        else:
                            subset = data[x_var].dropna()
                            if not subset.empty:
                                ax_kde = ax.twinx()
                                ax_kde.set_ylabel("")
                                ax_kde.set_yticklabels([])
                                ax_kde.set_xticks([])
                                ax_kde.set_yticks([])
                                ax_kde.set_frame_on(False)
                                
                                kde = gaussian_kde(subset)
                                x_vals = np.linspace(subset.min(), subset.max(), 100)
                                ax_kde.plot(x_vals, kde(x_vals), color=color_map[None])
                                ax_kde.fill_between(x_vals, kde(x_vals), color=color_map[None], alpha=0.2)
                        
                        # Add variable name to diagonal
                        ax.text(0.5, 0.5, x_var, transform=ax.transAxes,
                                ha='center', va='center', fontsize=12, fontweight='bold',
                                color=self.colors['text_color'])

                    # Lower triangle (Scatter plots)
                    elif i > j:
                        if hue_col:
                            for group in hue_groups:
                                subset = data[data[hue_col] == group]
                                ax.scatter(subset[x_var], subset[y_var],
                                           color=color_map[group], label=group, s=10, alpha=0.7)
                        else:
                            ax.scatter(data[x_var], data[y_var],
                                       color=color_map[None], s=10, alpha=0.7)

                    # Upper triangle (Correlation text)
                    else: # i < j
                        ax.set_axis_off() # Turn off axis for correlation text
                        # Call _plot_correlations to display text
                        self._plot_correlations(
                            x_var=x_var, y_var=y_var, data=data, hue=hue_col, ax=ax, palette=plot_palette, **kwargs
                        )

                    # Set labels for outer plots
                    if i == n_vars - 1: # Bottom row
                        ax.set_xlabel(x_var, color=self.colors['text_color'])
                    else:
                        ax.set_xticklabels([])

                    if j == 0: # Leftmost column
                        ax.set_ylabel(y_var, color=self.colors['text_color'])
                    else:
                        ax.set_yticklabels([])
            
            # Adjust layout to make space for legend
            fig.subplots_adjust(right=0.8)

            # Manual legend creation
            if hue_col:
                handles = []
                labels = []
                # Create dummy handles for the legend
                for group in hue_groups:
                    handles.append(plt.Line2D([0], [0], marker='o', color=color_map[group], linestyle=''))
                    labels.append(group)
                
                fig.legend(handles, labels, title=hue_col, loc='center right', bbox_to_anchor=(1.0, 0.5),
                           facecolor=self.colors['bg_color'], edgecolor=self.colors['detail_light_color'],
                           labelcolor=self.colors['text_color'])
                params["showlegend"] = False # Prevent BasePlotter from adding another legend

            fig.suptitle(title, y=1.02) # Main title
            self._apply_common_layout(fig, params)
            return fig

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