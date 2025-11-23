import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..base import BasePlotter

class BeeswarmPlot(BasePlotter):
    """
    A plotter for creating Beeswarm plots using Matplotlib.

    A beeswarm plot displays individual data points for a numerical variable,
    arranging them to avoid overlap, which gives a visual representation of
    the data's distribution and density.
    """

    def _calculate_offsets(self, values, point_diameter_y, spread_factor):
        """
        Calculates horizontal offsets for a single set of points to create a swarm.
        This algorithm works by binning points along the y-axis and spreading them
        horizontally within each bin.
        
        Args:
            values (np.array): The numerical data points for a single category.
            point_diameter_y (float): The effective diameter of a point in y-axis data coordinates.
            spread_factor (float): Controls the horizontal spread of points.
        
        Returns:
            np.array: The calculated horizontal offsets for each point.
        """
        if not hasattr(values, '__len__') or len(values) == 0:
            return np.array([])

        if point_diameter_y <= 0:
            return np.zeros(len(values))

        # Bin the data along the y-axis
        bins = np.arange(values.min(), values.max() + point_diameter_y, point_diameter_y)
        digitized = np.digitize(values, bins)
        
        offsets = np.zeros(len(values))
        
        # For each bin, calculate the horizontal positions
        for i in np.unique(digitized):
            # Get the indices of points in the current bin
            in_bin_indices = np.where(digitized == i)[0]
            num_points_in_bin = len(in_bin_indices)
            
            if num_points_in_bin == 0:
                continue

            # Arrange points symmetrically around the center
            if num_points_in_bin % 2 == 1:  # Odd number of points
                bin_offsets = np.arange(-(num_points_in_bin // 2), num_points_in_bin // 2 + 1)
            else:  # Even number of points
                bin_offsets = np.arange(-(num_points_in_bin / 2) + 0.5, num_points_in_bin / 2 + 0.5)

            # Apply the spread factor and assign to points
            offsets[in_bin_indices] = bin_offsets * spread_factor
            
        return offsets

    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        orientation: str = "vertical",
        point_size: int = 50,
        spread_factor: float = 0.05,
        **kwargs,
    ):
        """
        Generates a Beeswarm plot.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column name for the categorical variable.
            y (str): The column name for the numerical variable.
            orientation (str): 'vertical' (default). 'horizontal' is not yet implemented.
            point_size (int): Size of the scatter points (area in points^2).
            spread_factor (float): Controls the horizontal spread of the points. Tune for best appearance.
            **kwargs: Additional keyword arguments.
        """
        if self.library != 'matplotlib':
            raise NotImplementedError("Beeswarm plot is only implemented for matplotlib.")
        
        if orientation != 'vertical':
            raise NotImplementedError("Horizontal orientation is not yet implemented for Beeswarm plot.")

        params = self._process_common_params(**kwargs)
        fig, ax = self._create_figure(figsize=params.get("figsize", (10, 6)))

        categorical_var = x
        numerical_var = y
        
        categories = sorted(data[categorical_var].dropna().unique())
        n_categories = len(categories)
        colors = self.colors["category_color_scale"][:n_categories]
        color_map = dict(zip(categories, colors))
        
        # Heuristic to estimate point diameter in y-data coordinates.
        y_total_range = data[numerical_var].max() - data[numerical_var].min()
        if y_total_range == 0: 
            y_total_range = 1
        
        # This heuristic relates point size to the data range.
        # A larger point_size or smaller data range will result in a larger "diameter".
        # This value is critical for avoiding overlap. The denominator is a magic number for tuning.
        point_diameter_y = y_total_range * (np.sqrt(point_size) / 1500.0)

        all_x_pos = []
        all_y_pos = []
        all_colors = []

        for i, cat in enumerate(categories):
            subset = data[data[categorical_var] == cat]
            y_values = subset[numerical_var].dropna().to_numpy()
            
            if y_values.size == 0:
                continue

            # Calculate horizontal offsets for this category
            x_offsets = self._calculate_offsets(y_values, point_diameter_y, spread_factor)
            
            # Add category position to offsets
            x_pos = i + x_offsets
            
            all_x_pos.extend(x_pos)
            all_y_pos.extend(y_values)
            all_colors.extend([color_map[cat]] * len(y_values))

        # Plot all points at once for better performance
        ax.scatter(all_x_pos, all_y_pos, c=all_colors, s=point_size, alpha=0.8, **kwargs.get("scatter_kwargs", {}))

        ax.set_xticks(np.arange(n_categories))
        ax.set_xticklabels(categories)
        
        self._set_labels(
            ax,
            xlabel=categorical_var,
            ylabel=numerical_var,
            subplot_title=params.get("subplot_title", f"Beeswarm Plot of {numerical_var} by {categorical_var}")
        )
        self._apply_common_layout(fig, params)
        
        return fig
