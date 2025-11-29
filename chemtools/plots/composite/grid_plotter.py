import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Any, Optional

from ..base import BasePlotter

class GridPlotter(BasePlotter):
    """
    A plotter for arranging multiple chemtools plots in a dynamic grid layout.
    This class allows for flexible arrangement, dimension control, and thematic
    consistency across subplots.
    """

    def __init__(self, nrows: int = 1, ncols: int = 1, 
                 width_ratios: Optional[List[float]] = None, 
                 height_ratios: Optional[List[float]] = None,
                 subplot_titles: Optional[List[str]] = None,
                 **kwargs):
        """
        Initializes the GridPlotter.

        Args:
            nrows (int): Number of rows in the grid.
            ncols (int): Number of columns in the grid.
            width_ratios (List[float], optional): Ratios of subplot widths.
            height_ratios (List[float], optional): Ratios of subplot heights.
            subplot_titles (List[str], optional): Titles for each subplot, ordered row by row.
            **kwargs: Additional keyword arguments passed to the BasePlotter.
        """
        super().__init__(**kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.subplot_titles = subplot_titles if subplot_titles is not None else []

        self._figure = None
        self._axes = np.empty((nrows, ncols), dtype=object) # Store axes objects
        self._plot_configs = [] # List of (row, col, plotter_instance, plot_method_name, data, plot_kwargs)

        # Create the figure and initial grid structure
        self._create_grid_figure()

    def _create_grid_figure(self):
        """Creates the Matplotlib Figure and GridSpec layout."""
        # Process common parameters (especially figsize and overall title)
        params = self._process_common_params()
        self._figure = plt.figure(figsize=params.get("figsize"))

        gs_kwargs = {}
        if self.width_ratios:
            gs_kwargs['width_ratios'] = self.width_ratios
        if self.height_ratios:
            gs_kwargs['height_ratios'] = self.height_ratios

        gs = gridspec.GridSpec(self.nrows, self.ncols, figure=self._figure, **gs_kwargs)

        for r in range(self.nrows):
            for c in range(self.ncols):
                ax = self._figure.add_subplot(gs[r, c])
                self._axes[r, c] = ax
                # Apply base styles to individual axes
                ax.set_facecolor(self.colors['axes.facecolor'])
                ax.tick_params(axis='x', colors=self.colors['text_color'])
                ax.tick_params(axis='y', colors=self.colors['text_color'])
                ax.spines['left'].set_color(self.colors['text_color'])
                ax.spines['bottom'].set_color(self.colors['text_color'])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Set individual subplot title if provided
                title_idx = r * self.ncols + c
                if title_idx < len(self.subplot_titles):
                    ax.set_title(self.subplot_titles[title_idx], color=self.colors['text_color'],
                                 fontsize=plt.rcParams["axes.titlesize"])

        # Apply overall figure title if present
        if params.get("title"):
            self._figure.suptitle(
                params["title"],
                fontsize=plt.rcParams["figure.titlesize"],
                weight=plt.rcParams["figure.titleweight"],
                color=self.colors["text_color"],
            )

    def add_plot(self, row: int, col: int, 
                 plotter_instance: BasePlotter, 
                 plot_method_name: str,
                 data: pd.DataFrame, 
                 **plot_kwargs):
        """
        Adds a plot to a specific cell in the grid.

        Args:
            row (int): The row index for the subplot (0-indexed).
            col (int): The column index for the subplot (0-indexed).
            plotter_instance (BasePlotter): An instance of a chemtools plotter class.
            plot_method_name (str): The name of the plotting method to call on the plotter_instance.
            data (pd.DataFrame): The data to be passed to the plotting method.
            **plot_kwargs: Additional keyword arguments for the specific plotting method.
                           These will override any defaults set by the GridPlotter instance.
        """
        if not (0 <= row < self.nrows and 0 <= col < self.ncols):
            raise ValueError(f"Plot position ({row}, {col}) is out of grid bounds ({self.nrows}, {self.ncols}).")
        
        if not isinstance(plotter_instance, BasePlotter):
            raise TypeError("plotter_instance must be an instance of a class inheriting from BasePlotter.")

        # Ensure the plotter_instance's library matches the GridPlotter's library
        if plotter_instance.library != self.library:
            print(f"Warning: Plotter instance library '{plotter_instance.library}' does not match GridPlotter library '{self.library}'. Forcing plotter_instance's library to match.")
            plotter_instance.library = self.library
            # Re-initialize style for the plotter instance if library changed
            if self.library == "matplotlib":
                plotter_instance._init_matplotlib_style()
            elif self.library == "plotly":
                plotter_instance._init_plotly_style() # This would be complex for a subplot axes. Plotly is better as separate figures for now.

        if not hasattr(plotter_instance, plot_method_name):
            raise AttributeError(f"Plotter instance does not have method '{plot_method_name}'.")

        # Store the plot configuration for later rendering
        self._plot_configs.append((row, col, plotter_instance, plot_method_name, data, plot_kwargs))

    def render(self):
        """
        Renders all added plots onto their respective subplots in the grid.
        """
        if self.library == "matplotlib":
            for r, c, plotter_instance, plot_method_name, data, plot_kwargs in self._plot_configs:
                ax = self._axes[r, c]
                plot_method = getattr(plotter_instance, plot_method_name)

                # Call the plotting method, passing the specific axis
                # Override plotter_instance's internal figsize/title since GridPlotter controls figure.
                modified_plot_kwargs = plot_kwargs.copy()
                modified_plot_kwargs['figsize'] = None # Figure size controlled by GridPlotter
                modified_plot_kwargs['title'] = None # Figure title controlled by GridPlotter
                modified_plot_kwargs['subplot_title'] = None # Subplot title already set or handled here

                # Dynamically determine arguments to pass
                # Many plot methods take ax as an argument, but not all.
                # If the method accepts 'ax', pass it. Otherwise, assume it returns a figure.
                import inspect
                sig = inspect.signature(plot_method)
                if 'ax' in sig.parameters:
                    plot_method(data, ax=ax, **modified_plot_kwargs)
                else:
                    # If the method doesn't take 'ax', it might create its own figure.
                    # This scenario is problematic for subplots.
                    # For chemtools plotters, they should ideally support an 'ax' argument
                    # for integration into subplots.
                    print(f"Warning: Plot method '{plot_method_name}' does not accept 'ax' argument. "
                          f"It might create a new figure, potentially leading to unexpected results.")
                    # As a fallback, render to a temporary figure and try to copy, or raise error.
                    # For now, let's assume methods intended for subplots accept 'ax'.
                    raise NotImplementedError(f"Plot method '{plot_method_name}' for {plotter_instance.__class__.__name__} "
                                              f"must accept an 'ax' argument to be used in GridPlotter.")
            
            # Apply tight layout for the overall figure
            self._figure.tight_layout()
            return self._figure
        elif self.library == "plotly":
            # Plotly multi-subplot rendering is typically done via plotly.subplots.make_subplots
            # which returns a single figure. This is different from Matplotlib's ax-based approach.
            # GridPlotter would need a different rendering mechanism for Plotly.
            raise NotImplementedError("GridPlotter for Plotly is not yet implemented using this approach. "
                                      "Consider using plotly.subplots.make_subplots directly for Plotly grids.")
        else:
            raise ValueError(f"Unsupported plotting library: {self.library}")

    def show(self):
        """Displays the rendered grid plot."""
        if self._figure:
            plt.show()
        else:
            print("No figure to show. Call render() first.")

    def save(self, filepath: str, **kwargs):
        """Saves the rendered grid plot to a file."""
        if self._figure:
            self._figure.savefig(filepath, **kwargs)
        else:
            print("No figure to save. Call render() first.")
