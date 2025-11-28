import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ..base import BasePlotter 

class DensityPlot(BasePlotter):
    """
    A plotter for creating 2D density charts, including kernel density estimates (KDE),
    2D histograms, and hexbin plots using Matplotlib and Scipy.
    """

    def plot(self, data: pd.DataFrame, x: str, y: str, kind: str = 'kde', **kwargs):
        """
        Plots a 2D density chart.
        
        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            kind (str, optional): The type of density plot. Can be 'kde', 'hist2d', or 'hexbin'.
                                  Defaults to 'kde'.
            **kwargs: Additional keyword arguments for the specific plot type.
                - For 'kde': 
                    `bw_adjust` (float): Smoothing factor for KDE (default 1.0, scales kernel.factor).
                    `cmap` (str or Colormap): Colormap for the filled area (defaults to theme-generated).
                    `fill` (bool): Whether to fill the density area (default True).
                    `fill_method` (str): 'contourf' (default) or 'pcolormesh' for fill rendering.
                    `fill_alpha` (float): Alpha transparency for the filled area (default 0.8).
                    `levels` (int): Number of contour levels to draw (default 7).
                    `thresh` (float): Lowest iso-proportion level at which to draw a contour line (default 0.05).
                    `contour_line_color` (str): Color of the contour lines (defaults to theme text color).
                    `line_alpha` (float): Alpha transparency for the contour lines (default 0.7).
                - For 'hist2d': 
                    `bins` (int or tuple): Number of bins in each dimension (default 30).
                    `cmap` (str or Colormap): Colormap (defaults to theme-generated).
                - For 'hexbin': 
                    `gridsize` (int): The number of hexagons in the x-direction (default 20).
                    `cmap` (str or Colormap): Colormap (defaults to theme-generated).
        """
        params = self._process_common_params(**kwargs)
        
        if self.library != "matplotlib":
            raise NotImplementedError("2D Density plots are only implemented for matplotlib.")

        if x not in data.columns or y not in data.columns:
            raise ValueError(f"Columns '{x}' and '{y}' must be in the data.")

        # Clean data (remove NaNs) and get numpy arrays
        plot_data = data[[x, y]].dropna()
        x_data = plot_data[x].values
        y_data = plot_data[y].values

        fig, ax = self._create_figure(figsize=params["figsize"])

        if kind == 'kde':
            self._plot_kde(ax, x_data, y_data, **kwargs)
        elif kind == 'hist2d':
            self._plot_hist2d(fig, ax, x_data, y_data, **kwargs)
        elif kind == 'hexbin':
            self._plot_hexbin(fig, ax, x_data, y_data, **kwargs)
        else:
            raise ValueError(f"Unsupported plot kind: '{kind}'. Choose from 'kde', 'hist2d', or 'hexbin'.")

        # Apply labels using BasePlotter logic or fallbacks
        self._set_labels(
            ax, 
            subplot_title=params.get("subplot_title", f'2D Density Plot ({kind.upper()})'), 
            xlabel=params.get("xlabel", x), 
            ylabel=params.get("ylabel", y)
        )
        self._apply_common_layout(fig, params)
        return fig

    def _plot_kde(self, ax, x, y, **kwargs):
        """Internal method to handle KDE plotting."""
        # 1. Calculate KDE
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)

        # 2. Handle Bandwidth Adjustment (Replicating Seaborn's bw_adjust)
        bw_adjust = kwargs.get('bw_adjust', 1.0)
        kernel.set_bandwidth(bw_method=kernel.factor * bw_adjust)

        # 3. Create Grid with padding
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xpad = (xmax - xmin) * 0.1
        ypad = (ymax - ymin) * 0.1
        
        xi, yi = np.mgrid[xmin-xpad:xmax+xpad:100j, ymin-ypad:ymax+ypad:100j]
        positions = np.vstack([xi.ravel(), yi.ravel()])
        zi = np.reshape(kernel(positions).T, xi.shape)

        # 4. Plotting parameters
        cmap = kwargs.get('cmap', self.get_continuous_colormap())
        levels = kwargs.get('levels', 7) # Default changed to 7 for more detail
        thresh = kwargs.get('thresh', 0.05)
        contour_line_color = kwargs.get('contour_line_color', self.colors['text_color'])
        fill_alpha = kwargs.get('fill_alpha', 0.8)
        line_alpha = kwargs.get('line_alpha', 0.7)
        fill_method = kwargs.get('fill_method', 'contourf')

        zi_max = zi.max()
        level_values = np.linspace(thresh * zi_max, zi_max, levels)

        # Conditional fill logic
        if kwargs.get('fill', True):
            if fill_method == 'contourf':
                ax.contourf(xi, yi, zi, levels=level_values, cmap=cmap, alpha=fill_alpha)
            elif fill_method == 'pcolormesh':
                ax.pcolormesh(xi, yi, zi, shading='gouraud', cmap=cmap, alpha=fill_alpha)
            else:
                raise ValueError(f"Unsupported fill_method: '{fill_method}'. Choose from 'contourf' or 'pcolormesh'.")

        # Draw contour lines
        ax.contour(xi, yi, zi, levels=level_values, colors=contour_line_color, linewidths=0.5, alpha=line_alpha)

    def _plot_hist2d(self, fig, ax, x, y, **kwargs):
        """Internal method to handle 2D Histogram plotting."""
        bins = kwargs.get('bins', 30)
        cmap = kwargs.get('cmap', self.get_continuous_colormap()) # Use theme colormap as default
        h = ax.hist2d(x, y, bins=bins, cmap=cmap, cmin=1) # cmin=1 makes empty bins transparent
        fig.colorbar(h[3], ax=ax, label='Count')

    def _plot_hexbin(self, fig, ax, x, y, **kwargs):
        """Internal method to handle Hexbin plotting."""
        gridsize = kwargs.get('gridsize', 20)
        cmap = kwargs.get('cmap', self.get_continuous_colormap()) # Use theme colormap as default
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1) # mincnt=1 makes empty bins transparent
        fig.colorbar(hb, ax=ax, label='Count')

