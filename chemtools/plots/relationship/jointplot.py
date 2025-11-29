import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ..base import BasePlotter
from ..distribution.histogram import HistogramPlot # For marginal histograms
from ..relationship.scatterplot import ScatterPlot # For central scatter
from ..relationship.density import DensityPlot # For central and marginal KDE

class JointPlot(BasePlotter):
    """
    A plotter for creating joint plots (marginal plots), combining a 2D plot
    (e.g., scatter or 2D KDE) with 1D distribution plots (histograms or KDEs)
    along the margins.
    """

    def plot(self, data: pd.DataFrame, x: str, y: str, 
             central_kind: str = 'scatter', marginal_kind: str = 'hist',
             central_kwargs: dict = None, marginal_kwargs: dict = None,
             **kwargs):
        """
        Generates a joint plot showing the relationship between two variables (x, y)
        and their individual distributions on the margins.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The column name for the x-axis (main variable).
            y (str): The column name for the y-axis (main variable).
            central_kind (str): Type of central plot: 'scatter' or 'kde2d'. Defaults to 'scatter'.
            marginal_kind (str): Type of marginal plot: 'hist' or 'kde1d'. Defaults to 'hist'.
            central_kwargs (dict, optional): Additional kwargs for the central plot.
            marginal_kwargs (dict, optional): Additional kwargs for the marginal plots.
            **kwargs: Additional keyword arguments passed to the BasePlotter.
        """
        params = self._process_common_params(**kwargs)
        if self.library != "matplotlib":
            raise NotImplementedError("Joint plots are currently only implemented for matplotlib.")

        if x not in data.columns or y not in data.columns:
            raise ValueError(f"Columns '{x}' and '{y}' must be in the data.")
        
        # Initialize kwargs for safety
        central_kwargs = central_kwargs or {}
        marginal_kwargs = marginal_kwargs or {}

        # Set up GridSpec for the layout
        # The main plot takes 5 units, marginals take 1 unit.
        fig = plt.figure(figsize=params.get("figsize", (10, 10)))
        gs = fig.add_gridspec(2, 2,  width_ratios=(5, 1), height_ratios=(1, 5),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 0]) # Main plot (bottom-left)
        ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_main) # Top marginal
        ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_main) # Right marginal

        # --- Plot Central Data ---
        if central_kind == 'scatter':
            ax_main.scatter(data[x], data[y], 
                            color=central_kwargs.get('color', self.colors['theme_color']),
                            s=central_kwargs.get('s', 20),
                            alpha=central_kwargs.get('alpha', 0.6),
                            **{k: v for k, v in central_kwargs.items() if k not in ['color', 's', 'alpha']})
        elif central_kind == 'kde2d':
            density_plotter = DensityPlot(library='matplotlib', theme=self.theme)
            # Pass relevant kwargs to _plot_kde
            density_plotter._plot_kde(ax_main, data[x].values, data[y].values, **central_kwargs)
        else:
            raise ValueError(f"Unsupported central_kind: '{central_kind}'. Choose from 'scatter' or 'kde2d'.")

        # --- Plot Marginal Distributions ---
        if marginal_kind == 'hist':
            ax_hist_x.hist(data[x], bins=marginal_kwargs.get('bins', 30), 
                           color=marginal_kwargs.get('color', self.colors['accent_color']), 
                           alpha=marginal_kwargs.get('alpha', 0.6),
                           **{k: v for k, v in marginal_kwargs.items() if k not in ['bins', 'color', 'alpha']})
            ax_hist_y.hist(data[y], bins=marginal_kwargs.get('bins', 30), orientation='horizontal', 
                           color=marginal_kwargs.get('color', self.colors['accent_color']), 
                           alpha=marginal_kwargs.get('alpha', 0.6),
                           **{k: v for k, v in marginal_kwargs.items() if k not in ['bins', 'color', 'alpha']})
        elif marginal_kind == 'kde1d':
            # 1D KDE for x-axis marginal
            kde_x = gaussian_kde(data[x].dropna())
            x_vals = np.linspace(data[x].min(), data[x].max(), 100)
            ax_hist_x.plot(x_vals, kde_x(x_vals), 
                           color=marginal_kwargs.get('color', self.colors['accent_color']),
                           **{k: v for k, v in marginal_kwargs.items() if k not in ['color']})
            ax_hist_x.fill_between(x_vals, kde_x(x_vals), 
                                   color=marginal_kwargs.get('color', self.colors['accent_color']), 
                                   alpha=marginal_kwargs.get('alpha', 0.2))

            # 1D KDE for y-axis marginal
            kde_y = gaussian_kde(data[y].dropna())
            y_vals = np.linspace(data[y].min(), data[y].max(), 100)
            ax_hist_y.plot(kde_y(y_vals), y_vals, 
                           color=marginal_kwargs.get('color', self.colors['accent_color']),
                           **{k: v for k, v in marginal_kwargs.items() if k not in ['color']})
            ax_hist_y.fill_betweenx(y_vals, kde_y(y_vals), 
                                    color=marginal_kwargs.get('color', self.colors['accent_color']), 
                                    alpha=marginal_kwargs.get('alpha', 0.2))
        else:
            raise ValueError(f"Unsupported marginal_kind: '{marginal_kind}'. Choose from 'hist' or 'kde1d'.")


        # --- Apply Theming and Common Layout to all axes ---
        # Apply theme colors manually to each axes background
        ax_main.set_facecolor(self.colors['axes.facecolor'])
        ax_hist_x.set_facecolor(self.colors['axes.facecolor'])
        ax_hist_y.set_facecolor(self.colors['axes.facecolor'])

        # Turn off ticks and spines for marginals for a cleaner look
        ax_hist_x.set_axis_off()
        ax_hist_y.set_axis_off()

        # Set labels for the main plot
        self._set_labels(ax_main, 
                         xlabel=params.get('xlabel', x), 
                         ylabel=params.get('ylabel', y),
                         subplot_title=params.get('subplot_title', f"Joint Plot: {x} vs {y}"))
        
        # Apply other common layout settings like title, watermark etc.
        self._apply_common_layout(fig, params)
        
        # Ensure tight layout for all subplots
        fig.tight_layout()

        return fig
