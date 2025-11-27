import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

def calculate_grid_dimensions(n_plots: int) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for subplots."""
    n_rows = int(np.sqrt(n_plots))
    n_cols = int(np.ceil(n_plots / n_rows))
    return n_rows, n_cols


def get_colormap(n_colors: int, cmap_name: str = "viridis") -> np.ndarray:
    """Get colors from a colormap."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, n_colors))


def format_axis_ticks(ax, rotation: int = 45, ha: str = "right"):
    """Format axis ticks consistently."""
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)
    plt.setp(ax.get_yticklabels(), rotation=0)


def show_plot(fig, library="matplotlib"):
    """Consistently show plots across backends."""
    if library == "plotly":
        fig.show()
    else:
        plt.show()