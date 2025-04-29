import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chemtools.clustering.HierarchicalClustering import HierarchicalClustering
from chemtools.plots.clustering.plot_dendogram import DendrogramPlotter


def show_figure(fig, library):
    """Helper function to display figures consistently."""
    if library == "plotly":
        fig.show()  # Opens in browser
    else:  # matplotlib
        plt.show(block=True)


def test_hierarchical_clustering():
    # Generate sample data
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(25, 2)

    # Create and fit the model
    model = HierarchicalClustering(X)
    model.fit()

    # Test with both backends and different presets
    for library in ["matplotlib", "plotly"]:
        for preset in ["default", "minimal", "grid", "presentation"]:
            plotter = DendrogramPlotter(
                library=library, theme="light", style_preset=preset  # or "dark"
            )

            fig = plotter.plot_dendrogram(
                model,
                title=f"Dendrogram ({preset} style)",
                orientation="top",
                color_threshold=0.5,
            )

            # Show or save the figure
            if library == "plotly":
                fig.show()
            else:
                plt.show(block=False)
                plt.pause(2)  # Show each style for 2 seconds
                plt.close()


if __name__ == "__main__":
    test_hierarchical_clustering()
