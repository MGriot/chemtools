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

    # Test with both backends
    for library in ["matplotlib", "plotly"]:
        plotter = DendrogramPlotter(library=library)
        fig = plotter.plot_dendrogram(
            model,
            figsize=(10, 7),      # works for both backends
            width=800,            # works for both backends
            height=600,           # works for both backends
            title="Hierarchical Clustering Dendrogram",
            orientation="top",
            color_threshold=0.5,
            labels=None          # optional: custom labels
        )
        show_figure(fig, library)

if __name__ == "__main__":
    test_hierarchical_clustering()
