import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chemtools.clustering.HierarchicalClustering import HierarchicalClustering
from chemtools.plots.clustering.plot_dendogram import DendrogramPlotter

def test_hierarchical_clustering():
    # Generate sample data
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(25, 2)

    # Create and fit the model
    model = HierarchicalClustering(X)
    model.fit()

    # Create plotter and plot the dendrogram
    plotter = DendrogramPlotter(library="matplotlib")
    fig = plotter.plot_dendrogram(
        model,
        figsize=(10, 7),
        title="Hierarchical Clustering Dendrogram",
        truncate_mode="level",
        p=3,
        orientation="top"
    )
    plt.show(block=True)  # This will keep the window open

if __name__ == "__main__":
    test_hierarchical_clustering()
