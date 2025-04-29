import numpy as np
from matplotlib import pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chemtools.clustering.HierarchicalClustering import HierarchicalClustering
from chemtools.plots.clustering.dendogram_plot import dendogram_plot

def test_hierarchical_clustering():
    # Generate sample data
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(100, 2)

    # Create and fit the model
    model = HierarchicalClustering(X)
    model.fit()

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    dendogram_plot(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


if __name__ == "__main__":
    test_hierarchical_clustering()
