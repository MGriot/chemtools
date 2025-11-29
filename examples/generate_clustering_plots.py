import os
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.clustering.plot_dendogram import DendrogramPlotter
from chemtools.clustering.hierarchical_clustering import HierarchicalClustering

def generate_clustering_plots():
    """
    This script generates and saves example dendrogram plots for hierarchical clustering.
    """
    print("--- Generating Clustering Plots ---")
    output_dir = "doc/img/plots/clustering"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data and Model ---
    X = np.random.rand(12, 5)
    model = HierarchicalClustering(X)
    model.fit()
    labels = [f'Sample_{i+1}' for i in range(12)]

    for theme in themes:
        print(f"\nGenerating dendrogram for theme: {theme}...")
        try:
            plotter = DendrogramPlotter(theme=theme)
            fig = plotter.plot_dendrogram(model, labels=labels, subplot_title=f"Dendrogram ({theme})")
            
            filename = f"dendrogram_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating dendrogram for theme {theme}: {e}")

    print("\n--- Clustering plot generation complete. ---")

if __name__ == "__main__":
    generate_clustering_plots()
