import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.relationship.scatterplot import ScatterPlot

def generate_scatter_plots():
    """
    This script generates and saves example 2D, 3D, and bubble scatter plots.
    """
    print("--- Generating Scatter Plots ---")
    output_dir = "doc/img/plots/relationship"
    os.makedirs(output_dir, exist_ok=True)

    # --- Sample Data ---
    data = pd.DataFrame({
        'x_axis': np.random.rand(50) * 10,
        'y_axis': np.random.rand(50) * 10 + 5,
        'z_axis': np.random.rand(50) * 5,
        'bubble_size': np.random.rand(50) * 800 + 50
    })
    
    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 1. 2D Scatter Plot ---
    print("\nGenerating 2D Scatter Plots...")
    for theme in themes:
        try:
            plotter = ScatterPlot(theme=theme)
            fig = plotter.plot_2d(data, x_column='x_axis', y_column='y_axis', subplot_title=f"2D Scatter Plot ({theme})")
            
            filename = f"scatter_2d_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating 2D scatter plot for theme {theme}: {e}")

    # --- 2. Bubble Chart ---
    print("\nGenerating Bubble Charts...")
    for theme in themes:
        try:
            plotter = ScatterPlot(theme=theme)
            fig = plotter.plot_2d(data, x_column='x_axis', y_column='y_axis', size_column='bubble_size', subplot_title=f"Bubble Chart ({theme})")
            
            filename = f"bubble_chart_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating bubble chart for theme {theme}: {e}")

    # --- 3. 3D Scatter Plot ---
    print("\nGenerating 3D Scatter Plots...")
    for theme in themes:
        try:
            plotter = ScatterPlot(library='matplotlib', theme=theme)
            fig = plotter.plot_3d(data, x_column='x_axis', y_column='y_axis', z_column='z_axis', subplot_title=f"3D Scatter Plot ({theme})")
            
            filename = f"scatter_3d_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating 3D scatter plot for theme {theme}: {e}")

    print("\n--- Scatter plot generation complete. ---")

if __name__ == "__main__":
    generate_scatter_plots()
