import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from chemtools.plots.relationship.heatmap import HeatmapPlot
from chemtools.plots.categorical.mosaic import MosaicPlot

def generate_categorical_plots():
    """
    This script generates plots for purely categorical data using the new
    and updated plotters.
    """
    print("--- Generating Categorical Plots ---")
    output_dir = "doc/img/plots/categorical"
    os.makedirs(output_dir, exist_ok=True)

    # --- Sample Data ---
    data = pd.DataFrame({
        'Hair Color': np.random.choice(['Black', 'Brown', 'Blonde', 'Red'], 100),
        'Eye Color': np.random.choice(['Brown', 'Blue', 'Green'], 100),
    })

    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 1. Heatmap of Co-occurrences ---
    print("\nGenerating Heatmap of Co-occurrences...")
    for theme in themes:
        try:
            plotter = HeatmapPlot(theme=theme)
            fig = plotter.plot_categorical(data, x_column='Hair Color', y_column='Eye Color', subplot_title=f"Co-occurrence Heatmap ({theme})")
            fig.savefig(os.path.join(output_dir, f"heatmap_categorical_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved categorical heatmap for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating categorical heatmap for theme {theme}: {e}")

    # --- 2. Mosaic Plot ---
    print("\nGenerating Mosaic Plots...")
    for theme in themes:
        # Matplotlib backend
        try:
            plotter_mpl = MosaicPlot(library='matplotlib', theme=theme)
            fig_mpl = plotter_mpl.plot(data, columns=['Hair Color', 'Eye Color'], subplot_title=f"Mosaic Plot (matplotlib, {theme})")
            fig_mpl.savefig(os.path.join(output_dir, f"mosaic_plot_mpl_{theme}.png"), bbox_inches='tight')
            plt.close(fig_mpl)
            print(f"  - Saved matplotlib mosaic plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating matplotlib mosaic plot for theme {theme}: {e}")

        # Plotly backend (Treemap)
        try:
            plotter_plotly = MosaicPlot(library='plotly', theme=theme)
            fig_plotly = plotter_plotly.plot(data, columns=['Hair Color', 'Eye Color'], title=f"Mosaic Plot (Treemap, {theme})")
            fig_plotly.write_image(os.path.join(output_dir, f"mosaic_plot_plotly_{theme}.png"))
            print(f"  - Saved plotly mosaic plot (treemap) for theme: {theme}")
        except ValueError as e:
            if "kaleido" in str(e):
                print("  - Skipping plotly mosaic plot: 'kaleido' package not found.")
            else:
                print(f"  - Error generating plotly mosaic plot for theme {theme}: {e}")
        except Exception as e:
            print(f"  - An unexpected error occurred while generating plotly mosaic plot for theme {theme}: {e}")


    print("\n--- All categorical plots have been generated. ---")

if __name__ == "__main__":
    generate_categorical_plots()