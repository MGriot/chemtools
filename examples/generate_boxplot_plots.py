import os
import pandas as pd
import numpy as np
from chemtools.plots.distribution.boxplot import BoxPlot
import matplotlib.pyplot as plt

def generate_boxplot_plots():
    """
    This script generates and saves example box plots.
    """
    print("--- Generating Box Plots ---")
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    data = pd.DataFrame({
        'Category': np.repeat(['A', 'B', 'C', 'D'], 50),
        'Value': np.concatenate([
            np.random.normal(5, 1, 50),
            np.random.normal(8, 2, 50),
            np.random.normal(4, 1.5, 50),
            np.random.normal(6, 1, 50)
        ])
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = BoxPlot(theme=theme)

        # 1. Box Plot
        try:
            fig = plotter.plot(data, x='Category', y='Value', subplot_title=f"Box Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"boxplot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved box plot")
        except Exception as e:
            print(f"  - Error generating box plot: {e}")

    print("\n--- Box plot generation complete. ---")

if __name__ == "__main__":
    generate_boxplot_plots()
