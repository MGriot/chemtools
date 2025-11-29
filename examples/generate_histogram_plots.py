import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.histogram import HistogramPlot

def generate_histogram_plots():
    """
    This script generates and saves example histograms and density plots.
    """
    print("--- Generating Histogram and Density Plots ---")
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    data = pd.DataFrame({'Value': np.random.randn(500) * 5 + 10})

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = HistogramPlot(theme=theme)

        # 1. Histogram
        try:
            fig = plotter.plot(data, column='Value', mode='hist', bins=20, subplot_title=f"Histogram ({theme})")
            fig.savefig(os.path.join(output_dir, f"histogram_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved histogram")
        except Exception as e:
            print(f"  - Error generating histogram: {e}")

        # 2. Density Plot
        try:
            fig = plotter.plot(data, column='Value', mode='density', subplot_title=f"Density Curve ({theme})")
            fig.savefig(os.path.join(output_dir, f"density_curve_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved density curve")
        except Exception as e:
            print(f"  - Error generating density curve: {e}")

    print("\n--- Histogram and Density plot generation complete. ---")

if __name__ == "__main__":
    generate_histogram_plots()
