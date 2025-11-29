import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.raincloud import RaincloudPlot

def generate_raincloud_plots():
    """
    This script generates and saves example raincloud plots.
    """
    print("--- Generating Raincloud Plots ---")
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    np.random.seed(0)
    data = pd.DataFrame({
        'Value': np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(2, 1.5, 50),
            np.random.normal(-1, 0.8, 50)
        ]),
        'Category': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = RaincloudPlot(library='matplotlib', theme=theme)

        # 1. Vertical Raincloud
        try:
            fig_v = plotter.plot(
                data,
                x='Category',
                y='Value',
                orientation='vertical',
                subplot_title=f"Vertical Raincloud ({theme})"
            )
            filename_v = f"raincloud_vertical_{theme}.png"
            filepath_v = os.path.join(output_dir, filename_v)
            fig_v.savefig(filepath_v, bbox_inches='tight')
            plt.close(fig_v)
            print(f"  - Saved {filename_v}")
        except Exception as e:
            print(f"  - Error generating vertical raincloud plot for theme {theme}: {e}")

        # 2. Horizontal Raincloud
        try:
            fig_h = plotter.plot(
                data,
                x='Value',
                y='Category',
                orientation='horizontal',
                subplot_title=f"Horizontal Raincloud ({theme})"
            )
            filename_h = f"raincloud_horizontal_{theme}.png"
            filepath_h = os.path.join(output_dir, filename_h)
            fig_h.savefig(filepath_h, bbox_inches='tight')
            plt.close(fig_h)
            print(f"  - Saved {filename_h}")
        except Exception as e:
            print(f"  - Error generating horizontal raincloud plot for theme {theme}: {e}")

    print("\n--- Raincloud plot generation complete. ---")

if __name__ == "__main__":
    generate_raincloud_plots()
