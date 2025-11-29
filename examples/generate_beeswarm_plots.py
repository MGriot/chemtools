import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.beeswarm import BeeswarmPlot

def generate_beeswarm_plots():
    """
    This script generates and saves example beeswarm plots.
    """
    print("--- Generating Beeswarm Plots ---")
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
        print(f"\nGenerating beeswarm plot for theme: {theme}...")
        try:
            plotter = BeeswarmPlot(library='matplotlib', theme=theme)
            fig = plotter.plot(
                data,
                x='Category',
                y='Value',
                subplot_title=f"Beeswarm Plot ({theme})"
            )
            
            filename = f"beeswarm_plot_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating beeswarm plot for theme {theme}: {e}")

    print("\n--- Beeswarm plot generation complete. ---")

if __name__ == "__main__":
    generate_beeswarm_plots()