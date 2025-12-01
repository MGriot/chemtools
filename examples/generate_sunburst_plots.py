import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.basic import SunburstPlot

def generate_sunburst_plots():
    """
    This script generates and saves example sunburst plots.
    """
    print("--- Generating Sunburst Plots ---")
    output_dir = "doc/img/plots/basic"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    np.random.seed(42)
    data = {
        'Market': np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'], 200, p=[0.25, 0.25, 0.2, 0.1, 0.1, 0.1]),
        'Meta-Supplier': np.random.choice(['Supplier X', 'Supplier Y', 'Supplier Z', 'Supplier W', 'Other Supplier'], 200, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'Material': [f'Mat{i}' for i in range(200)],
        'Status': np.random.choice(['Ok', 'Not Ok'], 200, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = SunburstPlot(theme=theme)

        # 1. Sunburst Plot
        try:
            fig = plotter.plot(
                df,
                cols=['Market', 'Meta-Supplier'],
                count_col='Material',
                status_col='Status',
                status_ok_val='Ok',
                subplot_title=f"Material Status Breakdown ({theme})",
                top_n_limits=(4, 5) # Adjusting limits for better visibility in example
            )
            
            filename = f"sunburst_plot_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating sunburst plot for theme {theme}: {e}")

    print("\n--- Sunburst plot generation complete. ---")

if __name__ == "__main__":
    generate_sunburst_plots()
