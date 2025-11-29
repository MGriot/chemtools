import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.basic.bar import BarPlot

def generate_bar_plots():
    """
    This script generates and saves example bar plots.
    """
    print("--- Generating Bar Plots ---")
    output_dir = "doc/img/plots/basic"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Data for Counts ---
    data_counts = pd.DataFrame({'Category': np.random.choice(['A', 'B', 'C', 'D'], 200)})

    # --- Data for Grouped/Stacked ---
    data_grouped = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Group': ['G1', 'G2', 'G1', 'G2', 'G1', 'G2'],
        'Value': [10, 12, 15, 18, 13, 9]
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = BarPlot(theme=theme)

        # 1. Plot Counts
        try:
            fig = plotter.plot_counts(data_counts, column='Category', subplot_title=f"Category Counts ({theme})")
            fig.savefig(os.path.join(output_dir, f"bar_plot_counts_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved bar plot (counts)")
        except Exception as e:
            print(f"  - Error generating counts bar plot: {e}")

        # 2. Grouped Plot
        try:
            fig = plotter.plot(data_grouped, x='Category', y='Value', color='Group', mode='group', subplot_title=f"Grouped Bar Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"bar_plot_grouped_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved bar plot (grouped)")
        except Exception as e:
            print(f"  - Error generating grouped bar plot: {e}")
            
        # 3. Stacked Plot
        try:
            fig = plotter.plot(data_grouped, x='Category', y='Value', color='Group', mode='stack', subplot_title=f"Stacked Bar Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"bar_plot_stacked_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved bar plot (stacked)")
        except Exception as e:
            print(f"  - Error generating stacked bar plot: {e}")

    print("\n--- Bar plot generation complete. ---")

if __name__ == "__main__":
    generate_bar_plots()
