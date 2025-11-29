import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.basic.pie import PiePlot

def generate_pie_plots():
    """
    This script generates and saves example pie and donut charts.
    """
    print("--- Generating Pie Plots ---")
    output_dir = "doc/img/plots/basic"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    data = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [45, 25, 15, 15]
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = PiePlot(theme=theme)

        # 1. Pie Plot
        try:
            fig = plotter.plot(data, names_column='Category', values_column='Value', subplot_title=f"Pie Chart ({theme})")
            fig.savefig(os.path.join(output_dir, f"pie_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved pie chart")
        except Exception as e:
            print(f"  - Error generating pie chart: {e}")

        # 2. Donut Plot
        try:
            fig = plotter.plot(data, names_column='Category', values_column='Value', hole=0.4, subplot_title=f"Donut Chart ({theme})")
            fig.savefig(os.path.join(output_dir, f"donut_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved donut chart")
        except Exception as e:
            print(f"  - Error generating donut chart: {e}")

    print("\n--- Pie plot generation complete. ---")

if __name__ == "__main__":
    generate_pie_plots()
