import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.basic.line import LinePlot

def generate_line_plots():
    """
    This script generates and saves example line, dot, and area plots.
    """
    print("--- Generating Line Plots ---")
    output_dir = "doc/img/plots/basic"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    data = pd.DataFrame({
        'Time': np.arange(20),
        'Value': (np.arange(20) + np.random.randn(20) * 2).cumsum()
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = LinePlot(theme=theme)

        # 1. Line Plot
        try:
            fig = plotter.plot(data, x_column='Time', y_column='Value', mode='line', subplot_title=f"Line Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"line_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved line plot")
        except Exception as e:
            print(f"  - Error generating line plot: {e}")

        # 2. Dot Plot
        try:
            fig = plotter.plot(data, x_column='Time', y_column='Value', mode='dot', subplot_title=f"Dot Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"dot_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved dot plot")
        except Exception as e:
            print(f"  - Error generating dot plot: {e}")
            
        # 3. Area Plot
        try:
            fig = plotter.plot(data, x_column='Time', y_column='Value', mode='area', subplot_title=f"Area Chart ({theme})")
            fig.savefig(os.path.join(output_dir, f"area_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved area chart")
        except Exception as e:
            print(f"  - Error generating area chart: {e}")

    print("\n--- Line plot generation complete. ---")

if __name__ == "__main__":
    generate_line_plots()
