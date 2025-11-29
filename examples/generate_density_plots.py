import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from chemtools.plots.relationship import DensityPlot

def generate_density_plots():
    """
    This script generates and saves example 2D density plots (KDE, Hist2D, Hexbin).
    """
    print("--- Generating 2D Density Plots ---")
    output_dir = "doc/img/plots/relationship"
    os.makedirs(output_dir, exist_ok=True)

    # --- Sample Data ---
    np.random.seed(42)
    data = pd.DataFrame({
        'x_var': np.random.normal(loc=10, scale=2, size=1000),
        'y_var': np.random.normal(loc=10, scale=3, size=1000)
    })

    themes = ['classic_professional_light', 'classic_professional_dark']
    plot_kinds = ['kde', 'hist2d', 'hexbin']

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = DensityPlot(library='matplotlib', theme=theme)
        
        for kind in plot_kinds:
            try:
                print(f"  - Generating {kind} plot...")
                fig = plotter.plot(
                    data,
                    x='x_var',
                    y='y_var',
                    kind=kind,
                    subplot_title=f"2D Density Plot ({kind.upper()}) ({theme})"
                )
                
                filename = f"density_{kind}_{theme}.png"
                filepath = os.path.join(output_dir, filename)
                
                fig.savefig(filepath, bbox_inches='tight')
                plt.close(fig)
                print(f"    ... Plot saved to {filepath}")
                
            except Exception as e:
                print(f"    ... Error generating {kind} plot for {theme}: {e}")

    print("\n--- 2D Density plot generation complete. ---")

if __name__ == "__main__":
    generate_density_plots()
