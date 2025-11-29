import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.violin import ViolinPlot

def generate_violin_plots():
    """
    This script generates and saves example enhanced violin plots.
    """
    print("--- Generating Violin Plots ---")
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    np.random.seed(10)
    data = pd.DataFrame({
        'Category': np.repeat(['Group A', 'Group B', 'Group C'], 70),
        'Value': np.concatenate([
            np.random.normal(5, 1.5, 70),
            np.random.normal(8, 2, 70),
            np.random.normal(5.5, 1.2, 70)
        ])
    })

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        plotter = ViolinPlot(theme=theme)

        # 1. Enhanced Violin Plot (Jitter + Mean)
        try:
            fig = plotter.plot(
                data, 
                y='Value', 
                x='Category', 
                show_jitter=True, 
                show_mean=True,
                show_n=True,
                subplot_title=f"Enhanced Violin Plot ({theme})"
            )
            theme_suffix = "light" if "light" in theme else "dark"
            fig.savefig(os.path.join(output_dir, f"violin_plot_classic_professional_{theme_suffix}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved enhanced violin plot")
        except Exception as e:
            print(f"  - Error generating enhanced violin plot: {e}")

        # 2. Violin Plot with Statistical Annotations
        try:
            fig = plotter.plot(
                data,
                y='Value',
                x='Category',
                perform_stat_test=True,
                subplot_title=f"Violin Plot with T-Tests ({theme})"
            )
            theme_suffix = "light" if "light" in theme else "dark"
            fig.savefig(os.path.join(output_dir, f"violin_stats_{theme_suffix}.png"), bbox_inches='tight')
            plt.close(fig)
            print("  - Saved violin plot with stats")
        except Exception as e:
            print(f"  - Error generating violin plot with stats: {e}")

    print("\n--- Violin plot generation complete. ---")

if __name__ == "__main__":
    generate_violin_plots()
