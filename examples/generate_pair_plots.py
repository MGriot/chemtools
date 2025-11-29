import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.relationship.pairplot import PairPlot

def generate_pair_plots():
    """
    This script generates and saves example pair plots.
    """
    print("--- Generating Pair Plots ---")
    output_dir = "doc/img/plots/relationship"
    os.makedirs(output_dir, exist_ok=True)

    # --- Sample Data (Iris-like) ---
    np.random.seed(42)
    data = pd.DataFrame({
        'Sepal Length': np.random.normal(5.8, 0.8, 150),
        'Sepal Width': np.random.normal(3.0, 0.4, 150),
        'Petal Length': np.random.normal(3.7, 1.7, 150),
        'Petal Width': np.random.normal(1.2, 0.7, 150),
        'Species': np.random.choice(['Setosa', 'Versicolor', 'Virginica'], 150, p=[0.33, 0.33, 0.34])
    })
    
    themes = ["classic_professional_light", "classic_professional_dark"]

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        try:
            # --- Plot 1: Pair Plot with legend ---
            plotter_legend = PairPlot(theme=theme, figsize=(10, 10))
            fig_legend = plotter_legend.plot(
                data, 
                hue='Species', 
                showlegend=True, 
                title=f"Pair Plot with Legend ({theme})"
            )
            
            filename_legend = f"pairplot_with_legend_{theme}.png"
            filepath_legend = os.path.join(output_dir, filename_legend)
            
            fig_legend.savefig(filepath_legend, bbox_inches='tight')
            plt.close(fig_legend)
            print(f"  - Saved {filename_legend}")

            # --- Plot 2: Pair Plot without legend ---
            # For the main plot_types.md page, a version without a large legend might be better
            plotter_no_legend = PairPlot(theme=theme, figsize=(9, 9))
            fig_no_legend = plotter_no_legend.plot(
                data, 
                hue='Species', 
                showlegend=False, 
                title=f"Pair Plot without Legend ({theme})"
            )

            filename_no_legend = f"pairplot_{theme}.png"
            filepath_no_legend = os.path.join(output_dir, filename_no_legend)
            
            fig_no_legend.savefig(filepath_no_legend, bbox_inches='tight')
            plt.close(fig_no_legend)
            print(f"  - Saved {filename_no_legend}")

        except Exception as e:
            print(f"  - Error generating pair plots for theme {theme}: {e}")

    print("\n--- Pair plot generation complete. ---")

if __name__ == "__main__":
    generate_pair_plots()
