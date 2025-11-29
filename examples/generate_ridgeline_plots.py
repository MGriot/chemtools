import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.ridgeline import RidgelinePlot

def generate_ridgeline_plots():
    """
    This script generates and saves example ridgeline plots.
    """
    print("--- Generating Ridgeline Plots ---")
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)

    # --- Sample Data ---
    data_simple = pd.DataFrame({
        'Value': np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1.5, 100),
            np.random.normal(-2, 0.8, 100)
        ]),
        'Category': ['Group A'] * 100 + ['Group B'] * 100 + ['Group C'] * 100
    })

    data_annotated = pd.DataFrame({
        'Price': np.concatenate([
            np.random.gamma(4, 500, 100),
            np.random.gamma(6, 400, 100),
            np.random.gamma(5, 600, 100),
            np.random.gamma(7, 300, 100)
        ]),
        'Adjective': ['Nice'] * 100 + ['Spacious'] * 100 + ['Clean'] * 100 + ['Modern'] * 100
    })
    
    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 1. Simple Ridgeline Plot ---
    print("\nGenerating Simple Ridgeline Plots...")
    for theme in themes:
        try:
            plotter = RidgelinePlot(theme=theme)
            fig = plotter.plot(data_simple, x='Value', y='Category', overlap=0.6, title=f"Simple Ridgeline ({theme})")
            
            filename = f"ridgeline_simple_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating simple ridgeline for theme {theme}: {e}")

    # --- 2. Annotated Ridgeline Plot ---
    print("\nGenerating Annotated Ridgeline Plots...")
    annotations = {
        'title': "Adjectives vs. Rental Prices",
        'description': "Distribution of rental prices based on descriptive adjectives.",
        'xlabel': "Rent in USD",
        'credit': "Data: Fictional rental listings."
    }
    for theme in themes:
        try:
            plotter = RidgelinePlot(theme=theme, figsize=(8, 7))
            fig = plotter.plot_annotated(
                data_annotated, 
                x='Price', 
                y='Adjective',
                annotations=annotations,
                show_legend=True
            )
            
            filename = f"ridgeline_annotated_{theme}.png"
            filepath = os.path.join(output_dir, filename)

            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating annotated ridgeline for theme {theme}: {e}")

    print("\n--- Ridgeline plot generation complete. ---")

if __name__ == "__main__":
    generate_ridgeline_plots()
