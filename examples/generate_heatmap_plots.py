import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.relationship.heatmap import HeatmapPlot

def generate_heatmap_plots():
    """
    This script generates and saves example numerical and categorical heatmaps.
    """
    print("--- Generating Heatmap Plots ---")
    
    # Define themes to generate plots for
    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 1. Numerical Heatmap ---
    output_dir_relationship = "doc/img/plots/relationship"
    os.makedirs(output_dir_relationship, exist_ok=True)
    print("\nGenerating Numerical Heatmaps (e.g., Correlation Matrix)...")

    # Sample numerical data for correlation
    numerical_data = pd.DataFrame(np.random.rand(100, 5), columns=[f'Var{i+1}' for i in range(5)])
    corr_matrix = numerical_data.corr()

    for theme in themes:
        try:
            plotter = HeatmapPlot(theme=theme)
            fig = plotter.plot(corr_matrix, annot=True, subplot_title=f"Correlation Matrix ({theme})")
            
            filename = f"heatmap_{theme}.png"
            filepath = os.path.join(output_dir_relationship, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating numerical heatmap for theme {theme}: {e}")

    # --- 2. Categorical Heatmap ---
    output_dir_categorical = "doc/img/plots/categorical"
    os.makedirs(output_dir_categorical, exist_ok=True)
    print("\nGenerating Categorical Heatmaps (Co-occurrence)...")
    
    # Sample categorical data
    categorical_data = pd.DataFrame({
        'Hair Color': np.random.choice(['Black', 'Brown', 'Blonde', 'Red'], 200),
        'Eye Color': np.random.choice(['Brown', 'Blue', 'Green', 'Hazel'], 200),
    })

    for theme in themes:
        try:
            plotter = HeatmapPlot(theme=theme)
            fig = plotter.plot_categorical(categorical_data, 
                                           x_column='Hair Color', 
                                           y_column='Eye Color', 
                                           annot=True, 
                                           subplot_title=f"Co-occurrence ({theme})")
            
            filename = f"heatmap_categorical_{theme}.png"
            filepath = os.path.join(output_dir_categorical, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating categorical heatmap for theme {theme}: {e}")

    print("\n--- Heatmap plot generation complete. ---")

if __name__ == "__main__":
    generate_heatmap_plots()
