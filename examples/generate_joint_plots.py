import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.relationship.jointplot import JointPlot

def generate_joint_plots():
    """
    This script generates and saves example Joint Plots (marginal plots).
    """
    print("--- Generating Joint Plots ---")
    output_dir = "doc/img/plots/relationship"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    np.random.seed(42)
    data = pd.DataFrame({
        'var_a': np.random.normal(0, 1, 300),
        'var_b': np.random.normal(5, 2, 300) + np.random.normal(0, 0.5, 300) * np.random.normal(0, 1, 300),
        'group': np.random.choice(['Group X', 'Group Y'], 300)
    })

    for theme in themes:
        print(f"\nGenerating joint plots for theme: {theme}...")
        plotter = JointPlot(theme=theme, figsize=(8, 8))

        # Example 1: Scatter plot with marginal histograms
        try:
            fig1 = plotter.plot(data, 
                                x='var_a', y='var_b', 
                                central_kind='scatter', marginal_kind='hist',
                                central_kwargs={'color': plotter.colors['theme_color'], 'alpha': 0.7, 's': 30},
                                marginal_kwargs={'color': plotter.colors['accent_color'], 'bins': 20, 'alpha': 0.6},
                                subplot_title=f"Joint Plot: Scatter with Histograms ({theme})")
            
            filename1 = f"joint_scatter_hist_{theme}.png"
            fig1.savefig(os.path.join(output_dir, filename1), bbox_inches='tight')
            plt.close(fig1)
            print(f"  - Saved {filename1}")
        except Exception as e:
            print(f"  - Error generating joint scatter hist plot for theme {theme}: {e}")

        # Example 2: 2D KDE with marginal 1D KDEs
        try:
            fig2 = plotter.plot(data, 
                                x='var_a', y='var_b', 
                                central_kind='kde2d', marginal_kind='kde1d',
                                central_kwargs={'cmap': 'Blues', 'levels': 10, 'fill_alpha': 0.8},
                                marginal_kwargs={'color': plotter.colors['accent_color'], 'lw': 1.5, 'alpha': 0.3},
                                subplot_title=f"Joint Plot: 2D KDE with 1D KDEs ({theme})")
            
            filename2 = f"joint_kde_kde_{theme}.png"
            fig2.savefig(os.path.join(output_dir, filename2), bbox_inches='tight')
            plt.close(fig2)
            print(f"  - Saved {filename2}")
        except Exception as e:
            print(f"  - Error generating joint kde kde plot for theme {theme}: {e}")

    print("\n--- Joint plot generation complete. ---")

if __name__ == "__main__":
    generate_joint_plots()
