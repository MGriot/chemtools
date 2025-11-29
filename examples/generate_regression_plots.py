import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.regression.regression_plots import RegressionPlots
from chemtools.regression.linear_regression import OLSRegression

def generate_regression_plots():
    """
    This script generates and saves example regression plots.
    """
    print("--- Generating Regression Plots ---")
    output_dir = "doc/img/plots/regression"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data and Model ---
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.flatten() + 1 + np.random.randn(50) * 2
    model = OLSRegression()
    model.fit(X, y)

    for theme in themes:
        print(f"\nGenerating regression plot for theme: {theme}...")
        try:
            plotter = RegressionPlots(model, theme=theme)
            fig = plotter.plot_regression_results(subplot_title=f"Regression Results ({theme})")
            
            filename = f"regression_results_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating regression plot for theme {theme}: {e}")

    print("\n--- Regression plot generation complete. ---")

if __name__ == "__main__":
    generate_regression_plots()
