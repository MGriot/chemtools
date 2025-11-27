import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.beeswarm import BeeswarmPlot
import os

def run_demo():
    """
    This script generates a sample beeswarm plot for demonstration purposes.
    """
    # Create sample data
    np.random.seed(0)
    data = pd.DataFrame({
        'Value': np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(2, 1.5, 50),
            np.random.normal(-1, 0.8, 50)
        ]),
        'Category': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
    })

    print("--- Beeswarm Plot Demo ---")

    output_dir = "doc/img/examples/beeswarm_demo"
    os.makedirs(output_dir, exist_ok=True)

    # --- Demo 1: Light Theme ---
    print("\n1. Generating beeswarm plot with light theme...")
    try:
        plotter_light = BeeswarmPlot(library='matplotlib', theme='classic_professional_light')
        fig1 = plotter_light.plot(
            data,
            x='Category',
            y='Value',
            subplot_title="Beeswarm Plot (Light Theme)"
        )
        fig1.savefig(os.path.join(output_dir, "beeswarm_plot_light.png"), bbox_inches='tight')
        plt.close(fig1)
        print("   ... Light theme plot generated.")
    except Exception as e:
        print(f"   ... Error generating light theme plot: {e}")


    # --- Demo 2: Dark Theme ---
    print("\n2. Generating beeswarm plot with dark theme...")
    try:
        plotter_dark = BeeswarmPlot(library='matplotlib', theme='oceanic_slate_dark')
        fig2 = plotter_dark.plot(
            data,
            x='Category',
            y='Value',
            subplot_title="Beeswarm Plot (Dark Theme)"
        )
        fig2.savefig(os.path.join(output_dir, "beeswarm_plot_dark.png"), bbox_inches='tight')
        plt.close(fig2)
        print("   ... Dark theme plot generated.")
    except Exception as e:
        print(f"   ... Error generating dark theme plot: {e}")

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    run_demo()