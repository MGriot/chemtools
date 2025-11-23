import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.raincloud import RaincloudPlot
import os # Import os for directory creation

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

print("--- Raincloud Plot Demo ---")

output_dir = "doc/img/examples/raincloud_demo"
os.makedirs(output_dir, exist_ok=True) # Create output directory

# --- Demo 1: Filled Violin ---
print("\n1. Generating filled raincloud plot...")
try:
    plotter_filled = RaincloudPlot(library='matplotlib', theme='classic_professional_light')
    fig1 = plotter_filled.plot(
        data,
        x='Category',
        y='Value',
        violin_filled=True,
        title="Filled Raincloud Plot",
        plot_offset=0.1,  # Adjusted for closer points and violin
        jitter_amount=0.02 # Adjusted for tighter points
    )
    fig1.savefig(os.path.join(output_dir, "filled_raincloud_plot.png"), bbox_inches='tight')
    plt.close(fig1)
    print("   ... Filled plot generated.")
except Exception as e:
    print(f"   ... Error generating filled plot: {e}")


# --- Demo 2: Unfilled (Contour) Violin ---
print("\n2. Generating unfilled (contour) raincloud plot...")
try:
    plotter_unfilled = RaincloudPlot(library='matplotlib', theme='oceanic_slate_dark')
    fig2 = plotter_unfilled.plot(
        data,
        x='Category',
        y='Value',
        violin_filled=False,
        title="Unfilled (Contour) Raincloud Plot",
        plot_offset=0.1,  # Adjusted for closer points and violin
        jitter_amount=0.02 # Adjusted for tighter points
    )
    fig2.savefig(os.path.join(output_dir, "unfilled_raincloud_plot.png"), bbox_inches='tight')
    plt.close(fig2)
    print("   ... Unfilled plot generated.")
except Exception as e:
    print(f"   ... Error generating unfilled plot: {e}")

print("\n--- Demo Complete ---")