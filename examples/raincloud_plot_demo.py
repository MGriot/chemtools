import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.distribution.raincloud import RaincloudPlot

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

# --- Demo 1: Filled Violin ---
print("\n1. Generating filled raincloud plot...")
try:
    plotter_filled = RaincloudPlot(library='matplotlib', theme='classic_professional_light')
    fig1 = plotter_filled.plot(
        data,
        x='Category',
        y='Value',
        violin_filled=True,
        title="Filled Raincloud Plot"
    )
    plt.show()
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
        title="Unfilled (Contour) Raincloud Plot"
    )
    plt.show()
    print("   ... Unfilled plot generated.")
except Exception as e:
    print(f"   ... Error generating unfilled plot: {e}")

print("\n--- Demo Complete ---")
