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

output_dir = "doc/img/plots/distribution"
os.makedirs(output_dir, exist_ok=True) # Create output directory

# --- Demo 1: Vertical Light Theme ---
print("\n1. Generating vertical light theme raincloud plot...")
try:
    plotter_light = RaincloudPlot(library='matplotlib', theme='classic_professional_light')
    fig1 = plotter_light.plot(
        data,
        x='Category',
        y='Value',
        orientation='vertical',
        violin_filled=True,
    )
    fig1.savefig(os.path.join(output_dir, "raincloud_vertical_classic_professional_light.png"), bbox_inches='tight')
    plt.close(fig1)
    print("   ... Vertical light theme plot generated.")
except Exception as e:
    print(f"   ... Error generating vertical light theme plot: {e}")


# --- Demo 2: Vertical Dark Theme ---
print("\n2. Generating vertical dark theme raincloud plot...")
try:
    plotter_dark = RaincloudPlot(library='matplotlib', theme='classic_professional_dark')
    fig2 = plotter_dark.plot(
        data,
        x='Category',
        y='Value',
        orientation='vertical',
        violin_filled=True,
    )
    fig2.savefig(os.path.join(output_dir, "raincloud_vertical_classic_professional_dark.png"), bbox_inches='tight')
    plt.close(fig2)
    print("   ... Vertical dark theme plot generated.")
except Exception as e:
    print(f"   ... Error generating vertical dark theme plot: {e}")

# --- Demo 3: Horizontal Light Theme ---
print("\n3. Generating horizontal light theme raincloud plot...")
try:
    plotter_light_h = RaincloudPlot(library='matplotlib', theme='classic_professional_light')
    fig3 = plotter_light_h.plot(
        data,
        x='Value',
        y='Category',
        orientation='horizontal',
        violin_filled=True,
    )
    fig3.savefig(os.path.join(output_dir, "raincloud_horizontal_classic_professional_light.png"), bbox_inches='tight')
    plt.close(fig3)
    print("   ... Horizontal light theme plot generated.")
except Exception as e:
    print(f"   ... Error generating horizontal light theme plot: {e}")

# --- Demo 4: Horizontal Dark Theme ---
print("\n4. Generating horizontal dark theme raincloud plot...")
try:
    plotter_dark_h = RaincloudPlot(library='matplotlib', theme='classic_professional_dark')
    fig4 = plotter_dark_h.plot(
        data,
        x='Value',
        y='Category',
        orientation='horizontal',
        violin_filled=True,
    )
    fig4.savefig(os.path.join(output_dir, "raincloud_horizontal_classic_professional_dark.png"), bbox_inches='tight')
    plt.close(fig4)
    print("   ... Horizontal dark theme plot generated.")
except Exception as e:
    print(f"   ... Error generating horizontal dark theme plot: {e}")


print("\n--- Demo Complete ---")
