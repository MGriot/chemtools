import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.plots.temporal.run_chart import RunChartPlot

def generate_temporal_plots():
    """
    This script generates and saves example run charts.
    """
    print("--- Generating Temporal Plots ---")
    output_dir = "doc/img/plots/temporal"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    data = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50)),
        'Measurement': np.random.randn(50).cumsum() + 50
    })

    for theme in themes:
        print(f"\nGenerating run chart for theme: {theme}...")
        try:
            plotter = RunChartPlot(theme=theme)
            fig = plotter.plot(data, 
                               time_column='Date', 
                               value_column='Measurement', 
                               subplot_title=f"Run Chart ({theme})")
            
            filename = f"run_chart_{theme}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating run chart for theme {theme}: {e}")

    print("\n--- Temporal plot generation complete. ---")

if __name__ == "__main__":
    generate_temporal_plots()
