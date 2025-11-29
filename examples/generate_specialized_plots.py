import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import all the plotter and model classes
from chemtools.plots.specialized.bullet import BulletPlot
from chemtools.plots.specialized.dual_axis import DualAxisPlot
from chemtools.plots.specialized.funnel import FunnelPlot
from chemtools.plots.specialized.parallel_coordinates import ParallelCoordinatesPlot

def generate_specialized_plots():
    """
    This script generates a sample plot for each of the specialized plot types
    and saves them in the 'doc/img/plots' directory.
    """
    print("--- Generating Specialized Plots ---")
    themes = ["classic_professional_light", "classic_professional_dark"]

    output_dir = "doc/img/plots/specialized"
    os.makedirs(output_dir, exist_ok=True)
    
    # Bullet Plot
    print("\nGenerating Bullet Plot...")
    for theme in themes:
        try:
            plotter = BulletPlot(theme=theme)
            fig = plotter.plot(value=275, target=250, ranges=[150, 225, 300], subplot_title=f"Revenue ({theme})")
            fig.savefig(os.path.join(output_dir, f"bullet_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved bullet plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating bullet plot for theme {theme}: {e}")

    # Dual Axis Plot
    print("\nGenerating Dual Axis Plot...")
    dual_axis_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Sales': [150, 200, 180, 220, 250, 210],
        'Growth (%)': [1.5, 1.8, 1.2, 2.0, 2.2, 1.9]
    })
    for theme in themes:
        try:
            plotter = DualAxisPlot(theme=theme)
            fig = plotter.plot(dual_axis_data, x_column='Month', y1_column='Sales', y2_column='Growth (%)', subplot_title=f"Sales and Growth ({theme})")
            fig.savefig(os.path.join(output_dir, f"dual_axis_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved dual axis plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating dual axis plot for theme {theme}: {e}")

    # Funnel Plot
    print("\nGenerating Funnel Plot...")
    funnel_data = pd.DataFrame({
        'Stage': ['Website Visits', 'Downloads', 'Registrations', 'Purchases'],
        'Value': [10000, 4000, 1500, 500]
    })
    for theme in themes:
        try:
            plotter = FunnelPlot(theme=theme)
            fig = plotter.plot(funnel_data, stage_column='Stage', values_column='Value', subplot_title=f"Sales Funnel ({theme})")
            fig.savefig(os.path.join(output_dir, f"funnel_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved funnel plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating funnel plot for theme {theme}: {e}")

    # Parallel Coordinates Plot
    print("\nGenerating Parallel Coordinates Plot...")
    pc_data = pd.DataFrame(np.random.rand(30, 3), columns=['A', 'B', 'C'])
    pc_data['class'] = np.random.choice(['X', 'Y', 'Z'], 30)
    for theme in themes:
        try:
            plotter = ParallelCoordinatesPlot(theme=theme)
            fig = plotter.plot(pc_data, class_column='class', subplot_title=f"Parallel Coordinates ({theme})")
            fig.savefig(os.path.join(output_dir, f"parallel_coordinates_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved parallel coordinates plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating parallel coordinates plot for theme {theme}: {e}")

    print("\n--- All specialized plots have been generated. ---")

if __name__ == "__main__":
    generate_specialized_plots()