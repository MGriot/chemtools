import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import GridPlotter and other plotters to use in the grid
from chemtools.plots.composite.grid_plotter import GridPlotter
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.basic.line import LinePlot
from chemtools.plots.relationship.scatterplot import ScatterPlot
from chemtools.plots.basic.pie import PiePlot
from chemtools.plots.distribution.histogram import HistogramPlot

def generate_grid_plots():
    """
    This script generates and saves example plots using the GridPlotter.
    """
    print("--- Generating Grid Plots ---")
    output_dir = "doc/img/plots/composite"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Sample Data ---
    np.random.seed(42)
    data1 = pd.DataFrame({'Category': np.random.choice(['A', 'B', 'C', 'D'], 50)})
    data2 = pd.DataFrame({'Time': np.arange(20), 'Value': np.random.randn(20).cumsum()})
    data3 = pd.DataFrame({'X': np.random.rand(100), 'Y': np.random.rand(100)})
    data4 = pd.DataFrame({'Task': ['Task1', 'Task2', 'Task3'], 'Progress': [45, 30, 25]})
    data5 = pd.DataFrame({'Value': np.random.randn(200) * 5 + 10})

    for theme in themes:
        print(f"\nGenerating grid plots for theme: {theme}...")
        
        # Example 1: Simple 2x2 grid
        try:
            grid_plotter = GridPlotter(
                nrows=2, ncols=2, figsize=(12, 10), 
                subplot_titles=[f"Counts ({theme})", f"Time Series ({theme})", f"Scatter ({theme})", f"Pie ({theme})"]
                theme=theme, title=f"Example Grid Plot ({theme})"
            )

            # Re-initialize plotters for each theme to ensure correct styling
            bar_plotter = BarPlot(theme=theme)
            line_plotter = LinePlot(theme=theme)
            scatter_plotter = ScatterPlot(theme=theme)
            pie_plotter = PiePlot(theme=theme)

            grid_plotter.add_plot(row=0, col=0, plotter_instance=bar_plotter, plot_method_name='plot_counts', 
                                  data=data1, column='Category') # subplot_title not needed here due to grid_plotter's own subplot_titles

            grid_plotter.add_plot(row=0, col=1, plotter_instance=line_plotter, plot_method_name='plot', 
                                  data=data2, x_column='Time', y_column='Value')

            grid_plotter.add_plot(row=1, col=0, plotter_instance=scatter_plotter, plot_method_name='plot_2d', 
                                  data=data3, x_column='X', y_column='Y')

            grid_plotter.add_plot(row=1, col=1, plotter_instance=pie_plotter, plot_method_name='plot', 
                                  data=data4, names_column='Task', values_column='Progress')

            fig = grid_plotter.render()
            filename = f"grid_plotter_example_{theme}.png"
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating simple grid plot for theme {theme}: {e}")

        # Example 2: Grid with custom ratios and mixed plot types
        try:
            grid_plotter_ratios = GridPlotter(
                nrows=2, ncols=2, figsize=(14, 8),
                width_ratios=[1, 2], height_ratios=[2, 1],
                subplot_titles=[f"Histogram ({theme})", f"Line ({theme})", f"Bar ({theme})", f"Scatter ({theme})"]
                theme=theme, title=f"Grid Plot with Custom Ratios ({theme})"
            )

            histogram_plotter = HistogramPlot(theme=theme)
            
            grid_plotter_ratios.add_plot(row=0, col=0, plotter_instance=histogram_plotter, plot_method_name='plot', 
                                         data=data5, column='Value', mode='hist')
            grid_plotter_ratios.add_plot(row=0, col=1, plotter_instance=line_plotter, plot_method_name='plot', 
                                         data=data2, x_column='Time', y_column='Value', marginal_kwargs={'color': 'green'}) # Pass through kwargs
            grid_plotter_ratios.add_plot(row=1, col=0, plotter_instance=bar_plotter, plot_method_name='plot', 
                                         data=data1.groupby('Category')['Category'].count().reset_index(name='Count'), x='Category', y='Count')
            grid_plotter_ratios.add_plot(row=1, col=1, plotter_instance=scatter_plotter, plot_method_name='plot_2d', 
                                         data=data3, x_column='X', y_column='Y')
            
            fig_ratios = grid_plotter_ratios.render()
            filename_ratios = f"grid_plotter_ratios_example_{theme}.png"
            fig_ratios.savefig(os.path.join(output_dir, filename_ratios), bbox_inches='tight')
            plt.close(fig_ratios)
            print(f"  - Saved {filename_ratios}")
        except Exception as e:
            print(f"  - Error generating grid plot with ratios for theme {theme}: {e}")

    print("\n--- Grid plot generation complete. ---")

if __name__ == "__main__":
    generate_grid_plots()
