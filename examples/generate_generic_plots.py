import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import all the plotter classes
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.basic.line import LinePlot
from chemtools.plots.basic.pie import PiePlot
from chemtools.plots.distribution.histogram import HistogramPlot
from chemtools.plots.distribution.boxplot import BoxPlot
from chemtools.plots.relationship.scatterplot import ScatterPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot
from chemtools.plots.violin import ViolinPlot

def generate_generic_plots():
    """
    This script generates a sample plot for each of the common plot types
    and saves them in the 'doc/img/plots' directory.
    """
    print("--- Generating Generic Plots ---")

    # --- Sample Data ---
    data_numeric = pd.DataFrame({
        'A': np.random.randn(100) + 1,
        'B': np.random.randn(100) * 2,
        'C': np.random.randn(100) + 5,
    })
    data_categorical = pd.DataFrame({
        'Category': ['Cat1', 'Cat2', 'Cat3', 'Cat4'] * 25,
        'Value': np.random.rand(100) * 10,
        'Group': ['A', 'B'] * 50
    })
    data_mixed = pd.concat([data_numeric, data_categorical], axis=1)

    # Define themes to use for generating plots
    themes = ["classic_professional_light", "classic_professional_dark"]
    
    # --- Plot Generation ---
    
    # 1. Bar Plot
    output_dir = "doc/img/plots/basic"
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating Bar Plots...")
    for theme in themes:
        try:
            plotter = BarPlot(theme=theme)
            # Simple bar plot of counts
            fig = plotter.plot_counts(data_categorical, 'Category', subplot_title=f"Category Counts ({theme})")
            fig.savefig(os.path.join(output_dir, f"bar_plot_counts_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Grouped bar plot
            fig = plotter.plot(data_mixed, x='Category', y='Value', color='Group', mode='group', subplot_title=f"Grouped Bar Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"bar_plot_grouped_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved bar plots for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating bar plots for theme {theme}: {e}")

    # 2. Line Plot
    print("\nGenerating Line Plots...")
    line_data = pd.DataFrame({'x': range(20), 'y': np.random.randn(20).cumsum()})
    for theme in themes:
        try:
            plotter = LinePlot(theme=theme)
            fig = plotter.plot(line_data, 'x', 'y', subplot_title=f"Line Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"line_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved line plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating line plot for theme {theme}: {e}")

    # 3. Pie Plot
    print("\nGenerating Pie Plots...")
    pie_data = data_categorical.groupby('Category')['Value'].sum().reset_index()
    for theme in themes:
        try:
            plotter = PiePlot(theme=theme)
            fig = plotter.plot(pie_data, names_column='Category', values_column='Value', subplot_title=f"Pie Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pie_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved pie plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating pie plot for theme {theme}: {e}")

    # 4. Histogram
    output_dir = "doc/img/plots/distribution"
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating Histograms...")
    for theme in themes:
        try:
            plotter = HistogramPlot(theme=theme)
            fig = plotter.plot(data_numeric, 'A', subplot_title=f"Histogram ({theme})")
            fig.savefig(os.path.join(output_dir, f"histogram_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved histogram for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating histogram for theme {theme}: {e}")

    # 5. Box Plot
    print("\nGenerating Box Plots...")
    for theme in themes:
        try:
            plotter = BoxPlot(theme=theme)
            fig = plotter.plot(data_mixed, x='Category', y='Value', subplot_title=f"Box Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"boxplot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved box plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating box plot for theme {theme}: {e}")

    # 6. Violin Plot
    print("\nGenerating Violin Plots...")
    for theme in themes:
        try:
            plotter = ViolinPlot(theme=theme)
            fig = plotter.plot(data_mixed, y='Value', x='Category', show_jitter=True, show_mean=True, subplot_title=f"Violin Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"violin_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved violin plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating violin plot for theme {theme}: {e}")

    # 7. Scatter Plot
    output_dir = "doc/img/plots/relationship"
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating Scatter Plots...")
    for theme in themes:
        try:
            plotter = ScatterPlot(theme=theme)
            fig = plotter.plot_2d(data_numeric, 'A', 'B', subplot_title=f"Scatter Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"scatter_plot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved scatter plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating scatter plot for theme {theme}: {e}")

    # 8. Heatmap
    print("\nGenerating Heatmaps...")
    corr_matrix = data_numeric.corr()
    for theme in themes:
        try:
            plotter = HeatmapPlot(theme=theme)
            fig = plotter.plot(corr_matrix, subplot_title=f"Heatmap ({theme})")
            fig.savefig(os.path.join(output_dir, f"heatmap_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved heatmap for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating heatmap for theme {theme}: {e}")

    print("\n--- All generic plots have been generated. ---")

if __name__ == "__main__":
    generate_generic_plots()
