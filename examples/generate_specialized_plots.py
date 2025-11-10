import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Import all the plotter and model classes
from chemtools.plots.clustering.plot_dendogram import DendrogramPlotter
from chemtools.clustering.HierarchicalClustering import HierarchicalClustering
from chemtools.plots.geographical.map import MapPlot
from chemtools.plots.regression.regression_plots import RegressionPlots
from chemtools.regression.LinearRegression import OLSRegression
from chemtools.plots.relationship.pairplot import PairPlot
from chemtools.plots.specialized.bullet import BulletPlot
from chemtools.plots.specialized.dual_axis import DualAxisPlot
from chemtools.plots.specialized.funnel import FunnelPlot
from chemtools.plots.specialized.parallel_coordinates import ParallelCoordinatesPlot
from chemtools.plots.temporal.run_chart import RunChartPlot

def generate_specialized_plots():
    """
    This script generates a sample plot for each of the specialized plot types
    and saves them in the 'doc/img/plots' directory.
    """
    print("--- Generating Specialized Plots ---")
    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- Dendrogram ---
    output_dir = "doc/img/plots/clustering"
    print("\nGenerating Dendrogram...")
    X = np.random.rand(10, 4)
    model = HierarchicalClustering(X)
    model.fit()
    for theme in themes:
        try:
            plotter = DendrogramPlotter(theme=theme)
            fig = plotter.plot_dendrogram(model, subplot_title=f"Dendrogram ({theme})")
            fig.savefig(os.path.join(output_dir, f"dendrogram_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved dendrogram for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating dendrogram for theme {theme}: {e}")

    # --- Regression Plots ---
    output_dir = "doc/img/plots/regression"
    print("\nGenerating Regression Plots...")
    X_reg = np.linspace(0, 10, 50).reshape(-1, 1)
    y_reg = 2 * X_reg.flatten() + 1 + np.random.randn(50) * 2
    reg_model = OLSRegression()
    reg_model.fit(X_reg, y_reg)
    for theme in themes:
        try:
            plotter = RegressionPlots(reg_model, theme=theme)
            fig = plotter.plot_regression_results(subplot_title=f"Regression Results ({theme})")
            fig.savefig(os.path.join(output_dir, f"regression_results_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved regression plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating regression plot for theme {theme}: {e}")

    # --- Pair Plot ---
    output_dir = "doc/img/plots/relationship"
    print("\nGenerating Pair Plot...")
    pair_data = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
    pair_data['species'] = np.random.choice(['setosa', 'versicolor'], 100)
    for theme in themes:
        try:
            plotter = PairPlot(theme=theme)
            fig = plotter.plot(pair_data, hue='species', title=f"Pair Plot ({theme})")
            fig.savefig(os.path.join(output_dir, f"pairplot_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved pair plot for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating pair plot for theme {theme}: {e}")

    # --- Specialized Plots ---
    output_dir = "doc/img/plots/specialized"
    
    # Bullet Plot
    print("\nGenerating Bullet Plot...")
    for theme in themes:
        try:
            plotter = BulletPlot(theme=theme)
            fig = plotter.plot(value=275, target=250, ranges=[150, 225, 300], title=f"Revenue ({theme})")
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

    # --- Temporal Plot ---
    output_dir = "doc/img/plots/temporal"
    print("\nGenerating Run Chart...")
    run_chart_data = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50)),
        'Measurement': np.random.randn(50).cumsum() + 50
    })
    for theme in themes:
        try:
            plotter = RunChartPlot(theme=theme)
            fig = plotter.plot(run_chart_data, time_column='Date', value_column='Measurement', subplot_title=f"Run Chart ({theme})")
            fig.savefig(os.path.join(output_dir, f"run_chart_{theme}.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved run chart for theme: {theme}")
        except Exception as e:
            print(f"  - Error generating run chart for theme {theme}: {e}")
            
    # --- Geographical Plots (Plotly only) ---
    output_dir = "doc/img/plots/geographical"
    print("\nGenerating Geographical Plots (Plotly only)...")
    try:
        # Choropleth
        map_data = px.data.gapminder().query("year==2007")
        plotter = MapPlot(library='plotly', theme='classic_professional_light')
        fig = plotter.plot_choropleth(map_data, locations_column='iso_alpha', values_column='lifeExp', title="Choropleth Map")
        fig.write_image(os.path.join(output_dir, "choropleth_map.png"))
        print("  - Saved choropleth map")

        # Scatter Geo
        scatter_geo_data = pd.DataFrame({
            'City': ['London', 'New York', 'Tokyo', 'Sydney'],
            'lat': [51.5074, 40.7128, 35.6895, -33.8688],
            'lon': [-0.1278, -74.0060, 139.6917, 151.2093],
            'Population': [8.9, 8.4, 13.9, 5.3]
        })
        fig = plotter.plot_scatter_geo(scatter_geo_data, lat_column='lat', lon_column='lon', title="Scatter Geo Plot")
        fig.write_image(os.path.join(output_dir, "scatter_geo_map.png"))
        print("  - Saved scatter geo map")
    except ValueError as e:
        if "kaleido" in str(e):
            print("  - Skipping geographical plots: 'kaleido' package not found.")
            print("    To generate these plots, please install it: pip install --upgrade kaleido")
        else:
            print(f"  - Error generating geographical plots: {e}")
    except Exception as e:
        print(f"  - An unexpected error occurred while generating geographical plots: {e}")


    print("\n--- All specialized plots have been generated. ---")

if __name__ == "__main__":
    generate_specialized_plots()
