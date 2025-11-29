import os
import pandas as pd
import plotly.express as px
from chemtools.plots.geographical.map import MapPlot

def generate_geographical_plots():
    """
    This script generates and saves example geographical plots (choropleth and scatter geo).
    """
    print("--- Generating Geographical Plots (Plotly) ---")
    output_dir = "doc/img/plots/geographical"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]

    # --- 2. Sample Data (outside loop as it's static) ---
    map_data = px.data.gapminder().query("year==2007")
    scatter_geo_data = pd.DataFrame({
        'City': ['London', 'New York', 'Tokyo', 'Sydney', 'Cairo'],
        'lat': [51.5074, 40.7128, 35.6895, -33.8688, 30.0444],
        'lon': [-0.1278, -74.0060, 139.6917, 151.2093, 31.2357],
        'Population (M)': [8.9, 8.4, 13.9, 5.3, 9.5]
    })

    for theme in themes:
        print(f"\nGenerating geographical plots for theme: {theme}...")
        plotter = MapPlot(library='plotly', theme=theme)

        # --- 1. Choropleth Map ---
        try:
            print(f"  - Generating Choropleth map for theme {theme}...")
            fig_choro = plotter.plot_choropleth(
                map_data, 
                locations_column='iso_alpha',
                values_column='lifeExp',
                title=f"Global Life Expectancy (2007) ({theme})"
            )
            
            filename_choro = f"choropleth_map_{theme}.png"
            fig_choro.write_image(os.path.join(output_dir, filename_choro))
            print(f"    - Saved {filename_choro}")

        except ValueError as e:
            if "kaleido" in str(e):
                print("    - Skipping choropleth plot: 'kaleido' package not found.")
                print("      To generate these plots, please install it: pip install --upgrade kaleido")
            else:
                print(f"    - Error generating choropleth map for theme {theme}: {e}")
        except Exception as e:
            print(f"    - An unexpected error occurred while generating choropleth map for theme {theme}: {e}")

        # --- 2. Scatter Geo Plot ---
        try:
            print(f"  - Generating Scatter Geo plot for theme {theme}...")
            fig_scatter = plotter.plot_scatter_geo(
                scatter_geo_data, 
                lat_column='lat',
                lon_column='lon',
                size='Population (M)',
                color='City',
                hover_name='City',
                title=f"Major City Locations & Populations ({theme})"
            )
            
            filename_scatter = f"scatter_geo_map_{theme}.png"
            fig_scatter.write_image(os.path.join(output_dir, filename_scatter))
            print(f"    - Saved {filename_scatter}")

        except ValueError as e:
            if "kaleido" in str(e):
                print("    - Skipping scatter geo plot: 'kaleido' package not found.")
            else:
                print(f"    - Error generating scatter geo plot for theme {theme}: {e}")
        except Exception as e:
            print(f"    - An unexpected error occurred while generating scatter geo plot for theme {theme}: {e}")

    print("\n--- Geographical plot generation complete. ---")

if __name__ == "__main__":
    generate_geographical_plots()
