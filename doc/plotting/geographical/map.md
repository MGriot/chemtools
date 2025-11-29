# Geographical Plots (Maps)

Geographical plots are used to visualize data on a map. The `MapPlot` class in `chemtools` supports two main types of map plots, both of which are best used with the `plotly` backend for interactivity.

**Note:** To save these plots as static images, you will also need the `kaleido` package (`pip install --upgrade kaleido`).

## Choropleth Map

A choropleth map displays divided geographical areas or regions that are colored in relation to a data variable. This is a great way to visualize how a measurement varies across a geographic area.

### Usage
```python
from chemtools.plots.geographical import MapPlot
import plotly.express as px

# Sample data from plotly express
data = px.data.gapminder().query("year==2007")

# Create Plot
plotter = MapPlot(library='plotly')
fig = plotter.plot_choropleth(data, 
                            locations_column='iso_alpha', 
                            values_column='lifeExp', 
                            title="Global Life Expectancy in 2007")
fig.write_image("choropleth_map.png")
```

### Parameters for `plot_choropleth`
- `data` (pd.DataFrame): The input DataFrame.
- `locations_column` (str): The column containing country names or ISO codes.
- `values_column` (str): The column with the numerical values to plot.
- `**kwargs`: Additional keyword arguments passed to `plotly.express.choropleth`.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/geographical/choropleth_map_classic_professional_dark.png">
  <img alt="Choropleth Map" src="../../img/plots/geographical/choropleth_map_classic_professional_light.png">
</picture>

---

## Geo Scatter Plot

A geo scatter plot renders points on a map based on their latitude and longitude. It's useful for showing the location of specific events or objects.

### Usage
```python
from chemtools.plots.geographical import MapPlot
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'City': ['London', 'New York', 'Tokyo', 'Sydney'],
    'lat': [51.5074, 40.7128, 35.6895, -33.8688],
    'lon': [-0.1278, -74.0060, 139.6917, 151.2093],
    'Population': [8.9, 8.4, 13.9, 5.3]
})

# Create Plot
plotter = MapPlot(library='plotly')
fig = plotter.plot_scatter_geo(data, 
                               lat_column='lat', 
                               lon_column='lon',
                               color='City',
                               size='Population',
                               title="Major City Locations & Populations")
fig.write_image("scatter_geo_map.png")
```

### Parameters for `plot_scatter_geo`
- `data` (pd.DataFrame): The input DataFrame.
- `lat_column` (str): The column with latitude values.
- `lon_column` (str): The column with longitude values.
- `**kwargs`: Additional keyword arguments passed to `plotly.express.scatter_geo`, such as `color`, `size`, `hover_name`, etc.

### Example Output
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../img/plots/geographical/scatter_geo_map_classic_professional_dark.png">
  <img alt="Geo Scatter Plot" src="../../img/plots/geographical/scatter_geo_map_classic_professional_light.png">
</picture>
