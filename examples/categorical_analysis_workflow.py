import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration import EDA
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot

def categorical_analysis_workflow():
    """
    An example script that reproduces a typical categorical data analysis workflow
    using chemtools, without saving any files.
    """
    print("--- Categorical Analysis Workflow with Chemtools ---")

    # 1. Load and create sample data
    print("\n1. Creating sample data...")
    data = pd.DataFrame({
        'Category1': np.random.choice(['A', 'B', 'C'], 200),
        'Category2': np.random.choice(['X', 'Y', 'Z', 'W'], 200),
    })
    print("Sample Data Head:")
    print(data.head())

    # 2. Summary statistics
    print("\n2. Summary Statistics:")
    eda = EDA(data)
    
    # Using chemtools EDA class for a quick overview
    print("\n--- Using chemtools.exploration.EDA ---")
    categorical_summary = eda.get_categorical_summary()
    print(categorical_summary)

    # Mimicking pandas value_counts() for detailed counts
    print("\n--- Mimicking pandas value_counts() ---")
    print("\nValue counts for Category1:")
    print(data['Category1'].value_counts())
    print("\nNormalized value counts for Category1:")
    print(data['Category1'].value_counts(normalize=True))

    # 3. Visualize distributions (Count Plot)
    print("\n3. Visualizing single variable distribution (Count Plot)...")
    bar_plotter = BarPlot()
    fig1 = bar_plotter.plot_counts(data, 'Category1', subplot_title="Distribution of Category1")
    plt.show()

    # 4. Relationship between two categorical variables (Grouped Count Plot)
    print("\n4. Visualizing relationship between two variables (Grouped Count Plot)...")
    # To mimic a count plot with a hue, we first need to get the counts per group.
    grouped_counts = data.groupby(['Category1', 'Category2']).size().reset_index(name='count')
    
    # The BarPlot.plot method needs a y-value, so we use the calculated 'count'.
    fig2 = bar_plotter.plot(grouped_counts, x='Category1', y='count', color='Category2', mode='group', subplot_title="Counts of Category2 within Category1")
    plt.show()

    # 5. Crosstab and heatmap
    print("\n5. Crosstab and Heatmap...")
    
    # Using chemtools EDA to get the contingency table (crosstab)
    crosstab_df = eda.get_crosstab('Category1', 'Category2')
    print("\nContingency Table (Crosstab):")
    print(crosstab_df)

    # Using the enhanced HeatmapPlot with annotations
    print("\nDisplaying heatmap of the crosstab...")
    heatmap_plotter = HeatmapPlot()
    fig3 = heatmap_plotter.plot(crosstab_df, annot=True, subplot_title="Heatmap of Co-occurrences")
    plt.show()
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    categorical_analysis_workflow()
