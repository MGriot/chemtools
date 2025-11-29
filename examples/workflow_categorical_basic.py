import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration.eda import ExploratoryDataAnalysis
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot
import os

def run_basic_categorical_workflow():
    """
    An example script that reproduces a typical categorical data analysis workflow
    using chemtools.
    """
    print("--- Basic Categorical Analysis Workflow with Chemtools ---")

    output_dir = "doc/img/plots/categorical"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]

    # 1. Create sample data
    print("\n1. Creating sample data...")
    data = pd.DataFrame({
        'Category1': np.random.choice(['A', 'B', 'C'], 200),
        'Category2': np.random.choice(['X', 'Y', 'Z', 'W'], 200),
    })
    print("Sample Data Head:")
    print(data.head())

    # 2. Summary statistics
    print("\n2. Summary Statistics:")
    eda = ExploratoryDataAnalysis(data)
    
    print("\n--- Chemtools EDA Categorical Summary (All Columns) ---")
    print(eda.get_categorical_summary())

    print("\n--- Value counts for 'Category1' ---")
    print("\nValue counts for Category1:")
    print(data['Category1'].value_counts())
    print("\nNormalized value counts for Category1:")
    print(data['Category1'].value_counts(normalize=True))

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")
        
        # 3. Visualize distributions (Count Plot)
        try:
            bar_plotter = BarPlot(theme=theme)
            fig1 = bar_plotter.plot_counts(data, 'Category1', subplot_title=f"Distribution of Category1 ({theme})")
            filename1 = f"category1_count_plot_{theme}.png"
            fig1.savefig(os.path.join(output_dir, filename1), bbox_inches='tight')
            plt.close(fig1)
            print(f"  - Saved {filename1}")
        except Exception as e:
            print(f"  - Error generating count plot for theme {theme}: {e}")

        # 4. Relationship between two categorical variables (Grouped Count Plot)
        try:
            grouped_counts = data.groupby(['Category1', 'Category2']).size().reset_index(name='count')
            fig2 = bar_plotter.plot(grouped_counts, x='Category1', y='count', color='Category2', mode='group', subplot_title=f"Counts of Category2 within Category1 ({theme})")
            filename2 = f"grouped_count_plot_{theme}.png"
            fig2.savefig(os.path.join(output_dir, filename2), bbox_inches='tight')
            plt.close(fig2)
            print(f"  - Saved {filename2}")
        except Exception as e:
            print(f"  - Error generating grouped count plot for theme {theme}: {e}")

        # 5. Crosstab and heatmap
        try:
            crosstab_df = eda.get_crosstab('Category1', 'Category2')
            heatmap_plotter = HeatmapPlot(theme=theme)
            fig3 = heatmap_plotter.plot(crosstab_df, annot=True, subplot_title=f"Heatmap of Co-occurrences ({theme})")
            filename3 = f"crosstab_heatmap_{theme}.png"
            fig3.savefig(os.path.join(output_dir, filename3), bbox_inches='tight')
            plt.close(fig3)
            print(f"  - Saved {filename3}")
        except Exception as e:
            print(f"  - Error generating heatmap for theme {theme}: {e}")
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    run_basic_categorical_workflow()
