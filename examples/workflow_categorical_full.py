import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration.eda import ExploratoryDataAnalysis
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot
from chemtools.stats import ChiSquaredTest
import os

def run_full_categorical_workflow():
    """
    An example script that demonstrates the full categorical data analysis workflow
    using chemtools, including 100% stacked bar charts and Chi-Squared test.
    """
    print("--- Full Categorical Analysis Workflow with Chemtools ---")

    output_dir = "doc/img/plots/categorical"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]

    # 1. Create sample data
    print("\n1. Creating sample data...")
    data = pd.DataFrame({
        'Education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], 300),
        'JobSatisfaction': np.random.choice(['Low', 'Medium', 'High'], 300, p=[0.3, 0.4, 0.3]),
        'Gender': np.random.choice(['Male', 'Female'], 300),
    })
    print("Sample Data Head:")
    print(data.head())

    # 2. Summary statistics (using EDA)
    print("\n2. Summary Statistics:")
    eda = ExploratoryDataAnalysis(data)
    
    print("\n--- Chemtools EDA Categorical Summary (All Columns) ---")
    print(eda.get_categorical_summary())

    print("\n--- Value counts for 'Education' ---")
    print("\nValue counts for Education:")
    print(data['Education'].value_counts())
    print("\nNormalized value counts for Education:")
    print(data['Education'].value_counts(normalize=True))

    print("\n--- Value counts for 'JobSatisfaction' ---")
    print("\nValue counts for JobSatisfaction:")
    print(data['JobSatisfaction'].value_counts())
    print("\nNormalized value counts for JobSatisfaction:")
    print(data['JobSatisfaction'].value_counts(normalize=True))

    # 3. Contingency Table (Crosstab)
    print("\n3. Generating Contingency Table for Education vs. JobSatisfaction:")
    crosstab_df = pd.crosstab(data['Education'], data['JobSatisfaction'])
    print(crosstab_df)

    for theme in themes:
        print(f"\nGenerating plots for theme: {theme}...")

        # 4. 100% Stacked Bar Chart
        try:
            bar_plotter = BarPlot(theme=theme)
            fig_100_stacked = bar_plotter.plot_crosstab(
                crosstab_df, 
                stacked=True, 
                normalize=True, 
                subplot_title=f"100% Stacked Bar Chart: Job Satisfaction by Education ({theme})"
            )
            filename = f"stacked_bar_chart_{theme}.png"
            fig_100_stacked.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close(fig_100_stacked)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating stacked bar chart for theme {theme}: {e}")

        # 6. Heatmap of Co-occurrences (using the enhanced HeatmapPlot)
        try:
            heatmap_plotter = HeatmapPlot(theme=theme)
            fig_heatmap = heatmap_plotter.plot(crosstab_df, annot=True, subplot_title=f"Heatmap of Co-occurrences ({theme})")
            filename = f"crosstab_heatmap_{theme}.png"
            fig_heatmap.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close(fig_heatmap)
            print(f"  - Saved {filename}")
        except Exception as e:
            print(f"  - Error generating heatmap for theme {theme}: {e}")

    # 5. Chi-Squared Test and Cramér's V (not plot related, runs once)
    print("\n5. Performing Chi-Squared Test for Education vs. JobSatisfaction:")
    chi2_test = ChiSquaredTest()
    chi2_test.fit(x=data['Education'], y=data['JobSatisfaction'])
    print(chi2_test.summary)
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    run_full_categorical_workflow()
