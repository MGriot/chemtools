import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration.EDA import ExploratoryDataAnalysis
from chemtools.plots.basic.bar import BarPlot
from chemtools.plots.relationship.heatmap import HeatmapPlot
from chemtools.stats import ChiSquaredTest
import os # Import os for directory creation

def categorical_analysis_full_workflow():
    """
    An example script that demonstrates the full categorical data analysis workflow
    using chemtools, including 100% stacked bar charts and Chi-Squared test.
    """
    print("--- Full Categorical Analysis Workflow with Chemtools ---")

    output_dir = "doc/img/examples/categorical_analysis"
    os.makedirs(output_dir, exist_ok=True) # Create output directory

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
    
    # Using chemtools EDA class for a quick overview of all categorical columns
    print("\n--- Chemtools EDA Categorical Summary (All Columns) ---")
    print(eda.get_categorical_summary())

    # Mimicking pandas value_counts() for detailed counts of specific columns
    print("\n--- Mimicking pandas value_counts() for 'Education' ---")
    print("\nValue counts for Education:")
    print(data['Education'].value_counts())
    print("\nNormalized value counts for Education:")
    print(data['Education'].value_counts(normalize=True))

    print("\n--- Mimicking pandas value_counts() for 'JobSatisfaction' ---")
    print("\nValue counts for JobSatisfaction:")
    print(data['JobSatisfaction'].value_counts())
    print("\nNormalized value counts for JobSatisfaction:")
    print(data['JobSatisfaction'].value_counts(normalize=True))

    # 3. Contingency Table (Crosstab)
    print("\n3. Generating Contingency Table for Education vs. JobSatisfaction:")
    crosstab_df = pd.crosstab(data['Education'], data['JobSatisfaction'])
    print(crosstab_df)

    # 4. 100% Stacked Bar Chart
    print("\n4. Plotting 100% Stacked Bar Chart for Education vs. JobSatisfaction:")
    bar_plotter = BarPlot(theme='classic_professional_light')
    fig_100_stacked = bar_plotter.plot_crosstab(
        crosstab_df, 
        stacked=True, 
        normalize=True, 
        subplot_title="100% Stacked Bar Chart: Job Satisfaction by Education"
    )
    fig_100_stacked.savefig(os.path.join(output_dir, "stacked_bar_chart.png"), bbox_inches='tight')
    plt.close(fig_100_stacked)

    # 5. Chi-Squared Test and Cramér's V
    print("\n5. Performing Chi-Squared Test for Education vs. JobSatisfaction:")
    chi2_test = ChiSquaredTest()
    chi2_test.fit(x=data['Education'], y=data['JobSatisfaction'])
    print(chi2_test.summary)

    # 6. Heatmap of Co-occurrences (using the enhanced HeatmapPlot)
    print("\n6. Displaying heatmap of the crosstab with annotations...")
    heatmap_plotter = HeatmapPlot(theme='classic_professional_light')
    fig_heatmap = heatmap_plotter.plot(crosstab_df, annot=True, subplot_title="Heatmap of Co-occurrences")
    fig_heatmap.savefig(os.path.join(output_dir, "crosstab_heatmap.png"), bbox_inches='tight')
    plt.close(fig_heatmap)
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    categorical_analysis_full_workflow()