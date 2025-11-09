import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chemtools.exploration.EDA import ExploratoryDataAnalysis

# --- 1. Generate Dummy Data ---
print("---" + " 1. Generating Dummy Data " + "---")
# Create a dataframe with a mix of numerical and categorical data
data = pd.DataFrame({
    'category_A': np.random.choice(['A', 'B', 'C'], 100),
    'category_B': np.random.choice(['X', 'Y', 'Z'], 100),
    'value1': np.random.rand(100) * 100,
    'value2': np.random.rand(100) * 50,
    'value3': np.random.rand(100) * 20,
    'timestamp': pd.to_datetime(np.arange(100), unit='D', origin='2023-01-01')
})
# Add some missing values to test missing value analysis
data.loc[5:10, 'value1'] = np.nan
data.loc[15:20, 'category_A'] = np.nan
print("Dummy data created successfully.\n")


# --- 2. Instantiate EDA class ---
eda = ExploratoryDataAnalysis(data)
print("---" + " 2. ExploratoryDataAnalysis class instantiated " + "---" + "\n")


# --- 3. Test Variable Classification and Summaries (New & Improved) ---
print("---" + " 3.1. Test `classify_variables` " + "---")
numerical_cols, categorical_cols = eda.classify_variables()
print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)
print("-" * 30)

print("\n---" + " 3.2. Test `get_univariate_summary` (Numerical) " + "---")
univariate_summary = eda.get_univariate_summary()
print(univariate_summary)
print("-" * 30)

print("\n---" + " 3.3. Test `get_categorical_summary` (New Method) " + "---")
categorical_summary = eda.get_categorical_summary()
print(categorical_summary)
print("-" * 30)

print("\n---" + " 3.4. Test Missing Values Analysis " + "---")
missing_summary = eda.get_missing_values_summary()
print("Missing Values Summary:")
print(missing_summary)
print("Plotting missing values matrix...")
eda.plot_missing_values()
plt.show()
print("-" * 30)


# --- 4. Test Bivariate and Mixed-Type Analysis (New & Improved) ---
print("\n---" + " 4.1. Test Numerical vs. Numerical Analysis " + "---")
print("Correlation Matrix:")
correlation_matrix = eda.get_correlation_matrix()
print(correlation_matrix)
print("Plotting heatmap of correlation matrix...")
heatmap_plotter = eda.heatmap_plotter()
heatmap_plotter.plot(correlation_matrix, title="Correlation Heatmap")
plt.show()
print("-" * 30)

print("\n---" + " 4.2. Test Categorical vs. Categorical Analysis (New Method) " + "---")
print("Crosstab between 'category_A' and 'category_B':")
crosstab = eda.get_crosstab('category_A', 'category_B')
print(crosstab)
print("Visualizing the crosstab...")
# Visualize the crosstab using the new themed method
bar_plotter = eda.barchart_plotter()
bar_plotter.plot_crosstab(crosstab, stacked=True, subplot_title="Crosstab: category_A vs. category_B")
plt.show()
print("-" * 30)

print("\n---" + " 4.3. Test Numerical vs. Categorical Analysis (New Methods) " + "---")
print("Summary of 'value1' grouped by 'category_A':")
num_by_cat_summary = eda.get_numerical_by_categorical_summary('value1', 'category_A')
print(num_by_cat_summary)

print("\nPlotting 'value1' by 'category_A' using the new high-level method...")
print("Generating Box Plot...")
eda.plot_numerical_by_categorical('value1', 'category_A', plot_type='box')
plt.show()

print("Generating Violin Plot (using plotly)...")
fig = eda.plot_numerical_by_categorical('value2', 'category_B', plot_type='violin', plotter_kwargs={'library': 'plotly'})
fig.show()
print("-" * 30)


# --- 5. Test New Plotter Factories ---
print("\n---" + " 5. Test New Plotter Factories " + "---")
print("Testing Pie Chart Plotter:")
pie_plotter = eda.pie_chart_plotter()
pie_data = data['category_A'].value_counts().reset_index()
pie_data.columns = ['category_A', 'count']
pie_plotter.plot(pie_data, names_column='category_A', values_column='count', title="Pie Chart of category_A")
plt.show()

print("\n---" + " All new EDA method tests are complete! " + "---")
