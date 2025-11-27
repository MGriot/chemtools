import pandas as pd
import matplotlib.pyplot as plt
from chemtools.semantic.model import HybridSemanticModel
from chemtools.utils.sql_builder import SqlModelBuilder
from chemtools.plots.basic.bar import BarPlot
import os # Import os for directory creation

# 1. Define a complex SQL Schema (String)
sql_script = """
CREATE TABLE Products (
    ProductID INT PRIMARY KEY,
    Category VARCHAR(50),
    Price DECIMAL(10,2)
);

CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    Region VARCHAR(50),
    Segment VARCHAR(50)
);

CREATE TABLE Sales (
    TransactionID INT PRIMARY KEY,
    CustomerID INT,
    ProductID INT,
    Amount DECIMAL(10,2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
);
"""

print("--- 1. Parsing SQL to Build Model ---")
# This automatically creates empty tables and relationships
model = SqlModelBuilder.parse_sql_schema(sql_script, engine='pandas')

print("\n--- 2. Loading Actual Data ---")
# Now we overwrite the empty schema with real data
df_prod = pd.DataFrame({
    'ProductID': [101, 102], 
    'Category': ['Electronics', 'Furniture']
})
df_cust = pd.DataFrame({
    'CustomerID': [1, 2], 
    'Region': ['North', 'South']
})
df_sales = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4],
    'CustomerID': [1, 1, 2, 2],  # Cust 1 (North), Cust 2 (South)
    'ProductID': [101, 102, 101, 102], # 101 (Elec), 102 (Furn)
    'Amount': [100, 200, 150, 300]
})

# Update the tables in the model
model.add_table('Products', df_prod)
model.add_table('Customers', df_cust)
model.add_table('Sales', df_sales)

# Define aggregation logic
model.add_measure('Total Revenue', 'Sales', 'Amount', 'sum')

print("\n--- 3. Visualization ---")
# model.visualize() # This might be an interactive visualization, skip for automated saving

print("\n--- 4. Calculation (Pandas) ---")
# Logic: "Revenue from North Region"
# Path: Customers(North) -> Sales
res_pandas = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
print(f"Revenue (North): {res_pandas} (Expected: 300)")

print("\n--- 5. Engine Switch -> Polars ---")
model.switch_backend('polars')

print("\n--- 6. Calculation (Polars) ---")
res_polars = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
print(f"Revenue (North): {res_polars} (Expected: 300)")

print("\n--- 7. Expanding Test with Plotting ---")
# Switch back to pandas for easier manipulation for this example
model.switch_backend('pandas')

# Create a flat table for plotting
df_merged = pd.merge(model.tables['Sales'], model.tables['Customers'], on='CustomerID')
df_region_revenue = df_merged.groupby('Region')['Amount'].sum().reset_index()

print("\nAggregated data for plotting:")
print(df_region_revenue)

output_dir = "doc/img/examples/semantic_model" # Define output_dir for this script
os.makedirs(output_dir, exist_ok=True) # Create output directory


# Use the BarPlot plotter from chemtools
try:
    plotter = BarPlot(theme='classic_professional_light')
    fig = plotter.plot(data=df_region_revenue, x='Region', y='Amount', title='Total Revenue by Region')
    fig.savefig(os.path.join(output_dir, "total_revenue_by_region.png"), bbox_inches='tight')
    plt.close(fig)
    print("\nSuccessfully generated and displayed the plot.")
except Exception as e:
    print(f"\nAn error occurred during plotting: {e}")
