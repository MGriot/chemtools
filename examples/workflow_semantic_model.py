import pandas as pd
import matplotlib.pyplot as plt
from chemtools.semantic.model import HybridSemanticModel
from chemtools.utils.sql_builder import SqlModelBuilder
from chemtools.plots.basic.bar import BarPlot
import os

def run_semantic_model_workflow():
    """
    An example script that demonstrates the full workflow of the HybridSemanticModel
    including data loading, calculations, and plotting.
    """
    print("--- HybridSemanticModel Workflow with Chemtools ---")

    output_dir = "doc/img/plots/semantic_model"
    os.makedirs(output_dir, exist_ok=True)

    themes = ["classic_professional_light", "classic_professional_dark"]

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
    model = SqlModelBuilder.parse_sql_schema(sql_script, engine='pandas')

    print("\n--- 2. Loading Actual Data ---")
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
        'CustomerID': [1, 1, 2, 2],
        'ProductID': [101, 102, 101, 102],
        'Amount': [100, 200, 150, 300]
    })

    model.add_table('Products', df_prod)
    model.add_table('Customers', df_cust)
    model.add_table('Sales', df_sales)

    model.add_measure('Total Revenue', 'Sales', 'Amount', 'sum')

    print("\n--- 3. Visualization ---")
    # model.visualize() # This might be an interactive visualization, skip for automated saving

    print("\n--- 4. Calculation (Pandas) ---")
    res_pandas = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
    print(f"Revenue (North): {res_pandas} (Expected: 300)")

    print("\n--- 5. Engine Switch -> Polars ---")
    model.switch_backend('polars')

    print("\n--- 6. Calculation (Polars) ---")
    res_polars = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
    print(f"Revenue (North): {res_polars} (Expected: 300)")

    print("\n--- 7. Expanding Test with Plotting ---")
    model.switch_backend('pandas')

    df_merged = pd.merge(model.tables['Sales'], model.tables['Customers'], on='CustomerID')
    df_region_revenue = df_merged.groupby('Region')['Amount'].sum().reset_index()

    print("\nAggregated data for plotting:")
    print(df_region_revenue)

    for theme in themes:
        try:
            plotter = BarPlot(theme=theme)
            fig = plotter.plot(data=df_region_revenue, x='Region', y='Amount', subplot_title=f"Total Revenue by Region ({theme})")
            filename = f"total_revenue_by_region_{theme}.png"
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close(fig)
            print(f"  - Successfully generated and saved plot for theme: {theme}")
        except Exception as e:
            print(f"  - An error occurred during plotting for theme {theme}: {e}")

    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    run_semantic_model_workflow()
