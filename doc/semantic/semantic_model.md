# Semantic Model

The `chemtools.semantic` module introduces a powerful `HybridSemanticModel` for building BI-style semantic layers that enable cross-filtering and measure calculation across multiple related data tables. This model is designed for flexibility, supporting both Pandas and Polars backends, and leverages graph theory to manage complex data relationships.

## `HybridSemanticModel`

The `HybridSemanticModel` acts as a central hub for defining data relationships, business logic (measures), and enabling dynamic queries across a schema of interconnected tables.

### Key Features:
-   **Backend Agnostic**: Seamlessly switch between `pandas` and `polars` execution engines.
-   **Graph-based Relationships**: Utilizes NetworkX to model table relationships, supporting Star/Snowflake schemas and multiple relationships between tables (e.g., different date roles).
-   **Measure Definition**: Define aggregated metrics (e.g., 'sum', 'mean') on specific columns within tables.
-   **Cross-Filtering**: Automatically propagates filters across related tables based on the defined graph, enabling complex analytical queries.
-   **Visualization**: Generate a visual representation of the defined schema topology.

## Usage

Here's an example demonstrating how to set up, populate, and query a `HybridSemanticModel`:

```python
import pandas as pd
from chemtools.semantic.model import HybridSemanticModel
from chemtools.utils.sql_builder import SqlModelBuilder

# 1. Define your SQL Schema (or define tables and relationships manually)
sql_script = """
CREATE TABLE Products (
    ProductID INT PRIMARY KEY,
    Category VARCHAR(50)
);

CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    Region VARCHAR(50)
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

# 2. Build the model from SQL schema (or add tables and relationships directly)
model = SqlModelBuilder.parse_sql_schema(sql_script, engine='pandas')

# 3. Load actual data into the model
df_prod = pd.DataFrame({'ProductID': [101, 102], 'Category': ['Electronics', 'Furniture']})
df_cust = pd.DataFrame({'CustomerID': [1, 2], 'Region': ['North', 'South']})
df_sales = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4],
    'CustomerID': [1, 1, 2, 2],
    'ProductID': [101, 102, 101, 102],
    'Amount': [100, 200, 150, 300]
})

model.add_table('Products', df_prod)
model.add_table('Customers', df_cust)
model.add_table('Sales', df_sales)

# 4. Define measures
model.add_measure('Total Revenue', 'Sales', 'Amount', 'sum')

# 5. Visualize the schema
model.visualize(title="Sales Data Model Schema")

# 6. Perform calculations with cross-filtering
# Calculate total revenue for 'North' region customers
north_revenue = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
print(f"Total Revenue for North Region: {north_revenue}")

# Switch backend to Polars and re-calculate
model.switch_backend('polars')
polars_north_revenue = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
print(f"Total Revenue for North Region (Polars): {polars_north_revenue}")

```

## API Reference

### `HybridSemanticModel` Class

```python
class HybridSemanticModel:
    def __init__(self, engine: Literal['pandas', 'polars'] = 'pandas')
    def switch_backend(self, target_engine: str)
    def add_table(self, name: str, df: Any)
    def add_relationship(self, parent: str, child: str, on: str, role: str = 'default')
    def add_measure(self, name: str, table: str, column: str, agg: str = 'sum')
    def calculate(self, measure_name: str, filters: Dict[str, Dict[str, Any]] = None, active_role: str = 'default')
    def visualize(self, title="Semantic Model Schema")
```

#### Parameters:
-   `engine` (`Literal['pandas', 'polars']`): The backend data manipulation library to use ('pandas' or 'polars'). Defaults to 'pandas'.
-   `name` (`str`): The name of the table or measure.
-   `df` (`Any`): A Pandas or Polars DataFrame.
-   `parent` (`str`): The name of the parent table in a relationship (dimension table).
-   `child` (`str`): The name of the child table in a relationship (fact table).
-   `on` (`str`): The column name on which the relationship is based (join key).
-   `role` (`str`): A descriptive role for the relationship (e.g., 'OrderDate', 'ShipDate'). Defaults to 'default'.
-   `table` (`str`): The table where the measure's column resides.
-   `column` (`str`): The column on which the measure's aggregation is performed.
-   `agg` (`str`): The aggregation function ('sum', 'mean', 'count', 'max', 'min').
-   `measure_name` (`str`): The name of the measure to calculate.
-   `filters` (`Dict[str, Dict[str, Any]]`): A dictionary of filters to apply, e.g., `{'Customers': {'Region': 'North'}}`.
-   `active_role` (`str`): The preferred role for resolving ambiguous relationships. Defaults to 'default'.
-   `title` (`str`): Title for the visualization plot.

### `SqlModelBuilder` (Utility Class)

For convenience, the `chemtools.utils.sql_builder.SqlModelBuilder` class can be used to parse SQL `CREATE TABLE` statements and automatically set up the `HybridSemanticModel` with tables and relationships.

```python
class SqlModelBuilder:
    @staticmethod
    def parse_sql_schema(sql_text: str, engine='pandas') -> HybridSemanticModel
```
See `chemtools/utils/sql_builder.py` for more details.

## Citing Sources
This model is inspired by concepts found in modern Business Intelligence (BI) tools and data warehousing principles.
