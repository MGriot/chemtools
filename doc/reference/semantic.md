# Semantic Modeling Module Reference (`chemtools.semantic`)

The `chemtools.semantic` module provides the `HybridSemanticModel` for building BI-style semantic layers, enabling cross-filtering and measure calculation across multiple related data tables. This model supports both Pandas and Polars backends and uses graph theory to manage complex data relationships.

---

## `HybridSemanticModel` Class

A central hub for defining data relationships, business logic (measures), and enabling dynamic queries across a schema of interconnected tables.

### `HybridSemanticModel(engine: Literal['pandas', 'polars'] = 'pandas')`

*   **Parameters:**
    *   `engine` (`Literal['pandas', 'polars']`, optional): The backend data manipulation library to use. Can be `'pandas'` or `'polars'`. Defaults to `'pandas'`.

### Methods

*   **`switch_backend(self, target_engine: str)`**
    *   Switches the underlying data processing backend between 'pandas' and 'polars'.
    *   **Parameters:** `target_engine` (`str`): The target engine ('pandas' or 'polars').

*   **`add_table(self, name: str, df: Any)`**
    *   Adds a DataFrame (Pandas or Polars) to the semantic model as a named table.
    *   **Parameters:**
        *   `name` (`str`): The name to assign to the table.
        *   `df` (`Any`): The Pandas or Polars DataFrame.

*   **`add_relationship(self, parent: str, child: str, on: str, role: str = 'default')`**
    *   Defines a one-to-many relationship between two tables.
    *   **Parameters:**
        *   `parent` (`str`): The name of the parent table (dimension table).
        *   `child` (`str`): The name of the child table (fact table).
        *   `on` (`str`): The common column name on which the relationship is based (join key).
        *   `role` (`str`, optional): A descriptive role for the relationship, useful for multiple relationships between the same tables. Defaults to `'default'`.

*   **`add_measure(self, name: str, table: str, column: str, agg: str = 'sum')`**
    *   Defines an aggregated measure within the semantic model.
    *   **Parameters:**
        *   `name` (`str`): The name of the measure (e.g., 'Total Revenue').
        *   `table` (`str`): The name of the table where the measure's `column` resides.
        *   `column` (`str`): The column on which the aggregation is performed.
        *   `agg` (`str`, optional): The aggregation function to apply (e.g., `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`). Defaults to `'sum'`.

*   **`calculate(self, measure_name: str, filters: Dict[str, Dict[str, Any]] = None, active_role: str = 'default') -> Any`**
    *   Calculates the value of a defined measure, optionally applying filters across related tables.
    *   **Parameters:**
        *   `measure_name` (`str`): The name of the measure to calculate.
        *   `filters` (`Dict[str, Dict[str, Any]]`, optional): A dictionary of filters to apply, where keys are table names and values are dictionaries of column-value pairs (e.g., `{'Customers': {'Region': 'North'}}`).
        *   `active_role` (`str`, optional): The role to use for resolving ambiguous relationships (if multiple exist). Defaults to `'default'`.
    *   **Returns:** `Any`: The calculated value of the measure.

*   **`visualize(self, title="Semantic Model Schema")`**
    *   Generates a visual representation of the defined schema topology using the NetworkX graph.
    *   **Parameters:** `title` (`str`, optional): Title for the visualization plot.

### Usage Example

```python
import pandas as pd
from chemtools.semantic.model import HybridSemanticModel
from chemtools.utils.sql_builder import SqlModelBuilder

# Define a simple SQL schema
sql_script = """
CREATE TABLE Products ( ProductID INT PRIMARY KEY, Category VARCHAR(50) );
CREATE TABLE Customers ( CustomerID INT PRIMARY KEY, Region VARCHAR(50) );
CREATE TABLE Sales (
    TransactionID INT PRIMARY KEY, CustomerID INT, ProductID INT, Amount DECIMAL(10,2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
);
"""
# Build model from SQL schema
model = SqlModelBuilder.parse_sql_schema(sql_script, engine='pandas')

# Load sample data
df_prod = pd.DataFrame({'ProductID': [101, 102], 'Category': ['Electronics', 'Furniture']})
df_cust = pd.DataFrame({'CustomerID': [1, 2], 'Region': ['North', 'South']})
df_sales = pd.DataFrame({'TransactionID': [1, 2, 3, 4], 'CustomerID': [1, 1, 2, 2], 'ProductID': [101, 102, 101, 102], 'Amount': [100, 200, 150, 300]})

model.add_table('Products', df_prod)
model.add_table('Customers', df_cust)
model.add_table('Sales', df_sales)

model.add_measure('Total Revenue', 'Sales', 'Amount', 'sum')

north_revenue = model.calculate('Total Revenue', filters={'Customers': {'Region': 'North'}})
print(f"Total Revenue for North Region: {north_revenue}")
```

---

<h2><code>SqlModelBuilder</code> Utility Class</h2>

A utility class to parse SQL <code>CREATE TABLE</code> statements and automatically set up the <code>HybridSemanticModel</code> with tables and relationships.

<h3><code>SqlModelBuilder.parse_sql_schema(sql_text: str, engine='pandas') -> HybridSemanticModel</code></h3>

*   <b>Parameters:</b>
    *   <code>sql_text</code> (<code>str</code>): A string containing SQL <code>CREATE TABLE</code> statements.
    *   <code>engine</code> (<code>str</code>, optional): The backend engine for the <code>HybridSemanticModel</code> ('pandas' or 'polars').
*   <b>Returns:</b> <code>HybridSemanticModel</code>: An initialized semantic model with tables and relationships defined from the SQL schema.
