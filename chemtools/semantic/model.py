import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import re
from typing import Dict, Any, List, Literal, Optional

class HybridSemanticModel:
    """
    A BI-style Semantic Engine that allows cross-filtering between DataFrames. 
    
    Features:
    - Backend agnostic: Switch between 'pandas' and 'polars' on the fly.
    - Graph-based: Uses NetworkX to model relationships (Star/Snowflake schemas).
    - Multi-Relational: Supports multiple active relationships between tables.
    - Visualization: Renders the schema topology.
    """

    def __init__(self, engine: Literal['pandas', 'polars'] = 'pandas'):
        self.engine = engine
        self.tables = {}
        # MultiDiGraph allows multiple edges between nodes (e.g., OrderDate and ShipDate)
        self.graph = nx.MultiDiGraph()
        self.measures = {}
        
        # Mapping aggregation strings to library specific methods
        self._agg_map = {
            'sum': {'pandas': 'sum', 'polars': 'sum'},
            'mean': {'pandas': 'mean', 'polars': 'mean'},
            'count': {'pandas': 'count', 'polars': 'count'},
            'max': {'pandas': 'max', 'polars': 'max'},
            'min': {'pandas': 'min', 'polars': 'min'}
        }

    # ------------------------------------------------------------------
    # Data Management & Engine Switching
    # ------------------------------------------------------------------
    
    def switch_backend(self, target_engine: str):
        """Hot-swaps the backend library (Pandas <-> Polars)."""
        if target_engine == self.engine:
            return
            
        print(f"[System] Switching Engine: {self.engine} -> {target_engine}...")
        
        for name, df in self.tables.items():
            if target_engine == 'polars' and self.engine == 'pandas':
                self.tables[name] = pl.from_pandas(df)
            elif target_engine == 'pandas' and self.engine == 'polars':
                self.tables[name] = df.to_pandas()
                
        self.engine = target_engine
        print(f"[System] Successfully converted {len(self.tables)} tables.")

    def add_table(self, name: str, df: Any):
        """Adds a table, converting it to the current engine format if necessary."""
        if self.engine == 'polars' and isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        elif self.engine == 'pandas' and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            
        self.tables[name] = df
        self.graph.add_node(name)
        print(f"[Model] Added table: '{name}'")

    def add_relationship(self, parent: str, child: str, on: str, role: str = 'default'):
        """
        Defines a relationship: Parent (Dimension) filters Child (Fact).
        :param role: A name for this specific link (useful for multiple date keys).
        """
        if parent not in self.tables or child not in self.tables:
            raise ValueError("Both tables must be added before linking.")
            
        self.graph.add_edge(parent, child, key=role, join_key=on)
        print(f"[Model] Linked: {parent} --[{role}]--> {child} (on {on})")

    def add_measure(self, name: str, table: str, column: str, agg: str = 'sum'):
        """Defines a calculation logic."""
        if agg not in self._agg_map:
            raise ValueError(f"Aggregation '{agg}' not supported.")
        self.measures[name] = {'table': table, 'col': column, 'agg': agg}

    # ------------------------------------------------------------------
    # Internal Abstractions
    # ------------------------------------------------------------------

    def _get_unique_keys(self, df: Any, col: str):
        if self.engine == 'pandas':
            return df[col].unique()
        else:
            return df.select(pl.col(col)).unique()

    def _filter_by_list(self, df: Any, col: str, values: Any):
        if self.engine == 'pandas':
            return df[df[col].isin(values)]
        else:
            if isinstance(values, pl.DataFrame): values = values.to_series()
            return df.filter(pl.col(col).is_in(values))

    def _filter_by_val(self, df: Any, col: str, value: Any):
        if self.engine == 'pandas':
            return df[df[col] == value]
        else:
            return df.filter(pl.col(col) == value)

    # ------------------------------------------------------------------
    # The Calculation Core
    # ------------------------------------------------------------------

    def calculate(self, measure_name: str, filters: Dict[str, Dict[str, Any]] = None, active_role: str = 'default'):
        """
        Calculates a measure with cross-filtering applied.
        :param active_role: If multiple paths exist, prefer edges with this key (e.g. 'ShipDate').
        """
        if measure_name not in self.measures:
            raise ValueError(f"Measure {measure_name} not found")
            
        m = self.measures[measure_name]
        target_table = m['table']
        current_df = self.tables[target_table]

        if filters:
            for source_table, criteria in filters.items():
                col_name, val = list(criteria.items())[0]

                if source_table == target_table:
                    current_df = self._filter_by_val(current_df, col_name, val)
                else:
                    try:
                        path = nx.shortest_path(self.graph, source=source_table, target=target_table)
                        
                        # 1. Filter Source Dimension
                        source_df = self.tables[source_table]
                        filtered_source = self._filter_by_val(source_df, col_name, val)
                        
                        keys_to_keep = None

                        # 2. Traverse Graph
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            
                            # Edge Selection Logic
                            valid_edges = self.graph[u][v]
                            selected_edge = None
                            
                            if active_role in valid_edges:
                                selected_edge = valid_edges[active_role]
                            elif 'default' in valid_edges:
                                selected_edge = valid_edges['default']
                            else:
                                first_key = list(valid_edges.keys())[0]
                                selected_edge = valid_edges[first_key]
                                
                            join_key = selected_edge['join_key']
                            
                            # Propagate Keys
                            if i == 0:
                                keys_to_keep = self._get_unique_keys(filtered_source, join_key)
                            else:
                                # Logic for intermediate tables (Snowflake) would go here
                                # For now, assuming star schema or simple propagation
                                pass

                        # 3. Filter Target Fact
                        final_edge_key = selected_edge['join_key']
                        current_df = self._filter_by_list(current_df, final_edge_key, keys_to_keep)

                    except nx.NetworkXNoPath:
                        print(f"Warning: No path from {source_table} to {target_table}")

        # Aggregation
        col = m['col']
        agg_type = m['agg']
        
        if self.engine == 'pandas':
            if current_df.empty: return 0
            return getattr(current_df[col], agg_type)()
        else:
            if current_df.height == 0: return 0
            expr = getattr(pl.col(col), agg_type)()
            return current_df.select(expr).item()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, title="Semantic Model Schema"):
        """Draws the entity relationship diagram."""
        if not self.graph.nodes:
            print("Graph is empty.")
            return

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, k=0.8, seed=42)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=2500, node_color='#E3F2FD', edgecolors='#1565C0')
        nx.draw_networkx_labels(self.graph, pos, font_weight='bold')
        
        # Edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='#90A4AE', arrowstyle='-|>', arrowsize=20)
        
        # Edge Labels (Join Keys + Roles)
        edge_labels = {}
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            label_text = f"{data['join_key']}\n({key})" if key != 'default' else data['join_key']
            edge_labels[(u, v)] = label_text
            
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()