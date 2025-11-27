import pandas as pd
import polars as pl
import re
from typing import Dict, Any, List, Literal, Optional
from chemtools.semantic.model import HybridSemanticModel

class SqlModelBuilder:
    """
    Static Utility to parse SQL CREATE statements and populate the HybridSemanticModel.
    """
    
    @staticmethod
    def parse_sql_schema(sql_text: str, engine='pandas') -> HybridSemanticModel:
        """
        Parses SQL CREATE TABLE statements to build a Semantic Model.
        
        Logic:
        1. Identify TABLE names.
        2. Identify Columns to build empty DataFrames (Schema).
        3. Identify FOREIGN KEYs to build Relationships.
        """
        model = HybridSemanticModel(engine=engine)
        
        # Normalize string
        sql_text = sql_text.replace('\n', ' ')
        
        # Regex Patterns
        # Capture: CREATE TABLE [IF NOT EXISTS] TableName ( ... )
        table_pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?['`]?(\w+)['`]?\s*\((.*?)\);", re.IGNORECASE)
        
        # Capture: FOREIGN KEY (Col) REFERENCES Parent(ParentCol)
        # Using a simpler regex to avoid escaping issues.
        fk_pattern = re.compile(r"FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)", re.IGNORECASE)

        # Capture: Column definitions (Simplified: Name Type ...)
        col_pattern = re.compile(r"['`]?(\w+)['`]?\s+\w+")

        # Store FKs to process after all tables exist
        pending_relationships = []

        # 1. Parse Tables
        for match in table_pattern.finditer(sql_text):
            table_name = match.group(1)
            body = match.group(2)
            
            # Extract columns to create empty schema
            # We split by comma, but respect parentheses (simplistic approach)
            columns = []
            for segment in body.split(','):
                segment = segment.strip()
                # Skip constraint lines for column extraction
                if segment.upper().startswith(('CONSTRAINT', 'PRIMARY', 'FOREIGN', 'KEY', 'UNIQUE')):
                    continue
                
                col_match = col_pattern.match(segment)
                if col_match:
                    columns.append(col_match.group(1))
            
            # Create Empty DataFrame with Schema
            if engine == 'pandas':
                df_schema = pd.DataFrame(columns=columns)
            else:
                # Polars requires schema dict for empty creation usually, 
                # but from_pandas is safer for dynamic schema inference without types
                df_schema = pl.from_pandas(pd.DataFrame(columns=columns))
            
            model.add_table(table_name, df_schema)
            
            # 2. Parse Foreign Keys in the body
            for fk_match in fk_pattern.finditer(body):
                local_col = fk_match.group(1)
                parent_table = fk_match.group(2)
                parent_col = fk_match.group(3)
                
                # Store: Source=Parent(Dim), Target=Child(Fact), Key=ParentCol
                # Note: In SQL, FK is on Child. In Logic, Parent filters Child.
                pending_relationships.append({
                    'parent': parent_table, 
                    'child': table_name, 
                    'on': parent_col, # Assuming join keys match or using parent's key
                    'role': 'default' 
                })

        # 3. Apply Relationships
        for rel in pending_relationships:
            try:
                model.add_relationship(
                    parent=rel['parent'], 
                    child=rel['child'], 
                    on=rel['on'],
                    role=rel['role']
                )
            except ValueError as e:
                print(f"Error linking {rel['parent']} -> {rel['child']}: {e}")

        return model
