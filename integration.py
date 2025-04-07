"""
Module for MySQL integration using SQLAlchemy, allowing import and export of DataFrames.

Features:
  - **Data Import**: Loads data from a MySQL table into a pandas DataFrame.
  - **Data Export**: Writes a DataFrame to a MySQL table, creating the schema if necessary.
  - **Type Mapping**: Converts pandas types to MySQL-compatible types (e.g., integer, float, text).
"""

import pandas as pd
from sqlalchemy import create_engine, text, Integer, Float, Text
import logging
import numpy as np

def import_data(host, user, password, database, table):
    """
    Imports data from a MySQL table into a DataFrame.
    
    Args:
        host (str): MySQL server address (e.g., 'localhost').
        user (str): MySQL username.
        password (str): User password.
        database (str): Name of the source database.
        table (str): Name of the table to be read.
        
    Returns:
        pd.DataFrame: DataFrame with the table data. Returns None in case of error.
    """
    try:
        # Create the connection engine using mysqlconnector as the driver
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
        df = pd.read_sql_table(table, engine)
        logging.info(f"Data from table {table} imported successfully.")
        return df
    except Exception as e:
        logging.error("Import failed: %s", e)
        return None

def export_data(host, user, password, table, schema, df):
    """
    Exports a DataFrame to a MySQL table, creating the schema if necessary.
    
    Steps:
        1. Replace NaN with None (compatible with MySQL).
        2. Convert categorical columns to strings.
        3. Create the schema (database) if it does not exist.
        4. Map pandas types to SQLAlchemy/MySQL types.
        5. Write the data to the specified table.
    
    Args:
        host (str): MySQL server address.
        user (str): Username.
        password (str): User password.
        table (str): Name of the destination table.
        schema (str): Name of the destination schema (database).
        df (pd.DataFrame): DataFrame to be exported.
    """
    try:
        # Preprocessing for MySQL compatibility
        df = df.replace({np.nan: None})  # MySQL does not recognize NaN
        df = df.astype({col: 'object' for col in df.select_dtypes(['category']).columns})  # Convert categories to strings
        
        # Create the connection engine
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{schema}")
        
        # Create the schema if it does not exist
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {schema}"))  # Execute SQL command
            conn.commit()
        
        # Map pandas types to SQLAlchemy/MySQL types
        dtype_mapping = {col: _map_pandas_to_sql(df[col]) for col in df.columns}
        
        # Export the DataFrame to the table
        df.to_sql(
            name=table,
            con=engine,
            if_exists='replace',  # Replace the table if it exists
            index=False,          # Do not write DataFrame index
            dtype=dtype_mapping   # Mapped types for each column
        )
        logging.info(f"Data exported to {schema}.{table} successfully.")
    except Exception as e:
        logging.error("Error during export: %s", e)

def _map_pandas_to_sql(series):
    """
    Maps pandas data types to SQLAlchemy/MySQL types.
    
    Rules:
        - Integer columns -> Integer()
        - Float columns -> Float()
        - All other types (strings, categories) -> Text()
        
    Args:
        series (pd.Series): Series to be mapped.
        
    Returns:
        SQLAlchemy Type: Corresponding type for the column in MySQL.
    """
    if pd.api.types.is_integer_dtype(series):
        return Integer()  # e.g., INT in MySQL
    elif pd.api.types.is_float_dtype(series):
        return Float()     # e.g., FLOAT in MySQL
    else:
        return Text()      # e.g., TEXT/VARCHAR in MySQL
