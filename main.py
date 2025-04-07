"""
Main script (main.py) for executing the preprocessing, analysis, and MySQL integration pipeline.

Features:
  - **Data Reading**: Loads data from a CSV file or a MySQL table, based on configuration.
  - **Preprocessing**: 
      - Data type conversion (e.g., forced categorical columns).
      - Handling of missing values (removal or imputation using mean/mode).
      - Outlier detection (using IQR, Z-score, or clustering methods).
  - **Exploratory Analysis**: Generates a statistical report and distribution plots (density and Q-Q plot).
  - **MySQL Integration**: Exports processed data to a new table/schema in MySQL.
"""

import pandas as pd
import numpy as np
import logging
from data_processor import DataProcessor
from data_analyzer import DataAnalyzer
import integration as mySQL

# Configure logging to track execution with timestamps and severity levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    
    # =========================================================================
    # PROJECT SETTINGS
    # =========================================================================
    
    # Adjustable parameters to control the pipeline:
    # - remove_missing: Remove rows with missing data if True; impute values if False.
    # - remove_outliers: Remove detected outliers if True.
    # - impute_method: "mean" for mean imputation (numerical) or "mode" for mode imputation (categorical and numerical).
    # - outlier_method: Defines the outlier detection strategy ("IQR", "zscore" or "clustering").
    # - force_categorical: Columns to be converted to 'category' type (improves performance on large datasets).
    # - mysql_import/mysql_export: Control MySQL integration.
    
    config = {
        "remove_missing": False,         # True: remove missing; False: impute values
        "remove_outliers": True,         # True: remove detected outliers
        "impute_method": "mode",         # Imputation method ("mean" or "mode")
        "outlier_method": "clustering",  # Outlier detection method
        "zscore_threshold": 3,           # Threshold for Z-score (used if outlier_method = "zscore")
        "force_categorical": ["customer_id", "product_id", "category_id", "review_score"], 

        # MySQL integration settings
        "mysql_import": False,           # Import data from MySQL? If False, reads from CSV.
        "mysql_export": True,            # Export processed data to MySQL?
        "mysql_host": "localhost",       # MySQL server address
        "mysql_user": " ",            # MySQL username
        "mysql_password": " ",      # User password
        "mysql_database": " ",     # Database to import from
        "mysql_table_import": " ",   # Source table in MySQL
        "mysql_table_export": " ", # Destination table for export
        "mysql_schema_export": " " # Destination schema (created if it does not exist)
    }

    # =========================================================================
    # DATA READING (CSV or MySQL)
    # =========================================================================
    
    try:
        if config["mysql_import"]:
            # Import data from MySQL using the specified credentials and table
            df = mySQL.import_data(
                host=config["mysql_host"],
                user=config["mysql_user"],
                password=config["mysql_password"],
                database=config["mysql_database"],
                table=config["mysql_table_import"]
            )
            logging.info("Data successfully imported from MySQL.")
        else:
            # Load data from a local CSV (e.g., synthetic_online_retail_data.csv)
            df = pd.read_csv("synthetic_online_retail_data.csv")
            logging.info("Data successfully read from the CSV file.")
    except Exception as e:
        logging.error("Data reading failed: %s", e)
        return  # Stop execution in case of error

    # =========================================================================
    # DATA PROCESSING (Cleaning and Transformation)
    # =========================================================================
    
    try:
        # Initialize the processor with the defined configuration
        processor = DataProcessor(
            remove_missing=config["remove_missing"],
            remove_outliers=config["remove_outliers"],
            impute_method=config["impute_method"],
            force_categorical=config["force_categorical"],
            outlier_method=config["outlier_method"],
            zscore_threshold=config.get("zscore_threshold", 3)
        )
        # Execute the pipeline: correct types, handle missing/outliers and return a clean DataFrame
        df_clean = processor.process_data(df)
        logging.info("Data processed successfully.")
    except Exception as e:
        logging.error("Error during processing: %s", e)
        return

    # =========================================================================
    # LOCAL SAVING OF RESULTS (CSV)
    # =========================================================================
    
    try:
        # Save the processed DataFrame and metadata (removed data)
        df_clean.to_csv("clean_data.csv", index=False)                # Save the clean data DataFrame                     
        processor.missing_df.to_csv("missing_data.csv", index=False)    # Save the rows with missing values
        processor.outliers_df.to_csv("outliers_data.csv", index=False)  # Save the rows with outliers
        logging.info("CSVs saved: clean_data.csv, missing_data.csv, outliers_data.csv") 
    except Exception as e:
        logging.error("Failed to save CSVs: %s", e)

    # =========================================================================
    # EXPLORATORY ANALYSIS (Report and Plots)
    # =========================================================================
    
    try:
        analyzer = DataAnalyzer()
        # Generate statistical report (outlier counts, descriptive statistics)
        report = analyzer.analyze(df_clean, processor.outliers_count)
        
        # Save report as a .txt file
        with open("report.txt", "w") as f:
            for col, info in report.items():
                f.write(f"\n=== {col} ===\n")
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")
        logging.info("Report 'report.txt' generated.")
        
        # Generate distribution plots for numerical columns
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            analyzer.analyze_distribution(df_clean[col], col)  # Saves PNGs in the current directory
        logging.info("Distribution plots generated.")
    except Exception as e:
        logging.error("Error during analysis: %s", e)

    # =========================================================================
    # EXPORT TO MYSQL (If enabled)
    # =========================================================================
    
    if config["mysql_export"]:
        try:
            # Preprocessing for MySQL compatibility:
            # - Replace NaN with None (MySQL does not accept NaN)
            # - Convert categorical columns to strings (avoids type errors)
            df_clean = df_clean.replace({np.nan: None})
            df_clean = df_clean.astype({col: 'object' for col in df_clean.select_dtypes(['category']).columns})

            # Export to the specified table
            mySQL.export_data(
                host=config["mysql_host"],
                user=config["mysql_user"],
                password=config["mysql_password"],
                table=config["mysql_table_export"],
                schema=config["mysql_schema_export"],
                df=df_clean
            )
            logging.info(f"Data exported to table {config['mysql_table_export']}.")
        except Exception as e:
            logging.error("MySQL export failed: %s", e)

if __name__ == "__main__":
    main()
