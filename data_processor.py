"""
Module for data preprocessing, including type conversion, handling of missing values, and outlier detection.

Features:
  - **Type Conversion**: Forces categorical columns, converts to datetime and numeric.
  - **Missing Value Handling**: Removes rows with missing values or imputes using mean/mode.
  - **Outlier Detection**: Uses methods such as IQR, Z-score, or DBSCAN (clustering) to identify outliers.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

# Attempt to import DBSCAN (optional dependency for clustering-based outlier detection)
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None
    logging.warning("Scikit-learn is not installed. The 'clustering' method for outlier detection will not be available.")

class DataProcessor:
    """
    Class for data preprocessing.

    Attributes:
        remove_missing (bool): Remove rows with missing values if True.
        remove_outliers (bool): Remove detected outliers if True.
        impute_method (str): Imputation method ("mean" for average, "mode" for mode).
        force_categorical (list): List of columns to be converted to 'category'.
        outlier_method (str): Outlier detection method ("IQR", "zscore" or "clustering").
        zscore_threshold (float): Threshold for Z-score (used if outlier_method="zscore").
        missing_df (pd.DataFrame): Stores rows removed due to missing values.
        outliers_df (pd.DataFrame): Stores rows identified as outliers.
        outliers_count (dict): Count of outliers per column.
    """
    
    def __init__(self, remove_missing=True, remove_outliers=True, impute_method="mean", force_categorical=None, outlier_method="IQR", zscore_threshold=3):
        self.remove_missing = remove_missing
        self.remove_outliers = remove_outliers
        self.impute_method = impute_method
        self.force_categorical = force_categorical or []  # Columns to be converted to 'category'
        self.missing_df = pd.DataFrame()  # DataFrame to store rows removed due to missing values
        self.outliers_df = pd.DataFrame()  # DataFrame to store detected outliers
        self.outliers_count = {}  # Count of outliers per column
        self.outlier_method = outlier_method  # Outlier detection method
        self.zscore_threshold = zscore_threshold  # Threshold for Z-score (default: 3)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the complete preprocessing pipeline:
        1. Fix column types.
        2. Handle missing values.
        3. Detect and remove outliers (if configured).
        Returns the processed DataFrame.
        """
        df = self._fix_column_types(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)
        return df

    def _fix_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts column types based on the following rules:
        - Columns in `force_categorical` become 'category'.
        - Attempts to convert to datetime (format '%Y-%m-%d').
        - Converts non-numeric columns to numeric if possible (>=90% success rate).
        - Remaining 'object' columns become 'category'.
        """
        # Force categorical columns
        for col in self.force_categorical:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert to datetime and numeric
        for col in df.columns:
            if col in self.force_categorical:
                continue  # Skip columns already converted to 'category'
            
            # Skip columns already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try to convert to datetime
            try:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='raise')
                continue  # Success: move to next column
            except:
                pass
            
            # Try to convert to numeric (if 90% of values can be converted)
            try:
                series_numeric = pd.to_numeric(df[col], errors='coerce')
                if series_numeric.notna().mean() >= 0.9:
                    df[col] = series_numeric
                    continue
            except Exception as e:
                logging.warning(f"Error converting {col} to numeric: {e}")
            
            # Convert 'object' to 'category' as fallback
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values:
        - If "remove_missing=True", removes rows with missing values.
        - Otherwise, imputes using mean (numeric columns) or mode (others).
        """
        self.missing_df = df[df.isnull().any(axis=1)]  # Save removed data
        
        if self.remove_missing:
            return df.dropna()
        else:
            df_clean = df.copy()
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    # Imputation using mean (numeric) or mode (others)
                    if self.impute_method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df_clean[col] = df_clean[col].fillna(df[col].mean())
                    elif self.impute_method == "mode":
                        df_clean[col] = df_clean[col].fillna(df[col].mode()[0])
            return df_clean

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in numerical columns using configurable methods:
        - **IQR**: Interquartile range (limits: Q1 - 1.5*IQR and Q3 + 1.5*IQR).
        - **Z-score**: Values beyond `zscore_threshold` (default: 3) are outliers.
        - **Clustering**: Uses DBSCAN to identify isolated points (requires scikit-learn).
        If `remove_outliers=True`, removes the identified rows.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns
        clean_df = df.copy()
        all_outlier_indices = set()  # Indices of all rows with outliers
        
        for col in numeric_cols:
            try:
                # IQR method (default)
                if self.outlier_method == "IQR":
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (df[col] < lower) | (df[col] > upper)
                
                # Z-score method
                elif self.outlier_method == "zscore":
                    col_data = df[col].dropna()
                    z_scores = np.abs(stats.zscore(col_data))
                    outlier_mask = z_scores > self.zscore_threshold
                    outlier_indices = col_data.index[outlier_mask]
                    outlier_mask = df.index.isin(outlier_indices)
                
                # Clustering method (DBSCAN)
                elif self.outlier_method == "clustering" and DBSCAN is not None:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    col_values = df[[col]].dropna()
                    scaled = scaler.fit_transform(col_values)
                    clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled)
                    outlier_mask = pd.Series(clustering.labels_, index=col_values.index) == -1
                    temp_mask = pd.Series(False, index=df.index)
                    temp_mask.loc[col_values.index] = outlier_mask
                    outlier_mask = temp_mask
                # Fallback to IQR if method is invalid
                else:
                    logging.warning(f"Method {self.outlier_method} not recognized. Using IQR.")
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (df[col] < lower) | (df[col] > upper)

                # Register outliers
                indices = df.index[outlier_mask].tolist()
                self.outliers_count[col] = len(indices)  # Count per column
                all_outlier_indices.update(indices)  # Accumulate indices

            except Exception as e:
                logging.error(f"Error detecting outliers in {col}: {e}")

        # Remove outliers and update outliers_df
        self.outliers_df = df.loc[list(all_outlier_indices)]
        return clean_df.drop(list(all_outlier_indices)) if self.remove_outliers else clean_df
