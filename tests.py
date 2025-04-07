"""
Unit test script to validate the functionalities of the DataProcessor and DataAnalyzer modules.

Tested functionalities:
  - DataProcessor:
    - Automatic correction of column types (numeric, categorical, dates).
    - Detection and removal of outliers using the Z-score method.
  - DataAnalyzer: (to be implemented)
    - Statistical analysis for numerical, categorical, and date columns.
    - Generation of distribution plots.
    - Statistical tests (T-test, ANOVA, Chi-square).

The tests ensure that transformations and analyses work as configured.
"""

import unittest
import pandas as pd
import numpy as np
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """
    Test class for the DataProcessor module.

    Methods:
        setUp(): Prepares test data before each execution.
        test_fix_column_types(): Verifies the conversion of column types.
        test_handle_outliers_zscore(): Tests removal of outliers using the Z-score method.
    """
    
    def setUp(self):
        """
        Sets up a test DataFrame with:
          - Numeric column ('A') containing an outlier (1000).
          - Numeric column ('B') with a missing value (None).
          - Date column ('C') in the 'YYYY-MM-DD' format.
        """
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 1000],          # Numeric column with an outlier
            'B': [10, 20, None, 40],        # Numeric column with a missing value
            'C': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']
        })
        # Explicitly convert column 'C' to datetime
        self.df['C'] = pd.to_datetime(self.df['C'], format='%Y-%m-%d')  
    
    def test_fix_column_types(self):
        """Tests if column types are converted correctly."""
        # Configure the processor to force column 'B' as categorical
        processor = DataProcessor(force_categorical=['B'])
        df_fixed = processor._fix_column_types(self.df.copy())
        
        # Checks:
        self.assertTrue(pd.api.types.is_numeric_dtype(df_fixed['A']), "Column 'A' should be numeric.")
        self.assertEqual(df_fixed['B'].dtype.name, 'category', "Column 'B' should be categorical.")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_fixed['C']), "Column 'C' should be datetime.")
    
    def test_handle_outliers_zscore(self):
        """Tests the removal of outliers using Z-score with a custom threshold."""
        # Configure the processor to remove outliers using Z-score (threshold: 1.7)
        processor = DataProcessor(
            remove_outliers=True,
            outlier_method="zscore",
            zscore_threshold=1.7  # Low threshold to capture the outlier 1000 in column 'A'
        )
        df_no_outliers = processor._handle_outliers(self.df.copy())
        
        # Verify that the outlier (1000) was removed from column 'A'
        self.assertNotIn(1000, df_no_outliers['A'].values, "Outlier 1000 was not removed.")

if __name__ == '__main__':
    unittest.main()
