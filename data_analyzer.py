"""
Module for exploratory data analysis (EDA) and report generation.

Features:
  - **Quantitative Analysis**: Mean, median, standard deviation, skewness, kurtosis, normality (Shapiro-Wilk).
  - **Qualitative Analysis**: Mode, category frequency, unique values.
  - **Temporal Analysis**: Date range, total days covered.
  - **Visualization**: Density plots (KDE) and Q-Q plots for distribution.
  - **Statistical Tests**: T-test, ANOVA, Chi-square for group comparison.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class DataAnalyzer:
    """
    Class for exploratory data analysis (EDA).

    Attributes:
        None public. Methods focus on generating reports and plots.
    
    Main Methods:
        analyze(): Generates a complete report with numerical, categorical, and temporal analyses.
        analyze_distribution(): Creates distribution plots (KDE and Q-Q plot).
        perform_ttest(), perform_anova(), perform_chi_square(): Statistical tests.
    """
    
    def analyze(self, df: pd.DataFrame, outliers_count: dict) -> dict:
        """
        Generates a consolidated report with descriptive statistics and diagnostics.
        
        Report structure:
            - missing_values: Count of missing values per column.
            - outliers: Count of outliers per column (provided externally).
            - Individual analysis for each column (numerical, categorical, or temporal).
            - correlation: Correlation matrix among numerical columns.
        
        Args:
            df (pd.DataFrame): Processed DataFrame (after preprocessing).
            outliers_count (dict): Outlier count per column (from DataProcessor).
            
        Returns:
            dict: Structured report as a dictionary, ready for export as text/JSON.
        """
        report = {
            "missing_values": df.isnull().sum().to_dict(),
            "outliers": outliers_count
        }
        
        # Analyze each column based on its type (date, numerical, or categorical)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                report[col] = self._datetime_analysis(df[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                report[col] = self._numeric_analysis(df[col], outliers_count.get(col, 0))
            else:
                report[col] = self._categorical_analysis(df[col])
        
        # Add correlation matrix (only for numerical columns)
        numeric_cols = df.select_dtypes(include=np.number).columns
        report["correlation"] = (
            df[numeric_cols].corr(numeric_only=True).to_dict()
            if len(numeric_cols) > 0
            else {}
        )
        
        return report

    def _numeric_analysis(self, series: pd.Series, outlier_count: int) -> dict:
        """
        Calculates descriptive statistics for numerical variables.
        
        Metrics included:
            - Central Tendency: mean, median.
            - Dispersion: standard deviation, variance, coefficient of variation (CV).
            - Distribution shape: skewness, kurtosis.
            - Normality: Shapiro-Wilk test (p-value).
            - Outliers: Count provided by preprocessing.
        
        Args:
            series (pd.Series): Numerical column to be analyzed.
            outlier_count (int): Number of outliers detected in the column.
            
        Returns:
            dict: Dictionary with all calculated metrics.
        """
        mean_val = series.mean()
        std = series.std()
        variance = series.var()
        cv = std / mean_val if mean_val != 0 else np.nan  # Avoid division by zero
        
        # Normality test (applied only if sufficient data)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(series.dropna())
            normality = {"shapiro_stat": shapiro_stat, "shapiro_p": shapiro_p}
        except Exception as e:
            logging.warning("Shapiro-Wilk test failed: %s", e)
            normality = {"shapiro_stat": None, "shapiro_p": None}
        
        return {
            "type": "numeric",
            "mean": mean_val,
            "median": series.median(),
            "std_dev": std,
            "variance": variance,
            "coefficient_of_variation": cv,
            "min": series.min(),
            "max": series.max(),
            "25th_percentile": series.quantile(0.25),
            "75th_percentile": series.quantile(0.75),
            "skewness": stats.skew(series.dropna()),
            "kurtosis": stats.kurtosis(series.dropna()),
            "normality": normality,
            "missing": series.isnull().sum(),
            "outliers": outlier_count
        }

    def _categorical_analysis(self, series: pd.Series) -> dict:
        """
        Analyzes categorical or text variables.
        
        Metrics included:
            - Mode: most frequent value.
            - Diversity: number of unique categories.
            - Frequencies: absolute and percentage frequency of each category.
        
        Args:
            series (pd.Series): Categorical column to be analyzed.
            
        Returns:
            dict: Dictionary with statistics and category distribution.
        """
        abs_freq = series.value_counts(dropna=True).to_dict()
        percent_freq = (series.value_counts(dropna=True, normalize=True) * 100).round(2).to_dict()
        
        return {
            "type": "categorical",
            "mode": series.mode()[0] if not series.empty else None,
            "unique_categories": series.nunique(),
            "unique_values": series.dropna().unique().tolist(),
            "absolute_frequency": abs_freq,
            "percentage_frequency": percent_freq,
            "missing": series.isnull().sum()
        }

    def _datetime_analysis(self, series: pd.Series) -> dict:
        """
        Analyzes date/time columns.
        
        Metrics included:
            - Range: minimum and maximum dates.
            - Duration: total number of days covered.
        
        Args:
            series (pd.Series): Date column to be analyzed.
            
        Returns:
            dict: Dictionary with temporal statistics.
        """
        return {
            "type": "date",
            "min_date": series.min().strftime('%Y-%m-%d') if not series.empty else None,
            "max_date": series.max().strftime('%Y-%m-%d') if not series.empty else None,
            "days_covered": (series.max() - series.min()).days if not series.empty else 0,
            "missing": series.isnull().sum()
        }
    
    def analyze_distribution(self, series: pd.Series, column_name: str, output_dir: str = 'plots'):
        """
        Generates plots to assess the distribution of a numerical variable.
        
        Produced plots:
            1. **KDE (Kernel Density Estimate)**: Shows the density distribution.
            2. **Q-Q Plot**: Compares the data distribution with a theoretical normal distribution.
        
        Args:
            series (pd.Series): Data to be plotted (numerical).
            column_name (str): Column name (used in the title and file name).
            output_dir (str): Output directory to save the plots (default: 'plots').
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create the directory if it doesn't exist
        
        plt.figure(figsize=(10, 4))
        
        # Subplot 1: Density plot (KDE)
        plt.subplot(1, 2, 1)
        sns.kdeplot(series.dropna(), fill=True)
        plt.title(f'Density - {column_name}')
        
        # Subplot 2: Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(series.dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot - {column_name}')
        
        plt.tight_layout()
        plot_file = f"{output_dir}/{column_name}_distribution.png"
        plt.savefig(plot_file)  # Save the combined figure
        plt.close()
        logging.info("Plots saved in: %s", plot_file)

    def perform_ttest(self, group1: pd.Series, group2: pd.Series):
        """
        Performs an independent T-test.
        
        Typical use: Comparing means between two groups (e.g., control vs treatment).
        
        Args:
            group1 (pd.Series): Sample from the first group.
            group2 (pd.Series): Sample from the second group.
            
        Returns:
            dict: T statistic and p-value. p-value < 0.05 indicates a significant difference.
        """
        t_stat, p_value = stats.ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
        return {"t_stat": t_stat, "p_value": p_value}

    def perform_anova(self, df: pd.DataFrame, group_col: str, value_col: str):
        """
        Performs an ANOVA test to compare means across three or more groups.
        
        Typical use: Check if there is a significant difference among categorical groups.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            group_col (str): Column name that defines the groups.
            value_col (str): Numerical column to be compared.
            
        Returns:
            dict: F statistic and p-value. p-value < 0.05 indicates a difference among groups.
        """
        groups = [group[value_col].dropna() for name, group in df.groupby(group_col)]
        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": f_stat, "p_value": p_value}

    def perform_chi_square(self, observed, expected):
        """
        Performs a Chi-square test to assess association between categorical variables.
        
        Typical use: Check if the observed distribution differs from the expected distribution.
        
        Args:
            observed (array-like): Observed frequencies.
            expected (array-like): Expected frequencies (e.g., theoretical distribution).
            
        Returns:
            dict: Chi-square statistic and p-value. p-value < 0.05 indicates significance.
        """
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        return {"chi2": chi2, "p_value": p_value}
