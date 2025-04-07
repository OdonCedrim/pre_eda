# 🔍 EDA-Pipeline: Automated Data Preprocessing & EDA Toolkit

![GitHub stars](https://img.shields.io/github/stars/OdonCedrim/pre_eda?style=social)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-success)

An integrated pipeline for **cleaning, exploratory analysis (EDA)**, and **MySQL integration** of structured datasets. Built for data scientists seeking to automate repetitive preprocessing tasks and generate rapid insights.

---

## 🚀 Features

### **Smart Preprocessing**
- **Auto Type Conversion**:  
  Convert columns to datetime, numeric, or categorical types intelligently.
- **Missing Values**:  
  Remove rows or impute missing values using mean/mode.
- **Outlier Detection**:  
  Choose from **IQR**, **Z-score**, or **DBSCAN** (clustering-based) methods.
- **MySQL Compatibility**:  
  Export cleaned data to MySQL with automatic type mapping.

### **Exploratory Analysis (EDA)**
- **Statistical Reports**:  
  Descriptive stats, normality tests (Shapiro-Wilk), and correlation matrices.
- **Visualizations**:  
  Auto-generated density plots (KDE) and Q-Q plots for numeric columns.
- **Flexible Outputs**:  
  Save results as CSV, TXT, or PNG files.

### **Integration**
- **MySQL Support**:  
  Import/export tables and auto-create schemas.
- **Scalable**:  
  Handles large datasets efficiently with categorical optimization.
