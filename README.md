# 🔍 EDA-Pipeline: Automated Data Preprocessing & EDA Toolkit

![GitHub stars](https://img.shields.io/github/stars/your-username/EDA-Pipeline?style=social)
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

---

## 📂 Example Dataset

Tested with the **[Online Retail Data](https://www.kaggle.com/datasets/ertugrulesol/online-retail-data)** (541k transactions):
- **Key Challenges**:  
  - Missing `CustomerID` (24.9% of records).  
  - Outliers in `Quantity` (e.g., orders > 10,000 units).  
  - Temporal patterns in `InvoiceDate`.

**Sample Output**:  
![Density Plot](plots/Quantity_distribution.png)  
*Example visualization of outlier detection in the `Quantity` column.*

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/EDA-Pipeline.git
   cd EDA-Pipeline
