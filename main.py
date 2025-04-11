import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('crime_data.csv')

# Initial EDA
print("First few rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nChecking for missing values...")
print(df.isnull().sum())
print("\nChecking for duplicates:", df.duplicated().sum())
print("\nSummarizing the data:")
print(df.describe())
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Preprocessing
df['START_DATE'] = pd.to_datetime(df['START_DATE'])
df['END_DATE'] = pd.to_datetime(df['END_DATE'])
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'])

# Create a clean copy of the DataFrame to avoid SettingWithCopyWarning
df_clean = df.dropna(subset=['START_DATE', 'LATITUDE', 'LONGITUDE']).copy()

# Extract year and month (no warning now because df_clean is a copy)
df_clean['START_YEAR'] = df_clean['START_DATE'].dt.year
df_clean['START_MONTH'] = df_clean['START_DATE'].dt.month_name()

# Keep relevant columns
key_columns = ['OFFENSE', 'SHIFT', 'METHOD', 'BLOCK', 'WARD', 'NEIGHBORHOOD_CLUSTER', 
               'LATITUDE', 'LONGITUDE', 'START_DATE', 'END_DATE', 'START_YEAR', 'START_MONTH']
df_clean = df_clean[key_columns]

# Outlier detection
numeric_cols = ['LATITUDE', 'LONGITUDE']
z_scores = (df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std()
outliers = (np.abs(z_scores) > 3).any(axis=1)
print(f"\nNumber of spatial outliers found: {outliers.sum()}")
df_clean = df_clean[~outliers]
