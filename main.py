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

#OBJECTIVE 1
# TIMELY CRIME INCIDENTS

# Load the data
df = pd.read_csv('crime_data.csv')

# Convert REPORT_DAT to datetime
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], errors='coerce')

# Extract time features
df['MONTH'] = df['REPORT_DAT'].dt.month_name()
df['DAY_OF_WEEK'] = df['REPORT_DAT'].dt.day_name()
df['HOUR'] = df['REPORT_DAT'].dt.hour

# Set seaborn theme
sns.set_theme(style="whitegrid")


# -------------------------
# 1. Line Plot - Crimes by Month
# -------------------------
plt.figure(figsize=(10, 6))
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_counts = df['MONTH'].value_counts().reindex(month_order)
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker="o", color="skyblue")
plt.title('Crime Incidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['DAY_OF_WEEK'].value_counts().reindex(day_order)

# Use pastel color palette from Seaborn
colors = sns.color_palette("pastel", len(day_counts))

# Plot with matplotlib to avoid seaborn warnings
plt.figure(figsize=(10, 6))
plt.bar(day_counts.index, day_counts.values, color=colors)
plt.title('Crime Incidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Count crimes by shift
shift_order = ['MIDNIGHT', 'DAY', 'EVENING']
shift_counts = df['SHIFT'].value_counts().reindex(shift_order)

# Use a different color palette (coolwarm)
colors = sns.color_palette("coolwarm", len(shift_counts))

# Horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(shift_counts.index, shift_counts.values, color=colors)
plt.title('Crime Incidents by Shift')
plt.xlabel('Number of Incidents')
plt.ylabel('Shift')
plt.tight_layout()
plt.show()


#OBJECTIVE 2
# DISTRIBUTION OF CRIME TYPES

# Prepare data
offense_counts = df['OFFENSE'].value_counts()
top_crimes = offense_counts.nlargest(5).index
df_top = df[df['OFFENSE'].isin(top_crimes)]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Chart 1: Distribution of All Crime Types ---
colors = sns.color_palette("pastel", len(offense_counts))
axes[0].barh(offense_counts.index, offense_counts.values, color=colors)
axes[0].set_title('Distribution of Crime Types', fontsize=16)
axes[0].set_xlabel('Number of Incidents')
axes[0].set_ylabel('Crime Type')
axes[0].invert_yaxis()

# --- Chart 2: Top 5 Crime Types by Shift ---
sns.countplot(
    ax=axes[1],
    x='OFFENSE',
    hue='SHIFT',
    data=df_top,
    order=top_crimes,
    palette='coolwarm'
)
axes[1].set_title('Top 5 Crime Types by Shift', fontsize=16)
axes[1].set_xlabel('Crime Type')
axes[1].set_ylabel('Number of Incidents')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(title='Shift')

# Adjust layout
plt.tight_layout()
plt.show()

#OBJECTIVE 3
# CRIME COUNT BY NEIGHBOURHOOD

# Prepare data
top_neighborhoods = df['NEIGHBORHOOD_CLUSTER'].value_counts().nlargest(10)

# Light pastel color palette
colors = sns.color_palette('pastel', len(top_neighborhoods))

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    top_neighborhoods.values,
    labels=top_neighborhoods.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'white'}
)
plt.title('Top 10 Neighborhoods by Crime Count', fontsize=16)
plt.tight_layout()
plt.show()



# Ensure you have your cleaned DataFrame (replace with actual cleaned df if needed)
# We'll assume df_clean already exists — if not, use df instead.
top_offenses = df_clean['OFFENSE'].value_counts().nlargest(5).index
df_filtered = df_clean[df_clean['OFFENSE'].isin(top_offenses)]

# OBJECTIVE 4: LATITUDE AND LONGITUDE DISTRIBUTION
plt.figure(figsize=(14, 6))

# Latitude boxplot
plt.subplot(1, 2, 1)
sns.boxplot(
    data=df_filtered,
    x='OFFENSE',
    y='LATITUDE',
    hue='OFFENSE',
    palette='Set2',
    legend=False
)
plt.title('Latitude Distribution by Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Latitude')
plt.xticks(rotation=30)

# Longitude boxplot
plt.subplot(1, 2, 2)
sns.boxplot(
    data=df_filtered,
    x='OFFENSE',
    y='LONGITUDE',
    hue='OFFENSE',
    palette='Set2',
    legend=False
)
plt.title('Longitude Distribution by Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Longitude')
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
