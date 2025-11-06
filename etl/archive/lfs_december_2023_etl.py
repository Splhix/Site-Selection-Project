import pandas as pd
import os

# File paths
raw_path = "data/raw/lfs_december_2023_539248763514.csv"
cleaned_path = "data/cleaned/lfs_december_2023_cleaned.csv"

# Load CSV
df = pd.read_csv(raw_path)

# Rename columns for simplicity
df.rename(columns={
    'Region': 'Region',
    'Survey Year': 'Survey_Year',
    'C09-Work Indicator': 'Work_Indicator',
    'C10-Job Indicator': 'Job_Indicator',
    'C11 - Location of Work (Province, Municipality)': 'Work_Location'
}, inplace=True)

# Keep only relevant columns
columns_to_keep = ['Region', 'Survey_Year', 'Work_Indicator', 'Job_Indicator', 'Work_Location']
df_cleaned = df[columns_to_keep].copy()

# Drop rows where Work_Indicator is missing
df_cleaned = df_cleaned[df_cleaned['Work_Indicator'].notna()]

# Convert indicators to numeric, replacing NaNs with 0
df_cleaned['Work_Indicator'] = pd.to_numeric(df_cleaned['Work_Indicator'], errors='coerce').fillna(0)
df_cleaned['Job_Indicator'] = pd.to_numeric(df_cleaned['Job_Indicator'], errors='coerce').fillna(0)

# Fill missing location with 'Unknown'
df_cleaned['Work_Location'] = df_cleaned['Work_Location'].fillna('Unknown')

# Save cleaned file
df_cleaned.to_csv(cleaned_path, index=False)
print(f"[✅] Cleaned LFS data saved to: {cleaned_path}")
