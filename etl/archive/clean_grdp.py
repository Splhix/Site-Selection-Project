import pandas as pd
import os

# File paths
raw_path = "data/raw/More-GRDP.xlsx"
cleaned_path = "data/cleaned/more_grdp_cleaned2.csv"

# Load sheet
df_raw = pd.read_excel(raw_path, sheet_name="2B5CPGD1")

# Drop empty rows
df_raw = df_raw.dropna(how='all')

# The first column is Region
region_col = df_raw.columns[0]

# Define years manually
years = list(range(2000, 2024))  # 2000–2023
n_years = len(years)

# Split into current and constant blocks
current_cols = df_raw.columns[1:1+n_years]
constant_cols = df_raw.columns[1+n_years:1+2*n_years]

# Melt both blocks
current_long = df_raw[[region_col] + list(current_cols)].melt(
    id_vars=region_col, var_name="Year", value_name="GRDP")
current_long["PriceType"] = "Current Prices"

constant_long = df_raw[[region_col] + list(constant_cols)].melt(
    id_vars=region_col, var_name="Year", value_name="GRDP")
constant_long["PriceType"] = "Constant 2018 Prices"

# Combine
df_cleaned = pd.concat([current_long, constant_long], ignore_index=True)

# Rename columns
df_cleaned.rename(columns={region_col: "Region"}, inplace=True)

# Replace the auto-melted 'Year' values with the real sequence of years
df_cleaned["Year"] = df_cleaned.groupby("PriceType").cumcount().map(lambda i: years[i % n_years])

# Clean Region names
df_cleaned["Region"] = df_cleaned["Region"].astype(str).str.strip()
df_cleaned["Region"] = df_cleaned["Region"].str.replace(r"\.+", "", regex=True)

# Drop rows with missing GRDP
df_cleaned = df_cleaned.dropna(subset=["GRDP"])

# Save cleaned file
df_cleaned.to_csv(cleaned_path, index=False)
print(f"[✅] Cleaned GRDP data saved to: {cleaned_path}")
