import pandas as pd
import os

# File paths
raw_path = "data/raw/Table-3.-2018-2021-and-2023p-Total-Annual-Family-Income-by-Per-Capita-Income-Decile-and-by-Region-Province-and-HUC.xlsx"
cleaned_path = "data/cleaned/family_income_cleaned.csv"

# Load
df_raw = pd.read_excel(raw_path, sheet_name="Table 3")

# Drop metadata rows
df_raw = df_raw.iloc[4:].copy()
df_raw.rename(columns={df_raw.columns[0]: "Region"}, inplace=True)

# Define years and income groups
years = [2018, 2021, 2023]
groups = ["All Income Groups","1st Decile","2nd Decile","3rd Decile","4th Decile",
          "5th Decile","6th Decile","7th Decile","8th Decile","9th Decile","10th Decile"]

# Build tidy dataframe
tidy_list = []
for i, yr in enumerate(years):
    start = 1 + i*11   # columns for that year
    end = start + 11
    block = df_raw.iloc[:, [0] + list(range(start, end))].copy()
    block = block.melt(id_vars="Region", var_name="IncomeGroup", value_name="Value")
    block["Year"] = yr
    # Replace header row values with proper names
    block["IncomeGroup"] = groups * (len(block) // len(groups))
    tidy_list.append(block)

df_cleaned = pd.concat(tidy_list, ignore_index=True)

# Drop rows where Region is NaN or header-like
df_cleaned = df_cleaned[df_cleaned["Region"].notna()]

# Save
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
df_cleaned.to_csv(cleaned_path, index=False)

print(f"[✅] Cleaned family income data saved to: {cleaned_path}")
