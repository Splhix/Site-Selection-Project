import pandas as pd
from pathlib import Path

# === File paths ===
raw_path = "data/raw/PAGASA Data/NAIA Daily Data.csv"   # input file
output_path = "data/cleaned/PAGASA Cleaned Data/Pasay Daily Weather Data_Cleaned.csv"  # output file

# === Load CSV ===
df = pd.read_csv(raw_path)

# === Remove duplicates ===
df_clean = df.drop_duplicates(keep="first")

# === Drop rows with ANY missing data ===
df_clean = df_clean.dropna(how="any")

# === Add new columns (do not alter existing ones) ===
df_clean["Region"] = "NCR"
df_clean["City/Province"] = "Pasay"

# === Export cleaned file ===
df_clean.to_csv(output_path, index=False)

# === Quick log ===
print("Cleaning complete!")
print(f"Input rows:  {len(df)}")
print(f"Output rows: {len(df_clean)}")
print(f"Saved cleaned file as: {output_path}")
