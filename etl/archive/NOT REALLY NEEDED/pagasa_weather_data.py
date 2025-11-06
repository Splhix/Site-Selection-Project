import pandas as pd
from pathlib import Path

# === File paths ===
# Replace these with your actual file locations if needed
files = [
    "data/cleaned/PAGASA Cleaned Data/Alabat Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Ambulong Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Baler Radar Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Cabanatuan Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Casiguran Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Clark Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Iba Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Infanta Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Muñoz Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Olongapo Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Pasay Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Manila Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Cavite Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Quezon Daily Weather Data_Cleaned.csv",
    "data/cleaned/PAGASA Cleaned Data/Tanay Daily Weather Data_Cleaned.csv",
]

output_path = "data/cleaned/pagasa_weather_data_cleaned.csv"

# === Load and concatenate ===
dfs = [pd.read_csv(f) for f in files]
compiled_df = pd.concat(dfs, ignore_index=True)

# === Sort by Year column in descending order ===
compiled_df = compiled_df.sort_values(by="YEAR", ascending=False).reset_index(drop=True)

# === Save final merged dataset ===
compiled_df.to_csv(output_path, index=False)

print("Compilation complete!")
print(f"Total rows: {len(compiled_df)}")
print(f"Total columns: {len(compiled_df.columns)}")
print(f"Saved to: {output_path}")
