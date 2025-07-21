import pandas as pd
import numpy as np
import os

# File paths
raw_folder = "data/raw/"
cleaned_folder = "data/cleaned/"
output_file = "housing_units_region_city_flat.csv"

files = [
    "Housing Units Data v2 - part 1.xlsx",
    "Housing Units Data v2 - part 2.xlsx",
    "Housing Units Data v2 - part 3.xlsx"
]

rows = []

for file in files:
    print(f"[📂] Processing: {file}")
    df = pd.read_excel(os.path.join(raw_folder, file))

    # Use second column (index 1) as the geographic location
    location_col = df.columns[1]
    data_cols = [col for i, col in enumerate(df.columns) if i != 1]

    current_region = None

    for _, row in df.iterrows():
        location = str(row[location_col]).strip()

        if location.startswith('..') and not location.startswith('....'):
            # Region level
            current_region = location.replace('..', '').strip()
        elif location.startswith('....'):
            # City/Province under Region
            city_province = location.replace('....', '').replace(' *', '').strip()
            entry = {
                'Region': current_region,
                'City/Province': city_province,
                'Year': 2020
            }
            for col in data_cols:
                entry[col] = row[col]
            rows.append(entry)

# Create flat DataFrame
flat_df = pd.DataFrame(rows)

# Clean missing values
flat_df.replace(['-999', -999, -999.0, '-999.0', 'N/A', '', 'na', 'NA'], np.nan, inplace=True)

# Save to cleaned folder
flat_df.to_csv(os.path.join(cleaned_folder, output_file), index=False)
print(f"\n[✅] Cleaned Region + City/Province output saved to: {output_file}")