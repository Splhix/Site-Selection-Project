import pandas as pd
import os

# File paths
raw_folder = "data/raw/"
cleaned_folder = "data/cleaned/"
output_file = "housing_units_structure_combined_cleaned.csv"

# Input files
part1_file = os.path.join(raw_folder, "Housing Units Data v2 - part 1.xlsx")
part3_file = os.path.join(raw_folder, "Housing Units Data v2 - part 3.xlsx")

# Common cleaning function for part 1 and 3
def extract_clean_data(filepath):
    df = pd.read_excel(filepath, skiprows=2)

    housing_type_col = df.columns[0]
    location_col = df.columns[1]
    data_cols = df.columns[2:]

    current_type = None
    current_region = None
    cleaned_rows = []

    for _, row in df.iterrows():
        housing_type = str(row[housing_type_col]).strip()
        location = str(row[location_col]).strip()

        if housing_type and housing_type.lower() != 'nan':
            current_type = housing_type
            current_region = None
            continue

        if location.startswith('..') and not location.startswith('....'):
            current_region = location.replace('..', '').strip()
        elif location.startswith('....'):
            city = location.replace('....', '').strip()

            # Skip invalid or footnote-like cities
            if any(bad in city.lower() for bad in ['cluster', 'note', 'footnote', '***', 'others', '8 are']):
                continue

            # Skip if required fields are missing
            if not current_type or not current_region or not city:
                continue

            entry = {
                'Housing_Unit_Type': current_type.title(),
                'Region': current_region.title(),
                'City/Province': city.title(),
                'Year': 2020
            }

            for col in data_cols:
                entry[col] = row[col]
            cleaned_rows.append(entry)

    return pd.DataFrame(cleaned_rows)

# Extract and clean both datasets
df_part1 = extract_clean_data(part1_file)
df_part3 = extract_clean_data(part3_file)

# Merge them horizontally
merged = pd.merge(
    df_part1,
    df_part3,
    on=['Region', 'City/Province', 'Housing_Unit_Type', 'Year'],
    how='outer',
    suffixes=('_Part1', '_Part3')
)

# Standardize missing values
merged.replace(['-999', -999, 'N/A', '', 'na', 'NA'], pd.NA, inplace=True)

# Save to cleaned folder
os.makedirs(cleaned_folder, exist_ok=True)
merged.to_csv(os.path.join(cleaned_folder, output_file), index=False)
print(f"[✅] Cleaned + merged housing units saved to: {output_file}")
