import pandas as pd
import os
import re


# === File Paths ===
part1_path = "data/raw/Housing Units Data v2 - part 1.xlsx"
part3_path = "data/raw/Housing Units Data v2 - part 3.xlsx"
output_path = "data/cleaned/housing_units_structure_combined_cleaned.csv"


# === Standardize Housing Unit Type ===
def normalize_housing_type(htype):
    htype = htype.strip().lower()
    if htype in ['Apartment/Accessoria/Rowhouse', 'apartment/accessoria/row house', 'apartment/accessoria/rowhouse']:
        return 'Apartment/Accessoria/Row House'
    if htype in ['single', 'single house', 'detached house']:
        return 'Single House'
    if htype in ['duplex', 'duplex unit']:
        return 'Duplex'
    return htype.title()


# === Extract and Clean ===
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
            current_type = normalize_housing_type(housing_type)
            current_region = None
            continue


        if location.startswith('..') and not location.startswith('....'):
            current_region = re.sub(r'^\.+', '', location).strip()
        elif location.startswith('....'):
            city = re.sub(r'^\.+', '', location).strip()
            if any(bad in city.lower() for bad in ['cluster', 'note', 'footnote', '***', 'others', '8 are']):
                continue
            if not current_type or not current_region or not city:
                continue


            entry = {
                'Housing_Unit_Type': current_type,
                'Region': current_region.title(),
                'City/Province': city.title(),
                'Year': 2020
            }


            for col in data_cols:
                entry[col] = row[col]


            cleaned_rows.append(entry)


    return pd.DataFrame(cleaned_rows)


# === Process and Merge ===
df1 = extract_clean_data(part1_path)
df3 = extract_clean_data(part3_path)


# Standardize merge keys
merge_keys = ['Region', 'City/Province', 'Housing_Unit_Type', 'Year']
for df in [df1, df3]:
    for col in merge_keys:
        df[col] = df[col].astype(str).str.strip()
    df['Housing_Unit_Type'] = df['Housing_Unit_Type'].apply(normalize_housing_type)


# Merge horizontally
merged_df = pd.merge(
    df1,
    df3,
    on=merge_keys,
    how='outer',
    suffixes=('_Part1', '_Part3')
)


# Clean nulls
merged_df.replace(['-999', -999, 'N/A', '', 'na', 'NA'], pd.NA, inplace=True)


# Save final output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"[✅] Cleaned merged file saved to: {output_path}")

