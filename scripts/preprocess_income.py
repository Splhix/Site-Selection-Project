#!/usr/bin/env python3
"""
Preprocess family income data.
Based on CP2 methods guide section 1.2.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns


def main():
    # File paths
    raw_path = "data/raw/Feasibility/Table 1. 2018, 2021 and 2023p Average Annual Family Income, by Per Capita Income Decile Class and by Region, Province and HUCs.xlsx"
    output_path = "data/cleaned/income_per_hh_panel.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw data
    print(f"Loading income data from: {raw_path}")
    df = pd.read_excel(raw_path, sheet_name=0)  # First sheet
    
    # Basic cleaning - remove header rows and metadata
    df = df.iloc[4:].copy()  # Skip first 4 rows (headers/metadata)
    df = df.dropna(how='all')
    
    # Rename first column to Region
    df.rename(columns={df.columns[0]: "Region"}, inplace=True)
    
    # Define years and income groups
    years = [2018, 2021, 2023]
    groups = ["All Income Groups", "1st Decile", "2nd Decile", "3rd Decile", "4th Decile",
              "5th Decile", "6th Decile", "7th Decile", "8th Decile", "9th Decile", "10th Decile"]
    
    # Build tidy dataframe
    tidy_list = []
    for i, yr in enumerate(years):
        start = 1 + i * 11  # columns for that year
        end = start + 11
        block = df.iloc[:, [0] + list(range(start, end))].copy()
        block = block.melt(id_vars="Region", var_name="IncomeGroup", value_name="Value")
        block["Year"] = yr
        # Replace header row values with proper names
        block["IncomeGroup"] = groups * (len(block) // len(groups))
        tidy_list.append(block)
    
    df_cleaned = pd.concat(tidy_list, ignore_index=True)
    
    # Drop rows where Region is NaN or header-like
    df_cleaned = df_cleaned[df_cleaned["Region"].notna()]
    df_cleaned = df_cleaned[~df_cleaned["Region"].str.contains("Region|Province|City", case=False, na=False)]
    
    # Standardize geographic names
    df_cleaned = standardize_geo(df_cleaned)
    
    # Filter for target regions
    target_regions = ['NCR', 'Region III (Central Luzon)', 'Region IV-A (CALABARZON)']
    df_cleaned = df_cleaned[df_cleaned['Region'].isin(target_regions)]
    
    # Clean numeric data
    df_cleaned['Value'] = pd.to_numeric(df_cleaned['Value'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['Value'])
    
    # Focus on "All Income Groups" for panel data
    df_panel = df_cleaned[
        df_cleaned['IncomeGroup'] == 'All Income Groups'
    ].copy()
    
    # Rename Value column to standard name
    df_panel['INC_income_per_hh'] = df_panel['Value']
    
    # Prepare output columns
    output_cols = ['Region', 'Province', 'City', 'Year', 'INC_income_per_hh']
    
    # If Province and City columns don't exist, create them from Region
    if 'Province' not in df_panel.columns:
        df_panel['Province'] = df_panel['Region']  # Use region as province for now
    if 'City' not in df_panel.columns:
        df_panel['City'] = df_panel['Region']  # Use region as city for now
    
    df_output = df_panel[output_cols].copy()
    
    # Sort by Region, Province, City, Year
    df_output = df_output.sort_values(['Region', 'Province', 'City', 'Year']).reset_index(drop=True)
    
    # Save cleaned panel data
    df_output.to_csv(output_path, index=False)
    print(f"✅ Income panel data cleaned and saved to: {output_path}")
    print(f"   - {len(df_output)} records processed")
    print(f"   - Years: {sorted(df_output['Year'].unique())}")
    print(f"   - Regions: {df_output['Region'].unique()}")


if __name__ == "__main__":
    main()
