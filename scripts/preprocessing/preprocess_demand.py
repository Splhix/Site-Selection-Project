#!/usr/bin/env python3
"""
Preprocess demand data (occupied housing units).
Based on CP2 methods guide section 1.3.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns


def main():
    # File paths
    raw_path = "data/raw/Demand/2_SR on_Housing Characteristics_Statistical Tables_by Region_revised_PMMJ_CRD-approved_0 (1).xlsx"
    output_path = "data/cleaned/demand_occupied_units_region_panel.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw data
    print(f"Loading demand data from: {raw_path}")
    
    # Try to read the Excel file - may need to specify sheet name
    try:
        df = pd.read_excel(raw_path, sheet_name=0)  # First sheet
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Trying alternative sheet names...")
        try:
            df = pd.read_excel(raw_path, sheet_name="Table 1")  # Common sheet name
        except:
            df = pd.read_excel(raw_path, sheet_name="Housing Units")  # Alternative
    
    # Basic cleaning
    df = df.dropna(how='all')
    
    # Standardize geographic names
    df = standardize_geo(df)
    
    # Filter for target regions
    target_regions = ['NCR', 'Region III (Central Luzon)', 'Region IV-A (CALABARZON)']
    df = df[df['Region'].isin(target_regions)]
    
    # Look for occupied units or housing units columns
    occupied_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['occupied', 'housing', 'units', 'total'])]
    
    if not occupied_cols:
        raise ValueError("Could not find occupied housing units column")
    
    # Use the first relevant column found
    occupied_col = occupied_cols[0]
    df['occupied_units'] = pd.to_numeric(df[occupied_col], errors='coerce')
    
    # Remove rows with missing data
    df = df.dropna(subset=['occupied_units'])
    
    # Create panel data structure (assuming single year for now, but keeping structure for extrapolation)
    # Group by region to get regional totals
    regional_demand = df.groupby('Region')['occupied_units'].sum().reset_index()
    regional_demand['Year'] = 2020  # Assuming 2020 data
    regional_demand['OccupiedUnits'] = regional_demand['occupied_units']
    
    # Prepare output
    output_cols = ['Region', 'Year', 'OccupiedUnits']
    df_output = regional_demand[output_cols].copy()
    
    # Sort by Region, Year
    df_output = df_output.sort_values(['Region', 'Year']).reset_index(drop=True)
    
    # Save cleaned panel data
    df_output.to_csv(output_path, index=False)
    print(f"✅ Demand panel data cleaned and saved to: {output_path}")
    print(f"   - {len(df_output)} regions processed")
    print(f"   - Total occupied units: {df_output['OccupiedUnits'].sum():,.0f}")
    print(f"   - Regions: {df_output['Region'].unique()}")


if __name__ == "__main__":
    main()
