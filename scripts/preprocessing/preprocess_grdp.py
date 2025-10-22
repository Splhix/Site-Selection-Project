#!/usr/bin/env python3
"""
Preprocess GRDP (Gross Regional Domestic Product) data.
Based on CP2 methods guide section 1.1.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns


def main():
    # File paths
    raw_path = "data/raw/Per Capita GRDP 2022-2024.csv"
    output_path = "data/cleaned/grdp_pc_region_panel.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw data
    print(f"Loading GRDP data from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Basic cleaning
    df = df.dropna(how='all')
    
    # Standardize geographic names
    df = standardize_geo(df)
    
    # Filter for required regions
    target_regions = ['NCR', 'Region III (Central Luzon)', 'Region IV-A (CALABARZON)']
    df = df[df['Region'].isin(target_regions)].copy()
    
    # Look for year columns and GRDP columns
    year_cols = [col for col in df.columns if 'year' in col.lower() or col.isdigit()]
    grdp_cols = [col for col in df.columns if 'grdp' in col.lower() or 'per capita' in col.lower()]
    
    # Create panel data structure
    panel_data = []
    
    for _, row in df.iterrows():
        region = row['Region']
        
        # Process each year column
        for col in df.columns:
            if col.isdigit() and int(col) >= 2000:  # Year columns
                year = int(col)
                value = row[col]
                
                if pd.notna(value):
                    panel_data.append({
                        'Region': region,
                        'Year': year,
                        'GRDP_per_capita': pd.to_numeric(value, errors='coerce')
                    })
    
    # Create panel DataFrame
    df_panel = pd.DataFrame(panel_data)
    df_panel = df_panel.dropna(subset=['GRDP_per_capita'])
    
    # Sort by Region, Year
    df_panel = df_panel.sort_values(['Region', 'Year']).reset_index(drop=True)
    
    # Save cleaned panel data
    df_panel.to_csv(output_path, index=False)
    print(f"✅ GRDP panel data cleaned and saved to: {output_path}")
    print(f"   - {len(df_panel)} records processed")
    print(f"   - Years: {sorted(df_panel['Year'].unique())}")
    print(f"   - Regions: {df_panel['Region'].unique()}")


if __name__ == "__main__":
    main()
