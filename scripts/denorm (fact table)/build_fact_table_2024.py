#!/usr/bin/env python3
"""
Build unified fact table for 2024.
Based on CP2 methods guide section 4.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, minmax_norm, validate_required_columns


def main():
    # Input file paths (using extrapolated data)
    IN_INCOME = "data/extrapolated/income_per_hh_2024.csv"
    IN_PRICE = "data/cleaned/feasibility/housing_prices_completed_PURE_INHERITANCE.csv"  # Price already at 2024
    IN_GRDP = "data/extrapolated/grdp_pc_region_2024.csv"
    IN_DEMAND = "data/extrapolated/demand_occupied_units_region_2024.csv"
    IN_LABOR = "data/extrapolated/labor_employment_region_2024.csv"
    IN_HAZ = "data/cleaned/risk/risk_clean_2024_PARTIAL_from_3uploads.csv.xlsx"
    
    # Output file path
    OUT = "data/curated/fact_table_2024.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    
    print("Loading data files...")
    
    # Load base income data
    try:
        base = pd.read_csv(IN_INCOME)
        print(f"✅ Loaded income data: {len(base)} records")
    except Exception as e:
        print(f"❌ Error loading income data: {e}")
        return
    
    # Load price data
    try:
        price = pd.read_csv(IN_PRICE)
        print(f"✅ Loaded price data: {len(price)} records")
    except Exception as e:
        print(f"❌ Error loading price data: {e}")
        return
    
    # Load hazard data
    try:
        haz = pd.read_excel(IN_HAZ)
        print(f"✅ Loaded hazard data: {len(haz)} records")
    except Exception as e:
        print(f"❌ Error loading hazard data: {e}")
        return
    
    # Load GRDP data
    try:
        grdp = pd.read_csv(IN_GRDP)
        print(f"✅ Loaded GRDP data: {len(grdp)} records")
    except Exception as e:
        print(f"❌ Error loading GRDP data: {e}")
        return
    
    # Load demand data
    try:
        dem = pd.read_csv(IN_DEMAND)
        print(f"✅ Loaded demand data: {len(dem)} records")
    except Exception as e:
        print(f"❌ Error loading demand data: {e}")
        return
    
    # Load labor data
    try:
        lab = pd.read_csv(IN_LABOR)
        print(f"✅ Loaded labor data: {len(lab)} records")
    except Exception as e:
        print(f"❌ Error loading labor data: {e}")
        return
    
    # Standardize geographic names in all datasets
    base = standardize_geo(base)
    price = standardize_geo(price)
    haz = standardize_geo(haz)
    grdp = standardize_geo(grdp)
    dem = standardize_geo(dem)
    lab = standardize_geo(lab)
    
    print("Merging datasets...")
    
    # Merge datasets
    df = (base.merge(price, on=['Region', 'Province', 'City'], how='left')
               .merge(haz, on=['Region', 'Province', 'City'], how='left')
               .merge(grdp, on=['Region'], how='left')
               .merge(dem, on=['Region'], how='left')
               .merge(lab, on=['Region'], how='left'))
    
    print(f"✅ Merged dataset: {len(df)} records")
    
    # Create economy score
    if 'GRDP_per_capita_2024' in df.columns:
        df['GRDPScore'] = minmax_norm(df['GRDP_per_capita_2024'])
    else:
        df['GRDPScore'] = 0.5  # Default neutral score
    
    # Use actual employment rate from labor data
    if 'EmploymentRate_2024' not in df.columns:
        # Fallback: create employment rate proxy using GRDP
        df['EmploymentRate_2024'] = df['GRDP_per_capita_2024'] / df['GRDP_per_capita_2024'].mean()
        df['EmploymentRate_2024'] = df['EmploymentRate_2024'].clip(0.5, 1.5)  # Reasonable bounds
    
    # Create economy score (60% employment + 40% GRDP)
    df['EconomyScore'] = 0.60 * minmax_norm(df['EmploymentRate_2024']) + 0.40 * df['GRDPScore']
    
    # Create demand score (normalized occupied units)
    if 'OccupiedUnits_2024' in df.columns:
        df['DemandScore'] = minmax_norm(df['OccupiedUnits_2024'])
    else:
        df['DemandScore'] = 0.5  # Default neutral score
    
    # Ensure hazard safety score exists
    if 'HazardSafety_NoFault' not in df.columns:
        df['HazardSafety_NoFault'] = 0.5  # Default neutral score
    
    # Select and order columns for output
    output_cols = [
        'Region', 'Province', 'City',
        'INC_income_per_hh_2024',
        'PRICE_median_2024_final',
        'GRDP_per_capita_2024',
        'demand_occupied_units_region_2024',
        'EmploymentRate_2024',
        'GRDPScore',
        'EconomyScore',
        'DemandScore',
        'HazardSafety_NoFault'
    ]
    
    # Add any additional hazard columns that exist
    hazard_cols = [col for col in df.columns if 'Risk' in col or 'Safety' in col]
    for col in hazard_cols:
        if col not in output_cols:
            output_cols.append(col)
    
    # Create final output dataframe
    df_output = df[output_cols].copy()
    
    # Sort by Region, Province, City
    df_output = df_output.sort_values(['Region', 'Province', 'City']).reset_index(drop=True)
    
    # Save unified fact table
    df_output.to_csv(OUT, index=False)
    print(f"✅ Unified fact table saved to: {OUT}")
    print(f"   - {len(df_output)} cities processed")
    print(f"   - Columns: {list(df_output.columns)}")
    print(f"   - Regions: {df_output['Region'].unique()}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   - Income range: {df_output['INC_income_per_hh_2024'].min():,.0f} - {df_output['INC_income_per_hh_2024'].max():,.0f}")
    print(f"   - Price range: {df_output['PRICE_median_2024_final'].min():,.0f} - {df_output['PRICE_median_2024_final'].max():,.0f}")
    print(f"   - Economy score range: {df_output['EconomyScore'].min():.3f} - {df_output['EconomyScore'].max():.3f}")
    print(f"   - Demand score range: {df_output['DemandScore'].min():.3f} - {df_output['DemandScore'].max():.3f}")
    print(f"   - Hazard safety range: {df_output['HazardSafety_NoFault'].min():.3f} - {df_output['HazardSafety_NoFault'].max():.3f}")


if __name__ == "__main__":
    main()
