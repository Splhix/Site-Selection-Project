#!/usr/bin/env python3
"""
Preprocess hazard and seismic data.
Based on CP2 methods guide section 1.4.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns


def main():
    # File paths
    hazard_path = "data/raw/Risk/Environmental_Hazards - Sheet1.csv"
    fault_path = "data/raw/Risk/Fault Line Distance Data.xlsx"
    earthquake_path = "data/raw/Risk/PHIVOLCS Data/phivolcs_earthquake_data.csv"
    output_path = "data/cleaned/risk/risk_clean_2024_PARTIAL_from_3uploads.csv.xlsx"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Loading hazard data...")
    
    # Load environmental hazards data
    try:
        df_hazard = pd.read_csv(hazard_path)
        print(f"✅ Loaded environmental hazards from: {hazard_path}")
    except Exception as e:
        print(f"⚠️  Could not load environmental hazards: {e}")
        df_hazard = pd.DataFrame()
    
    # Load fault line distance data
    try:
        df_fault = pd.read_excel(fault_path)
        print(f"✅ Loaded fault line data from: {fault_path}")
    except Exception as e:
        print(f"⚠️  Could not load fault line data: {e}")
        df_fault = pd.DataFrame()
    
    # Load earthquake data
    try:
        df_earthquake = pd.read_csv(earthquake_path)
        print(f"✅ Loaded earthquake data from: {earthquake_path}")
    except Exception as e:
        print(f"⚠️  Could not load earthquake data: {e}")
        df_earthquake = pd.DataFrame()
    
    # Process environmental hazards
    if not df_hazard.empty:
        df_hazard = standardize_geo(df_hazard)
        
        # Look for hazard level columns (1-3 scale)
        hazard_cols = [col for col in df_hazard.columns if any(keyword in col.lower() 
                       for keyword in ['flood', 'surge', 'landslide', 'level', 'risk'])]
        
        # Create standardized hazard columns
        for col in hazard_cols:
            if 'flood' in col.lower():
                df_hazard['FloodRisk'] = pd.to_numeric(df_hazard[col], errors='coerce')
            elif 'surge' in col.lower():
                df_hazard['StormSurgeRisk'] = pd.to_numeric(df_hazard[col], errors='coerce')
            elif 'landslide' in col.lower():
                df_hazard['LandslideRisk'] = pd.to_numeric(df_hazard[col], errors='coerce')
    
    # Process fault line data
    if not df_fault.empty:
        df_fault = standardize_geo(df_fault)
        
        # Look for distance column
        distance_cols = [col for col in df_fault.columns if any(keyword in col.lower() 
                         for keyword in ['distance', 'km', 'fault'])]
        
        if distance_cols:
            df_fault['FaultDistance_km'] = pd.to_numeric(df_fault[distance_cols[0]], errors='coerce')
    
    # Process earthquake data
    if not df_earthquake.empty:
        df_earthquake = standardize_geo(df_earthquake)
        
        # Look for earthquake-related columns
        eq_cols = [col for col in df_earthquake.columns if any(keyword in col.lower() 
                  for keyword in ['magnitude', 'depth', 'count', 'events'])]
        
        # Create earthquake risk metrics
        if 'magnitude' in str(eq_cols).lower():
            df_earthquake['EarthquakeRisk'] = pd.to_numeric(df_earthquake[eq_cols[0]], errors='coerce')
    
    # Combine all hazard data
    df_combined = pd.DataFrame()
    
    # Start with environmental hazards if available
    if not df_hazard.empty:
        df_combined = df_hazard[['Region', 'Province', 'City']].copy()
        
        # Add hazard columns
        for col in ['FloodRisk', 'StormSurgeRisk', 'LandslideRisk']:
            if col in df_hazard.columns:
                df_combined[col] = df_hazard[col]
    
    # Merge fault data
    if not df_fault.empty and not df_combined.empty:
        df_combined = df_combined.merge(
            df_fault[['Region', 'Province', 'City', 'FaultDistance_km']], 
            on=['Region', 'Province', 'City'], 
            how='left'
        )
    elif not df_fault.empty:
        df_combined = df_fault[['Region', 'Province', 'City', 'FaultDistance_km']].copy()
    
    # Merge earthquake data
    if not df_earthquake.empty and not df_combined.empty:
        df_combined = df_combined.merge(
            df_earthquake[['Region', 'Province', 'City', 'EarthquakeRisk']], 
            on=['Region', 'Province', 'City'], 
            how='left'
        )
    elif not df_earthquake.empty:
        df_combined = df_earthquake[['Region', 'Province', 'City', 'EarthquakeRisk']].copy()
    
    # Filter for target regions
    target_regions = ['NCR', 'Region III (Central Luzon)', 'Region IV-A (CALABARZON)']
    df_combined = df_combined[df_combined['Region'].isin(target_regions)]
    
    # Create hazard safety score (inverse of risk, normalized)
    risk_cols = ['FloodRisk', 'StormSurgeRisk', 'LandslideRisk', 'EarthquakeRisk']
    for col in risk_cols:
        if col in df_combined.columns:
            # Convert risk to safety (higher risk = lower safety)
            df_combined[f'{col}_Safety'] = 1 - (df_combined[col] / 3.0)  # Assuming 1-3 scale
            df_combined[f'{col}_Safety'] = df_combined[f'{col}_Safety'].clip(0, 1)
    
    # Create overall hazard safety score (average of individual safety scores)
    safety_cols = [col for col in df_combined.columns if col.endswith('_Safety')]
    if safety_cols:
        df_combined['HazardSafety_NoFault'] = df_combined[safety_cols].mean(axis=1)
    else:
        df_combined['HazardSafety_NoFault'] = 0.5  # Default neutral score
    
    # Sort by Region, Province, City
    df_combined = df_combined.sort_values(['Region', 'Province', 'City']).reset_index(drop=True)
    
    # Save as Excel file (matching the expected output format)
    df_combined.to_excel(output_path, index=False)
    print(f"✅ Hazard data cleaned and saved to: {output_path}")
    print(f"   - {len(df_combined)} cities processed")
    print(f"   - Hazard columns: {[col for col in df_combined.columns if 'Risk' in col or 'Safety' in col]}")
    print(f"   - Regions: {df_combined['Region'].unique()}")


if __name__ == "__main__":
    main()
