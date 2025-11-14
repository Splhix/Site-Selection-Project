#!/usr/bin/env python3
"""
Preprocess hazard and seismic data.
Based on CP2 methods guide section 1.4.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns, minmax_norm


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
        
        # Look for earthquake-related columns for proper risk calculation
        eq_events_col = None
        eq_m5plus_col = None  
        eq_depth_col = None
        
        for col in df_earthquake.columns:
            col_lower = col.lower()
            if 'events' in col_lower and '50km' in col_lower:
                eq_events_col = col
            elif 'm5plus' in col_lower and '50km' in col_lower:
                eq_m5plus_col = col
            elif 'depth' in col_lower and '50km' in col_lower:
                eq_depth_col = col
        
        # Calculate earthquake risk using proper weighted formula
        # EarthquakeRisk = 0.45*EQ_m5plus + 0.35*EQ_freq + 0.20*EQ_depthR
        
        # Normalize components (0-1 scale)
        if eq_events_col and eq_events_col in df_earthquake.columns:
            EQ_freq = minmax_norm(pd.to_numeric(df_earthquake[eq_events_col], errors='coerce'))
        else:
            EQ_freq = 0.5  # Default medium risk
            
        if eq_m5plus_col and eq_m5plus_col in df_earthquake.columns:
            EQ_m5plus = minmax_norm(pd.to_numeric(df_earthquake[eq_m5plus_col], errors='coerce'))
        else:
            EQ_m5plus = 0.5  # Default medium risk
            
        if eq_depth_col and eq_depth_col in df_earthquake.columns:
            # For depth: deeper = safer, so we invert the normalization
            EQ_depthR = 1 - minmax_norm(pd.to_numeric(df_earthquake[eq_depth_col], errors='coerce'))
        else:
            EQ_depthR = 0.5  # Default medium risk
        
        # Apply methodology weights: 45% magnitude, 35% frequency, 20% depth
        df_earthquake['EarthquakeRisk'] = (
            0.45 * EQ_m5plus.fillna(0.5) + 
            0.35 * EQ_freq.fillna(0.5) + 
            0.20 * EQ_depthR.fillna(0.5)
        ).clip(0, 1)
    
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
    
    # Create hazard safety score using proper weighted formula from methodology
    # First, ensure all required risk columns exist and are normalized to 0-1 scale
    required_risks = ['FloodRisk', 'StormSurgeRisk', 'LandslideRisk', 'EarthquakeRisk']
    
    # Normalize risk columns to 0-1 scale (assuming input is 1-3 scale)
    for col in required_risks:
        if col in df_combined.columns:
            # Convert from 1-3 scale to 0-1 normalized risk
            df_combined[col] = (df_combined[col] - 1.0) / 2.0
            df_combined[col] = df_combined[col].clip(0, 1)
        else:
            df_combined[col] = 0.5  # Default medium risk if missing
    
    # Calculate HydroRisk (hydrometeorological risk) - weighted average of flood and storm surge
    df_combined['HydroRisk'] = 0.5 * df_combined['FloodRisk'] + 0.5 * df_combined['StormSurgeRisk']
    
    # Calculate composite hazard risk using methodology weights
    # HazardRisk_NoFault = 0.40*EarthquakeRisk + 0.40*HydroRisk + 0.20*LandslideRisk
    df_combined['HazardRisk_NoFault'] = (
        0.40 * df_combined['EarthquakeRisk'] + 
        0.40 * df_combined['HydroRisk'] + 
        0.20 * df_combined['LandslideRisk']
    ).clip(0, 1)
    
    # Convert to safety score (higher = safer)
    df_combined['HazardSafety_NoFault'] = (1 - df_combined['HazardRisk_NoFault']).clip(0, 1)
    
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
