#!/usr/bin/env python3
"""
Extrapolate labor (employment rate) data to 2024.
Based on CP2 methods guide section 4.4.
"""

import pandas as pd
import numpy as np
import os
from utils import ets_forecast_to_2024, apply_cagr_forward


def to_2024(group):
    """Extrapolate a group's data to 2024 using ETS, CAGR, or carry-forward."""
    yv = group[['Year', 'EmploymentRate']].dropna().sort_values('Year')
    yrs = yv['Year'].unique()
    
    if len(yrs) >= 4 and ets_forecast_to_2024 is not None:
        # Use ETS if we have >=4 annual points
        v2024 = ets_forecast_to_2024(yv.rename(columns={'EmploymentRate': 'Value'}))
        method = "ETS"
    elif len(yrs) >= 2:
        # Use CAGR if we have >=2 points
        v2024 = apply_cagr_forward(yv['EmploymentRate'].tolist(), 2024 - int(yrs.max()))
        method = "CAGR"
    else:
        # Carry forward if insufficient data
        v2024 = float(yv['EmploymentRate'].iloc[-1])
        method = "CarryForward"
    
    return pd.Series({
        "EmploymentRate_2024": v2024,
        "extrapolation_method": method
    })


def main():
    # File paths
    INP = "data/cleaned/labor_employment_region_panel.csv"
    OUT = "data/extrapolated/labor_employment_region_2024.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    
    print("Loading labor panel data...")
    
    # Load panel data
    try:
        df = pd.read_csv(INP)
        print(f"✅ Loaded labor panel data: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading labor panel data: {e}")
        return
    
    print("Extrapolating to 2024...")
    
    # Apply extrapolation by region
    out = df.groupby('Region', as_index=False).apply(to_2024)
    
    # Sort by Region
    out = out.sort_values('Region').reset_index(drop=True)
    
    # Save extrapolated data
    out.to_csv(OUT, index=False)
    print(f"✅ Labor extrapolated to 2024 and saved to: {OUT}")
    print(f"   - {len(out)} regions processed")
    
    # Print method summary
    method_counts = out['extrapolation_method'].value_counts()
    print("\n📊 Extrapolation Methods Used:")
    for method, count in method_counts.items():
        print(f"   - {method}: {count} regions")
    
    print(f"\n💼 Employment rate 2024 range: {out['EmploymentRate_2024'].min():.3f} - {out['EmploymentRate_2024'].max():.3f}")


if __name__ == "__main__":
    main()
