#!/usr/bin/env python3
"""
Extrapolate income per household data to 2024.
Based on CP2 methods guide section 4.2.
"""

import pandas as pd
import numpy as np
import os
from utils import apply_cagr_forward


def to_2024(group):
    """Extrapolate a group's data to 2024 using CAGR or carry-forward."""
    yv = group[['Year', 'INC_income_per_hh']].dropna().sort_values('Year')
    yrs = yv['Year'].values
    
    if len(yrs) >= 2:
        # Use CAGR if we have >=2 points
        v2024 = apply_cagr_forward(yv['INC_income_per_hh'].tolist(), 2024 - int(yrs.max()))
        method = "CAGR"
    else:
        # Carry forward if insufficient data
        v2024 = float(yv['INC_income_per_hh'].iloc[-1])
        method = "CarryForward"
    
    return pd.Series({
        "INC_income_per_hh_2024": v2024,
        "extrapolation_method": method
    })


def main():
    # File paths
    INP = "data/cleaned/income_per_hh_panel.csv"
    OUT = "data/extrapolated/income_per_hh_2024.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    
    print("Loading income panel data...")
    
    # Load panel data
    try:
        df = pd.read_csv(INP)
        print(f"✅ Loaded income panel data: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading income panel data: {e}")
        return
    
    print("Extrapolating to 2024...")
    
    # Apply extrapolation by Region, Province, City
    out = (df.groupby(['Region', 'Province', 'City'], as_index=False)
             .apply(to_2024))
    
    # Sort by Region, Province, City
    out = out.sort_values(['Region', 'Province', 'City']).reset_index(drop=True)
    
    # Save extrapolated data
    out.to_csv(OUT, index=False)
    print(f"✅ Income extrapolated to 2024 and saved to: {OUT}")
    print(f"   - {len(out)} cities processed")
    
    # Print method summary
    method_counts = out['extrapolation_method'].value_counts()
    print("\n📊 Extrapolation Methods Used:")
    for method, count in method_counts.items():
        print(f"   - {method}: {count} cities")
    
    print(f"\n💰 Income per household 2024 range: {out['INC_income_per_hh_2024'].min():,.0f} - {out['INC_income_per_hh_2024'].max():,.0f}")


if __name__ == "__main__":
    main()
