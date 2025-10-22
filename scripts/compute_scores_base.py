#!/usr/bin/env python3
"""
Compute base scores with IPR (Income-to-Payment Ratio).
Based on CP2 methods guide section 5.
"""

import pandas as pd
import numpy as np
import os
from utils import amort, minmax_norm, validate_required_columns


def main():
    # Input and output file paths
    IN_FACT = "data/curated/fact_table_2024.csv"
    OUT = "data/curated/with scores/fact_table_FULL_FINAL.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    
    # Constants
    BASE_RATE_20Y = 0.085  # 8.5% annual
    TERM_YEARS = 20
    
    print("Loading fact table...")
    
    # Load fact table
    try:
        df = pd.read_csv(IN_FACT)
        print(f"✅ Loaded fact table: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading fact table: {e}")
        return
    
    # Validate required columns
    required_cols = ['INC_income_per_hh_2024', 'PRICE_median_2024_final', 
                     'EconomyScore', 'DemandScore', 'HazardSafety_NoFault']
    validate_required_columns(df, required_cols)
    
    print("Computing IPR and scores...")
    
    # Calculate monthly income
    income_mo = df['INC_income_per_hh_2024'] / 12.0
    
    # Calculate monthly amortization payment
    A = amort(df['PRICE_median_2024_final'], BASE_RATE_20Y, TERM_YEARS)
    
    # Calculate IPR (Income-to-Payment Ratio)
    df['IPR_20yr'] = income_mo / A
    
    # Create feasibility score (normalized IPR)
    df['FeasibilityScore'] = minmax_norm(df['IPR_20yr'])
    
    # Create profitability score (40% Economy + 40% Feasibility + 20% Demand)
    df['ProfitabilityScore'] = (0.40 * df['EconomyScore'] + 
                               0.40 * df['FeasibilityScore'] + 
                               0.20 * df['DemandScore'])
    
    # Create final city score (50% Profitability + 50% Hazard Safety)
    df['FinalCityScore'] = (0.50 * df['ProfitabilityScore'] + 
                           0.50 * df['HazardSafety_NoFault'])
    
    # Add additional metrics for analysis
    df['MonthlyIncome'] = income_mo
    df['MonthlyPayment'] = A
    df['AffordabilityScore'] = df['FeasibilityScore']  # Alias for compatibility
    
    # Sort by final score (descending)
    df = df.sort_values('FinalCityScore', ascending=False).reset_index(drop=True)
    
    # Add ranking
    df['Rank'] = range(1, len(df) + 1)
    
    # Save scored fact table
    df.to_csv(OUT, index=False)
    print(f"✅ Scored fact table saved to: {OUT}")
    print(f"   - {len(df)} cities processed")
    
    # Print top 10 cities
    print("\n🏆 Top 10 Cities by Final Score:")
    top_10 = df.head(10)[['Rank', 'City', 'Region', 'FinalCityScore', 'IPR_20yr', 'FeasibilityScore']]
    for _, row in top_10.iterrows():
        print(f"   {row['Rank']:2d}. {row['City']} ({row['Region']}) - Score: {row['FinalCityScore']:.3f}, IPR: {row['IPR_20yr']:.2f}")
    
    # Print summary statistics
    print("\n📊 Score Summary:")
    print(f"   - IPR range: {df['IPR_20yr'].min():.2f} - {df['IPR_20yr'].max():.2f}")
    print(f"   - Feasibility score range: {df['FeasibilityScore'].min():.3f} - {df['FeasibilityScore'].max():.3f}")
    print(f"   - Profitability score range: {df['ProfitabilityScore'].min():.3f} - {df['ProfitabilityScore'].max():.3f}")
    print(f"   - Final city score range: {df['FinalCityScore'].min():.3f} - {df['FinalCityScore'].max():.3f}")
    
    # Print affordability insights
    affordable_cities = df[df['IPR_20yr'] >= 3.0]  # IPR >= 3.0 is generally affordable
    print(f"\n💰 Affordability Insights:")
    print(f"   - Cities with IPR >= 3.0 (affordable): {len(affordable_cities)}")
    print(f"   - Cities with IPR >= 2.0: {len(df[df['IPR_20yr'] >= 2.0])}")
    print(f"   - Cities with IPR >= 1.5: {len(df[df['IPR_20yr'] >= 1.5])}")


if __name__ == "__main__":
    main()
