#!/usr/bin/env python3
"""
Preprocess housing prices data.
Based on CP2 methods guide section 1.5.
"""

import pandas as pd
import numpy as np
import os
from utils import standardize_geo, validate_required_columns


def main():
    # File paths
    raw_path = "data/raw/Feasibility/Housing_v2.csv"
    output_path = "data/cleaned/feasibility/housing_prices_completed_PURE_INHERITANCE.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw data
    print(f"Loading housing prices data from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Basic cleaning
    df = df.dropna(how='all')
    
    # Standardize geographic names
    df = standardize_geo(df)
    
    # Filter for target regions
    target_regions = ['NCR', 'Region III (Central Luzon)', 'Region IV-A (CALABARZON)']
    df = df[df['Region'].isin(target_regions)]
    
    # Find price column (look for median or price-related columns)
    price_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['price', 'median', 'cost', 'value'])]
    
    if not price_cols:
        raise ValueError("Could not find price column in housing data")
    
    # Use the first price column found
    price_col = price_cols[0]
    df['PRICE_median_2024_final'] = pd.to_numeric(df[price_col], errors='coerce')
    
    # Remove rows with missing price data
    df = df.dropna(subset=['PRICE_median_2024_final'])
    
    # Outlier inspection and filtering
    # Remove prices that are too low (< 100,000) or too high (> 50,000,000)
    df = df[(df['PRICE_median_2024_final'] >= 100000) & 
            (df['PRICE_median_2024_final'] <= 50000000)]
    
    # Group by city and calculate median price
    city_prices = df.groupby(['Region', 'Province', 'City'])['PRICE_median_2024_final'].median().reset_index()
    
    # Ensure minimum count sanity check (at least 3 listings per city)
    city_counts = df.groupby(['Region', 'Province', 'City']).size().reset_index(name='count')
    city_prices = city_prices.merge(city_counts, on=['Region', 'Province', 'City'])
    city_prices = city_prices[city_prices['count'] >= 3]  # Minimum 3 listings
    
    # Prepare output
    output_cols = ['Region', 'Province', 'City', 'PRICE_median_2024_final']
    df_output = city_prices[output_cols].copy()
    
    # Sort by Region, Province, City
    df_output = df_output.sort_values(['Region', 'Province', 'City']).reset_index(drop=True)
    
    # Save cleaned data
    df_output.to_csv(output_path, index=False)
    print(f"✅ Housing prices data cleaned and saved to: {output_path}")
    print(f"   - {len(df_output)} cities processed")
    print(f"   - Price range: {df_output['PRICE_median_2024_final'].min():,.0f} - {df_output['PRICE_median_2024_final'].max():,.0f}")
    print(f"   - Regions: {df_output['Region'].unique()}")


if __name__ == "__main__":
    main()
