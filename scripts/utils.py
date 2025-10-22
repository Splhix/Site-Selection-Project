#!/usr/bin/env python3
"""
Utility functions for CP2 Construction Site Selection pipeline.
Common helper functions used across multiple scripts.
"""

import pandas as pd
import numpy as np
import re


def standardize_geo(df, region='Region', province='Province', city='City'):
    """
    Standardize geographic names: strip, title-case, collapse whitespace, replace known aliases.
    
    Args:
        df: DataFrame with geographic columns
        region: Column name for Region
        province: Column name for Province  
        city: Column name for City
        
    Returns:
        DataFrame with cleaned geographic columns
    """
    df = df.copy()
    
    # Known aliases mapping
    region_aliases = {
        'NCR': 'NCR',
        'National Capital Region': 'NCR',
        'Region III': 'Region III (Central Luzon)',
        'Central Luzon': 'Region III (Central Luzon)',
        'Region IV-A': 'Region IV-A (CALABARZON)',
        'CALABARZON': 'Region IV-A (CALABARZON)',
        'Region IVA': 'Region IV-A (CALABARZON)',
    }
    
    province_aliases = {
        'Metro Manila': 'Metro Manila',
        'NCR': 'Metro Manila',
        'Bulacan': 'Bulacan',
        'Pampanga': 'Pampanga',
        'Tarlac': 'Tarlac',
        'Nueva Ecija': 'Nueva Ecija',
        'Bataan': 'Bataan',
        'Zambales': 'Zambales',
        'Aurora': 'Aurora',
        'Cavite': 'Cavite',
        'Laguna': 'Laguna',
        'Batangas': 'Batangas',
        'Rizal': 'Rizal',
        'Quezon': 'Quezon',
    }
    
    # Clean and standardize each geographic column
    for col_name, col_key in [(region, 'region'), (province, 'province'), (city, 'city')]:
        if col_name in df.columns:
            # Strip whitespace and title case
            df[col_name] = df[col_name].astype(str).str.strip().str.title()
            
            # Collapse multiple whitespace
            df[col_name] = df[col_name].str.replace(r'\s+', ' ', regex=True)
            
            # Apply aliases
            if col_key == 'region':
                df[col_name] = df[col_name].map(region_aliases).fillna(df[col_name])
            elif col_key == 'province':
                df[col_name] = df[col_name].map(province_aliases).fillna(df[col_name])
    
    return df


def minmax_norm(s: pd.Series) -> pd.Series:
    """
    Min-max normalize a pandas Series to [0,1] range.
    
    Args:
        s: Input Series to normalize
        
    Returns:
        Normalized Series with values in [0,1] range
    """
    s = s.astype(float)
    mask = s.notna()
    
    if mask.sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    
    lo, hi = s[mask].min(), s[mask].max()
    
    if hi == lo:
        return pd.Series(np.nan, index=s.index)
    
    return (s - lo) / (hi - lo)


def amort(price: pd.Series, r_annual: float, years: int = 20) -> pd.Series:
    """
    Calculate fixed-rate mortgage (annuity) payment per month.
    
    Args:
        price: Series of property prices
        r_annual: Annual interest rate (e.g., 0.085 for 8.5%)
        years: Loan term in years (default: 20)
        
    Returns:
        Series of monthly payment amounts
    """
    r = r_annual / 12.0
    n = years * 12
    
    # Avoid division by zero if r==0 (edge case)
    if abs(r) < 1e-12:
        return price / n
    
    return price * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def safe_log(s: pd.Series, min_val: float = 1e-6) -> pd.Series:
    """
    Safe log transformation that handles zeros and negative values.
    
    Args:
        s: Input Series
        min_val: Minimum value to use for clipping (default: 1e-6)
        
    Returns:
        Log-transformed Series
    """
    s_clipped = s.clip(lower=min_val)
    return np.log(s_clipped)


def inv_log(s: pd.Series) -> pd.Series:
    """
    Inverse log transformation (exponential).
    
    Args:
        s: Log-transformed Series
        
    Returns:
        Original scale Series
    """
    return np.exp(s)


def safe_logit(s: pd.Series, min_val: float = 1e-6, max_val: float = 1-1e-6) -> pd.Series:
    """
    Safe logit transformation for bounded [0,1] data.
    
    Args:
        s: Input Series (should be in [0,1] range)
        min_val: Minimum value for clipping
        max_val: Maximum value for clipping
        
    Returns:
        Logit-transformed Series
    """
    s_clipped = s.clip(lower=min_val, upper=max_val)
    return np.log(s_clipped / (1 - s_clipped))


def inv_logit(s: pd.Series) -> pd.Series:
    """
    Inverse logit transformation (sigmoid).
    
    Args:
        s: Logit-transformed Series
        
    Returns:
        Original [0,1] scale Series
    """
    return 1 / (1 + np.exp(-s))


def clip_percent(val: float, min_pct: float = 0.0, max_pct: float = 100.0) -> float:
    """
    Clip a value to percentage range [0, 100].
    
    Args:
        val: Value to clip
        min_pct: Minimum percentage (default: 0.0)
        max_pct: Maximum percentage (default: 100.0)
        
    Returns:
        Clipped value
    """
    return max(min_pct, min(max_pct, val))


def enforce_positive(s: pd.Series, min_val: float = 1e-6) -> pd.Series:
    """
    Ensure all values in Series are positive.
    
    Args:
        s: Input Series
        min_val: Minimum positive value to use
        
    Returns:
        Series with all values >= min_val
    """
    return s.clip(lower=min_val)


def winsorize_series(s: pd.Series, lower_pct: float = 0.05, upper_pct: float = 0.95) -> pd.Series:
    """
    Winsorize a Series by clipping extreme values.
    
    Args:
        s: Input Series
        lower_pct: Lower percentile for clipping (default: 0.05)
        upper_pct: Upper percentile for clipping (default: 0.95)
        
    Returns:
        Winsorized Series
    """
    lower_bound = s.quantile(lower_pct)
    upper_bound = s.quantile(upper_pct)
    return s.clip(lower=lower_bound, upper=upper_bound)


def validate_required_columns(df: pd.DataFrame, required_cols: list) -> None:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def ensure_numeric(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Ensure specified columns are numeric, converting if necessary.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of column names to convert to numeric
        
    Returns:
        DataFrame with numeric columns converted
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for filesystem
    """
    # Replace invalid characters with underscores
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


# --------------------
# GROWTH / FORECAST HELPERS
# --------------------
def cagr(value_start: float, value_end: float, periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        value_start: Starting value
        value_end: Ending value
        periods: Number of periods
        
    Returns:
        CAGR as decimal (e.g., 0.05 for 5%)
    """
    try:
        if value_start <= 0 or periods <= 0:
            return None
        return (value_end / value_start) ** (1/periods) - 1
    except Exception:
        return None


def apply_cagr_forward(series, years_forward: int) -> float:
    """
    Given a chronological list/Series with at least two points, compute CAGR between 
    first and last and project forward by years_forward.
    
    Args:
        series: List or Series of values in chronological order
        years_forward: Number of years to project forward
        
    Returns:
        Projected value
    """
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return None
    r = cagr(float(s.iloc[0]), float(s.iloc[-1]), len(s)-1)
    if r is None:
        return None
    return float(s.iloc[-1]) * ((1 + r) ** years_forward)


# OPTIONAL: ETS utilities if statsmodels is available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    def ets_forecast_to_2024(year_value_df: pd.DataFrame, year_col='Year', value_col='Value') -> float:
        """
        Forecast to 2024 with simple ETS(A,N,N). 
        
        Args:
            year_value_df: DataFrame with year and value columns
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            Forecasted value for 2024
        """
        yv = year_value_df.sort_values(year_col)
        end_year = int(yv[year_col].max())
        steps = 2024 - end_year
        
        if steps <= 0:  # already have 2024
            return float(yv.loc[yv[year_col]==2024, value_col].iloc[0])
        
        model = ExponentialSmoothing(yv[value_col].astype(float), trend='add', seasonal=None)
        fit = model.fit(optimized=True)
        fc = fit.forecast(steps)
        return float(fc.iloc[-1])
        
except ImportError:
    def ets_forecast_to_2024(*args, **kwargs):
        """Fallback when statsmodels is not available."""
        return None
