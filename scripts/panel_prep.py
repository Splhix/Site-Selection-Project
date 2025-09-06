# panel_prep.py
import pandas as pd

def ensure_panel(df: pd.DataFrame, id_col: str, year_col: str, value_col: str):
    df = df[[id_col, year_col, value_col]].dropna()
    df[year_col] = df[year_col].astype(int)
    df = df.sort_values([id_col, year_col]).drop_duplicates([id_col, year_col], keep="last")
    return df
