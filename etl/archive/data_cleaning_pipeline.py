import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

# ensure output directory exists
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# required columns for validation
REQUIRED_COLUMNS = ["Region", "Province", "City"]

# rounding rules
ROUND_DECIMALS = {
    "IPR_20yr": 3,
    "HazardSafety_NoFault": 3,
    "GRDP_grdp_pc_2024_const": 2,
    "INC_income_per_hh_2024": 2,
}

# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_dataset(file_name: str):
    raw_path = RAW_DIR / file_name
    df = pd.read_csv(raw_path)
    print(f"Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 1. Strip whitespace from text columns
    text_cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # --- 2. Validate presence of key columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- 3. Standardize capitalization for key location fields
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].str.title()

    # --- 4. Handle duplicates
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first")

    # --- 5. Convert numeric columns safely
    for col in df.columns:
        if df[col].dtype == "object":
            # skip clear text columns
            if col in REQUIRED_COLUMNS or "Name" in col or "Scenario" in col or "Model" in col:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception as e:
                print(f"⚠️ Skipped numeric conversion for {col}: {e}")

    # --- 6. Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fill_log = []
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fill_log.append(f"Filled {col} with median ({median_val})")

    # --- 7. Round numeric columns
    for col, dec in ROUND_DECIMALS.items():
        if col in df.columns:
            df[col] = df[col].round(dec)

    # --- 8. Sort by geography
    df = df.sort_values(REQUIRED_COLUMNS).reset_index(drop=True)

    # --- 9. Optional: enforce year consistency (default 2024)
    if "year" in df.columns:
        if 2024 in df["year"].unique():
            df = df[df["year"] == 2024]
        else:
            print("⚠️ Warning: No 2024 records found; keeping all years.")

    # --- 10. Save cleaned output
    cleaned_name = file_name.replace("_RAW", "_CLEAN")
    output_path = CLEANED_DIR / cleaned_name
    df.to_csv(output_path, index=False)

    # --- 11. Save log summary
    log_path = CLEANED_DIR / (cleaned_name.replace(".csv", "_LOG.txt"))
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Cleaning log for {file_name}\n")
        log.write(f"Timestamp: {datetime.now()}\n")
        log.write(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        log.write("Missing values per column:\n")
        log.write(df.isna().sum().to_string())
        log.write("\n\n")
        if fill_log:
            log.write("Median fill operations:\n" + "\n".join(fill_log))
        else:
            log.write("No missing numeric values filled.\n")

    # --- 12. Console summary
    print(f"✅ Cleaned file saved to: {output_path}")
    print(f"📄 Log saved to: {log_path}")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.isna().sum())

    return df


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Example: clean dummy fact table
    clean_dataset("dummy_fact_table_app_ready_RAW.csv")
