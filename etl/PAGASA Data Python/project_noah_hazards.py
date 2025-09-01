import os
import re
import pandas as pd

# ========= User Paths =========
noah_path = "data/raw/project_noah_hazards.csv"
psgc_path = "data/raw/2020_NCR_Geographical/PSGC-July-2025-Publication-Datafile.xlsx"
psgc_sheet = "PSGC"
output_path = "data/cleaned/project_noah_hazards_cleaned.csv"

# ========= Helpers =========
def ensure_dir_for(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def extract_digits(code: str) -> str:
    if pd.isna(code):
        return ""
    return re.sub(r"\D", "", str(code))

def first_n_digits(d: str, n: int) -> str:
    return d[:n] if d and len(d) >= n else ""

def safe_get_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Expected one of {candidates}, got {df.columns}")

def normalize_region_name(region_2digit: str) -> str:
    mapping = {
        "13": "NCR",
        "03": "Region III (Central Luzon)",
        "04": "Region IV-A (Calabarzon)",
    }
    return mapping.get(region_2digit, None)

# ========= Load Project NOAH hazards =========
noah = pd.read_csv(noah_path, dtype=str)
noah = noah.applymap(lambda x: x.strip() if isinstance(x, str) else x)

if "adm4_pcode" not in noah.columns or "date" not in noah.columns:
    raise ValueError("Input CSV must contain 'adm4_pcode' and 'date' columns")

noah["adm4_digits"] = noah["adm4_pcode"].map(extract_digits)
noah["reg_code2"] = noah["adm4_digits"].map(lambda d: first_n_digits(d, 2))

allowed_regions = {"13", "03", "04"}
noah = noah[noah["reg_code2"].isin(allowed_regions)].copy()

noah["Region"] = noah["reg_code2"].map(normalize_region_name)

# ========= Date parsing =========
noah["_dt"] = pd.to_datetime(noah["date"], errors="coerce")
noah["Year"] = noah["_dt"].dt.year
noah["Month"] = noah["_dt"].dt.month
noah["Day"] = noah["_dt"].dt.day
noah.drop(columns=["_dt", "date"], inplace=True, errors="ignore")

# ========= Load PSGC master =========
psgc = pd.read_excel(psgc_path, sheet_name=psgc_sheet, dtype=str)
psgc = psgc.applymap(lambda x: x.strip() if isinstance(x, str) else x)

name_col = safe_get_col(psgc, ["Name", "name"])
level_col = safe_get_col(psgc, ["Level", "Geographic Level", "Administrative Level", "level"])
corr_col = safe_get_col(psgc, ["Correspondence Code", "CorrespondenceCode", "Corr Code", "corr_code"])

psgc["corr_digits"] = psgc[corr_col].map(extract_digits)

# Province table (for Regions III & IV-A)
prov_mask = psgc[level_col].str.lower().str.contains("prov", na=False)
psgc_prov = psgc[prov_mask].copy()
psgc_prov["key_rp4"] = psgc_prov["corr_digits"].map(lambda d: first_n_digits(d, 4))
psgc_prov = psgc_prov[["key_rp4", name_col]].drop_duplicates()
psgc_prov.columns = ["key_rp4", "Province"]

# City table
city_mask = psgc[level_col].str.lower().str.contains("city", na=False)
psgc_city = psgc[city_mask].copy()
psgc_city["key_rpc6"] = psgc_city["corr_digits"].map(lambda d: first_n_digits(d, 6))
psgc_city = psgc_city[["key_rpc6", name_col]].drop_duplicates()
psgc_city.columns = ["key_rpc6", "City"]

# Keys in NOAH
noah["key_rp4"] = noah["adm4_digits"].map(lambda d: first_n_digits(d, 4))
noah["key_rpc6"] = noah["adm4_digits"].map(lambda d: first_n_digits(d, 6))

# Province handling
noah["Province"] = None
noah.loc[noah["reg_code2"] == "13", "Province"] = "Metro Manila"
mask_non_ncr = noah["reg_code2"].isin({"03", "04"})
noah.loc[mask_non_ncr, "Province"] = noah.loc[mask_non_ncr, "key_rp4"].map(
    dict(psgc_prov[["key_rp4", "Province"]].values)
)

# City handling
noah = noah.merge(psgc_city, on="key_rpc6", how="left")

# ========= Clean "City" names =========
if "City" in noah.columns:
    # Remove "City of " at the start
    noah["City"] = noah["City"].str.replace(r"^City of\s+", "", regex=True)

# ========= Clean-up =========
noah.drop(columns=["adm4_digits", "reg_code2", "key_rp4", "key_rpc6"], inplace=True, errors="ignore")
noah = noah.drop_duplicates()

# Drop rows with missing essential info
essential = ["adm4_pcode", "Region", "Province", "City", "Year", "Month", "Day"]
noah = noah.dropna(subset=essential)

# ========= Export =========
ensure_dir_for(output_path)
noah.to_csv(output_path, index=False)

print(f"Done. Cleaned dataset saved to: {output_path}")
print(f"Rows: {len(noah):,} | Columns: {len(noah.columns)}")
