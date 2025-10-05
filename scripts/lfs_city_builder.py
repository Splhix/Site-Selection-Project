# lfs_city_by_work_only.py
import pandas as pd, numpy as np, re
import os, glob

"""Build monthly city aggregates for full years 2021–2023 (Jan–Dec)."""

# ========= CONFIG =========
YEARS = [2021, 2022, 2023]

# Discover monthly LFS files under data/raw/Workforce/<year>/
MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

def infer_month_from_name(filename):
    base = os.path.basename(filename).lower()
    # Prefer month names
    for name, idx in MONTH_NAME_TO_NUM.items():
        if name in base:
            return idx
    # Try to catch patterns with _MM_ or -MM- or MM as a standalone token
    m = re.search(r"(?<!\d)(1[0-2]|0?[1-9])(?!\d)", base)
    if m:
        mm = int(m.group(1))
        if 1 <= mm <= 12:
            return mm
    return None

def discover_lfs_files(years):
    mapping = {}
    for y in years:
        dir_path = os.path.join("data", "raw", "Workforce", str(y))
        for fp in sorted(glob.glob(os.path.join(dir_path, "*.csv"))):
            m = infer_month_from_name(fp)
            if m is None:
                continue
            key = f"{y}-{m:02d}"
            # Keep first seen per (year, month)
            if key not in mapping:
                mapping[key] = fp
    return mapping

LFS_FILES = discover_lfs_files(YEARS)

OUT_MAIN = "lfs_city_monthly_agg_2021_2023_BYCODE_WORKONLY.csv"

# Scope codes (every code appears monthly even if no rows)
VALID_CODES = [
    7501,3901,3902,3903,3904,3905,3906,3907,3908,3909,3910,3911,3912,3913,3914,
    7601,7602,7502,7401,7402,7603,7503,7604,7605,7403,7404,7405,7607,7504,803,
    1410,1412,1420,4903,4908,4919,4926,4917,5401,5409,5416,6916,7107,1005,1014,
    1031,2103,2105,2104,2106,2108,2109,2119,2122,3403,3404,3405,3424,3425,3428,
    5624,5647,5802
]

# ========= HELPERS =========
def extract_code(val):
    """Extract leftmost 3–5 digits from '3901 Manila...' or handle '3901.0'."""
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r"^(\d{3,5})", s)
    if m: return int(m.group(1))
    try: return int(float(s))
    except: return np.nan

def find_col(df, requires_all=None, fallback_contains=None):
    """Tolerant finder: prefer ALL tokens; fallback to single token."""
    requires_all = requires_all or []
    for c in df.columns:
        cl = c.lower()
        if all(tok.lower() in cl for tok in requires_all):
            return c
    if fallback_contains:
        for c in df.columns:
            if fallback_contains.lower() in c.lower():
                return c
    return None

def nec_flags(nec_series):
    """NEC: 1=employed, 2=unemployed; LF = 1 or 2."""
    s = nec_series.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)  # '1.0' -> '1'
    emp = s.eq("1")
    unemp = s.eq("2")
    lf = emp | unemp
    return emp, unemp, lf

# ========= PROCESS =========
frames = []

for ym, path in LFS_FILES.items():
    df = pd.read_csv(path)
    year, month = map(int, ym.split("-"))

    # Detect columns across month variants
    col_age = find_col(df, ["age", "last", "birthday"]) or find_col(df, fallback_contains="age")
    col_wt  = find_col(df, ["final", "weight"]) or find_col(df, fallback_contains="weight")
    col_nec = find_col(df, ["new", "employment", "criteria"]) or find_col(df, fallback_contains="employment criteria")
    col_loc = find_col(df, ["location", "work"]) or find_col(df, ["province", "municipality"])
    if not all([col_age, col_wt, col_nec, col_loc]):
        raise KeyError(f"Missing required columns in {path}. Found: {list(df.columns)}")

    # Age >= 15
    df = df[pd.to_numeric(df[col_age], errors="coerce") >= 15].copy()

    # Work-location code (only employed will have this)
    work_code = df[col_loc].apply(extract_code).astype("Int64")

    # Flags & weights
    emp, unemp, lf = nec_flags(df[col_nec])
    w = pd.to_numeric(df[col_wt], errors="coerce").fillna(0.0)

    # Keep only rows with a valid work code AND in scope
    # (Unemployed typically have blank work location; we don't assign them.)
    mask_in_scope = work_code.isin(VALID_CODES)
    # Row-level contributions for employed by work location
    rows = pd.DataFrame({
        "year": year,
        "month": month,
        "loc_code": work_code.where(mask_in_scope, np.nan),
        "EMP_w": w * emp.to_numpy(dtype=int),
        "EMP_n": emp.to_numpy(dtype=int),
    })
    rows = rows.dropna(subset=["loc_code"])
    rows["loc_code"] = rows["loc_code"].astype("Int64")

    if not rows.empty:
        frames.append(rows)

# Aggregate EMP by work location
if frames:
    emp_agg = (pd.concat(frames, ignore_index=True)
                 .groupby(["year","month","loc_code"], as_index=False)
                 .sum(numeric_only=True))
else:
    emp_agg = pd.DataFrame(columns=["year","month","loc_code","EMP_w","EMP_n"])

# ========= ENSURE COMPLETE PANEL (Jan–Dec × 2021–2023 × all codes) =========
months = list(range(1, 13))
panel = pd.MultiIndex.from_product([YEARS, months, VALID_CODES],
                                   names=["year","month","loc_code"]).to_frame(index=False)

out = panel.merge(emp_agg, on=["year","month","loc_code"], how="left")

# Keep only truthful fields; set unknowns to NaN (not computable by work-location)
for col in ["EMP_w","EMP_n"]:
    out[col] = pd.to_numeric(out[col], errors="coerce")  # keep NaN if no employed rows that month

# Add placeholders (NaN) for fields that are NOT identifiable by work-location
out["UNEMP_w"]  = np.nan
out["LF_w"]     = np.nan
out["POP15p_w"] = np.nan
out["UNEMP_n"]  = np.nan
out["LF_n"]     = np.nan
out["POP15p_n"] = np.nan

# Order columns
out = out[[
    "year","month","loc_code",
    "EMP_w","EMP_n",
    "UNEMP_w","LF_w","POP15p_w",
    "UNEMP_n","LF_n","POP15p_n",
]]

out.to_csv(OUT_MAIN, index=False)
print(f"[OK] Saved {OUT_MAIN} with {len(out)} rows (2021–2023 Jan–Dec panel by work-location; no invented LF/UNEMP/POP15+)")
