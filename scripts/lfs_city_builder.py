# lfs_city_builder.py  (panel-safe, with backfill + QA)
import pandas as pd, numpy as np, re
from pathlib import Path

# ========= CONFIG =========
PSGC_FILE = "data/cleaned/Revised Cleaned Files/PSGC_master_aliases (1).csv"
LFS_PSGC_MAP = "data/reference/LFS-PSGC Code Matching.xlsx"
LFS_FILES = {
    "2024-01": "data/raw/Workforce/lfs_january_2024_714186541812.csv",
    "2024-02": "data/raw/Workforce/lfs_february_2024_1155652079312.csv",
    "2024-03": "data/raw/Workforce/lfs_march_2024_1259594005757.csv",
    "2024-04": "data/raw/Workforce/lfs_april_2024_1612512661365.csv",
    "2024-05": "data/raw/Workforce/lfs_may_2024_1523919684245.csv",
    "2024-06": "data/raw/Workforce/lfs_june_2024_712879761172.csv",
}
OUT_MAIN = "lfs_city_monthly_agg_2024.csv"
OUT_MISSING_CODES = "lfs_city_missing_codes_by_month.csv"
OUT_GAPS = "lfs_city_missing_cities_by_month.csv"

# Optional: enforce the same 50 cities per month (panel)
EXPECTED_CITIES = [
 "CALOOCAN CITY","CITY OF MANILA","LAS PINAS CITY","MAKATI CITY","MALABON CITY","MANDALUYONG CITY",
 "MARIKINA CITY","MUNTINLUPA CITY","NAVOTAS CITY","PARANAQUE CITY","PASAY CITY","PASIG CITY",
 "QUEZON CITY","SAN JUAN CITY","TAGUIG CITY","VALENZUELA CITY","BALANGA CITY","MALOLOS CITY",
 "MEYCAUAYAN CITY","SAN JOSE DEL MONTE CITY","CABANATUAN CITY","GAPAN CITY","PALAYAN CITY",
 "SAN JOSE CITY","SCIENCE CITY OF MUNOZ","ANGELES CITY","MABALACAT CITY","SAN FERNANDO CITY",
 "TARLAC CITY","OLONGAPO CITY","BATANGAS CITY","LIPA CITY","TANAUAN CITY","BACOOR CITY","CAVITE CITY",
 "CITY OF CARMONA","DASMARINAS CITY","GENERAL TRIAS CITY","IMUS CITY","TAGAYTAY CITY","TRECE MARTIRES CITY",
 "BINAN CITY","CABUYAO CITY","CALAMBA CITY","SAN PABLO CITY","SAN PEDRO CITY","SANTA ROSA CITY",
 "LUCENA CITY","TAYABAS CITY","ANTIPOLO CITY"
]

# ========= HELPERS =========
def extract_loc_code(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r"^(\d{3,5})", s)  # allow up to 5 just in case
    if m: return int(m.group(1))
    try: return int(float(s))        # handles "3901.0"
    except: return np.nan

def find_col(df, includes, fallback_contains=None):
    for c in df.columns:
        cname = c.lower()
        if all(sub.lower() in cname for sub in includes):
            return c
    if fallback_contains:
        for c in df.columns:
            if fallback_contains.lower() in c.lower():
                return c
    return None

def std_geo(df):
    for col in ["Region","Province","City"]:
        if df[col] == "NCR":
            df[col] = "NCR"
        elif df[col] == "Central Luzon":
            df[col] = "Region III (Central Luzon)"
        elif df[col] == "CALABARZON":
            df[col] = "Region IV-A (CALABARZON)"
        else:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

# ========= LOAD PSGC & VALID TUPLES =========
psgc = pd.read_csv(PSGC_FILE, dtype=str)
psgc = std_geo(psgc)
valid_keys = set(tuple(x) for x in psgc[["Region","Province","City"]].itertuples(index=False, name=None))

# ========= LOAD CODE MAPPER (Excel) =========
xls = pd.ExcelFile(LFS_PSGC_MAP)
map_df = pd.read_excel(xls, xls.sheet_names[0])
map_df.columns = [c.strip() for c in map_df.columns]

# Detect columns
def pick(colnames, alts):
    for c in colnames:
        if c.lower() in [a.lower() for a in alts]: return c
    for c in colnames:
        cl = c.lower()
        if any(a.lower() in cl for a in alts): return c
    return None

code_col = pick(map_df.columns, ["loc_code","lfs_code","code","c11","c12"])
region_col = pick(map_df.columns, ["Region"])
prov_col   = pick(map_df.columns, ["Province","Prov"])
city_col   = pick(map_df.columns, ["City","City/Municipality","LGU","city_raw","CityName"])

if code_col is None:  # last resort: first mostly-numeric col
    pct_numeric = {c: pd.to_numeric(map_df[c], errors="coerce").notna().mean() for c in map_df.columns}
    code_col = max(pct_numeric, key=pct_numeric.get)

mapper = map_df.copy()
mapper["loc_code"] = pd.to_numeric(mapper[code_col].apply(extract_loc_code), errors="coerce").astype("Int64")

if not (region_col and prov_col and city_col):
    raise KeyError("Your LFS-PSGC Code Matching file must contain Region/Province/City columns.")

mapper["Region"] = mapper[region_col]
mapper["Province"] = mapper[prov_col]
mapper["City"] = mapper[city_col]
mapper = std_geo(mapper)
mapper = mapper[["loc_code","Region","Province","City"]].dropna().drop_duplicates("loc_code")

# ========= PASS 1: READ ALL MONTHS, MERGE, BUILD 'OBSERVED' BACKFILL =========
rows = []
missing_codes = []  # per-month missing loc_codes (for your mapper patch)
raw_loc_text_samples = []  # optional: to help identify what the LFS string looked like

for ym, path in LFS_FILES.items():
    if not Path(path).exists():
        print(f"[WARN] Missing {path}, skipping")
        continue

    df = pd.read_csv(path)
    year, month = map(int, ym.split("-"))

    col_age = find_col(df, ["age as of last birthday"]) or find_col(df, ["age"])
    col_wt  = find_col(df, ["final weight"]) or find_col(df, ["weight"])
    col_nec = find_col(df, ["new employment criteria"]) or find_col(df, ["employment criteria"])
    col_loc = find_col(df, ["location of work","province"]) or find_col(df, ["location of work"])

    if not all([col_age, col_wt, col_nec, col_loc]):
        raise KeyError(f"Missing required columns in {path}")

    # Population 15+
    df = df[pd.to_numeric(df[col_age], errors="coerce") >= 15].copy()
    df["loc_code"] = df[col_loc].apply(extract_loc_code).astype("Int64")

    # Merge to mapper (first try)
    j = df.merge(mapper, on="loc_code", how="left", suffixes=("_LFS",""))
    # collect missing codes for this month
    miss = j.loc[j["City"].isna(), ["loc_code", col_loc]].dropna().copy()
    miss["year"] = year; miss["month"] = month
    if not miss.empty:
        missing_codes.append(miss[["year","month","loc_code"]])
        # keep a small sample of raw location text to aid mapping
        raw_loc_text_samples.append(miss.head(100).rename(columns={col_loc:"raw_loc_text"}))

    # Keep ALL rows here; backfill happens after we build observed map
    j["year"] = year; j["month"] = month
    j["NEC_col"] = col_nec
    j["WT_col"] = col_wt
    rows.append(j[["year","month","loc_code","Region","Province","City","NEC_col","WT_col"]])

# Build 'observed' mapper from rows that DID map in any month
all_rows = pd.concat(rows, ignore_index=True)
obs_map = (all_rows.dropna(subset=["City"])
                    .groupby("loc_code")[["Region","Province","City"]]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.dropna().iloc[0])
                    .reset_index())

# ========= PASS 2: REPROCESS MONTHS WITH OBSERVED BACKFILL =========
frames = []
for ym, path in LFS_FILES.items():
    if not Path(path).exists(): 
        continue

    df = pd.read_csv(path)
    year, month = map(int, ym.split("-"))

    col_age = find_col(df, ["age as of last birthday"]) or find_col(df, ["age"])
    col_wt  = find_col(df, ["final weight"]) or find_col(df, ["weight"])
    col_nec = find_col(df, ["new employment criteria"]) or find_col(df, ["employment criteria"])
    col_loc = find_col(df, ["location of work","province"]) or find_col(df, ["location of work"])

    df = df[pd.to_numeric(df[col_age], errors="coerce") >= 15].copy()
    df["loc_code"] = df[col_loc].apply(extract_loc_code).astype("Int64")

    # Merge mapper, then backfill from observed map
    j = df.merge(mapper, on="loc_code", how="left", suffixes=("_LFS",""))
    j = j.merge(obs_map, on="loc_code", how="left", suffixes=("","_OBS"))

    # Use mapped columns; if NA, use observed backfill
    for col in ["Region","Province","City"]:
        j[col] = j[col].where(j[col].notna(), j[f"{col}_OBS"])
    j.drop(columns=[c for c in j.columns if c.endswith("_OBS")], inplace=True)

    # Keep mapped + valid tuples
    j = j[j["Region"].notna() & j["Province"].notna() & j["City"].notna()].copy()
    j = std_geo(j)
    j = j[j.apply(lambda r: (r["Region"], r["Province"], r["City"]) in valid_keys, axis=1)].copy()

    emp = j[col_nec].astype(str).str.strip().isin(["1"])
    unemp = j[col_nec].astype(str).str.strip().isin(["2"])
    lf = emp | unemp
    w = pd.to_numeric(j[col_wt], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "year": year,
        "month": month,
        "Region": j["Region"],
        "Province": j["Province"],
        "City": j["City"],
        "POP15p_w": w,
        "EMP_w": w * emp.astype(int),
        "UNEMP_w": w * unemp.astype(int),
        "LF_w": w * lf.astype(int),
        "POP15p_n": 1,
        "LF_n": lf.astype(int),
    })
    frames.append(out)

agg = (pd.concat(frames, ignore_index=True)
         .groupby(["year","month","Region","Province","City"], as_index=False)
         .sum(numeric_only=True))

# ========= OPTIONAL: enforce 50 cities per month (fill zeros for missing ones) =========
if EXPECTED_CITIES:
    # derive expected (Region/Province) for those cities from PSGC
    psgc_sub = psgc[psgc["City"].isin(EXPECTED_CITIES)][["Region","Province","City"]].drop_duplicates()
    all_months = agg[["year","month"]].drop_duplicates()
    panel = (all_months.assign(key=1)
                .merge(psgc_sub.assign(key=1), on="key", how="left")
                .drop("key", axis=1))
    agg = panel.merge(agg, on=["year","month","Region","Province","City"], how="left")
    # fill zeros for missing
    for col in ["POP15p_w","EMP_w","UNEMP_w","LF_w","POP15p_n","LF_n"]:
        agg[col] = agg[col].fillna(0)

# ========= SAVE =========
agg.to_csv(OUT_MAIN, index=False)

# Save per-month missing codes for your Excel mapper patching
if missing_codes:
    miss = pd.concat(missing_codes, ignore_index=True).drop_duplicates().sort_values(["year","month","loc_code"])
    miss.to_csv(OUT_MISSING_CODES, index=False)

# Missing cities per month (QA)
full_set = set(EXPECTED_CITIES) if EXPECTED_CITIES else set(agg["City"].unique())
gaps = []
for (y,m), g in agg.groupby(["year","month"]):
    have = set(g["City"].unique())
    for c in sorted(full_set - have):
        gaps.append({"year": y, "month": m, "missing_city": c})
pd.DataFrame(gaps).to_csv(OUT_GAPS, index=False)

print(f"[OK] Saved {OUT_MAIN} ({len(agg)} rows)")
print(f"[OK] Missing loc_codes by month -> {OUT_MISSING_CODES}")
print(f"[OK] Missing cities by month  -> {OUT_GAPS}")
