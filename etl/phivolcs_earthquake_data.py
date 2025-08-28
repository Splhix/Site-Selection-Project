import pandas as pd
import os
import re

# === File Paths ===
raw_path = "data/raw/phivolcs_earthquake_data.xlsx"
output_path = "data/cleaned/phivolcs_earthquake_data_cleaned.csv"

# === Canonical Region Labels we will ALLOW in the final file ===
REGION_LABELS_ALLOWED = {
    "NCR": "NCR",
    "III": "Region III (Central Luzon)",
    "IV-A": "Region IV-A (CALABARZON)",
}

ALLOWED_CODES = set(REGION_LABELS_ALLOWED.keys())  # {"NCR","III","IV-A"}

# --- Normalizers ---
def _norm_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', str(s)).strip().lower()
    s = s.replace('–','-').replace('—','-')
    s = s.replace('ñ','n').replace('á','a').replace('í','i').replace('ó','o').replace('ú','u')
    return s

def normalize_city_province(cp: str):
    if pd.isna(cp):
        return cp
    cp = re.sub(r'\s+', ' ', str(cp)).strip(' .,-')
    cp = re.sub(r'\bCity Of\b', 'City of', cp, flags=re.IGNORECASE)
    cp = cp.title()
    cp = re.sub(r'\b(Del|De|La|Las|Los)\b', lambda m: m.group(1).lower(), cp)
    return cp

def normalize_region_text_to_code(value: str) -> str:
    """
    From any Region text, return a canonical code among {NCR, III, IV-A} if detectable,
    else '' (unknown).
    """
    if pd.isna(value):
        return ""
    s = _norm_text(value)
    # Direct detects
    if s in {"ncr", "national capital region"}:
        return "NCR"
    if "central luzon" in s or s in {"iii","region iii","region 3","3"}:
        return "III"
    if "calabarzon" in s or s in {"iv-a","iv a","region iv-a","region iv a"}:
        return "IV-A"
    return ""

# Provinces we care about (only those mapping to the ALLOWED codes)
PROVINCE_TO_ALLOWED_CODE = {
    # Region III (Central Luzon)
    "aurora": "III", "bataan": "III", "bulacan": "III", "nueva ecija": "III", "pampanga": "III", "tarlac": "III", "zambales": "III",
    # Region IV-A (CALABARZON)
    "batangas": "IV-A", "cavite": "IV-A", "laguna": "IV-A", "quezon": "IV-A", "rizal": "IV-A",
}

# NCR cities (lowercase)
NCR_CITIES = {
    "caloocan","las piñas","las pinas","makati","malabon","mandaluyong","manila","marikina","muntinlupa",
    "navotas","parañaque","paranaque","pasay","pasig","quezon city","san juan","taguig","valenzuela","pateros"
}

# Some well-known HUCs inside III and IV-A that may appear without province
CITY_HINTS_TO_CODE = {
    # Region III examples (not exhaustive)
    "angeles city": "III", "balanga city": "III", "san fernando": "III", "olongapo city": "III",
    # Region IV-A examples (not exhaustive)
    "antipolo city": "IV-A", "lucena city": "IV-A", "calamba": "IV-A", "biñan city": "IV-A", "binan city": "IV-A",
}

def derive_code_from_cityprov(cp: str) -> str:
    """
    Best-effort derive Region code among {NCR, III, IV-A} from City/Province text.
    Returns '' if not confidently one of the allowed codes.
    """
    if pd.isna(cp):
        return ""
    raw = str(cp)
    norm = _norm_text(raw)

    # Direct NCR by city name
    if norm in NCR_CITIES or norm.replace(" city","") in NCR_CITIES:
        return "NCR"

    # City hints (III / IV-A)
    if norm in CITY_HINTS_TO_CODE:
        return CITY_HINTS_TO_CODE[norm]

    # "City, Province" -> use trailing province
    if "," in norm:
        tail = norm.split(",")[-1].strip()
        tail = re.sub(r'^(province of|prov\.? of)\s+', '', tail)
        code = PROVINCE_TO_ALLOWED_CODE.get(tail, "")
        if code:
            return code

    # Exact province
    code = PROVINCE_TO_ALLOWED_CODE.get(norm, "")
    if code:
        return code

    # Try find any province token inside the string
    for prov_key in PROVINCE_TO_ALLOWED_CODE.keys():
        if re.search(rf'\b{re.escape(prov_key)}\b', norm):
            return PROVINCE_TO_ALLOWED_CODE[prov_key]

    return ""

# === Helpers ===
def find_col(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, str(c), flags=re.IGNORECASE):
                return c
    return None

# === Extract and Clean ===
def extract_clean_data(filepath):
    xls = pd.ExcelFile(filepath)
    frames = []
    for sh in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sh)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    cols = list(raw.columns)
    location_col = find_col(cols, [r'^location$', r'location'])
    date_col = find_col(cols, [r'^date', r'date[_ ]?time', r'datetime', r'origin time'])
    region_col = find_col(cols, [r'^region$'])

    df = raw.copy()

    # --- City/Province from parentheses in Location; keep Location minus parentheses
    if location_col:
        df["City/Province"] = df[location_col].astype(str).str.extract(r'\((.*?)\)')[0].apply(normalize_city_province)
        df[location_col] = df[location_col].astype(str).str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()
    else:
        df["City/Province"] = pd.NA

    # --- Derive Region_Code (NCR, III, IV-A) from existing Region text (if any)
    if region_col:
        df["Region_Code"] = df[region_col].apply(normalize_region_text_to_code)
    else:
        df["Region_Code"] = ""

    # Fill missing codes using City/Province
    needs_fill = (df["Region_Code"] == "") | df["Region_Code"].isna()
    df.loc[needs_fill, "Region_Code"] = df.loc[needs_fill, "City/Province"].apply(derive_code_from_cityprov)

    # Map human-readable Region from the code
    df["Region"] = df["Region_Code"].map(lambda c: REGION_LABELS_ALLOWED.get(c, "Unknown"))

    # --- Date split
    if date_col:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        df["Year"] = dt.dt.year
        df["Month"] = dt.dt.month
        df["Day"] = dt.dt.day
        df["Time"] = dt.dt.time
        df.drop(columns=[date_col], inplace=True, errors='ignore')

    # --- HARD FILTER: keep only allowed Region_Code values
    df = df[df["Region_Code"].isin(ALLOWED_CODES)]

    # --- Remove any rows still missing code or with Unknown label (safety)
    df = df[(df["Region_Code"].notna()) & (df["Region_Code"] != "")]
    df = df[df["Region"] != "Unknown"]

    # --- Cleanups
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill residual text NaNs
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

    return df

# === Process and Save ===
if __name__ == "__main__":
    df = extract_clean_data(raw_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✅] Cleaned file saved to: {output_path}")
