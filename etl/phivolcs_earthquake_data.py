import pandas as pd
import os
import re

# === File Paths ===
raw_path = "data/raw/phivolcs_earthquake_data.xlsx"
output_path = "data/cleaned/phivolcs_earthquake_data_cleaned.csv"

# === Canonical Region Labels ===
REGION_LABELS = {
    "NCR": "NCR",
    "CAR": "CAR",
    "I": "Region I (Ilocos Region)",
    "II": "Region II (Cagayan Valley)",
    "III": "Region III (Central Luzon)",
    "IV-A": "Region IV-A (CALABARZON)",
    "IV-B": "Region IV-B (MIMAROPA)",
    "V": "Region V (Bicol Region)",
    "VI": "Region VI (Western Visayas)",
    "VII": "Region VII (Central Visayas)",
    "VIII": "Region VIII (Eastern Visayas)",
    "IX": "Region IX (Zamboanga Peninsula)",
    "X": "Region X (Northern Mindanao)",
    "XI": "Region XI (Davao Region)",
    "XII": "Region XII (SOCCSKSARGEN)",
    "XIII": "Region XIII (Caraga)",
    "BARMM": "BARMM",
}

# === Province -> Region Code ===
# Covers all provinces (including recent splits/legacy names)
PROVINCE_TO_REGION_CODE = {
    # CAR
    "abra": "CAR", "aparrri?": "CAR",  # guard for typos (won't match unless exact)
    "apayao": "CAR", "benguet": "CAR", "ifugao": "CAR", "kalinga": "CAR", "mountain province": "CAR",

    # Region I
    "ilocos norte": "I", "ilocos sur": "I", "la union": "I", "pangasinan": "I",

    # Region II
    "batanes": "II", "cagayan": "II", "isabela": "II", "nueva vizcaya": "II", "quirino": "II",

    # Region III
    "aurora": "III", "bataan": "III", "bulacan": "III", "nueva ecija": "III", "pampanga": "III", "tarlac": "III", "zambales": "III",

    # Region IV-A
    "batangas": "IV-A", "cavite": "IV-A", "laguna": "IV-A", "quezon": "IV-A", "rizal": "IV-A",

    # Region IV-B (MIMAROPA)
    "marinduque": "IV-B", "occidental mindoro": "IV-B", "oriental mindoro": "IV-B", "palawan": "IV-B", "romblon": "IV-B",

    # Region V
    "albay": "V", "camarines norte": "V", "camarines sur": "V", "catanduanes": "V", "masbate": "V", "sorsogon": "V",

    # Region VI
    "aklan": "VI", "antique": "VI", "capiz": "VI", "guimaras": "VI", "iloilo": "VI", "negros occidental": "VI",

    # Region VII
    "bohol": "VII", "cebu": "VII", "negros oriental": "VII", "siquijor": "VII",

    # Region VIII
    "biliran": "VIII", "eastern samar": "VIII", "leyte": "VIII", "northern samar": "VIII",
    "samar": "VIII", "western samar": "VIII", "southern leyte": "VIII",

    # Region IX
    "zamboanga del norte": "IX", "zamboanga del sur": "IX", "zamboanga sibugay": "IX",

    # Region X
    "bukidnon": "X", "camiguin": "X", "lanao del norte": "X", "misamis occidental": "X", "misamis oriental": "X",

    # Region XI
    "davao de oro": "XI", "compostela valley": "XI",  # legacy name
    "davao del norte": "XI", "davao del sur": "XI", "davao occidental": "XI", "davao oriental": "XI",

    # Region XII
    "cotabato": "XII", "north cotabato": "XII",  # synonyms
    "sarangani": "XII", "south cotabato": "XII", "sultan kudarat": "XII",

    # Region XIII (Caraga)
    "agusan del norte": "XIII", "agusan del sur": "XIII", "dinagat islands": "XIII",
    "surigao del norte": "XIII", "surigao del sur": "XIII",

    # BARMM
    "basilan": "BARMM", "lanao del sur": "BARMM",
    "maguindanao": "BARMM", "maguindanao del norte": "BARMM", "maguindanao del sur": "BARMM",
    "sulu": "BARMM", "tawi-tawi": "BARMM",

    # Special: Isabela City (Basilan) is administratively Region IX but province is BARMM
    # We'll catch city-level below.
}

# === Cities that imply NCR regardless of province text ===
NCR_CITIES = {
    # 16 cities + 1 municipality
    "caloocan", "las piñas", "las pinas", "makati", "malabon", "mandaluyong",
    "manila", "marikina", "muntinlupa", "navotas", "parañaque", "paranaque",
    "pasay", "pasig", "quezon city", "san juan", "taguig", "valenzuela", "pateros"
}

# === High-profile HUCs / special cases (City -> Province or Region) ===
CITY_SPECIALS = {
    # Bicol & VisMin examples
    "naga city (camarines sur)": ("camarines sur", None),
    "naga city": (None, None),  # ambiguous; will try to disambiguate if province present
    "isabela city": (None, "IX"),  # Isabela City (Basilan) is Region IX
    "cotabato city": (None, "BARMM"),
    "davao city": (None, "XI"),
    "zamboanga city": (None, "IX"),
    "general santos city": (None, "XII"),
    "butuan city": (None, "XIII"),
    "surigao city": ("surigao del norte", None),
    "cagayan de oro": (None, "X"),
    "iloilo city": ("iloilo", None),
    "bacolod city": ("negros occidental", None),
    "dumaguete city": ("negros oriental", None),
    "cebu city": ("cebu", None),
    "lapu-lapu city": ("cebu", None),
    "mandaue city": ("cebu", None),
    "ormoc city": ("leyte", None),
    "tacloban city": ("leyte", None),
    "bago city": ("negros occidental", None),
    "antipolo city": ("rizal", None),
    "lucena city": ("quezon", None),
    "olongapo city": ("zambales", None),
    "santiago city": ("isabela", None),
    "biñan city": ("laguna", None), "binan city": ("laguna", None),
    "san jose del monte": ("bulacan", None),
    "iligan city": ("lanao del norte", None),
    "marawi city": ("lanao del sur", None),
    "tagum city": ("davao del norte", None),
    "panabo city": ("davao del norte", None),
    "samal city": ("davao del norte", None), "island garden city of samal": ("davao del norte", None),
    "valencia city": ("bukidnon", None),
    "gingoog city": ("misamis oriental", None),
    "ozamiz city": ("misamis occidental", None),
    "dipolog city": ("zamboanga del norte", None),
    "pagadian city": ("zamboanga del sur", None),
    "tandag city": ("surigao del sur", None),
    "bayugan city": ("agusan del sur", None),
    "cabadiangan?": (None, None),  # placeholder to avoid accidental regex capture
}

# --- Normalizers ---
def _norm_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', str(s)).strip().lower()
    s = s.replace(' city of ', ' ').replace(' city', ' city')  # keep 'city' tokens
    s = s.replace('ñ', 'n').replace('á', 'a').replace('í', 'i').replace('ó','o').replace('ú','u')
    s = s.replace('–', '-').replace('—', '-')
    s = re.sub(r'[.,]*$', '', s)  # trailing punctuation
    return s

def normalize_city_province(cp):
    if pd.isna(cp):
        return cp
    cp = re.sub(r'\s+', ' ', str(cp)).strip(' .,-')
    # Standardize common patterns
    cp = re.sub(r'\bCity Of\b', 'City of', cp, flags=re.IGNORECASE)
    cp = re.sub(r'\b,\s*Philippines\b', '', cp, flags=re.IGNORECASE)
    # Title case but keep common lowercase tokens
    cp = cp.title()
    # Fix "Del/De/La/Las/Los" casing in Philippine names
    cp = re.sub(r'\b(Del|De|La|Las|Los)\b', lambda m: m.group(1).lower(), cp)
    return cp

def _region_label_from_code(code: str) -> str:
    return REGION_LABELS.get(code, code)

# Core resolver
def resolve_region_from_cityprov(value: str) -> str:
    """
    Accepts a City/Province string (e.g., 'Tarlac City, Tarlac', 'Northern Samar', 'Quezon City')
    and returns a standardized region label (e.g., 'Region III (Central Luzon)').
    """
    if pd.isna(value):
        return 'Unknown'

    raw = str(value)
    norm = _norm_text(raw)

    # 1) NCR by city name
    # e.g., "Quezon City", "Manila", "Taguig"
    token_city = norm.replace(' city', '')
    if token_city in NCR_CITIES or norm in NCR_CITIES:
        return _region_label_from_code("NCR")

    # 2) Explicit CITY_SPECIALS (city -> province or region code)
    if norm in CITY_SPECIALS:
        prov, region_code = CITY_SPECIALS[norm]
        if region_code:
            return _region_label_from_code(region_code)
        if prov:
            code = PROVINCE_TO_REGION_CODE.get(prov, None)
            if code:
                return _region_label_from_code(code)

    # 3) If "City, Province" or "Municipality, Province" pattern — use the trailing province
    if ',' in norm:
        parts = [p.strip() for p in norm.split(',') if p.strip()]
        # try last part as province name
        tail = parts[-1]
        # remove 'province of ' prefix if present
        tail = re.sub(r'^(province of|prov\.? of)\s+', '', tail)
        code = PROVINCE_TO_REGION_CODE.get(tail, None)
        if code:
            return _region_label_from_code(code)

    # 4) If only province name was provided (or city equals province)
    # Normalize variants like 'Naga City (Camarines Sur)' already handled above;
    # here we try direct province lookups from full text
    # Strip common words to isolate potential province token
    candidates = [
        norm,
        re.sub(r'\b(city|province of|prov\.?|municipality of)\b', '', norm).strip(),
        re.sub(r'\b(city|province|municipality)\b', '', norm).strip(),
    ]
    for cand in candidates:
        cand = re.sub(r'\s+', ' ', cand).strip()
        if not cand:
            continue
        code = PROVINCE_TO_REGION_CODE.get(cand, None)
        if code:
            return _region_label_from_code(code)

    # 5) Try disambiguating special "Naga City" if accompanied by 'camarines sur' in original string
    if 'naga city' in norm and 'camarines sur' in norm:
        return _region_label_from_code(PROVINCE_TO_REGION_CODE['camarines sur'])

    # 6) Last resort: try to find any province token inside the string
    for prov_key in PROVINCE_TO_REGION_CODE.keys():
        if re.search(rf'\b{re.escape(prov_key)}\b', norm):
            return _region_label_from_code(PROVINCE_TO_REGION_CODE[prov_key])

    # 7) Unknown
    return 'Unknown'

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

    # Identify important columns
    location_col = find_col(cols, [r'location'])
    date_col = find_col(cols, [r'^date', r'date[_ ]?time', r'time', r'origin time', r'datetime'])

    df = raw.copy()

    # Extract City/Province from parentheses in Location
    df["City/Province"] = df[location_col].astype(str).str.extract(r'\((.*?)\)')[0].apply(normalize_city_province)

    # Remove parentheses content from Location
    df[location_col] = df[location_col].astype(str).str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()

    # Assign Region based on City/Province (robust resolver)
    df["Region"] = df["City/Province"].apply(resolve_region_from_cityprov)

    # Process Date_Time into separate columns (if present)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month
        df["Day"] = df[date_col].dt.day
        df["Time"] = df[date_col].dt.time
        df.drop(columns=[date_col], inplace=True, errors='ignore')

    # Drop duplicates and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing text fields with 'Unknown'
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

    return df

# === Process and Save ===
if __name__ == "__main__":
    df = extract_clean_data(raw_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✅] Cleaned file saved to: {output_path}")
