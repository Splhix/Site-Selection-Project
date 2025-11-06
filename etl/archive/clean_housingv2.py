import pandas as pd
import os

# --------------------
# File paths
# --------------------
raw_path = "data/raw/Housing_v2 (1).csv"
cleaned_path = "data/cleaned/housing_v2_cleaned.csv"

# --------------------
# Config (tweak as needed)
# --------------------
MIN_PRICE = 500_000         # drop obvious junk
MAX_PRICE = 100_000_000     # cap extreme outliers

# Provinces by region (project scope)
CENTRAL_LUZON = {
    "Zambales","Bataan","Pampanga","Tarlac","Nueva Ecija","Bulacan","Aurora"
}
CALABARZON = {
    "Cavite","Laguna","Batangas","Rizal","Quezon"
}
NCR_CITIES = {
    "Caloocan","Las Piñas","Makati","Malabon","Mandaluyong","Manila","Marikina",
    "Muntinlupa","Navotas","Parañaque","Pasay","Pasig","Quezon City","San Juan",
    "Taguig","Valenzuela","Pateros"
}

# --------------------
# Helpers
# --------------------
def parse_location(loc: str):
    """
    Very light parser for 'Location' -> city, province.
    Heuristics:
      - If last token is a known province: province=that, city=prev token (if any)
      - If last token is an NCR city: province='NCR', city=that token
      - Else: city=last token, province=None
    """
    if not isinstance(loc, str) or not loc.strip():
        return pd.Series({"city": None, "province": None})

    parts = [p.strip() for p in loc.split(",") if p.strip()]
    last = parts[-1] if parts else None

    if last in CENTRAL_LUZON or last in CALABARZON:
        city = parts[-2] if len(parts) >= 2 else None
        province = last
    elif last in NCR_CITIES:
        city = last
        province = "NCR"
    else:
        city = last
        province = None

    return pd.Series({"city": city, "province": province})

def infer_region(city, province):
    if province == "NCR" or (isinstance(city, str) and city in NCR_CITIES):
        return "NCR"
    if province in CENTRAL_LUZON:
        return "Central Luzon"
    if province in CALABARZON:
        return "CALABARZON"
    return None

# --------------------
# Load
# --------------------
df = pd.read_csv(raw_path)

# Standardize columns
rename_map = {
    "Description": "description",
    "Location": "location",
    "Price": "price",
    "Bedrooms": "bedrooms",
    "Bathrooms": "bathrooms",
    "Floor Area": "floor_area",
    "Land Area": "land_area",
    "Latitude": "latitude",
    "Longitude": "longitude",
}
df.rename(columns=rename_map, inplace=True)

cols_keep = ["description","location","price","bedrooms","bathrooms",
             "floor_area","land_area","latitude","longitude"]
df = df[cols_keep].copy()

# Trim strings
df["description"] = df["description"].astype(str).str.strip()
df["location"] = df["location"].astype(str).str.strip()

# Coerce numerics
for c in ["price","bedrooms","bathrooms","floor_area","land_area","latitude","longitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Basic filters
before = len(df)
df = df[df["price"].between(MIN_PRICE, MAX_PRICE, inclusive="both")]
# keep if at least one of floor_area or land_area is positive (or allow NaN with the other present)
df = df[~((df["floor_area"].fillna(0) <= 0) & (df["land_area"].fillna(0) <= 0))]
after = len(df)

# Deduplicate
df.drop_duplicates(subset=["description","location","price","floor_area","land_area"], keep="first", inplace=True)

# Parse location
parsed = df["location"].apply(parse_location)
df = pd.concat([df, parsed], axis=1)

# Normalize capitalization (simple title case)
for c in ["city","province"]:
    df[c] = df[c].apply(lambda x: x.title() if isinstance(x, str) else x)

# Region + in_scope tag
df["region"] = df.apply(lambda r: infer_region(r["city"], r["province"]), axis=1)
df["in_scope"] = df["region"].isin({"NCR","Central Luzon","CALABARZON"})

# Derived metrics
df["price_per_sqm_floor"] = df.apply(
    lambda r: r["price"]/r["floor_area"] if pd.notna(r["price"]) and pd.notna(r["floor_area"]) and r["floor_area"]>0 else pd.NA,
    axis=1
)
df["price_per_sqm_land"] = df.apply(
    lambda r: r["price"]/r["land_area"] if pd.notna(r["price"]) and pd.notna(r["land_area"]) and r["land_area"]>0 else pd.NA,
    axis=1
)
df["price_per_bedroom"] = df.apply(
    lambda r: r["price"]/r["bedrooms"] if pd.notna(r["price"]) and pd.notna(r["bedrooms"]) and r["bedrooms"]>0 else pd.NA,
    axis=1
)
df["bb_ratio"] = df.apply(
    lambda r: r["bedrooms"]/r["bathrooms"] if pd.notna(r["bedrooms"]) and pd.notna(r["bathrooms"]) and r["bathrooms"]>0 else pd.NA,
    axis=1
)
df["far"] = df.apply(
    lambda r: r["floor_area"]/r["land_area"] if pd.notna(r["floor_area"]) and pd.notna(r["land_area"]) and r["land_area"]>0 else pd.NA,
    axis=1
)

# Final order
order = [
    "region","in_scope","province","city",
    "location","description",
    "price","bedrooms","bathrooms","floor_area","land_area",
    "price_per_sqm_floor","price_per_sqm_land","price_per_bedroom","bb_ratio","far",
    "latitude","longitude"
]
df_cleaned = df[order].copy()

# Save
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
df_cleaned.to_csv(cleaned_path, index=False)

print(f"[✅] Cleaned housing data saved to: {cleaned_path}")
print(f"[i] Rows input: {before:,} | kept after price/area filters: {after:,} | final after dedupe: {len(df_cleaned):,}")
print(f"[i] In-scope rows (NCR/Central Luzon/CALABARZON): {int(df_cleaned['in_scope'].sum())} of {len(df_cleaned)}")
