import pandas as pd
import re
from pathlib import Path

# === File paths ===
src = Path("data/cleaned/family_income_cleaned_v2.csv")   # your uploaded file
out_long = Path("data/cleaned/family_income_ready.csv")   # tidy
out_wide = Path("data/cleaned/family_income_ready_wide.csv")

# ---------- Helpers ----------
def normalize_space(s: str) -> str:
    if pd.isna(s): return s
    return re.sub(r"\s+", " ", str(s).strip())

def normalize_area_name(a: str) -> str:
    """Standardize city/province/HUC/region display names consistently."""
    if pd.isna(a): return a
    a = normalize_space(a)
    # Common patterns: "City of X" -> "X City"
    a = re.sub(r"^City of\s+", "", a, flags=re.IGNORECASE)
    # Ensure "City" suffix where appropriate (optional, conservative)
    # Handle known variants like "Mandaluyong" -> "Mandaluyong City" if needed later
    # For now, keep as given after removing "City of"
    return a

def normalize_region(r: str) -> str:
    """Map any region label to the canonical display used across your project."""
    if pd.isna(r): return r
    s = normalize_space(str(r)).upper()
    s = re.sub(r"[\.\-]", "", s)
    s = re.sub(r"\s+", " ", s)

    # Canonical outputs you decided to use:
    # "NCR", "Region III (Central Luzon)", "Region IV-A (CALABARZON)"
    if s in {"NCR", "NATIONAL CAPITAL REGION", "NATIONAL CAPITAL REGION NCR"}:
        return "NCR"
    if s in {"REGION III", "CENTRAL LUZON", "REGION III CENTRAL LUZON"}:
        return "Region III (Central Luzon)"
    if s in {"REGION IVA", "REGION IV A", "CALABARZON"}:
        return "Region IV-A (CALABARZON)"
    # If this row is actually a Region name in PSA style already:
    if "REGION III (CENTRAL LUZON)".upper() in s:
        return "Region III (Central Luzon)"
    if "REGION IV-A (CALABARZON)".upper() in s:
        return "Region IV-A (CALABARZON)"
    if "NATIONAL CAPITAL REGION (NCR)" in s:
        return "NCR"
    return None  # not a region label

def detect_area_level(area: str) -> str:
    """
    Heuristic to tag row as Region / Province / HUC/City.
    We’ll set Region if name maps via normalize_region; otherwise, detect 'City' strings.
    """
    if pd.isna(area): return "Unknown"
    reg = normalize_region(area)
    if reg is not None:
        return "Region"
    a = area.lower()
    if " city" in a or a.startswith("city ") or a.endswith(" city"):
        return "HUC/City"
    return "Province"  # default fallback for PSA tables

def parse_year(y):
    """Handle 2023p -> year=2023, prelim_flag=True; numeric years -> prelim_flag=False."""
    prelim = False
    if pd.isna(y): return None, prelim
    s = str(y).strip()
    if s.endswith("p") or s.endswith("P"):
        prelim = True
        s = s[:-1]
    try:
        year = int(s)
    except:
        year = None
    return year, prelim

# ---------- Load ----------
df = pd.read_csv(src)

# Expecting columns like: Area, Year, IncomeGroup, Value
# Make robust renames if headers differ slightly
rename_guess = {
    "area": "Area",
    "AREA": "Area",
    "region/province/huc/city": "Area",
    "Region/Province/HUC/City": "Area",
    "year": "Year",
    "YEAR": "Year",
    "Income Group": "IncomeGroup",
    "income group": "IncomeGroup",
    "IncomeGroup": "IncomeGroup",
    "value": "Value",
    "VALUE": "Value",
}
for k, v in rename_guess.items():
    if k in df.columns and v not in df.columns:
        df = df.rename(columns={k: v})

required = {"Area", "Year", "IncomeGroup", "Value"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")

# ---------- Clean fields ----------
df["Area"] = df["Area"].apply(normalize_area_name)

# Parse year + prelim flag
year_parsed = df["Year"].apply(parse_year)
df["Year"] = year_parsed.apply(lambda t: t[0])
df["Prelim_Flag"] = year_parsed.apply(lambda t: t[1])

# Coerce numeric Value
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Tag area level & derive Region tag for easier joins later
df["Area_Level"] = df["Area"].apply(detect_area_level)

# Create a Region column where possible:
# - If Area_Level == 'Region' -> Region = canonical region
# - Else leave blank for now; you'll fill using a province→region or city→region map later
df["Region"] = df["Area"].apply(lambda a: normalize_region(a) if detect_area_level(a)=="Region" else None)

# Filter out rows with Year missing or Value missing (optional, you can keep NaNs for extrapolation if preferred)
# df = df[df["Year"].notna() & df["Value"].notna()]

# ---------- Keep tidy (long) version ----------
long_cols = ["Area", "Area_Level", "Region", "Year", "IncomeGroup", "Value", "Prelim_Flag"]
df_long = df[long_cols].copy()

# ---------- Wide version (per Area × IncomeGroup) ----------
wide = df_long.pivot_table(
    index=["Area", "Area_Level", "Region", "IncomeGroup"],
    columns="Year",
    values="Value",
    aggfunc="first"
).reset_index()

# Rename year columns for clarity (only if they exist)
rename_years = {}
for col in wide.columns:
    if isinstance(col, int):
        rename_years[col] = f"Value_{col}"
wide = wide.rename(columns=rename_years)

# Percent changes (only compute where both endpoints exist)
if {"Value_2018", "Value_2021"}.issubset(wide.columns):
    wide["pct_change_2018_2021"] = (wide["Value_2021"] - wide["Value_2018"]) / wide["Value_2018"] * 100
else:
    wide["pct_change_2018_2021"] = pd.NA

if {"Value_2021", "Value_2023"}.issubset(wide.columns):
    wide["pct_change_2021_2023"] = (wide["Value_2023"] - wide["Value_2021"]) / wide["Value_2021"] * 100
else:
    wide["pct_change_2021_2023"] = pd.NA

# ---------- Save ----------
out_long.parent.mkdir(parents=True, exist_ok=True)
df_long.to_csv(out_long, index=False)
wide.to_csv(out_wide, index=False)

print(f"[✅] Saved tidy to: {out_long}")
print(f"[✅] Saved wide  to: {out_wide}")

# Optional: quick QA prints
print("\nSample tidy rows:")
print(df_long.head(8))
print("\nUnique Area levels:", df_long["Area_Level"].value_counts().to_dict())
print("\nYears found:", sorted(df_long["Year"].dropna().unique()))
print("\nIncome groups:", sorted(df_long["IncomeGroup"].dropna().unique()))
