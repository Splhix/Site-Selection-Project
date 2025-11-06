import pandas as pd

# === File paths ===
raw_path = "data/cleaned/more_grdp_cleaned.csv"
cleaned_path = "data/cleaned/grdp_ready.csv"

# Load CSV
df = pd.read_csv(raw_path)

# --- 1. Inspect unique PriceType values ---
print("Unique PriceType values:", df["PriceType"].unique())

# --- 2. Standardize Region names ---
region_map = {
    "NCR": "NCR",
    "National Capital Region": "NCR",
    "REGION III": "Region III (Central Luzon)",
    "Region III": "Region III (Central Luzon)",
    "CENTRAL LUZON": "Region III (Central Luzon)",
    "REGION IV-A": "Region IV-A (CALABARZON)",
    "Region IV-A": "Region IV-A (CALABARZON)",
    "CALABARZON": "Region IV-A (CALABARZON)"
}
df["Region"] = df["Region"].replace(region_map)

# --- 3. Pivot PriceType into columns ---
df_pivot = df.pivot_table(
    index=["Region", "Year"],
    columns="PriceType",
    values="GRDP"
).reset_index()

# --- 4. Rename columns dynamically (so it doesn’t fail if names differ) ---
rename_map = {}
for col in df_pivot.columns:
    col_lower = str(col).lower()
    if "constant" in col_lower:
        rename_map[col] = "GRDP_Const2018"
    elif "current" in col_lower:
        rename_map[col] = "GRDP_Current"

df_pivot = df_pivot.rename(columns=rename_map)

# --- 5. Compute YoY Growth from constant prices ---
df_pivot = df_pivot.sort_values(by=["Region", "Year"])
if "GRDP_Const2018" in df_pivot.columns:
    df_pivot["GRDP_Growth"] = df_pivot.groupby("Region")["GRDP_Const2018"].pct_change() * 100
else:
    print("⚠ Warning: GRDP_Const2018 column not found. Check PriceType labels.")

# --- 6. Filter to your scope ---
scope_regions = ["National Capital Region (NCR)", "Region III (Central Luzon)", "Region IV-A (CALABARZON)"]
df_ready = df_pivot[df_pivot["Region"].isin(scope_regions)].copy()

# Save cleaned version
df_ready.to_csv(cleaned_path, index=False)
print(f"[✅] Cleaned GRDP data saved to: {cleaned_path}")
