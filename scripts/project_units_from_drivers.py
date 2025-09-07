# scripts/project_units_from_drivers.py
import argparse, pandas as pd, numpy as np, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True, help="housing_units_ready_v3.csv (2020 snapshot, includes Housing_Type)")
ap.add_argument("--grdp_forecast", required=False, help="forecasts/grdp_city_forecast.csv")
ap.add_argument("--cons_forecast", required=False, help="forecasts/consumption_city_forecast.csv")
ap.add_argument("--id-col", default="Area_slug")
ap.add_argument("--out", required=True, help="Output CSV with projected Total_Units (and Occupied, optional)")
ap.add_argument("--elasticity", type=float, default=0.7, help="Elasticity of supply to composite driver growth (0..1)")
ap.add_argument("--alpha-grdp", type=float, default=0.5, help="Weight on GRDP growth (if provided)")
ap.add_argument("--beta-cons", type=float, default=0.5, help="Weight on Consumption growth (if provided)")
ap.add_argument("--max-annual-growth", type=float, default=0.06, help="Cap annual supply growth (e.g., 0.06 = 6%)")
ap.add_argument("--min-annual-growth", type=float, default=0.0, help="Floor annual supply growth")
args = ap.parse_args()

snap = pd.read_csv(args.snapshot)

# Pick totals row per city (Housing_Type like "All Types Of Building")
mask_all = snap["Housing_Type"].astype(str).str.contains("All Types", case=False, na=False)
totals = snap.loc[mask_all, [args.id_col, "Area_display", "Region", "Year", "Total_Units", "Occupied_Housing_Units", "Number_of_Households", "Household_Population", "avg_household_size"]].copy()

if totals["Year"].nunique() != 1:
    raise SystemExit("Snapshot should have a single base year per city. Found years: %s" % totals["Year"].unique())
base_year = int(totals["Year"].iloc[0])

# Load driver forecasts
def tidy(fc, var_name):
    df = pd.read_csv(fc)
    df = df[df["type"]=="forecast"]
    df = df[[args.id_col, "Year", "Value"]].rename(columns={"Value": f"{var_name}_level"})
    return df.sort_values([args.id_col, "Year"])

def growth_from_levels(df, var):
    df = df.sort_values([args.id_col, "Year"])
    df[f"{var}_growth"] = df.groupby(args.id_col)[f"{var}_level"].pct_change()
    return df

g_tidy = tidy(args.grdp_forecast, "GRDP") if args.grdp_forecast else None
c_tidy = tidy(args.cons_forecast, "CONS") if args.cons_forecast else None

if g_tidy is not None: g_tidy = growth_from_levels(g_tidy, "GRDP")
if c_tidy is not None: c_tidy = growth_from_levels(c_tidy, "CONS")

if g_tidy is not None and c_tidy is not None:
    merge = pd.merge(g_tidy[[args.id_col,"Year","GRDP_growth"]],
                     c_tidy[[args.id_col,"Year","CONS_growth"]],
                     on=[args.id_col,"Year"], how="outer")
elif g_tidy is not None:
    merge = g_tidy[[args.id_col,"Year","GRDP_growth"]].copy()
elif c_tidy is not None:
    merge = c_tidy[[args.id_col,"Year","CONS_growth"]].copy()
else:
    raise SystemExit("No driver forecasts provided.")

# Fill initial missing growth with group median, then zeros for any residual NaNs
for col in ["GRDP_growth","CONS_growth"]:
    if col in merge.columns:
        merge[col] = merge.groupby(args.id_col)[col].transform(lambda s: s.fillna(s.median()))
        merge[col] = merge[col].fillna(0.0)

# Determine weights and elasticity
ag, bc = args.alpha_grdp, args.beta_cons
if "GRDP_growth" not in merge.columns: ag = 0.0
if "CONS_growth" not in merge.columns: bc = 0.0
if ag + bc == 0:
    raise SystemExit("All driver weights are zero; set at least one of --alpha-grdp or --beta-cons")
# normalize weights if both > 0
if ag > 0 and bc > 0:
    s = ag + bc
    ag, bc = ag/s, bc/s

# Composite driver growth
merge["driver_growth"] = 0.0
if "GRDP_growth" in merge.columns: merge["driver_growth"] += ag*merge["GRDP_growth"]
if "CONS_growth" in merge.columns: merge["driver_growth"] += bc*merge["CONS_growth"]

# Supply response with elasticity and caps
merge["supply_growth"] = args.elasticity * merge["driver_growth"]
merge["supply_growth"] = merge["supply_growth"].clip(lower=args.min_annual_growth, upper=args.max_annual_growth)

# Build projection
first_fore_year = int(merge["Year"].min())
if first_fore_year != base_year + 1:
    # Insert missing intermediate years if needed?
    # We assume forecasts start right after base.
    pass

base = totals[[args.id_col, "Area_display", "Region", "Total_Units", "Occupied_Housing_Units"]].rename(
    columns={"Total_Units":"Units", "Occupied_Housing_Units":"Occupied"}
)
base["Year"] = base_year

proj = base.copy()
for yr in sorted(merge["Year"].unique()):
    g = merge[merge["Year"]==yr][[args.id_col, "supply_growth"]]
    prev = proj[proj["Year"]==yr-1][[args.id_col, "Units","Occupied"]]
    step = pd.merge(prev, g, on=args.id_col, how="left")
    # apply growth to Units; Occupied keeps baseline occupancy rate unless you want alternative logic
    occ_rate = (step["Occupied"] / step["Units"]).fillna(0.95).clip(0.5, 1.0)
    step["Units"] = step["Units"] * (1.0 + step["supply_growth"].fillna(0.0))
    step["Occupied"] = (step["Units"] * occ_rate).clip(0, step["Units"])
    step["Year"] = yr
    step = step[[args.id_col, "Year", "Units", "Occupied"]]
    proj = pd.concat([proj, step], ignore_index=True)

# Attach labels and export
proj = proj.merge(totals[[args.id_col,"Area_display","Region"]].drop_duplicates(args.id_col), on=args.id_col, how="left")
proj = proj.sort_values([args.id_col, "Year"]).reset_index(drop=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
proj.to_csv(args.out, index=False)
print(f"Wrote {args.out}")
