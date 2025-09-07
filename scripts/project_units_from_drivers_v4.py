# scripts/project_units_from_drivers_v4.py
import argparse, pandas as pd, numpy as np, os

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True, help="housing_units_ready_v3.csv (2020 base)")
ap.add_argument("--grdp_forecast", required=False, help="forecasts/grdp_city_forecast.csv")
ap.add_argument("--cons_forecast", required=False, help="forecasts/consumption_city_forecast.csv")
ap.add_argument("--id-col", default="Area_slug")
ap.add_argument("--out", required=True)
ap.add_argument("--elasticity", type=float, default=0.7)
ap.add_argument("--alpha-grdp", type=float, default=0.5)
ap.add_argument("--beta-cons", type=float, default=0.5)
ap.add_argument("--max-annual-growth", type=float, default=0.06)
ap.add_argument("--min-annual-growth", type=float, default=0.00)
ap.add_argument("--lag-years", type=int, default=0, help="Lag for supply response to drivers (e.g., 1=use previous year's driver growth)")
ap.add_argument("--occ-cap", type=float, default=None, help="Optional occupancy ceiling (e.g., 0.95)")
args = ap.parse_args()

snap = pd.read_csv(args.snapshot)
mask_all = snap["Housing_Type"].astype(str).str.contains("All Types", case=False, na=False)
totals = snap.loc[mask_all, [args.id_col,"Area_display","Region","Year","Total_Units","Occupied_Housing_Units"]].copy()

if totals["Year"].nunique() != 1:
    raise SystemExit(f"Expected single base year; found {totals['Year'].unique()}")
base_year = int(totals["Year"].iloc[0])

base = totals.rename(columns={"Total_Units":"Units","Occupied_Housing_Units":"Occupied"})
base["occ_rate0"] = (base["Occupied"]/base["Units"]).clip(0.0,1.0)
if args.occ_cap is not None:
    base["occ_rate0"] = base["occ_rate0"].clip(upper=args.occ_cap)

def tidy(path):
    if not path: return None
    df = pd.read_csv(path)
    # keep both history and forecast to compute growth from base onward
    cols = [c for c in ["type", args.id_col, "Year", "Value"] if c in df.columns]
    df = df[cols].copy()
    return df

gr = tidy(args.grdp_forecast)
cs = tidy(args.cons_forecast)
if gr is None and cs is None:
    raise SystemExit("No driver files provided.")

# Prepare driver growth per city-year from BOTH history & forecast
def growth_from(df, varname):
    if df is None: 
        return None
    d = df.copy()
    d = d.sort_values([args.id_col, "Year"])
    d = d[[args.id_col,"Year","Value"]]
    d = d.rename(columns={"Value": f"{varname}_level"})
    d[f"{varname}_growth"] = d.groupby(args.id_col)[f"{varname}_level"].pct_change()
    return d

g = growth_from(gr, "GRDP")
c = growth_from(cs, "CONS")

# Merge the two (outer), so we have growth for 2021..end_year
if g is not None and c is not None:
    m = pd.merge(g[[args.id_col,"Year","GRDP_growth"]], c[[args.id_col,"Year","CONS_growth"]], on=[args.id_col,"Year"], how="outer")
else:
    m = g if g is not None else c

# Fill missing early growth with group medians; residual NaNs -> 0
for col in ["GRDP_growth","CONS_growth"]:
    if col in m.columns:
        m[col] = m.groupby(args.id_col)[col].transform(lambda s: s.fillna(s.median()))
        m[col] = m[col].fillna(0.0)

# Weighted composite
ag, bc = args.alpha_grdp, args.beta_cons
if "GRDP_growth" not in m.columns: ag = 0.0
if "CONS_growth" not in m.columns: bc = 0.0
if ag + bc == 0: 
    raise SystemExit("All driver weights are zero.")
if ag > 0 and bc > 0:
    s = ag + bc
    ag, bc = ag/s, bc/s
m["driver_growth"] = 0.0
if "GRDP_growth" in m.columns: m["driver_growth"] += ag*m["GRDP_growth"]
if "CONS_growth" in m.columns: m["driver_growth"] += bc*m["CONS_growth"]

# Apply lag if requested
if args.lag_years != 0:
    m = m.sort_values([args.id_col,"Year"])
    m["driver_growth_lagged"] = m.groupby(args.id_col)["driver_growth"].shift(args.lag_years)
    m["driver_growth"] = m["driver_growth_lagged"].fillna(0.0)

# Convert to supply growth with caps
m["supply_growth"] = (args.elasticity * m["driver_growth"]).clip(lower=args.min_annual_growth, upper=args.max_annual_growth)

# Determine projection horizon: from base+1 to last available driver year
end_year = int(m["Year"].max())
years = list(range(base_year, end_year+1))

# Build projection path
rows = []
prev = base[[args.id_col,"Units","Occupied","occ_rate0"]].copy()
prev["Year"] = base_year
rows.append(prev.copy())

growth_dict = m.set_index([args.id_col,"Year"])["supply_growth"].to_dict()

for yr in range(base_year+1, end_year+1):
    p = rows[-1][[args.id_col,"Units","occ_rate0"]].rename(columns={"Units":"Units_prev"})
    p["Year"] = yr
    p["supply_growth"] = p.apply(lambda r: growth_dict.get((r[args.id_col], yr), 0.0), axis=1)
    p["Units"] = p["Units_prev"] * (1.0 + p["supply_growth"])
    occ_rate = p["occ_rate0"]
    if args.occ_cap is not None:
        occ_rate = occ_rate.clip(upper=args.occ_cap)
    p["Occupied"] = (p["Units"] * occ_rate).clip(0, p["Units"])
    rows.append(p[[args.id_col,"Year","Units","Occupied","occ_rate0"]])

proj = pd.concat(rows, ignore_index=True)
proj = proj.merge(base[[args.id_col,"Area_display","Region"]], on=args.id_col, how="left")
proj = proj.sort_values([args.id_col,"Year"]).reset_index(drop=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
proj.to_csv(args.out, index=False)
print(f"Wrote {args.out} with years {base_year}..{end_year}, lag={args.lag_years}, cities={proj[args.id_col].nunique()}")
