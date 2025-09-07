# scripts/project_units_from_drivers_v2.py
import argparse, pandas as pd, numpy as np, os

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True)
ap.add_argument("--grdp_forecast", required=False)
ap.add_argument("--cons_forecast", required=False)
ap.add_argument("--id-col", default="Area_slug")
ap.add_argument("--out", required=True)
ap.add_argument("--elasticity", type=float, default=0.7)
ap.add_argument("--alpha-grdp", type=float, default=0.5)
ap.add_argument("--beta-cons", type=float, default=0.5)
ap.add_argument("--max-annual-growth", type=float, default=0.06)
ap.add_argument("--min-annual-growth", type=float, default=0.00)
args = ap.parse_args()

snap = pd.read_csv(args.snapshot)
mask_all = snap["Housing_Type"].astype(str).str.contains("All Types", case=False, na=False)
totals = snap.loc[mask_all, [args.id_col,"Area_display","Region","Year","Total_Units","Occupied_Housing_Units"]].copy()

if totals["Year"].nunique() != 1:
    raise SystemExit(f"Expected single base year; found {totals['Year'].unique()}")
base_year = int(totals["Year"].iloc[0])

base = totals.rename(columns={"Total_Units":"Units","Occupied_Housing_Units":"Occupied"})
base["occ_rate0"] = (base["Occupied"]/base["Units"]).clip(0.5,1.0)

def tidy(path, var):
    if not path: return None
    df = pd.read_csv(path)
    df = df[df["type"]=="forecast"][[args.id_col,"Year","Value"]].rename(columns={"Value":f"{var}_level"})
    df = df.sort_values([args.id_col,"Year"])
    df[f"{var}_growth"] = df.groupby(args.id_col)[f"{var}_level"].pct_change()
    return df

g = tidy(args.grdp_forecast, "GRDP")
c = tidy(args.cons_forecast, "CONS")
if g is None and c is None:
    raise SystemExit("No driver forecasts provided.")

merge = None
if g is not None and c is not None:
    merge = pd.merge(g[[args.id_col,"Year","GRDP_growth"]],
                     c[[args.id_col,"Year","CONS_growth"]],
                     on=[args.id_col,"Year"], how="outer")
else:
    merge = (g if g is not None else c).copy()

# Fill first missing growth with group medians and zeros otherwise
for col in ["GRDP_growth","CONS_growth"]:
    if col in merge.columns:
        merge[col] = merge.groupby(args.id_col)[col].transform(lambda s: s.fillna(s.median()))
        merge[col] = merge[col].fillna(0.0)

ag, bc = args.alpha_grdp, args.beta_cons
if "GRDP_growth" not in merge.columns: ag = 0.0
if "CONS_growth" not in merge.columns: bc = 0.0
if ag + bc == 0: raise SystemExit("All driver weights are zero.")
if ag > 0 and bc > 0:
    s = ag + bc
    ag, bc = ag/s, bc/s

merge["driver_growth"] = 0.0
if "GRDP_growth" in merge.columns: merge["driver_growth"] += ag*merge["GRDP_growth"]
if "CONS_growth" in merge.columns: merge["driver_growth"] += bc*merge["CONS_growth"]
merge["supply_growth"] = (args.elasticity * merge["driver_growth"]).clip(lower=args.min_annual_growth, upper=args.max_annual_growth)

end_year = int(merge["Year"].max())
years_full = list(range(base_year, end_year+1))

rows = []
prev = base[[args.id_col,"Units","Occupied","occ_rate0"]].copy()
prev["Year"] = base_year
rows.append(prev.copy())

growth_dict = merge.set_index([args.id_col,"Year"])["supply_growth"].to_dict()

for yr in range(base_year+1, end_year+1):
    p = rows[-1][[args.id_col,"Units","occ_rate0"]].rename(columns={"Units":"Units_prev"})
    p["Year"] = yr
    p["supply_growth"] = p.apply(lambda r: growth_dict.get((r[args.id_col], yr), 0.0), axis=1)
    p["Units"] = p["Units_prev"] * (1.0 + p["supply_growth"])
    p["Occupied"] = (p["Units"] * p["occ_rate0"]).clip(0, p["Units"])
    rows.append(p[[args.id_col,"Year","Units","Occupied","occ_rate0"]])

proj = pd.concat(rows, ignore_index=True)
proj = proj.merge(base[[args.id_col,"Area_display","Region"]], on=args.id_col, how="left")
proj = proj.sort_values([args.id_col,"Year"]).reset_index(drop=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
proj.to_csv(args.out, index=False)
print(f"Wrote {args.out} with years {base_year}..{end_year} and {proj[args.id_col].nunique()} cities.")
