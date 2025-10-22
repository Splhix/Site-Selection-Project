# scripts/project_population_from_drivers.py (v2)
import argparse, pandas as pd, os

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True, help="population_density_ready.csv (2020 snapshot)")
ap.add_argument("--grdp_forecast", required=False)
ap.add_argument("--cons_forecast", required=False)
ap.add_argument("--id-col", default="Area_slug")
ap.add_argument("--base-households-col", default="Number_of_Households")
ap.add_argument("--base-hhpop-col", default="Household_Population")
ap.add_argument("--base-hhsize-col", default="avg_household_size")
ap.add_argument("--out", required=True)
ap.add_argument("--elasticity", type=float, default=0.9, help="Elasticity of household growth to drivers")
ap.add_argument("--alpha-grdp", type=float, default=0.5)
ap.add_argument("--beta-cons", type=float, default=0.5)
ap.add_argument("--max-annual-growth", type=float, default=0.04, help="Cap annual HH growth (e.g., 4%)")
ap.add_argument("--min-annual-growth", type=float, default=0.00)
ap.add_argument("--lag-years", type=int, default=0)
ap.add_argument("--base-year", type=int, default=2020, help="Base year if snapshot has no Year column")
args = ap.parse_args()

snap = pd.read_csv(args.snapshot)
need = [args.id_col, args.base_households_col, args.base_hhpop_col, args.base_hhsize_col, "Region", "Area_display"]
for c in need:
    if c not in snap.columns:
        raise SystemExit(f"Snapshot missing required column: {c}")
if "Year" in snap.columns:
    if snap["Year"].nunique()!=1:
        raise SystemExit(f"Snapshot should have a single base year; found {snap['Year'].unique()}")
    base_year = int(snap["Year"].iloc[0])
else:
    base_year = int(args.base_year)

base = snap[[args.id_col,"Area_display","Region",args.base_households_col,args.base_hhpop_col,args.base_hhsize_col]].copy()
base = base.rename(columns={args.base_households_col:"Households0",
                            args.base_hhpop_col:"HHPop0",
                            args.base_hhsize_col:"HHSize0"})

def tidy(path, var):
    if not path: return None
    df = pd.read_csv(path)
    cols = [c for c in ["type", args.id_col, "Year", "Value"] if c in df.columns]
    df = df[cols].copy()
    df = df.sort_values([args.id_col,"Year"])
    df = df.rename(columns={"Value":f"{var}_level"})
    df[f"{var}_growth"] = df.groupby(args.id_col)[f"{var}_level"].pct_change()
    return df

g = tidy(args.grdp_forecast, "GRDP")
c = tidy(args.cons_forecast, "CONS")
if g is None and c is None:
    raise SystemExit("No driver files provided.")

if g is not None and c is not None:
    m = pd.merge(g[[args.id_col,"Year","GRDP_growth"]],
                 c[[args.id_col,"Year","CONS_growth"]],
                 on=[args.id_col,"Year"], how="outer")
else:
    m = g if g is not None else c

# Fill early gaps with medians, then zeros
for col in ["GRDP_growth","CONS_growth"]:
    if col in m.columns:
        m[col] = m.groupby(args.id_col)[col].transform(lambda s: s.fillna(s.median()))
        m[col] = m[col].fillna(0.0)

# Weighted composite
ag, bc = args.alpha_grdp, args.beta_cons
if "GRDP_growth" not in m.columns: ag = 0.0
if "CONS_growth" not in m.columns: bc = 0.0
if ag + bc == 0: raise SystemExit("All driver weights are zero.")
if ag > 0 and bc > 0:
    s = ag + bc
    ag, bc = ag/s, bc/s

m["driver_growth"] = 0.0
if "GRDP_growth" in m.columns: m["driver_growth"] += ag*m["GRDP_growth"]
if "CONS_growth" in m.columns: m["driver_growth"] += bc*m["CONS_growth"]

# Lag + cap
if args.lag_years != 0:
    m = m.sort_values([args.id_col,"Year"])
    m["driver_growth"] = m.groupby(args.id_col)["driver_growth"].shift(args.lag_years).fillna(0.0)

m["hh_growth"] = (args.elasticity * m["driver_growth"]).clip(lower=args.min_annual_growth, upper=args.max_annual_growth)

end_year = int(m["Year"].max())

# Build panel
rows = []
prev = base[[args.id_col,"Households0","HHPop0","HHSize0"]].copy()
prev["Year"] = base_year
prev = prev.rename(columns={"Households0":"Households","HHPop0":"Household_Population"})
rows.append(prev[[args.id_col,"Year","Households","Household_Population","HHSize0"]])

growth = m.set_index([args.id_col,"Year"])["hh_growth"].to_dict()

for yr in range(base_year+1, end_year+1):
    p = rows[-1][[args.id_col,"Households","HHSize0"]].rename(columns={"Households":"Households_prev"})
    p["Year"] = yr
    p["hh_growth"] = p.apply(lambda r: growth.get((r[args.id_col], yr), 0.0), axis=1)
    p["Households"] = p["Households_prev"] * (1.0 + p["hh_growth"])
    p["Household_Population"] = p["Households"] * p["HHSize0"]
    rows.append(p[[args.id_col,"Year","Households","Household_Population","HHSize0"]])

panel = pd.concat(rows, ignore_index=True)
panel = panel.merge(base[[args.id_col,"Area_display","Region"]], on=args.id_col, how="left").sort_values([args.id_col,"Year"])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
panel.to_csv(args.out, index=False)
print(f"Wrote {args.out} with {panel[args.id_col].nunique()} cities, years {base_year}..{end_year}.")
