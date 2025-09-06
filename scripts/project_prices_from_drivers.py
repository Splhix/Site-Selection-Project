# scripts/project_prices_from_drivers.py
import argparse, pandas as pd, numpy as np, os

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True, help="housing_prices_ready_city.csv (snapshot, no Year)")
ap.add_argument("--grdp_forecast", required=False, help="forecasts/grdp_city_forecast.csv")
ap.add_argument("--cons_forecast", required=False, help="forecasts/consumption_city_forecast.csv")
ap.add_argument("--base-col", default="price_median", help="Column in snapshot to anchor (e.g., price_median)")
ap.add_argument("--id-col", default="Area_slug")
ap.add_argument("--out", required=True, help="Output CSV with projected prices")
ap.add_argument("--alpha-grdp", type=float, default=0.5, help="Weight on GRDP growth")
ap.add_argument("--beta-cons", type=float, default=0.5, help="Weight on Consumption growth")
args = ap.parse_args()

snap = pd.read_csv(args.snapshot)
assert args.base_col in snap.columns, f"{args.base_col} not in snapshot"
assert args.id_col in snap.columns, f"{args.id_col} not in snapshot"

def tidy(fc, var_name):
    df = fc.copy()
    df = df[df["type"]=="forecast"]
    df = df[[args.id_col,"Year","Value","type","variable"]]
    df = df.rename(columns={"Value": f"{var_name}_level"})
    return df.sort_values([args.id_col,"Year"])

def growth_from_levels(df, var):
    df = df.sort_values([args.id_col,"Year"])
    df[f"{var}_growth"] = df.groupby(args.id_col)[f"{var}_level"].pct_change()
    return df

g_tidy = tidy(pd.read_csv(args.grdp_forecast), "GRDP") if args.grdp_forecast else None
c_tidy = tidy(pd.read_csv(args.cons_forecast), "CONS") if args.cons_forecast else None

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

# Fill first missing growth with group median to avoid dropping the first projected year
for col in ["GRDP_growth","CONS_growth"]:
    if col in merge.columns:
        merge[col] = merge.groupby(args.id_col)[col].transform(lambda s: s.fillna(s.median()))

# weights
ag, bc = args.alpha_grdp, args.beta_cons
if "GRDP_growth" not in merge.columns: ag = 0.0
if "CONS_growth" not in merge.columns: bc = 0.0
if ag + bc == 0:
    raise SystemExit("All driver weights are zero; set at least one of --alpha-grdp or --beta-cons")
if ag > 0 and bc > 0:
    s = ag + bc
    ag, bc = ag/s, bc/s

merge["price_growth"] = 0.0
if "GRDP_growth" in merge.columns: merge["price_growth"] += ag*merge["GRDP_growth"].fillna(0)
if "CONS_growth" in merge.columns: merge["price_growth"] += bc*merge["CONS_growth"].fillna(0)

# Anchor one year before first forecast
base = snap[[args.id_col, args.base_col]].drop_duplicates(args.id_col)
first_fore_year = int(merge["Year"].min())
base["Year"] = first_fore_year - 1
base = base.rename(columns={args.base_col: "Price"})

proj = pd.concat([base], ignore_index=True)
for yr in sorted(merge["Year"].unique()):
    g = merge[merge["Year"]==yr][[args.id_col,"price_growth"]]
    prev = proj[proj["Year"]==yr-1][[args.id_col,"Price"]]
    step = pd.merge(prev, g, on=args.id_col, how="left")
    step["Price"] = step["Price"] * (1.0 + step["price_growth"].fillna(0))
    step["Year"] = yr
    step = step[[args.id_col,"Year","Price"]]
    proj = pd.concat([proj, step], ignore_index=True)

# add labels
proj = proj.merge(snap[[args.id_col,"Area_display","Region"]].drop_duplicates(args.id_col), on=args.id_col, how="left")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
proj.sort_values([args.id_col,"Year"]).to_csv(args.out, index=False)
print(f"Wrote {args.out}")
