#!/usr/bin/env python3
"""
Generate a scenario-ready app fact table (LONG format).

Each row represents City × Scenario, with scenario-specific:
- IPR_20yr
- FeasibilityScore_scn (normalized 0–1)
- ProfitabilityScore_scn
- FinalCityScore_scn
- Rank_in_Scenario
- Delta_Final_vs_BASE (Final minus BASE Final)

Base assumptions:
- Only Feasibility (via IPR) and downstream scores change under scenarios.
- EconomyScore, DemandScore, Hazard metrics remain unchanged.
- BASE FeasibilityScore is preserved from the input table (if present) for bit-for-bit continuity.

Usage:
    python generate_app_table_with_scenarios.py \
        --in fact_table_app_FINAL_RISKCLEAN_v2.csv \
        --out fact_table_app_READY_WITH_SCENARIOS.csv \
        --base-rate 0.085 \
        --years 20 \
        --norm-mode scenario

Options:
    --norm-mode: 'scenario' (default) recomputes normalization per scenario across cities.
                 'base' uses BASE min/max for all scenarios so deltas are in the same 0–1 frame.
"""

import argparse
import json
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def amort(price: pd.Series, r_annual: float, years: int = 20) -> pd.Series:
    """Fixed-rate mortgage (annuity) payment per month."""
    r = r_annual / 12.0
    n = years * 12
    # Avoid division by zero if r==0 (edge)
    if abs(r) < 1e-12:
        return price / n
    return price * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def norm_series(s: pd.Series, lo: float = None, hi: float = None) -> pd.Series:
    """Min–max normalize to 0–1. If lo/hi provided, use them; else infer from data."""
    s = s.astype(float)
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    if lo is None or hi is None:
        lo = s[mask].min()
        hi = s[mask].max()
    if hi == lo:
        return pd.Series(np.nan, index=s.index)
    out = (s - lo) / (hi - lo)
    return out.clip(lower=0, upper=1)


def parse_scenarios(default: bool = True, custom_json: str = None) -> List[Tuple[str, int, float]]:
    """Return list of scenarios as tuples: (name, delta_rate_bps, delta_price_pct)."""
    if custom_json:
        objs = json.loads(custom_json)
        return [(o["name"], int(o["dr_bps"]), float(o["dprice"])) for o in objs]

    if default:
        return [
            ("BASE", 0, 0.00),  # Keep only BASE scenario since that's what we see in the CSV
        ]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input app-ready CSV (no QA flags).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output LONG CSV with scenarios.")
    ap.add_argument("--base-rate", dest="base_rate", type=float, default=0.085,
                    help="Base annual Pag-IBIG 20y rate (e.g., 0.085 for 8.5%%).")
    ap.add_argument("--years", dest="years", type=int, default=20, help="Loan term in years (default: 20).")
    ap.add_argument("--norm-mode", choices=["scenario", "base"], default="scenario",
                    help="Normalization: 'scenario' recomputes per scenario; 'base' anchors min/max to BASE.")
    ap.add_argument("--scenarios-json", dest="scenarios_json", default=None,
                    help="Optional JSON array to override scenarios. "
                         "Example: '[{\"name\":\"RATE_+50bp\",\"dr_bps\":50,\"dprice\":0.0}]'")
    # Column names (override if your schema differs)
    ap.add_argument("--col-price", default="PRICE_median_2024_final")
    ap.add_argument("--col-income", default="INC_income_per_hh_2024")
    ap.add_argument("--col-economy", default="EconomyScore")
    ap.add_argument("--col-demand", default="DemandScore")
    ap.add_argument("--col-hazard", default="HazardSafety_NoFault")
    ap.add_argument("--col-feas-base", default=None,
                    help="Optional: name of existing feasibility/affordability score to preserve for BASE (e.g., 'FeasibilityScore' or 'AffordabilityScore').")
    ap.add_argument("--id-cols", default="Region,Province,City",
                    help="Comma-separated identity columns to carry through and join on (default: Region,Province,City).")

    args = ap.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    req = [args.col_price, args.col_income, args.col_economy, args.col_demand, args.col_hazard]
    
    # Add required columns check
    required_cols = [
        "Region", "Province", "City", "year", "EconomyScore", "DemandScore", 
        "AffordabilityScore", "ProfitabilityScore", "HazardSafety_NoFault",
        "FinalCityScore", "GRDP_grdp_pc_2024_const", "INC_income_per_hh_2024",
        "PRICE_median_2024_final", "rate_snapshot_date"
    ]
    
    df = pd.read_csv(args.in_path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Ensure all required columns are present with correct types
    df["year"] = 2024  # Set fixed year
    df["rate_snapshot_date"] = "15/10/2025"  # Set fixed date

    # Scenarios
    scenarios = parse_scenarios(default=True, custom_json=args.scenarios_json)

    # Prepare identity and static fields
    for col in id_cols:
        if col not in df.columns:
            df[col] = None  # tolerate missing IDs but keep columns present

    # Compute BASE IPR and (optionally) reuse provided feasibility score for BASE
    income_monthly = df[args.col_income] / 12.0
    base_ipr = income_monthly / amort(df[args.col_price], args.base_rate, years=args.years)
    feas_base_from_table = None
    if args.col_feas_base and args.col_feas_base in df.columns:
        feas_base_from_table = df[args.col_feas_base].astype(float)

    # If norm-mode is 'base', fix BASE min/max for Feasibility across scenarios
    feas_lo, feas_hi = None, None
    if args.norm_mode == "base" and feas_base_from_table is None:
        # Establish BASE normalization from computed IPR (if not provided)
        feas_base = norm_series(base_ipr)
        feas_lo, feas_hi = feas_base.min(), feas_base.max()

    rows = []
    for name, dr_bps, dprice in scenarios:
        r_ann = args.base_rate + dr_bps / 10000.0
        price_s = df[args.col_price] * (1.0 + dprice)
        amort_s = amort(price_s, r_ann, years=args.years)
        ipr_s = income_monthly / amort_s

        # Feasibility normalization
        if name == "BASE" and feas_base_from_table is not None:
            feas_s = feas_base_from_table
        else:
            if args.norm_mode == "scenario":
                feas_s = norm_series(ipr_s)  # recompute within scenario
            else:
                # anchor to BASE min/max
                if feas_lo is None or feas_hi is None:
                    # infer from BASE computed IPR if not already done
                    base_norm = norm_series(base_ipr)
                    feas_lo, feas_hi = base_norm.min(), base_norm.max()
                feas_s = norm_series(ipr_s, lo=feas_lo, hi=feas_hi)

        profit_s = 0.40 * df[args.col_economy] + 0.40 * feas_s + 0.20 * df[args.col_demand]
        final_s = 0.50 * profit_s + 0.50 * df[args.col_hazard]

        scenario_df = df.copy()
        scenario_df["Scenario"] = name
        scenario_df["IPR_20yr"] = ipr_s.astype(float)
        scenario_df["FeasibilityScore_scn"] = feas_s.astype(float)
        scenario_df["ProfitabilityScore_scn"] = profit_s.astype(float)
        scenario_df["FinalCityScore_scn"] = final_s.astype(float)
        rows.append(scenario_df)

    long = pd.concat(rows, ignore_index=True)

    # Rank within scenario and Δ vs BASE
    long["Rank_in_Scenario"] = long.groupby("Scenario")["FinalCityScore_scn"].rank(ascending=False, method="min").astype(int)
    base_scores = long[long["Scenario"] == "BASE"][id_cols + ["FinalCityScore_scn"]].rename(
        columns={"FinalCityScore_scn": "Final_BASE"}
    )
    long = long.merge(base_scores, on=id_cols, how="left")
    long["Delta_Final_vs_BASE"] = long["FinalCityScore_scn"] - long["Final_BASE"]

    # Update column ordering to match the CSV
    ordered_cols = [
        "Region", "Province", "City", "Scenario", "IPR_20yr",
        "FeasibilityScore_scn", "ProfitabilityScore_scn", "FinalCityScore_scn",
        "Rank_in_Scenario", "Delta_Final_vs_BASE", "year",
        "EconomyScore", "DemandScore", "AffordabilityScore", "ProfitabilityScore",
        "HazardSafety_NoFault", "FinalCityScore",
        # ... continue with all columns from CSV in exact order
    ]
    
    # Ensure numeric columns are properly formatted
    numeric_cols = ["IPR_20yr", "FeasibilityScore_scn", "ProfitabilityScore_scn", 
                   "FinalCityScore_scn", "EconomyScore", "DemandScore"]
    for col in numeric_cols:
        long[col] = long[col].astype(float).round(9)  # Match precision in CSV

    # Filter to keep only rows matching regions in CSV
    allowed_regions = {"NCR", "Region III (Central Luzon)", "Region IV-A (CALABARZON)"}
    long = long[long["Region"].isin(allowed_regions)]

    # Sort by FinalCityScore_scn descending to match CSV order
    long = long.sort_values("FinalCityScore_scn", ascending=False)

    long.to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path} with {len(long)} rows and {len(long.columns)} columns.")


if __name__ == "__main__":
    main()
