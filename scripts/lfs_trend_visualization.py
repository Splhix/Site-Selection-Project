from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_PATH = Path("data/cleaned/Revised Cleaned Files/economy/lfs_city_monthly_agg_2024.csv")
OUT_DIR = Path("reports")


def load_2024(path: Path, region: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"Region", "Province", "City", "year", "month", "EMP_w"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df = df.loc[df["year"] == 2024, ["Region", "Province", "City", "month", "EMP_w"]].copy()
    if region:
        df = df.loc[df["Region"].astype(str).str.lower() == region.lower()].copy()
        if df.empty:
            raise ValueError(f"No rows for Region='{region}'. Check the exact Region name in the CSV.")

    # Ensure month order, numeric type, and create a PeriodIndex for plotting convenience
    df["month"] = df["month"].astype(int)
    df = df.loc[df["month"].between(1, 12)]
    df["period"] = pd.PeriodIndex(pd.to_datetime({"year": 2024, "month": df["month"], "day": 1}), freq="M")
    return df


def plot_regional_trends(df: pd.DataFrame, outfile: Path) -> None:
    grouped = (df.groupby(["Region", "period"], as_index=False)["EMP_w"].mean())
    pivoted = grouped.pivot(index="period", columns="Region", values="EMP_w").sort_index()

    plt.figure(figsize=(12, 6))
    for region in pivoted.columns:
        plt.plot(pivoted.index.to_timestamp(), pivoted[region], marker="o", linewidth=2, label=str(region))

    plt.title("LFS EMP_w Regional Mean Trends — 2024")
    plt.xlabel("Month (2024)")
    plt.ylabel("EMP_w (mean across cities)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_top_cities_small_multiples(df: pd.DataFrame, outfile: Path, top_n: int = 12) -> None:
    totals = (df.groupby(["Region", "Province", "City"], as_index=False)["EMP_w"].sum())
    top = totals.nlargest(top_n, "EMP_w")
    top_keys = set(zip(top["Region"], top["Province"], top["City"]))
    df_top = df[df.set_index(["Region", "Province", "City"]).index.isin(top_keys)].copy()

    # Aggregate to unique city-month to avoid duplicate period labels
    df_top_agg = (df_top.groupby(["Region", "Province", "City", "period"], as_index=False)["EMP_w"].sum())

    # Prepare panel with complete months per city
    idx_months = pd.period_range("2024-01", "2024-12", freq="M")
    panels = []
    for (r, p, c), g in df_top_agg.groupby(["Region", "Province", "City"], as_index=False):
        series = g.set_index("period")["EMP_w"].sort_index().reindex(idx_months)
        panel_g = pd.DataFrame({
            "period": idx_months,
            "Region": r,
            "Province": p,
            "City": c,
            "EMP_w": series.values,
        })
        panels.append(panel_g)
    panel = pd.concat(panels, ignore_index=True)
    # Ensure period is Period[M] and create a datetime version for Matplotlib
    panel["period"] = pd.PeriodIndex(panel["period"], freq="M")
    panel["period_ts"] = panel["period"].dt.to_timestamp()

    n = len(top)
    cols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 3 * rows + 1), sharex=True)
    axes_arr: Sequence = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in axes_arr:
        ax.axis("off")

    for i, ((r, p, c), g) in enumerate(panel.groupby(["Region", "Province", "City"])):
        ax = axes_arr[i]
        ax.axis("on")
        ax.plot(g["period_ts"], g["EMP_w"], marker="o", linewidth=1.8, color="#1f77b4")
        ax.set_title(f"{c}\n{p} — {r}", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for j in range(i + 1, rows * cols):
        axes_arr[j].axis("off")

    fig.suptitle("Top Cities by EMP_w — 2024", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LFS Trends Visualization for 2024")
    parser.add_argument("--input", type=Path, default=IN_PATH, help="Path to lfs_city_monthly_agg_2024.csv")
    parser.add_argument("--region", type=str, default=None, help="Optional Region name filter (exact match)")
    parser.add_argument("--top_n", type=int, default=12, help="Number of top cities to chart in small multiples")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR, help="Output directory for charts")
    args = parser.parse_args()

    df = load_2024(args.input, args.region)

    args.outdir.mkdir(parents=True, exist_ok=True)
    regional_out = args.outdir / ("lfs_trends_2024_regional" + (f"_{args.region}" if args.region else "") + ".png")
    top_out = args.outdir / ("lfs_trends_2024_top_cities" + (f"_{args.region}" if args.region else "") + ".png")

    plot_regional_trends(df, regional_out)
    plot_top_cities_small_multiples(df, top_out, top_n=args.top_n)

    print(f"Saved: {regional_out}")
    print(f"Saved: {top_out}")


if __name__ == "__main__":
    main()


