# etl/pipelines/50_lfs_project_2024_ts_bakeoff.py
"""
Finish 2024 monthly EMP_w per city using a per-city model bake-off.

Models compared (trained ONLY on data up to 2022; holdout = 2023):
  1) Seasonal-Naive (repeat last year's same month)
  2) Seasonal-Index (Level × monthly index from train years)
  3) ETS (Exponential Smoothing) with additive seasonality on log(EMP_w)

Selection rule (per city):
  - Compute MASE (primary) and sMAPE on 2023 holdout.
  - Choose ETS only if it beats Seasonal-Index by >= 5% on MASE AND beats Seasonal-Naive.
  - Else choose Seasonal-Index (robust default).

Then finish 2024 (Sep–Dec) using the chosen model trained on all data up to Aug 2024,
anchor to observed 2024 mean, and clip projections to safe bounds.

Outputs:
  data/ready/lfs_city_monthly_2024_PROJECTED_TS.csv
  data/ready/lfs_city_EMPonly_pooled_2024_TS.csv
  data/ready/lfs_city_TS_model_report.csv
"""

from __future__ import annotations
from pathlib import Path
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import numpy as np
import pandas as pd

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Config --------------------
IN = Path("data/cleaned/Revised Cleaned Files/economy/lfs_city_monthly_agg_2024.csv")   # <= your uploaded file
OUT_DIR = Path("data/ready")
OUT_MONTHLY = OUT_DIR / "lfs_city_monthly_2024_PROJECTED_TS.csv"
OUT_POOLED  = OUT_DIR / "lfs_city_EMPonly_pooled_2024_TS.csv"
OUT_REPORT  = OUT_DIR / "lfs_city_TS_model_report.csv"

SEASON_LEN = 12
MIN_POINTS_FOR_ETS = 18       # allow ETS with >=18 points (handles 2021-02..2022-12)
IMPROVE_THRESH = 0.05         # ETS must beat Seasonal-Index by >=5% MASE
CLIP_PCT = 0.10               # ±10% around city's observed 2024 band
USE_REGIONAL_CLIP = True      # if city has no observed 2024 (edge), clip to region P5–P95
# ------------------------------------------------

def yearmonth_index(g: pd.DataFrame) -> pd.Series:
    idx = pd.PeriodIndex(g["year"].astype(int).astype(str) + "-" + g["month"].astype(int).astype(str), freq="M")
    g = g.copy()
    g.index = idx
    return g

def seasonal_naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    last_season = train[-SEASON_LEN:].values
    reps = int(np.ceil(horizon / SEASON_LEN))
    return np.tile(last_season, reps)[:horizon]

def seasonal_index_components(train: pd.Series) -> tuple[pd.Series, float]:
    # monthly index from train
    df = pd.DataFrame({"EMP": train.values, "month": train.index.month, "year": train.index.year})
    ann = df.groupby("year")["EMP"].mean()
    df = df.merge(ann.rename("ann"), on="year")
    df["ratio"] = np.where(df["ann"] > 0, df["EMP"] / df["ann"], np.nan)
    idx = df.groupby("month")["ratio"].mean()  # 12 values
    level = train.mean() / np.nanmean(idx.loc[train.index.month])
    return idx, level

def seasonal_index_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    idx, level = seasonal_index_components(train)
    future_months = [((train.index[-1] + i).month) for i in range(1, horizon + 1)]
    return np.array([level * idx.get(m, np.nan) for m in future_months])

def ets_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if len(train) < MIN_POINTS_FOR_ETS:
        logger.info(f"Insufficient points for ETS: {len(train)} < {MIN_POINTS_FOR_ETS}")
        return np.full(horizon, np.nan)
    
    y = train.clip(lower=1e-6).astype(float)
    logger.info(f"Training ETS model with {len(y)} points for horizon {horizon}")
    
    try:
        if len(y) >= 24:  # Full seasonal model if we have 2+ years
            model = ExponentialSmoothing(
                np.log(y), 
                trend=None, 
                seasonal="add", 
                seasonal_periods=SEASON_LEN
            )
        else:  # Simple exponential smoothing for shorter series
            model = SimpleExpSmoothing(np.log(y))
        
        fit = model.fit(optimized=True)
        f = fit.forecast(horizon)
        logger.info("ETS forecast successful")
        return np.exp(f)
    except Exception as e:
        logger.error(f"ETS forecast failed: {str(e)}")
        return np.full(horizon, np.nan)

def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray) -> float:
    if len(insample) <= SEASON_LEN or len(y_true) == 0:
        return np.inf
    denom = np.mean(np.abs(insample[SEASON_LEN:] - insample[:-SEASON_LEN]))
    denom = denom if denom != 0 else 1e-6
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))

def anchor_level(series_2024: pd.Series, observed_mask: np.ndarray) -> pd.Series:
    obs_mean = np.nanmean(series_2024.values[observed_mask]) if observed_mask.any() else np.nan
    all_mean = np.nanmean(series_2024.values)
    if np.isfinite(obs_mean) and np.isfinite(all_mean) and all_mean > 0:
        return series_2024 * (obs_mean / all_mean)
    return series_2024

def clip_guardrail(val: float, obs_min: float | None, obs_max: float | None, reg_p5: float | None, reg_p95: float | None) -> float:
    lo = obs_min * (1 - CLIP_PCT) if np.isfinite(obs_min) else reg_p5
    hi = obs_max * (1 + CLIP_PCT) if np.isfinite(obs_max) else reg_p95
    if lo is not None:
        val = max(val, lo)
    if hi is not None:
        val = min(val, hi)
    return val

def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep/aggregate
    df = df.loc[df["year"].between(2021, 2024) & df["month"].between(1, 12),
                ["Region", "Province", "City", "year", "month", "EMP_w"]]
    df = (df.groupby(["Region", "Province", "City", "year", "month"], as_index=False)
            .agg(EMP_w=("EMP_w", "sum")))
    return df

def main():
    df = load_input(IN)
    
    # regional P5/P95 by month (from observed 2024) used for clipping fallback
    reg_stats = (df[df["year"] == 2024]
                 .groupby(["Region", "month"])["EMP_w"]
                 .describe(percentiles=[0.05, 0.95]).reset_index())
    reg_stats = reg_stats.rename(columns={"5%": "reg_p5", "95%": "reg_p95"})[["Region", "month", "reg_p5", "reg_p95"]]

    rows_monthly = []
    rows_report = []
    rows_pooled = []

    for (r, p, c), g0 in df.groupby(["Region", "Province", "City"]):
        logger.info(f"\nProcessing {r} - {p} - {c}")
        g = yearmonth_index(g0.sort_values(["year", "month"]))  # EMP_w Series aligned to PeriodIndex

        # Build 2024 panel (Jan..Dec) aligned to full-year index
        idx_2024 = pd.period_range("2024-01", "2024-12", freq="M")
        s2024 = g.set_index(g.index)["EMP_w"].reindex(idx_2024).astype(float)
        observed_mask = s2024.notna().values

        # === Backtest on 2023 (train: <=2022 / test: 2023) ===
        # Only if we have enough history
        g_hist = g[g.index.year <= 2023]["EMP_w"].astype(float)
        use_ets = False
        chosen = "seasonal_index"
        metrics = {"MASE_sNaive": np.nan, "MASE_sIndex": np.nan, "MASE_ETS": np.nan,
                   "sMAPE_sNaive": np.nan, "sMAPE_sIndex": np.nan, "sMAPE_ETS": np.nan}

        if len(g_hist) >= (SEASON_LEN * 2) and (g_hist.index.year.min() <= 2022) and (g_hist.index.year.max() >= 2023):
            train = g_hist[g_hist.index.year <= 2022]
            test  = g_hist[g_hist.index.year == 2023]
            if len(train) >= SEASON_LEN and len(test) == 12:
                # Forecasts for 2023
                f_sna  = seasonal_naive_forecast(train, 12)
                f_sidx = seasonal_index_forecast(train, 12)
                try:
                    f_ets = ets_forecast(train, 12)
                except Exception:
                    f_ets = np.full(12, np.nan)

                # Metrics
                metrics["MASE_sNaive"]  = mase(test.values, f_sna,  train.values)
                metrics["MASE_sIndex"]  = mase(test.values, f_sidx, train.values)
                metrics["MASE_ETS"]     = mase(test.values, f_ets,  train.values)
                metrics["sMAPE_sNaive"] = smape(test.values, f_sna)
                metrics["sMAPE_sIndex"] = smape(test.values, f_sidx)
                metrics["sMAPE_ETS"]    = smape(test.values, f_ets)

                # Selection rule
                if np.isfinite(metrics["MASE_ETS"]) and metrics["MASE_ETS"] < metrics["MASE_sIndex"] * (1 - IMPROVE_THRESH) and metrics["MASE_ETS"] < metrics["MASE_sNaive"]:
                    use_ets = True
                    chosen = "ETS(A,?,A)_log"
                else:
                    chosen = "seasonal_index"

        # === Forecast missing 2024 months with chosen model (train up to Aug 2024) ===
        g_train = g[g.index <= "2024-08"]["EMP_w"].astype(float)
        # Which months are missing?
        missing_idx = [m for m in idx_2024 if pd.isna(s2024.get(m))]
        horizon = len(missing_idx)

        if horizon > 0:
            if use_ets and len(g_train) >= MIN_POINTS_FOR_ETS:
                try:
                    f = ets_forecast(g_train, horizon)
                    chosen_runtime = "ETS(A,?,A)_log"
                except Exception:
                    f = seasonal_index_forecast(g_train, horizon)
                    chosen_runtime = "seasonal_index_fallback"
            else:
                f = seasonal_index_forecast(g_train, horizon)
                chosen_runtime = "seasonal_index"

            # Place forecasts into missing months (index-aligned, avoids length mismatch)
            for k, m in enumerate(missing_idx):
                s2024.loc[m] = f[k] if k < len(f) else np.nan

            # Anchor to observed 2024 mean
            s2024 = anchor_level(s2024, observed_mask)

            # City observed band for clipping
            if observed_mask.any():
                obs_vals = s2024.values[observed_mask]
                obs_min = float(np.nanmin(obs_vals))
                obs_max = float(np.nanmax(obs_vals))
            else:
                obs_min = obs_max = np.nan

            # Regional clip fallback
            r_stats = reg_stats[reg_stats["Region"] == r]
            reg_p5 = {row["month"]: row["reg_p5"] for _, row in r_stats.iterrows()} if USE_REGIONAL_CLIP else {}
            reg_p95 = {row["month"]: row["reg_p95"] for _, row in r_stats.iterrows()} if USE_REGIONAL_CLIP else {}

            # Clip only the projected months
            for m in missing_idx:
                s2024.loc[m] = clip_guardrail(
                    float(s2024.loc[m]),
                    obs_min if np.isfinite(obs_min) else None,
                    obs_max if np.isfinite(obs_max) else None,
                    reg_p5.get(m.month, None),
                    reg_p95.get(m.month, None)
                )
        else:
            chosen_runtime = "observed_only"

        # === Emit monthly rows with provenance ===
        for m in idx_2024:
            is_obs = bool(observed_mask[idx_2024.get_loc(m)])
            rows_monthly.append({
                "Region": r, "Province": p, "City": c,
                "year": m.year, "month": m.month,
                "EMP_w": float(s2024.loc[m]) if np.isfinite(s2024.loc[m]) else np.nan,
                "observed": is_obs,
                "projection_reliance": "" if is_obs else "time_series",
                "ts_model": "" if is_obs else chosen_runtime
            })

        # === Report row (model metrics and decision) ===
        rows_report.append({
            "Region": r, "Province": p, "City": c,
            "chosen_model": chosen,
            **metrics
        })

        # === Pooled annual (mean of 12 months) ===
        rows_pooled.append({
            "Region": r, "Province": p, "City": c,
            "EMP_w_2024": float(np.nanmean(s2024.values)),
            "months_observed_2024": int(np.sum(observed_mask)),
            "ts_model_selected": chosen
        })

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_monthly).to_csv(OUT_MONTHLY, index=False)
    pd.DataFrame(rows_pooled).to_csv(OUT_POOLED, index=False)
    pd.DataFrame(rows_report).to_csv(OUT_REPORT, index=False)

if __name__ == "__main__":
    main()
