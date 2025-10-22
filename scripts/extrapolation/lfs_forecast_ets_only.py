from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Hard-require statsmodels for ETS-only run
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


# -------------------- Config --------------------
IN = Path("data/cleaned/Revised Cleaned Files/economy/lfs_city_monthly_agg_2024.csv")
OUT_DIR = Path("data/ready")
OUT_MONTHLY = OUT_DIR / "lfs_city_monthly_2024_PROJECTED_ETS.csv"
OUT_POOLED  = OUT_DIR / "lfs_city_EMPonly_pooled_2024_ETS.csv"
OUT_REPORT  = OUT_DIR / "lfs_city_ETS_model_report.csv"

SEASON_LEN = 12
MIN_POINTS_FOR_SEASONAL_ETS = 24  # statsmodels seasonal ETS needs >=2 seasons
MIN_POINTS_FOR_SES = 6            # allow SES when history is short but non-trivial
CLIP_PCT = 0.10          # ±10% around city's observed 2024 band
USE_REGIONAL_CLIP = True
# ------------------------------------------------


def yearmonth_index(g: pd.DataFrame) -> pd.DataFrame:
    idx = pd.PeriodIndex(
        g["year"].astype(int).astype(str) + "-" + g["month"].astype(int).astype(str), freq="M"
    )
    g = g.copy()
    g.index = idx
    return g


def ets_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    # Ensure continuous monthly index and handle missing months
    if len(train) < MIN_POINTS_FOR_SES:
        return np.full(horizon, np.nan)

    # Build a continuous monthly index from first to last observed period
    first_p = train.index.min()
    last_p = train.index.max()
    full_idx = pd.period_range(first_p, last_p, freq="M")
    y = train.reindex(full_idx).astype(float)

    # If too sparse, bail out
    frac_missing = float(y.isna().mean()) if len(y) > 0 else 1.0
    if len(y) < MIN_POINTS_FOR_SES or frac_missing > 0.5:
        return np.full(horizon, np.nan)

    # Fill missing via seasonal month means then interpolate
    month_means = y.groupby(y.index.month).transform(lambda s: s.mean())
    y_filled = y.copy()
    y_filled = y_filled.fillna(month_means)
    y_filled = y_filled.interpolate(method="linear", limit_direction="both")

    # Final safeguards: drop any remaining NaNs and ensure positivity for log
    y_filled = y_filled.dropna()
    if len(y_filled) < MIN_POINTS_FOR_SES:
        return np.full(horizon, np.nan)
    y_filled = y_filled.clip(lower=1e-6)

    log_y = np.log(y_filled)

    # Choose model based on history length
    if len(log_y) >= MIN_POINTS_FOR_SEASONAL_ETS:
        # Seasonal ETS(A,?,A)
        model = ExponentialSmoothing(log_y, trend=None, seasonal="add", seasonal_periods=SEASON_LEN)
        fit = model.fit(optimized=True, use_brute=True)
        f = fit.forecast(horizon)
        return np.exp(f)
    else:
        # Fallback: Simple Exponential Smoothing on log-scale (no seasonality)
        ses = SimpleExpSmoothing(log_y)
        ses_fit = ses.fit(optimized=True)
        f = ses_fit.forecast(horizon)
        return np.exp(f)


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

    rows_monthly: list[dict] = []
    rows_report: list[dict] = []
    rows_pooled: list[dict] = []

    for (r, p, c), g0 in df.groupby(["Region", "Province", "City"]):
        g = yearmonth_index(g0.sort_values(["year", "month"]))

        # Build 2024 panel (Jan..Dec)
        idx_2024 = pd.period_range("2024-01", "2024-12", freq="M")
        s2024 = g.set_index(g.index)["EMP_w"].reindex(idx_2024).astype(float)
        observed_mask = s2024.notna().values

        # Backtest ETS on 2023 (train <= 2022 / test=2023)
        g_hist = g[g.index.year <= 2023]["EMP_w"].astype(float)
        metrics = {"MASE_ETS": np.nan, "sMAPE_ETS": np.nan, "ETS_status": "skipped"}
        if (g_hist.index.year.min() <= 2022) and (g_hist.index.year.max() >= 2023):
            train = g_hist[g_hist.index.year <= 2022]
            test  = g_hist[g_hist.index.year == 2023]
            if len(train) >= MIN_POINTS_FOR_SES and len(test) == 12:
                try:
                    f_ets = ets_forecast(train, 12)
                    metrics["MASE_ETS"] = mase(test.values, f_ets, train.values)
                    metrics["sMAPE_ETS"] = smape(test.values, f_ets)
                    metrics["ETS_status"] = "ok"
                except Exception as e:
                    metrics["ETS_status"] = f"fit_error: {type(e).__name__}"

        # Forecast missing 2024 months with ETS trained up to Aug 2024
        g_train = g[g.index <= "2024-08"]["EMP_w"].astype(float)
        missing_idx = [m for m in idx_2024 if pd.isna(s2024.get(m))]
        horizon = len(missing_idx)

        if horizon > 0:
            if len(g_train) >= MIN_POINTS_FOR_SES:
                try:
                    f = ets_forecast(g_train, horizon)
                    chosen_runtime = "ETS(A,?,A)_log"
                except Exception:
                    f = np.full(horizon, np.nan)
                    chosen_runtime = "ETS_fit_error"
            else:
                f = np.full(horizon, np.nan)
                chosen_runtime = "ETS_insufficient_history"

            # Place forecasts into missing months
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
                    float(s2024.loc[m]) if np.isfinite(s2024.loc[m]) else np.nan,
                    obs_min if np.isfinite(obs_min) else None,
                    obs_max if np.isfinite(obs_max) else None,
                    reg_p5.get(m.month, None),
                    reg_p95.get(m.month, None)
                )
        else:
            chosen_runtime = "observed_only"

        # Emit monthly rows
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

        # Report row
        rows_report.append({
            "Region": r, "Province": p, "City": c,
            **metrics
        })

        # Pooled annual
        rows_pooled.append({
            "Region": r, "Province": p, "City": c,
            "EMP_w_2024": float(np.nanmean(s2024.values)),
            "months_observed_2024": int(np.sum(observed_mask)),
        })

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_monthly).to_csv(OUT_MONTHLY, index=False)
    pd.DataFrame(rows_pooled).to_csv(OUT_POOLED, index=False)
    pd.DataFrame(rows_report).to_csv(OUT_REPORT, index=False)


if __name__ == "__main__":
    main()


