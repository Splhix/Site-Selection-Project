from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

# -------------------- Config --------------------
IN = Path("data/cleaned/economy/lfs_city_monthly_agg_2024.csv")
OUT_DIR = Path("data/extrapolated/economy/2025-2029")
OUT_MONTHLY = OUT_DIR / "lfs_city_monthly_2025_2029_PROJECTED_ETS.csv"
OUT_POOLED  = OUT_DIR / "lfs_city_EMPonly_pooled_2025_2029_ETS.csv"
OUT_REPORT  = OUT_DIR / "lfs_city_ETS_model_report.csv"

SEASON_LEN = 12
MIN_POINTS_FOR_SEASONAL_ETS = 24
MIN_POINTS_FOR_SES = 4
CLIP_PCT = 0.40                   # widened clip range
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
    """Fit ETS (A,A,A) when possible, fallback to SES if short history.
       Adds mild synthetic drift for very flat series."""
    if len(train) < MIN_POINTS_FOR_SES:
        return np.full(horizon, np.nan)

    first_p, last_p = train.index.min(), train.index.max()
    full_idx = pd.period_range(first_p, last_p, freq="M")
    y = train.reindex(full_idx).astype(float)

    frac_missing = float(y.isna().mean()) if len(y) > 0 else 1.0
    if len(y) < MIN_POINTS_FOR_SES or frac_missing > 0.5:
        return np.full(horizon, np.nan)

    # Fill gaps and ensure positive
    month_means = y.groupby(y.index.month).transform(lambda s: s.mean())
    y_filled = y.fillna(month_means).interpolate(method="linear", limit_direction="both")
    y_filled = y_filled.dropna().clip(lower=1e-6)
    if len(y_filled) < MIN_POINTS_FOR_SES:
        return np.full(horizon, np.nan)

    log_y = np.log(y_filled)

    if len(log_y) >= MIN_POINTS_FOR_SEASONAL_ETS:
        model = ExponentialSmoothing(
            log_y,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASON_LEN,
            damped_trend=False  # explicitly disable damping
        )
        fit = model.fit(optimized=True, use_brute=True)
    else:
        ses = SimpleExpSmoothing(log_y)
        fit = ses.fit(optimized=True)

    f = np.exp(fit.forecast(horizon))

    # --- Add small synthetic drift for nearly flat series (~+1.5% over 5 yrs) ---
    if np.std(f) / np.mean(f) < 0.01:  # <1% relative variation
        drift_factor = np.linspace(1.0, 1.015, horizon)
        f = f * drift_factor

    return f

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

def clip_guardrail(val: float, obs_min: float | None, obs_max: float | None,
                   reg_p5: float | None, reg_p95: float | None) -> float:
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

    reg_stats = (df[df["year"] == 2024]
                 .groupby(["Region", "month"])["EMP_w"]
                 .describe(percentiles=[0.05, 0.95])
                 .reset_index()
                 .rename(columns={"5%": "reg_p5", "95%": "reg_p95"}))[["Region", "month", "reg_p5", "reg_p95"]]

    rows_monthly, rows_report, rows_pooled = [], [], []

    for (r, p, c), g0 in df.groupby(["Region", "Province", "City"]):
        g = yearmonth_index(g0.sort_values(["year", "month"]))

        # --- Backtest ETS on 2023 ---
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

        # --- Prepare training data (pad to Dec 2024) ---
        g_train = g[g.index <= "2024-12"]["EMP_w"].astype(float)
        full_2024_idx = pd.period_range("2024-01", "2024-12", freq="M")
        g_train = g_train.reindex(full_2024_idx.union(g_train.index))
        g_train = g_train.interpolate(limit_direction="both")

        # --- Forecast 2025–2029 ---
        forecast_idx = pd.period_range("2025-01", "2029-12", freq="M").strftime("%Y-%m")
        horizon = len(forecast_idx)

        if len(g_train) >= MIN_POINTS_FOR_SES:
            try:
                f = ets_forecast(g_train, horizon)
                chosen_runtime = "ETS(A,A,A)_log"
            except Exception as e:
                f = np.full(horizon, np.nan)
                chosen_runtime = f"ETS_fit_error_{type(e).__name__}"
        else:
            f = np.full(horizon, np.nan)
            chosen_runtime = "ETS_insufficient_history"

        s_forecast = pd.Series(f, index=forecast_idx, dtype=float)

        # 2024 observed band
        idx_2024 = pd.period_range("2024-01", "2024-12", freq="M")
        s2024 = g.set_index(g.index)["EMP_w"].reindex(idx_2024).astype(float)
        obs_vals = s2024.dropna().values
        obs_min, obs_max = (float(np.nanmin(obs_vals)) if len(obs_vals) else np.nan,
                            float(np.nanmax(obs_vals)) if len(obs_vals) else np.nan)

        r_stats = reg_stats[reg_stats["Region"] == r]
        reg_p5 = {row["month"]: row["reg_p5"] for _, row in r_stats.iterrows()} if USE_REGIONAL_CLIP else {}
        reg_p95 = {row["month"]: row["reg_p95"] for _, row in r_stats.iterrows()} if USE_REGIONAL_CLIP else {}

        # --- Guardrails + monthly output ---
        for m_str in forecast_idx:
            y, m = map(int, m_str.split("-"))
            s_forecast.loc[m_str] = clip_guardrail(
                float(s_forecast.loc[m_str]) if np.isfinite(s_forecast.loc[m_str]) else np.nan,
                obs_min if np.isfinite(obs_min) else None,
                obs_max if np.isfinite(obs_max) else None,
                reg_p5.get(m, None),
                reg_p95.get(m, None)
            )
            rows_monthly.append({
                "Region": r, "Province": p, "City": c,
                "year": y, "month": m,
                "EMP_w": float(s_forecast.loc[m_str]) if np.isfinite(s_forecast.loc[m_str]) else np.nan,
                "observed": False,
                "projection_reliance": "time_series",
                "ts_model": chosen_runtime
            })

        rows_report.append({
            "Region": r, "Province": p, "City": c,
            **metrics
        })

        # Annual pooled mean
        for year in range(2025, 2030):
            annual_idx = [f"{year}-{m:02d}" for m in range(1, 13)]
            annual_vals = s_forecast.reindex(annual_idx).values
            rows_pooled.append({
                "Region": r, "Province": p, "City": c,
                f"EMP_w_{year}": float(np.nanmean(annual_vals)),
                f"months_observed_{year}": 0,
            })

        mean_forecast = np.nanmean(s_forecast.values)
        last_2024 = g_train.iloc[-1] if len(g_train) > 0 else np.nan
        print(f"{c} ({r}) — model={chosen_runtime}, mean forecast={mean_forecast:.2f}, last2024={last_2024:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_monthly).to_csv(OUT_MONTHLY, index=False)
    pd.DataFrame(rows_pooled).to_csv(OUT_POOLED, index=False)
    pd.DataFrame(rows_report).to_csv(OUT_REPORT, index=False)

    print("\n✅ Forecasting complete.")
    print(f"Saved monthly forecasts: {OUT_MONTHLY}")
    print(f"Saved pooled annual means: {OUT_POOLED}")
    print(f"Saved model diagnostics: {OUT_REPORT}")

if __name__ == "__main__":
    main()
