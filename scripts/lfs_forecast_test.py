# ts_finish_2024_empw.py
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# === inputs/outputs ===
IN  = Path("data/cleaned/Revised Cleaned Files/economy/lfs_city_monthly_agg_2024.csv")  # columns: year,month,Region,Province,City,EMP_w
OUT_MONTHLY = Path("data/ready/lfs_city_monthly_2024_PROJECTED_TS.csv")
OUT_POOLED  = Path("data/ready/lfs_city_EMPonly_pooled_2024_TS.csv")

MIN_POINTS = 24          # need at least 2 years to even consider ETS
SEASON_LEN = 12
IMPROVE_THRESH = 0.05    # 5% MASE improvement required over seasonal-index

def yearmonth_index(g):
    idx = pd.PeriodIndex(g["year"].astype(int).astype(str) + "-" + g["month"].astype(int).astype(str), freq="M")
    g = g.copy(); g.index = idx; return g

def seasonal_naive_forecast(train, horizon):
    # repeat last year's same months
    return train[-SEASON_LEN:].values

def seasonal_index_forecast(train, horizon):
    # build monthly ratios to annual mean over all years in train
    s = train.copy()
    df = pd.DataFrame({"EMP": s.values, "month": s.index.month, "year": s.index.year})
    ann = df.groupby("year")["EMP"].mean()
    df = df.merge(ann.rename("ann"), on="year")
    df["ratio"] = np.where(df["ann"]>0, df["EMP"]/df["ann"], np.nan)
    idx = df.groupby("month")["ratio"].mean()
    # level = mean(train) / mean(idx for observed months)
    m_obs = s.index.month
    level = s.mean() / np.nanmean(idx.loc[m_obs])
    # forecast next horizon by level * monthly index
    future_months = ((s.index[-1] + k).month for k in range(1, horizon+1))
    return np.array([level * idx.get(m, np.nan) for m in future_months])

def ets_forecast(train, horizon):
    # log-transform for stability; fallback if errors
    y = train.copy()
    y = y.clip(lower=1e-6)
    model = ExponentialSmoothing(np.log(y), trend=None, seasonal="add", seasonal_periods=SEASON_LEN)
    fit = model.fit(optimized=True, use_brute=True)
    f = fit.forecast(horizon)
    return np.exp(f)

def mase(y_true, y_pred, insample):
    # scale by seasonal naive MAE on insample
    if len(insample) <= SEASON_LEN:
        return np.inf
    denom = np.mean(np.abs(insample[SEASON_LEN:] - insample[:-SEASON_LEN]))
    if denom == 0: denom = 1e-6
    return np.mean(np.abs(y_true - y_pred)) / denom

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom==0, 1e-6, denom)
    return np.mean(2*np.abs(y_pred - y_true)/denom)

def anchor_level(complete_2024, observed_mask):
    # scale so observed mean stays unchanged
    obs_mean = complete_2024[observed_mask].mean()
    all_mean = complete_2024.mean()
    if np.isfinite(obs_mean) and np.isfinite(all_mean) and all_mean>0:
        return complete_2024 * (obs_mean / all_mean)
    return complete_2024

def clip_guardrails(series, obs_min, obs_max, reg_p5=None, reg_p95=None):
    lo = obs_min * 0.90 if np.isfinite(obs_min) else reg_p5
    hi = obs_max * 1.10 if np.isfinite(obs_max) else reg_p95
    if lo is not None: series = np.maximum(series, lo)
    if hi is not None: series = np.minimum(series, hi)
    return series

def main():
    df = pd.read_csv(IN)
    df = df.loc[df["year"].between(2021, 2024) & df["month"].between(1,12),
                ["Region","Province","City","year","month","EMP_w"]]
    # aggregate duplicates
    df = (df.groupby(["Region","Province","City","year","month"], as_index=False)
            .agg(EMP_w=("EMP_w","sum")))
    out_rows = []
    pooled_rows = []

    # regional P5/P95 per month (from observed 2024) for clipping fallback
    reg_stats = (df[df["year"]==2024]
                 .groupby(["Region","month"])["EMP_w"]
                 .describe(percentiles=[0.05,0.95]).reset_index())
    reg_stats = reg_stats[["Region","month","5%","95%"]].rename(columns={"5%":"reg_p5","95%":"reg_p95"})

    for (r,p,c), g in df.groupby(["Region","Province","City"]):
        g = yearmonth_index(g.sort_values(["year","month"]))
        # build 2024 panel Jan–Dec
        idx_2024 = pd.period_range("2024-01", "2024-12", freq="M")
        s2024 = pd.Series(index=idx_2024, dtype=float)
        s2024.loc[g.index.intersection(idx_2024)] = g.set_index(g.index)["EMP_w"].reindex(idx_2024).values
        observed = s2024.notna().values

        # Backtest on 2023 if enough history
        use_ets = False
        g_hist = g[g.index.year <= 2023]["EMP_w"].astype(float)
        if len(g_hist) >= MIN_POINTS and (g_hist.index.year.min() <= 2022) and (g_hist.index.year.max() >= 2023):
            train = g_hist[g_hist.index.year <= 2022]
            test  = g_hist[g_hist.index.year == 2023]
            if len(train)>=SEASON_LEN and len(test)==12:
                try:
                    f_ets  = ets_forecast(train, 12)
                except Exception:
                    f_ets  = np.full(12, np.nan)
                f_sidx = seasonal_index_forecast(train, 12)
                f_sna  = seasonal_naive_forecast(train, 12)

                m_ets  = mase(test.values, f_ets, train.values)
                m_sidx = mase(test.values, f_sidx, train.values)
                m_sna  = mase(test.values, f_sna, train.values)

                # choose ETS only if it beats seasonal-index by >= 5% and beats sNaive
                if np.isfinite(m_ets) and m_ets < m_sidx*(1-IMPROVE_THRESH) and m_ets < m_sna:
                    use_ets = True

        # Forecast Sep–Dec 2024 with chosen method
        # Train on all available up to Aug 2024 (or latest observed)
        g_train = g[g.index <= "2024-08"]["EMP_w"].astype(float)
        horizon = 12 - g[g.index.year==2024].shape[0]  # missing months in 2024
        if horizon < 0: horizon = 0

        if horizon > 0:
            if use_ets:
                try:
                    f_full = ets_forecast(g_train, horizon)
                    method = "ETS(A,?,A)_log"
                except Exception:
                    f_full = None
            else:
                f_full = seasonal_index_forecast(g_train, horizon)
                method = "seasonal_index"

            # place the forecasts into missing months
            months_missing = [m for m in idx_2024 if pd.isna(s2024.get(m))]
            for k, m in enumerate(months_missing):
                val = f_full[k] if f_full is not None and np.isfinite(f_full[k]) else np.nan
                s2024.loc[m] = val

            # anchor and clip
            s2024 = anchor_level(s2024, observed)
            obs_min = np.nanmin(s2024[np.where(observed)])
            obs_max = np.nanmax(s2024[np.where(observed)])
            # reg clip fallback
            reg_row = reg_stats[(reg_stats["Region"]==r)]
            reg_p5 = {row["month"]: row["reg_p5"] for _,row in reg_row.iterrows()}
            reg_p95= {row["month"]: row["reg_p95"] for _,row in reg_row.iterrows()}
            s_clipped = s2024.copy()
            for m in idx_2024:
                if not observed[idx_2024.get_loc(m)]:
                    s_clipped.loc[m] = clip_guardrails(
                        np.array([s2024.loc[m]]),
                        obs_min, obs_max,
                        reg_p5.get(m.month), reg_p95.get(m.month)
                    )[0]
            s2024 = s_clipped
        else:
            method = "observed_only"

        # write monthly rows
        for m in idx_2024:
            out_rows.append({
                "Region": r, "Province": p, "City": c,
                "year": m.year, "month": m.month,
                "EMP_w": float(s2024.loc[m]) if np.isfinite(s2024.loc[m]) else np.nan,
                "observed": bool(observed[idx_2024.get_loc(m)]),
                "projection_reliance": "time_series" if (not observed[idx_2024.get_loc(m)]) else "",
                "ts_model": method if (not observed[idx_2024.get_loc(m)]) else ""
            })

        # pooled (annual = mean of 12 months)
        pooled_rows.append({
            "Region": r, "Province": p, "City": c,
            "EMP_w_2024": float(np.nanmean(s2024.values)),
            "months_observed_2024": int(np.sum(observed)),
            "ts_model_selected": method
        })

    monthly = pd.DataFrame(out_rows)
    pooled  = pd.DataFrame(pooled_rows)
    OUT_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_MONTHLY, index=False)
    pooled.to_csv(OUT_POOLED, index=False)

if __name__ == "__main__":
    main()
