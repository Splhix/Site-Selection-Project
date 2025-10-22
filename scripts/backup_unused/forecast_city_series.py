# forecast_city_series.py
import argparse, os, sys, json, math, warnings, pathlib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils_guardrails import (
    clip_percent, enforce_positive, winsorize_series,
    safe_log, inv_log, safe_logit, inv_logit
)
from backtest import rolling_origin_forecast

warnings.filterwarnings("ignore")

try:
    import pmdarima as pm
    HAS_PM = True
except Exception:
    HAS_PM = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS
from statsmodels.tsa.arima.model import ARIMA

@dataclass
class Config:
    forecast_horizons: list
    common_horizon_end: Optional[int]
    bounded_percent: bool
    positive_only: bool
    min_train_for_arima: int = 10
    min_train_for_ets: int = 5
    backtest_min_train: int = 5

# --- zero-safe, bounded metrics for backtests ---
def _compute_scores_bounded(res_bt: pd.DataFrame):
    """
    Returns:
      smape_pct: sMAPE in percent
      mape_bounded: MAPE in [0,1] (each term min(|e|/max(|a|,eps), 1.0))
      wape_pct: WAPE in percent
      rmse: root mean squared error (units of target)
    """
    import numpy as np
    if res_bt is None or res_bt.empty:
        return {"smape_pct": np.nan, "mape_bounded": np.nan, "wape_pct": np.nan, "rmse": np.nan}

    a = res_bt["actual"].astype(float).to_numpy()
    f = res_bt["forecast"].astype(float).to_numpy()
    e = res_bt["error"].astype(float).to_numpy()
    eps = 1e-9

    smape_pct = 100.0 * np.mean(2.0*np.abs(f-a) / (np.abs(f)+np.abs(a)+eps))
    # bounded per-term: avoids ∞ when a=0 and caps tiny-actual explosions
    mape_terms = np.minimum(np.abs(e) / np.maximum(np.abs(a), eps), 1.0)
    mape_bounded = float(np.mean(mape_terms))  # 0..1 (NOT percent)
    wape_pct = 100.0 * (np.sum(np.abs(e)) / (np.sum(np.abs(a)) + eps))
    rmse = float(np.sqrt(np.mean(e**2)))
    return {"smape_pct": float(smape_pct), "mape_bounded": mape_bounded, "wape_pct": float(wape_pct), "rmse": rmse}


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        forecast_horizons = data.get("forecast_horizons", [3,5]),
        common_horizon_end = data.get("common_horizon_end", None),
        bounded_percent = bool(data.get("bounded_percent", False)),
        positive_only = bool(data.get("positive_only", True)),
        min_train_for_arima = int(data.get("min_train_for_arima", 10)),
        min_train_for_ets = int(data.get("min_train_for_ets", 5)),
        backtest_min_train = int(data.get("backtest_min_train", 5)),
    )

def fit_arima(y: pd.Series):
    if HAS_PM and len(y) >= 10:
        m = pm.auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True, information_criterion='aicc')
        return ("ARIMA(auto)", m)
    # fallback small grid
    best = None
    best_aic = np.inf
    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                try:
                    mod = ARIMA(y, order=(p,d,q)).fit()
                    if mod.aic < best_aic:
                        best_aic = mod.aic
                        best = ("ARIMA(%d,%d,%d)"%(p,d,q), mod)
                except:
                    pass
    return best

def fcst_arima(model, steps):
    label, m = model
    if "ARIMA(auto)" in label and HAS_PM:
        f = m.predict(n_periods=steps)
        return np.array(f)
    else:
        f = m.forecast(steps=steps)
        return np.array(f)

def fit_ets(y: pd.Series):
    m = ETS(y, trend="add", seasonal=None).fit(optimized=True)
    return ("ETS(A,A)", m)

def fcst_ets(model, steps):
    _, m = model
    return np.array(m.forecast(steps))

def _safe_forecast(fc, ys, steps):
    import numpy as np
    # If any NaN/inf appears, fall back to Linear (or Hold)
    if fc is None or not np.all(np.isfinite(fc)):
        years = ys.index.values
        if len(ys) >= 3 and np.std(ys.values.astype(float)) > 0:
            s, b = np.polyfit(years, ys.values.astype(float), 1)
            future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + 1 + steps)
            return s * future_years + b, "Linear(fallback)"
        else:
            return np.array([ys.iloc[-1]] * steps), "Hold(fallback)"
    return fc, None


def choose_and_fit(y: pd.Series, positive_only=True, bounded_percent=False):
    # optional transforms
    transform = safe_logit if bounded_percent else (safe_log if positive_only else None)
    inv_transform = inv_logit if bounded_percent else (inv_log if positive_only else None)
    y_mod = transform(y) if transform else y.copy()

    # HARD-DISABLE ARIMA to avoid NaNs/long runtimes on sparse series
    # (If you ever want ARIMA back, revert this block and reinstate the ARIMA branch.)
    if len(y_mod) >= 5:
        try:
            m = fit_ets(y_mod)
            return ("ETS", m, transform, inv_transform)
        except Exception:
            pass

    # Linear/Hold fallback
    return ("LinearOrHold", None, transform, inv_transform)


def forecast_series(y: pd.Series, cfg: Config, steps: int):
    positive_only = cfg.positive_only or (y.min() > 0)
    bounded_percent = cfg.bounded_percent

    method, model, transform, inv_transform = choose_and_fit(y, positive_only, bounded_percent)
    ys = transform(y) if transform else y

    if method == "ARIMA":
        fit_func = lambda train: fit_arima(train)
        fc_func = lambda m, s: fcst_arima(m, s)
        res_bt, _ = rolling_origin_forecast(ys, fit_func, fc_func, min_train=cfg.backtest_min_train, steps=1)
        scores = _compute_scores_bounded(res_bt)

        fc = fcst_arima(model, steps)
    elif method == "ETS":
        fit_func = lambda train: fit_ets(train)
        fc_func = lambda m, s: fcst_ets(m, s)
        res_bt, _ = rolling_origin_forecast(ys, fit_func, fc_func, min_train=cfg.backtest_min_train, steps=1)
        scores = _compute_scores_bounded(res_bt)

        fc = fcst_ets(model, steps)
        fc, _ov = _safe_forecast(fc, ys, steps)
        method = _ov or method

    else:
        # Linear or Hold
        rows = []
        years = ys.index.values
        if len(ys) >= 3:
            # linear
            slope, intercept = np.polyfit(years, ys.values.astype(float), deg=1)
            for i in range(cfg.backtest_min_train, len(ys)):
                t = ys.iloc[:i]
                s2, b2 = np.polyfit(t.index.values, t.values.astype(float), deg=1)
                yhat = s2*years[i] + b2
                rows.append({"year": years[i], "actual": ys.iloc[i], "forecast": yhat, "error": ys.iloc[i]-yhat})
            res_bt = pd.DataFrame(rows)
            if res_bt.empty:
                scores = {"mape": np.nan, "smape": np.nan, "rmse": np.nan}
            else:
                mape = np.mean(np.abs(res_bt["error"]/res_bt["actual"])) * 100.0
                smape = 100*np.mean(2*np.abs(res_bt["forecast"]-res_bt["actual"])/(np.abs(res_bt["forecast"])+np.abs(res_bt["actual"])))
                rmse = np.sqrt(np.mean(res_bt["error"]**2))
                scores = {"mape": mape, "smape": smape, "rmse": rmse}
            future_years = np.arange(int(years[-1])+1, int(years[-1])+1+steps)
            fc = slope*future_years + intercept
            method = "Linear"
            fc, _ov = _safe_forecast(fc, ys, steps)
            method = _ov or method

        else:
            res_bt = pd.DataFrame(columns=["year","actual","forecast","error"])
            scores = {"mape": np.nan, "smape": np.nan, "rmse": np.nan}
            fc = np.array([ys.iloc[-1]]*steps)
            method = "Hold"

    fc_inv = inv_transform(pd.Series(fc)) if inv_transform else pd.Series(fc)
    return method, res_bt, scores, fc_inv.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--id-col", default="Area_slug")
    ap.add_argument("--year-col", default="Year")
    ap.add_argument("--output", required=True)
    ap.add_argument("--method-cards", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variable-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.method_cards, exist_ok=True)

    df = pd.read_csv(args.input)
    id_col = args.id_col if args.id_col in df.columns else ("Area_display" if "Area_display" in df.columns else None)
    if id_col is None:
        raise ValueError(f"ID column '{args.id_col}' not found and no 'Area_display' fallback found.")
    if args.year_col not in df.columns:
        raise ValueError(f"Year column '{args.year_col}' not found.")
    if args.value_col not in df.columns:
        raise ValueError(f"Value column '{args.value_col}' not found.")

    keep_cols = [c for c in [id_col, "Area_display", "Region", "Province", args.year_col, args.value_col] if c in df.columns]
    df = df[keep_cols].dropna(subset=[args.year_col, args.value_col]).copy()
    df[args.year_col] = df[args.year_col].astype(int)

    out_rows, bt_rows = [], []
    for cid, g in df.groupby(id_col):
        y = g.set_index(args.year_col)[args.value_col].sort_index()
        variable_label = args.variable_name or args.value_col

        # Flags: auto-detect percent-ish
        bounded_percent = (y.max() <= 100 and y.min() >= 0 and variable_label.lower().endswith(("%","rate","ratio")))
        positive_only = cfg.positive_only or (y.min() > 0)
        cfg2 = cfg
        cfg2.bounded_percent = cfg.bounded_percent or bounded_percent
        cfg2.positive_only = positive_only

        last_year = int(y.index.max())
        if cfg2.common_horizon_end is not None:
            steps = max(0, int(cfg2.common_horizon_end) - last_year)
        else:
            steps = max(cfg2.forecast_horizons)

        method, res_bt, scores, fc = forecast_series(y, cfg2, steps)

        # Simple symmetric PI via RMSE
        rmse = scores.get("rmse", np.nan)
        if np.isnan(rmse):
            rmse = np.std(y.values - np.mean(y.values)) if len(y)>1 else 0.0
        z80, z95 = 1.28, 1.96
        fc_years = list(range(last_year+1, last_year+1+steps))
        fc80_lo = (fc - z80*rmse).tolist()
        fc80_hi = (fc + z80*rmse).tolist()
        fc95_lo = (fc - z95*rmse).tolist()
        fc95_hi = (fc + z95*rmse).tolist()

        for yr, val in y.items():
            out_rows.append({id_col: cid, "Year": int(yr), "Value": float(val),
                             "variable": variable_label, "type": "history", "method": method,
                             "pi80_lo": np.nan, "pi80_hi": np.nan, "pi95_lo": np.nan, "pi95_hi": np.nan})
        for i, yr in enumerate(fc_years):
            out_rows.append({id_col: cid, "Year": int(yr), "Value": float(fc[i]),
                             "variable": variable_label, "type": "forecast", "method": method,
                             "pi80_lo": fc80_lo[i], "pi80_hi": fc80_hi[i], "pi95_lo": fc95_lo[i], "pi95_hi": fc95_hi[i]})
        if not res_bt.empty:
            for _, r in res_bt.iterrows():
                bt_rows.append({id_col: cid, "Year": int(r["year"]), "actual": float(r["actual"]),
                                "forecast": float(r["forecast"]), "error": float(r["error"]),
                                "variable": variable_label, "method": method})

        area_display = g["Area_display"].iloc[0] if "Area_display" in g.columns else str(cid)
        mape_b = scores.get('mape_bounded', float('nan'))
        mape_str = f"{mape_b:.3f}" if np.isfinite(mape_b) else "N/A"
        md = f"""# Method Card — {area_display} ({variable_label})

**History:** {int(y.index.min())}–{int(y.index.max())} ({len(y)} points)  
**Chosen method:** {method}  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): {scores.get('smape_pct', float('nan')):.3f}
- MAPE (bounded, 0–1): {mape_str}
- WAPE (%): {scores.get('wape_pct', float('nan')):.3f}
- RMSE: {scores.get('rmse', float('nan')):.3f}

**Forecast horizon:** {fc_years[0] if fc_years else '—'}–{fc_years[-1] if fc_years else '—'}  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** {'Positive-only' if positive_only else 'Unbounded'}; {'Percent-bounded [0,100]' if bounded_percent else 'Not a percent series'}.
"""

        card_path = pathlib.Path(args.method_cards) / f"{cid}_{variable_label}.md"
        os.makedirs(os.path.dirname(card_path), exist_ok=True)
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(md)

    out_df = pd.DataFrame(out_rows).sort_values([id_col, "Year"]).reset_index(drop=True)
    out_df.to_csv(args.output, index=False)
    if bt_rows:
        bt_df = pd.DataFrame(bt_rows)
        bt_out = os.path.splitext(args.output)[0] + "_backtests.csv"
        bt_df.to_csv(bt_out, index=False)

if __name__ == "__main__":
    main()
