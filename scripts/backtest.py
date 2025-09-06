# backtest.py
import numpy as np
import pandas as pd

def rolling_origin_forecast(y, fit_func, fcst_func, min_train=5, steps=1):
    years = y.index.to_list()
    rows = []
    for i in range(min_train, len(y)):
        train = y.iloc[:i]
        test_year = years[i]
        model = fit_func(train)
        fcst = fcst_func(model, steps)[-1]  # 1-step ahead
        rows.append({"year": test_year, "actual": y.iloc[i], "forecast": fcst, "error": y.iloc[i]-fcst})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {"mape": np.nan, "smape": np.nan, "rmse": np.nan}
    mape = np.mean(np.abs(res["error"]/res["actual"])) * 100.0
    smape = 100*np.mean(2*np.abs(res["forecast"]-res["actual"])/(np.abs(res["forecast"])+np.abs(res["actual"])))
    rmse = np.sqrt(np.mean(res["error"]**2))
    return res, {"mape": mape, "smape": smape, "rmse": rmse}

def naive_forecast(y, steps=1):
    return np.array([y.iloc[-1]]*steps)

def drift_forecast(y, steps=1):
    if len(y) < 2:
        return naive_forecast(y, steps)
    first, last = y.iloc[0], y.iloc[-1]
    n = len(y)-1
    slope = (last-first)/n
    return np.array([last + slope*k for k in range(1, steps+1)])
