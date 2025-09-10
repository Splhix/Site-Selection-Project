# Method Card — Aurora (EQ_mean_mag)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 7.593
- MAPE (bounded, 0–1): 0.079
- WAPE (%): 7.916
- RMSE: 0.069

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
