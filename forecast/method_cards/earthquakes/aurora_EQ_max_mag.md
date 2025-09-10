# Method Card — Aurora (EQ_max_mag)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 3.636
- MAPE (bounded, 0–1): 0.039
- WAPE (%): 3.826
- RMSE: 0.099

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
