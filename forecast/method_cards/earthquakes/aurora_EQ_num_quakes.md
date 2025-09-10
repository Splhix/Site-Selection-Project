# Method Card — Aurora (EQ_num_quakes)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 8.471
- MAPE (bounded, 0–1): 0.090
- WAPE (%): 8.294
- RMSE: 0.561

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
