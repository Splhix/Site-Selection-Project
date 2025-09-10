# Method Card — Quezon (EQ_num_quakes)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 14.793
- MAPE (bounded, 0–1): 0.152
- WAPE (%): 14.638
- RMSE: 0.870

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
