# Method Card — Batangas (EQ_num_quakes)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 5.256
- MAPE (bounded, 0–1): 0.055
- WAPE (%): 5.276
- RMSE: 0.431

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
