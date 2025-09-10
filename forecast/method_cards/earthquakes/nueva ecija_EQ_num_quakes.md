# Method Card — Nueva Ecija (EQ_num_quakes)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 16.143
- MAPE (bounded, 0–1): 0.151
- WAPE (%): 15.179
- RMSE: 0.673

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
