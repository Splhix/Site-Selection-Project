# Method Card — Bataan (EQ_num_quakes)

**History:** 2016–2025 (10 points)  
**Chosen method:** ARIMA  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 13.694
- MAPE (bounded, 0–1): 0.163
- WAPE (%): 13.659
- RMSE: 0.777

**Forecast horizon:** 2026–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
