# Method Card — Quezon (WX_tmax_avg)

**History:** 2018–2023 (6 points)  
**Chosen method:** ETS  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 0.100
- MAPE (bounded, 0–1): 0.001
- WAPE (%): 0.101
- RMSE: 0.003

**Forecast horizon:** 2024–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
