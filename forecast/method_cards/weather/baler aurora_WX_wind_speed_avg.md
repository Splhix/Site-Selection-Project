# Method Card — Baler, Aurora (WX_wind_speed_avg)

**History:** 2018–2023 (6 points)  
**Chosen method:** ETS  
**Why:** Decision ladder + backtests.

**Backtest (rolling-origin):**
- sMAPE (%): 15.111
- MAPE (bounded, 0–1): 0.163
- WAPE (%): 16.346
- RMSE: 0.097

**Forecast horizon:** 2024–2029  
**Intervals:** 80% / 95% (RMSE-based)

**Guardrails:** Positive-only; Not a percent series.
