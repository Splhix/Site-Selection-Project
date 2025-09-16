# CP2 — Concise Method Cards

### GRDP
- **Datasets:** grdp_ready_city_EXPANDED.csv; grdp_city_forecast.csv; grdp_city_forecast_backtests.csv
- **Sources:** PSA regional GRDP; city expansion via consumption shares (2018–2023)
- **Method:** City allocation (NCR via consumption shares); per-LGU Linear trend forecast to 2029; 80/95% prediction intervals
- **Key Parameters:** Forecast horizon 2024–2029; linear trend per city
- **Validation:** Backtests present
- **Notes:** Forecasted variable: GRDP_Const2018 (real).

### Consumption
- **Datasets:** consumption_city_forecast.csv; consumption_city_forecast_backtests.csv
- **Sources:** PSA household consumption, real 2018 pesos
- **Method:** Model ladder (ETS/ARIMA/Linear) per LGU; select by lowest backtest MAPE
- **Key Parameters:** Forecast horizon 2024–2029; min–max sanity checks
- **Validation:** Median MAPE=0.77%, Mean MAPE=0.74%, n_cities=32
- **Notes:** Forecasted variable: Consumption_Const2018 (real).

### Prices
- **Datasets:** housing_prices_driver_projection_PATCHED.csv; housing_prices_driver_projection_GRDP70.csv; housing_prices_driver_projection_CONS70.csv
- **Sources:** Marketplace snapshots (2023 anchor) + macro drivers
- **Method:** Multiplicative blend of driver ratios: Price(t)=Price_2023 * ratio_GRDP^α * ratio_Cons^β
- **Key Parameters:** α=0.5, β=0.5 (central); GRDP70/CONS70 variants; base year 2023; region median impute for missing base
- **Validation:** Derived from GRDP/Consumption; sanity checks on CAGR distribution
- **Notes:** Used for Profitability & PTI; no model backtest required.

### Affordability (PTI)
- **Datasets:** price_to_income_CALIB_FIXED.csv; family_income_total_projected_CALIB_FIXED.csv; population_households_projection.csv
- **Sources:** Price panel; Family income totals (millions PHP); Households
- **Method:** Avg income per HH = (FamilyIncome_total_millions*1e6)/Households; PTI=Price/AvgIncome
- **Key Parameters:** Unit handling: millions→PHP; join keys Area_slug,Year
- **Validation:** Unit audit and coverage check 2024–2029
- **Notes:** family_income_calibration.json present but not used in final PTI pipeline.

### Hazard/Safety
- **Datasets:** hazard_panel_WITH_TOOLTIPS.csv; fault_distance_ready_FIXED.csv; env_hazards_ready_v2_tooltips_ascii.csv
- **Sources:** Project NOAH (Flood, Surge, Landslide); PHIVOLCS (faults)
- **Method:** HazardScore = 0.4*Fault_Proximity + 0.4*Hydromet + 0.2*Landslide; Safety = 1 - HazardScore
- **Key Parameters:** Indices scaled 0–1; impute missing with region→national medians
- **Validation:** Range checks [0,1]; coverage 32/32
- **Notes:** Tooltips carry NOAH text & nearest fault name/distance.

### Employment (context)
- **Datasets:** lfs_employment_rate_panel_PSA2025_SMOOTH.csv; lfs_employment_smoothing.json
- **Sources:** PSA Table 4, Employment Rate by Region (July 2024, Jan 2025, Apr 2025, Jul 2025); base_def: Employment Rate = Employed / Labor Force, regional
- **Method:** Regional average of 2024–2025; light smoothing/caps per JSON
- **Key Parameters:** Caps level [90,98] 2026–2029; drift shrink 0.5; +/-0.5pp/year cap
- **Validation:** Range checks; coverage by Region
- **Notes:** Used in Commercialization context only.

### Households
- **Datasets:** population_households_projection.csv
- **Sources:** Provided projection (cleaned)
- **Method:** Trend/driver-based projection (as provided); used directly
- **Key Parameters:** —
- **Validation:** Coverage 2024–2029; no extreme YoY spikes
- **Notes:** Used for demand pressure and market size.

### Housing Units (context)
- **Datasets:** housing_units_projection.csv; units_per_1k_population.csv
- **Sources:** Provided projections (cleaned)
- **Method:** Driver-based path with guardrails (as provided)
- **Key Parameters:** —
- **Validation:** Coverage & non-negative checks
- **Notes:** Used as supply proxy; not required for core scoring.