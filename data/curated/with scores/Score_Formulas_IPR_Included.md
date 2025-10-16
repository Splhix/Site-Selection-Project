# CP2 Site Selection — Score Computation (IPR & Amortization Edition)
**Anchor Year:** 2024  •  **Scope:** 50 HUC cities (NCR, Region III, Region IV-A)  
**Rates Snapshot:** 2025-10-15 (Pag-IBIG Fixed Rates)

---

## Normalization
For any variable x:
```
norm(x) = (x - min(x)) / (max(x) - min(x))
```

---

## Amortization (Pag-IBIG Rates)
Monthly amortization is computed using the fixed-rate mortgage formula:
```
A = P * [r(1+r)^n] / [(1+r)^n - 1]
```
Where:  
`A` = monthly payment (PHP)  
`P` = PRICE_median_2024_final (PHP)  
`r` = annual_rate / 12  
`n` = years × 12 months  

**Annual Rates**
```
1 yr – 5.75%     3 yrs – 6.25%
5 yrs – 6.50%     10 yrs – 7.125%
15 yrs – 7.75%    20 yrs – 8.50%
25 yrs – 9.125%   30 yrs – 9.75%
```

Generated columns:
```
Amort_1yr, Amort_3yr, Amort_5yr, Amort_10yr, Amort_15yr,
Amort_20yr, Amort_25yr, Amort_30yr
```

---

## Income-to-Payment Ratio (IPR)
Monthly income per household:
```
MonthlyIncome = INC_income_per_hh_2024 / 12
```
Income-to-Payment Ratio per term:
```
IPR_Y = MonthlyIncome / Amort_Y
```
We compute IPR for all terms (1–30 years).  
Primary affordability signal: **IPR_20yr**

Interpretation: higher IPR = more affordable  
(e.g., IPR ≈ 3.3 → P/I ≈ 30%).

---

## Profitability Model

### 1) EconomyScore
```
EmpScore  = norm(EMP_w_2024)
GRDPpc    = norm(GRDP_grdp_pc_2024_const)
EconomyScore = 0.60*EmpScore + 0.40*GRDPpc
```

### 2) DemandScore
```
DemandScore = norm(DEM_units_single_duplex_2024)
```

### 3) AffordabilityScore (IPR-Based)
```
AffordabilityScore = norm(IPR_20yr)
```

### 4) ProfitabilityScore
```
ProfitabilityScore = (EconomyScore + DemandScore + AffordabilityScore) / 3
```

---

## Hazard Risk Safety (used in Final Score; fault distance = gate-only)

### Component Risks
```
EQ_freq   = norm(RISK_events_50km_per_year)
EQ_m5plus = norm(RISK_m5plus_50km)
EQ_depthR = 1 - norm(RISK_avg_depth_km_50km)

EarthquakeRisk = 0.45*EQ_m5plus + 0.35*EQ_freq + 0.20*EQ_depthR

FloodRisk      = (RISK_Flood_Level_Num - 1)/2
StormSurgeRisk = (RISK_StormSurge_Level_Num - 1)/2
HydroRisk      = 0.5*FloodRisk + 0.5*StormSurgeRisk

LandslideRisk  = (RISK_Landslide_Level_Num - 1)/2
```

### Composite (Excludes Fault Distance)
```
HazardRisk_NoFault   = 0.40*EarthquakeRisk + 0.40*HydroRisk + 0.20*LandslideRisk
HazardSafety_NoFault = 1 - HazardRisk_NoFault
```

### Risk Gate (Reported Separately)
City is flagged **REVIEW** if any is true; otherwise **PASS**.
```
RISK_Fault_Distance_km < 1
OR RISK_Flood_Level_Num = 3
OR RISK_StormSurge_Level_Num = 3
OR RISK_events_50km_per_year >= 140
OR RISK_m5plus_50km >= 3
```

---

## Final City Score
```
FinalCityScore = 0.50*ProfitabilityScore + 0.50*HazardSafety_NoFault
```

**Interpretation:** higher = more attractive (site is profitable + safe).  
Show `RISK_Risk_Gate` (PASS/REVIEW) alongside for decision gating.

---

## Transparency & Guardrails
- Price policy: Pure inheritance (≥ 5 city listings → city median; else province → region → national).  
- Outlier control: Log-price IQR trimming per city before median.  
- Normalization: Min–max within 50 cities; uniform metrics left blank.  
- Sensitivity (recommended): ±25 bps on rates and ±10% on inherited prices.  
- Units: PHP for prices / amortizations; IPR unitless; scores 0–1 (higher = better).
