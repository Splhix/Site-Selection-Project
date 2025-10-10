# CP2 Site Selection — Score Computation (Income-Only Affordability)
**Anchor Year:** 2024  •  **Scope:** 50 HUC cities (NCR, Region III, Region IV-A)

---
## Normalization
For any variable x:
```
norm(x) = (x - min(x)) / (max(x) - min(x))
```

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

### 3) AffordabilityScore (Income-Only)
```
AffordabilityScore = norm(INC_income_per_hh_2024)
```

### 4) ProfitabilityScore
```
ProfitabilityScore = (EconomyScore + DemandScore + AffordabilityScore) / 3
```

---
## Hazard Risk Safety (used in Final Score; fault distance is gate-only)

### Component risks
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

### Composite (excludes fault distance)
```
HazardRisk_NoFault   = 0.40*EarthquakeRisk + 0.40*HydroRisk + 0.20*LandslideRisk
HazardSafety_NoFault = 1 - HazardRisk_NoFault
```

### Risk Gate (reported separately)
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

**Interpretation:** Higher = more attractive (market-viable + safer). Show `RISK_Risk_Gate` (PASS/REVIEW) alongside for decision gating.
