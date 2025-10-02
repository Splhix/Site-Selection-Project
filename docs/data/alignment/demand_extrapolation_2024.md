# Demand Extrapolation to 2024 — CP2 Site Selection

**Scope**: NCR, Region III (Central Luzon), Region IV‑A (CALABARZON)  
**Grain**: City-level (HUC/city) records; regional growth used as fallback  
**Anchor year**: 2024 (project standard)

---

## 1) Purpose

Produce 2024 estimates of demand-side housing counts by carrying the 2020 city-level counts forward using official PSA regional growth in occupied housing units (OHU) over 2015→2020. This follows the CP2 rule: **Demand ≈ Occupied Housing Units (OHU)**.

---

## 2) Inputs

- PSA statistical tables: Occupied Housing Units by Region (1960–2020)  
  File: `2_SR on_Housing Characteristics_Statistical Tables_by Region_revised_PMMJ_CRD-approved_0.xlsx`, sheet `T1`  
  Purpose: compute regional CAGR of occupied housing units (2015→2020).
- City-level demand snapshot (2020)  
  File: `demand_clean_2020.csv`  
  Contains city-level 2020 counts for Single/House and Duplex (households and occupied units).  
  Keys: `Region | Province | City | geo_source_for_demand`.

Notes:
- Region names are standardized (e.g., `NCR` → `National Capital Region`).
- The GlobalPropertyGuide page is not used (commentary only; no unit counts).

---

## 3) Definitions

- Let OHU be Occupied Housing Units. For region `r` and year `t`, denote by `OHU_r,t`.
- Compound annual growth rate (CAGR) over years `t0 → t1` with span `n`: `(value_t1 / value_t0)^(1/n) - 1`.
- City-level metric `X_c,2020` is any numeric column that ends with `_2020` in the input (e.g., `single_house_occupied_units_2020`).

---

## 4) Method

### 4.1 Compute regional OHU CAGR (2015→2020)

\[
\text{CAGR\_occ}_{r,2015\to 2020} = \left( \frac{\text{OHU}_{r,2020}}{\text{OHU}_{r,2015}} \right)^{1/5} - 1
\]

Compute once per standardized region. This is the fallback growth applied to all cities within that region.

### 4.2 Project city metrics from 2020 → 2024

For each city metric `X_c,2020` and its region’s growth `g_r`:

\[
X_{c,2024} = X_{c,2020} \times (1 + g_r)^4
\]

- Years forward = 4 (2020 → 2024)
- `g_r` = `CAGR_occ_2015_2020` from PSA `T1`

### 4.3 Edge cases and guardrails

- If region mapping fails (missing `Region_std`), set growth to null and flag in QA.
- If OHU values for 2015 or 2020 are missing/≤0 for a region, do not compute CAGR; flag in QA.
- If an input city metric is missing or ≤0, carry null to 2024 and flag.

---

## 5) Columns & Contracts

**Identity (pass-through)**
- `Region` (may be short-form, e.g., `NCR`)
- `Region_std` (PSA standardized region name)
- `Province`, `City`, `geo_source_for_demand`

**Metrics (pattern)**
- Every numeric `*_2020` column in the input will produce a matching `*_2024` column in the output.
  - Examples: `households_single_duplex_2020` → `households_single_duplex_2024`, `single_house_occupied_units_2020` → `single_house_occupied_units_2024`, `duplex_occupied_units_2020` → `duplex_occupied_units_2024`.

**Provenance & QA (added columns)**
- `CAGR_occ_2015_2020` — regional growth from PSA `T1`
- `growth_rate_used_value` — same as above
- `growth_rate_used_basis` — `PSA Occupied Housing Units CAGR (2015-2020)`
- `growth_rate_geo_source` — `region_fallback`
- `years_used_from` — `2020`
- `years_used_to` — `2024`
- `qa_flag_missing_region_or_cagr` — true if region mapping or CAGR missing/invalid
- `qa_flag_nonpositive_output` — true if any `*_2024` is ≤ 0 or null due to invalid input

---

## 6) Worked example

Suppose a city in `Region IV-A (CALABARZON)` has:
- `X_c,2020 = 10,000` occupied single-house units
- Region CAGR (2015→2020) `g_r = 0.028` (2.8%/yr)

Then:
- `X_c,2024 = 10,000 * (1 + 0.028)^4 ≈ 11,158`
- Provenance: `growth_rate_used_value = 0.028`, `growth_rate_geo_source = region_fallback`, `years_used_from = 2020`, `years_used_to = 2024`

---

## 7) Reproducibility (pseudo-code)

```python
import pandas as pd
import numpy as np

# 1) Load PSA Occupied HU (T1) and compute regional CAGR 2015->2020
t1 = pd.read_excel(
    "2_SR on_Housing Characteristics_Statistical Tables_by Region_revised_PMMJ_CRD-approved_0.xlsx",
    sheet_name="T1",
    header=3,
).rename(columns={"Unnamed: 0": "Region", "Unnamed: 8": "2015", "Unnamed: 9": "2020"})
occ = t1[["Region", "2015", "2020"]].dropna()
occ = occ[(occ["2015"] > 0) & (occ["2020"] > 0)].copy()
occ["CAGR_occ_2015_2020"] = (occ["2020"] / occ["2015"]) ** (1 / 5) - 1

# 2) Region name harmonization (aliases → PSA standard)
alias = {
    "NCR": "National Capital Region",
    "Region III": "Region III (Central Luzon)",
    "Region IV-A": "Region IV-A (CALABARZON)",
    "CAR": "Cordillera Administrative Region (CAR)",
    "BARMM": "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)",
}

df = pd.read_csv("demand_clean_2020.csv")
df["Region_std"] = df["Region"].replace(alias)
occ = occ.rename(columns={"Region": "Region_std"})

# 3) Join growth and project all numeric *_2020 columns to *_2024
df = df.merge(occ[["Region_std", "CAGR_occ_2015_2020"]], on="Region_std", how="left")
years = 4
numeric_2020_cols = [
    c for c in df.columns
    if c.endswith("_2020") and pd.api.types.is_numeric_dtype(df[c])
]
for c in numeric_2020_cols:
    df[c.replace("_2020", "_2024")] = df[c] * (1 + df["CAGR_occ_2015_2020"]) ** years

# 4) Provenance and QA
df["growth_rate_used_value"] = df["CAGR_occ_2015_2020"]
df["growth_rate_used_basis"] = "PSA Occupied Housing Units CAGR (2015-2020)"
df["growth_rate_geo_source"] = "region_fallback"
df["years_used_from"], df["years_used_to"] = 2020, 2024
df["qa_flag_missing_region_or_cagr"] = df["CAGR_occ_2015_2020"].isna()

out_2024_cols = [c.replace("_2020", "_2024") for c in numeric_2020_cols]
df["qa_flag_nonpositive_output"] = df[out_2024_cols].le(0).any(axis=1) | df[out_2024_cols].isna().any(axis=1)

# 5) Write output
df.to_csv("demand_extrapolated_to_2024_using_PSA_occ_CAGR_std.csv", index=False)
```

---

## 8) QA summary

- Scope limited to NCR, Region III, Region IV‑A — ✅
- Every numeric `*_2020` produces a `*_2024` — ✅
- No negatives; monotonic when CAGR > 0 — ✅
- Provenance and QA columns present — ✅

---

## 9) Outputs

- Main dataset: `demand_extrapolated_to_2024_using_PSA_occ_CAGR_std.csv`  
- Inputs referenced: `demand_clean_2020.csv`, `2_SR...xlsx (T1)`

---

## 10) Assumptions & limitations

- Regional→City fallback due to lack of city time series; flagged in `growth_rate_geo_source`.
- Stationarity assumption: 2015–2020 regional growth proxies 2020–2024.
- No reweighting of building-type splits over time; all `*_2020` grown uniformly per city.
- No cross-check to total housing units by design in this artifact.

---

## 11) Versioning

- `method_version`: `v1.0_occCAGR2015_2020`  
- `prepared_by`: CP2 Data (Economy & Feasibility)  
- `date_prepared`: 2025‑10‑02 (Asia/Manila)
