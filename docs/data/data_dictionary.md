# Data Dictionary

**Keys:** `Region`, `Province`, `City` (PSGC-aligned)  
**Anchor:** 2024 (projections to be added in the next phase)  
**Units:** Explicit per field; monetary values are nominal PHP unless noted.

---

## 1) Geography (applies to all tables)

- **Region** — string  
  PSA/PSGC region name.

- **Province** — string  
  PSA/PSGC province name.

- **City** — string  
  HUC/City name per PSGC master.

---

## 2) Demand (PSA 2020 — *Single + Duplex only*)

- **households_single_duplex_2020** — integer (households)  
  Households living in single or duplex dwellings (2020).

- **units_single_duplex_2020** — integer (units)  
  **Occupied** single + duplex housing units (stock, 2020). *Primary demand signal.*

- **single_house_households_2020** — integer (households)  
  Households in single detached units.

- **single_house_occupied_units_2020** — integer (units)  
  Occupied single detached units.

- **duplex_households_2020** — integer (households)  
  Households in duplex/semi-detached units.

- **duplex_occupied_units_2020** — integer (units)  
  Occupied duplex/semi-detached units.

- **geo_source_for_demand** — string  
  Source granularity note if applicable (e.g., `city`, `province`, `region`).

> **Scope note:** Apartments/condos are intentionally excluded by design.

---

## 3) Income (PSA Table 1 — Average Annual Family Income)

- **income_per_hh_2018** — number (PHP)  
  Average annual family income per household, 2018 (nominal).

- **income_per_hh_2021** — number (PHP)  
  Average annual family income per household, 2021 (nominal).

- **income_per_hh_2023p** — number (PHP)  
  Average annual family income per household, 2023 **provisional** (nominal).

- **geo_source_for_income** — string  
  Granularity used: `city` | `province` | `region`.

---

## 4) GRDP per Capita (Snapshot)

- **grdp_pc_2024_const** — number (PHP, 2018=100)  
  Per-capita GRDP in constant 2018 pesos, year 2024.

---

## 5) LFS 2024 (Monthly so far; annualization next phase)

**Weighted levels** fields end with `_w` (persons).  
**Sample counts** fields end with `_n` (sample size).

- **EMP_w / EMP_n** — number / integer  
  Employed (weighted) and sample size, by **work location**.

- **UNEMP_w / UNEMP_n** — number / integer  
  Unemployed (weighted) and sample size (**not assigned by work location**).

- **LF_w / LF_n** — number / integer  
  Labor force (weighted) and sample size.

- **POP15p_w / POP15p_n** — number / integer  
  Population 15+ (weighted) and sample size.

- **Year** — integer  
  Calendar year (2024 in scope).

- **Month** — integer  
  Month index (Feb–Aug currently present).

**Notes:**
- LFS uses **work-location** coding; unemployed typically lack a work-location assignment.
- Do **not** inflate `*_n` (sample counts). Annualization will record observed vs. projected months.

---

## 6) Risk (Static exposures)

- **Flood_Level_Num** — integer (ordinal)  
  Flood hazard level (lower = safer).

- **Landslide_Level_Num** — integer (ordinal)  
  Landslide hazard level (lower = safer).

- **StormSurge_Level_Num** — integer (ordinal)  
  Storm surge hazard level (lower = safer).

- **Fault_Distance_km** — number (km)  
  Distance to nearest active fault.

- **Nearest_Fault_Name** — string  
  Name of the nearest active fault.

- **(Earthquake 50km fields)** — numbers (various)  
  Event counts / average magnitude / depth within 50 km (context only).

---

## 7) Conventions

- **PSGC alignment** is required for all joins using `Region`, `Province`, `City`.
- Preserve **provenance** fields (e.g., `geo_source_for_income`) in all outputs.
- 2024-projected columns (e.g., `income_per_hh_2024`, `units_single_duplex_2024`) will be added in the next phase, each with matching growth provenance:
  - `growth_rate_used_*` (annual rate applied)
  - `growth_rate_geo_source_*` (`city`/`province`/`region`)
  - `years_used_*` (e.g., `2015–2020`, `2021–2023p`)
