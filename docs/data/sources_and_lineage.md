# Sources & Lineage

**Status:** Datasets are **cleaned and staged**, but **NOT joined yet**. Final joining will happen **after** we align all time-dependent fields to the **2024 anchor** (Demand/Income projections; LFS annualization).

**Keys:** All joins will use PSGC-aligned `Region`, `Province`, `City`.

---

## 1) PSGC Master
- **File:** `PSGC_master_aliases (1).csv`
- **Fields:** `Region`, `Province`, `City`
- **Use:** Canonical geography for **all** joins (left-join base).
- **Notes:** Any name not matching PSGC must be fixed before joining.

## 2) Demand (PSA 2020 — Single + Duplex)
- **File:** `demand_clean_2020.csv`
- **Fields (examples):** `households_single_duplex_2020`, `units_single_duplex_2020`, `single_house_*`, `duplex_*`
- **Use:** 2020 stock of **occupied Single + Duplex** units (primary demand signal).
- **Notes:** Apartments/condos intentionally excluded (client scope).

## 3) Income (PSA Table 1)
- **File:** `income_clean_2018_2021_2023p_ALIGNED_v3.csv`
- **Fields:** `income_per_hh_2018`, `income_per_hh_2021`, `income_per_hh_2023p`, `geo_source_for_income`
- **Use:** Average annual family income per household (nominal PHP).
- **Notes:** City → Province → Region fallback already applied; `geo_source_for_income` preserves provenance.

## 4) GRDP per Capita (Snapshot)
- **File:** `grdp_pc_2024_citymapped.csv`
- **Fields:** `grdp_pc_2024_const`
- **Use:** Per-capita GRDP, **constant 2018 pesos**, year **2024**.
- **Notes:** No projection needed.

## 5) LFS 2024 (Monthly aggregates)
- **File:** `lfs_city_monthly_agg_2024.csv`
- **Fields (examples):** `Year`, `Month`, `EMP_w/EMP_n`, `UNEMP_w/UNEMP_n`, `LF_w/LF_n`, `POP15p_w/POP15p_n`
- **Use:** 2024 monthly indicators (Feb–Aug currently present); annualization to follow.
- **Notes:** Employed by **work location**. Unemployed typically **not assigned** to a work location. Do **not** inflate `*_n` counts.

## 6) Risk (Static exposures)
- **File:** `risk_clean_2024_PARTIAL_from_3uploads.csv.xlsx`
- **Fields (examples):** `Flood_Level_Num`, `Landslide_Level_Num`, `StormSurge_Level_Num`, `Fault_Distance_km`, `Nearest_Fault_Name`
- **Use:** Hazard exposure features for ranking and tooltips.
- **Notes:** Ordinal scales (lower = safer). Tooltips to be finalized.

## 7) Earthquakes (Context, 50 km radius)
- **File:** `earthquakes_city_agg_50km.csv`
- **Fields (examples):** event counts, average magnitude, average depth
- **Use:** Context for seismic risk near each city.
- **Notes:** Ensure naming is PSGC-harmonized before joins.

---

## Planned Join & Lineage (to be executed after 2024 alignment)

1. **Normalize names** to PSGC (`Region`, `Province`, `City`).
2. **Planned join order (left joins):**  
   PSGC Master → Demand → Income → GRDP → LFS → Risk → (Earthquakes as needed)
3. **Quality gates (pre-join & post-join):**  
   - No duplicate keys on `(Region, Province, City)`  
   - Required fields present; no unexpected nulls  
   - Preserve provenance fields (e.g., `geo_source_for_income`)
4. **Projections pending:** Demand/Income to 2024 (PSA or CAGR); LFS annualization.
5. **Artifacts:** final build will publish `fact_table_clean_2024.csv` + updated data dictionary.
