# Join Manifest (Planned — not executed yet)

**Status:** The final join will run **after** we align time-dependent fields to **2024**.

**Primary key:** `(Region, Province, City)` — must match PSGC exactly.

**Planned join order (all LEFT JOINs):**
1) PSGC master
2) Demand (2020 Single+Duplex)
3) Income (2018/2021/2023p)
4) GRDP per capita (2024)
5) LFS (2024 annualized after projection)
6) Risk (exposures)
7) Earthquakes (context; optional)

**Rules:**
- Any duplicate key rows abort the build and are logged.
- Any non-PSGC names must be fixed before join.
- Strings trimmed; case must match PSGC; no fuzzy joins.

**Outputs (after join):**
- `fact_table_clean_2024.csv`
- Updated `data_dictionary.md` with 2024-projected fields and growth provenance.
