# Income per Household — 2024 Alignment Method

**Purpose:** Compute city-level `income_per_hh_2024` by carrying PSA-reported income forward to 2024 using simple, auditable growth rules.

## Inputs (required columns)
- `income_per_hh_2018` (nominal PHP, PSA)
- `income_per_hh_2021` (nominal PHP, PSA)
- `income_per_hh_2023p` (nominal PHP, PSA; 2023 provisional)
- `geo_source_for_income` (text: `city` | `province` | `region` indicating provenance)

Notes:
- All values are nominal (not CPI-adjusted). Do not deflate/inflate within this method.
- If a city lacks a PSA series, `geo_source_for_income` indicates the fallback aggregation level used upstream.

## Definitions
- Let `I_2018`, `I_2021`, `I_2023p` be the three input values for a city.
- Compound Annual Growth Rate (CAGR) over years `t0 → t1` with span `n` years: `(I_t1 / I_t0)^(1/n) - 1`.

## Step 1 — Compute growth candidates
- Short-run (closest to 2024, 2-year span):
  - `g_21_23p = (I_2023p / I_2021)^(1/2) - 1`
- Long-run (smoother, 5-year span):
  - `g_18_23p = (I_2023p / I_2018)^(1/5) - 1`

Handle missing safely:
- If either denominator or numerator is missing or ≤ 0, mark that CAGR as missing and set `qa_flag_missing_any_cagr = true`.

## Step 2 — Choose the rate (switch rule)
Default is the short-run rate `g_21_23p`. Switch to the long-run rate `g_18_23p` if either condition holds:
1. Extreme short-run: `abs(g_21_23p) > 0.15` (i.e., > 15% per year), or
2. Sign flip: `sign(g_21_23p) != sign(g_18_23p)`.

Record the decision:
- `growth_rate_used_income` = chosen CAGR
- `years_used_income` = `2021–2023p` or `2018–2023p`
- `switched_to_longer_cagr` = true/false
- `switch_reason` = `abs>15%`, `sign_flip`, or `both` (empty if not switched)
- `growth_rate_geo_source_income` = `geo_source_for_income`

Edge cases:
- If `g_21_23p` is missing but `g_18_23p` exists, use `g_18_23p`.
- If both are missing, set `growth_rate_used_income = null` and propagate a QA flag (see below).

## Step 3 — Carry forward to 2024
Compute:
- `income_per_hh_2024 = I_2023p * (1 + growth_rate_used_income)`

QA checks:
- `qa_flag_nonpositive_2024 = (income_per_hh_2024 <= 0 or null)`
- `qa_flag_missing_any_cagr = true` if any CAGR could not be computed due to missing/invalid inputs

## Outputs (add/overwrite columns)
- `income_per_hh_2024` (nominal PHP)
- `growth_rate_used_income`
- `years_used_income`
- `growth_rate_geo_source_income`
- `switched_to_longer_cagr`
- `switch_reason`
- `qa_flag_missing_any_cagr`
- `qa_flag_nonpositive_2024`

## Worked example
Given: `I_2018 = 220,000`, `I_2021 = 250,000`, `I_2023p = 270,000` (PHP)
- `g_21_23p = (270000/250000)^(1/2) - 1 ≈ 0.0392` (3.92%/yr)
- `g_18_23p = (270000/220000)^(1/5) - 1 ≈ 0.0417` (4.17%/yr)
- Switch test: `abs(0.0392) <= 0.15` and signs match → keep short-run
- `income_per_hh_2024 = 270000 * (1 + 0.0392) ≈ 280,584`
- Record: `years_used_income = 2021–2023p`, `switched_to_longer_cagr = false`

## Reproducibility checklist
- Use only PSA-provided nominal series for 2018, 2021, 2023p.
- Compute geometric-means (CAGR), not arithmetic averages.
- Apply the switch rule exactly as stated (15% threshold, sign flip).
- Do not CPI-adjust within this method; nominal-in to nominal-out.
- Emit all decision and QA columns alongside the output.

## Rationale (why this is safe/simple)
- Simple and audit-friendly: limited to PSA anchor years and geometric means.
- Robust to noise: long-horizon rate smooths short-run spikes when needed.
- Aligns with adviser guidance: “use PSA growth; else geometric mean across available years.”

## Alternatives (optional, not implemented here)
- CPI-based nominal carry (conservative baseline)
- Median of available CAGRs
- Robust log-trend (Theil–Sen) over {2018, 2021, 2023p}
- Hierarchical pooling to province/region (more complex)
