# CP2 Construction Site Selection - Scripts

This directory contains Python scripts for the CP2 Construction Site Selection pipeline, based on the methods guide for reproducible data processing.

## 📁 Script Organization

### Core Pipeline Scripts
These scripts follow the methods guide and produce the main output files:

| Script | Purpose | Output Files |
|--------|---------|--------------|
| `preprocess_grdp.py` | Clean GRDP per capita data | `data/cleaned/grdp_pc_region_panel.csv` |
| `preprocess_income.py` | Clean family income data | `data/cleaned/income_per_hh_panel.csv` |
| `preprocess_housing_prices.py` | Clean housing prices data | `data/cleaned/feasibility/housing_prices_completed_PURE_INHERITANCE.csv` |
| `preprocess_demand.py` | Clean demand (occupied units) data | `data/cleaned/demand_occupied_units_region_panel.csv` |
| `preprocess_hazard.py` | Clean hazard and seismic data | `data/cleaned/risk/risk_clean_2024_PARTIAL_from_3uploads.csv.xlsx` |
| `preprocess_labor.py` | Clean labor (employment) data | `data/cleaned/labor_employment_region_panel.csv` |
| `extrapolate_grdp.py` | Extrapolate GRDP to 2024 | `data/extrapolated/grdp_pc_region_2024.csv` |
| `extrapolate_income.py` | Extrapolate income to 2024 | `data/extrapolated/income_per_hh_2024.csv` |
| `extrapolate_demand.py` | Extrapolate demand to 2024 | `data/extrapolated/demand_occupied_units_region_2024.csv` |
| `extrapolate_labor.py` | Extrapolate labor to 2024 | `data/extrapolated/labor_employment_region_2024.csv` |
| `build_fact_table_2024.py` | Build unified fact table | `data/curated/fact_table_2024.csv` |
| `compute_scores_base.py` | Compute base scores with IPR | `data/curated/with scores/fact_table_FULL_FINAL.csv` |
| `generate_app_table_with_scenarios.py` | Generate scenario variations | `data/curated/with scores/app-ready/fact_table_app_READY_WITH_SCENARIOS.csv` |
| `add_recommendations.py` | Add client recommendations | `data/curated/with scores/app-ready/fact_table_app_READY_WITH_RECS.csv` |

### Specialized Scripts
These scripts handle specific data processing tasks:

| Script | Purpose | Output Files |
|--------|---------|--------------|
| `lfs_city_builder.py` | Build LFS city aggregates | `data/cleaned/economy/lfs_city_monthly_agg_2024.csv` |
| `lfs_forecast_ets_only.py` | ETS forecasting for LFS | `data/extrapolated/economy/lfs_city_monthly_2024_PROJECTED_ETS.csv` |
| `lfs_forecast_ts_bakeoff.py` | Time series model comparison | `data/extrapolated/economy/lfs_city_monthly_2024_PROJECTED_TS.csv` |

### Utility Scripts
| Script | Purpose |
|--------|---------|
| `utils.py` | Common utility functions (standardize_geo, minmax_norm, amort, etc.) |
| `cleanup_unused_scripts.py` | Identify and remove scripts not related to current outputs |

## 🚀 Quick Start

### Option 1: Use Makefile (Recommended)
```bash
# Run complete pipeline
make all

# Run base processing only
make base

# Generate scenarios from existing fact table
make scenarios

# Add recommendations
make recs

# Clean intermediate files
make clean
```

### Option 2: Run Scripts Individually
```bash
# 1. Preprocessing
python scripts/preprocess_grdp.py
python scripts/preprocess_income.py
python scripts/preprocess_housing_prices.py
python scripts/preprocess_demand.py
python scripts/preprocess_hazard.py

# 2. Build fact table
python scripts/build_fact_table_2024.py

# 3. Compute scores
python scripts/compute_scores_base.py

# 4. Generate scenarios
python scripts/generate_app_table_with_scenarios.py \
    --in data/curated/with\ scores/fact_table_FULL_FINAL.csv \
    --out data/curated/with\ scores/app-ready/fact_table_app_READY_WITH_SCENARIOS.csv

# 5. Add recommendations
python scripts/add_recommendations.py
```

## 📊 Data Flow

```
Raw Data → Preprocessing → Extrapolation → Fact Table → Scoring → Scenarios → Recommendations
    ↓           ↓              ↓            ↓          ↓         ↓           ↓
data/raw/  data/cleaned/  data/extrapolated/  data/curated/  Scores   Scenarios   Final App Data
```

## 🔧 Dependencies

Core dependencies (install with `pip install -r requirements.txt`):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `openpyxl` - Excel file handling
- `statsmodels` - Time series forecasting (for LFS scripts)

## 📋 Script Details

### Preprocessing Scripts
- **Input**: Raw data files in `data/raw/`
- **Output**: Cleaned panel data files in `data/cleaned/`
- **Purpose**: Standardize geographic names, clean numeric data, filter for target regions, create panel structure

### Extrapolation Scripts
- **Input**: Cleaned panel data files in `data/cleaned/`
- **Output**: 2024-aligned data files in `data/extrapolated/`
- **Purpose**: Use CAGR, ETS forecasting, or carry-forward to align all data to 2024

### Integration Scripts
- **Input**: Extrapolated data files
- **Output**: Unified fact table
- **Purpose**: Merge all data sources into single table for analysis

### Scoring Scripts
- **Input**: Fact table
- **Output**: Scored fact table with IPR and composite scores
- **Purpose**: Calculate feasibility, profitability, and final city scores

### Scenario Scripts
- **Input**: Scored fact table
- **Output**: Long-format table with multiple scenarios
- **Purpose**: Generate "what-if" scenarios for different interest rates and prices

### Recommendation Scripts
- **Input**: Scenario table
- **Output**: Table with client-friendly recommendations
- **Purpose**: Add prescriptive labels and explanatory text

## 🧹 Maintenance

### Clean Up Unused Scripts
```bash
python scripts/cleanup_unused_scripts.py
```

This will:
1. Analyze all scripts in the directory
2. Identify scripts not related to current outputs
3. Optionally move or delete unused scripts

### Scripts That Can Be Removed
The following scripts are not related to current output files and can be safely removed:
- `forecast_city_series.py` - Generic forecasting tool
- `project_population_from_drivers.py` - Population projection
- `project_units_from_drivers.py` - Housing units projection
- `project_prices_from_drivers.py` - Price projection
- `backtest.py` - Forecasting utility
- `lfs_trend_visualization.py` - Visualization script
- `panel_prep.py` - Utility script
- `utils_guardrails.py` - Replaced by `utils.py`
- `env_check.py` - Environment check utility

## 🐛 Troubleshooting

### Common Issues

1. **Missing input files**: Ensure raw data files exist in `data/raw/`
2. **Column name mismatches**: Check that input files have expected column names
3. **Memory issues**: For large datasets, consider processing in chunks
4. **Permission errors**: Ensure write permissions for output directories

### Debug Mode
Add `--verbose` flag to scripts for detailed output:
```bash
python scripts/preprocess_grdp.py --verbose
```

## 📝 Notes

- All scripts follow the methods guide specifications
- Geographic names are standardized using PSGC conventions
- Scores are normalized to [0,1] range using min-max normalization
- IPR (Income-to-Payment Ratio) uses 8.5% annual rate and 20-year term
- Target regions: NCR, Region III (Central Luzon), Region IV-A (CALABARZON)
