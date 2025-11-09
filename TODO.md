# TODO: Modify LFS ETS Forecast Script to Forecast Up to 2029

## Steps to Complete

1. **Update training data selection**: Change from training up to "2024-08" to "2024-12" to use all available data. ✅
2. **Set forecast horizon and index**: Define forecast_idx as pd.period_range("2025-01", "2029-12", freq="M") and horizon=60. ✅
3. **Modify forecasting logic**: For each city, forecast 60 months and store in a series for 2025-2029. ✅
4. **Update monthly output emission**: Emit rows for 2025-2029 with observed=False, projection_reliance="time_series", and appropriate ts_model. ✅
5. **Update pooled annual output**: Compute annual means for each year from 2025 to 2029 instead of just 2024. ✅
6. **Adjust anchoring and clipping**: Keep logic, but note that for future years, no observed data, so anchoring may not apply, and clipping uses regional stats. ✅
7. **Update output file names**: Change OUT_MONTHLY to "lfs_city_monthly_2025_2029_PROJECTED_ETS.csv" and OUT_POOLED to "lfs_city_EMPonly_pooled_2025_2029_ETS.csv". ✅
8. **Run the modified script**: Execute to generate outputs and verify up to 2029.
9. **Verify outputs**: Check that monthly and pooled files contain data for 2025-2029.
