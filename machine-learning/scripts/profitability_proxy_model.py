# ---------------------------------------------------------
# CP2 Construction Site Selection – Profitability Proxy-Based Model
# ---------------------------------------------------------
# Objective:
# Predict ProfitabilityScore_scn for new cities using raw economic,
# demand, and hazard variables (no pillar scores).
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(
    project_root,
    "data",
    "curated",
    "with scores",
    "app-ready",
    "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv"
)

df = pd.read_csv(data_path)

# We'll focus on a consistent scenario + unit model
SCENARIO_FILTER = "BASE"
UNITMODEL_FILTER = "MARKET_MEDIAN"  # change if needed

# Output folder for trained models
OUTPUT_DIR = Path(project_root) / "machine-learning" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "ProfitabilityScore_scn"

# Raw/proxy-based feature columns (plus one derived ratio)
BASE_FEATURE_COLS = [
    "GRDP_grdp_pc_2024_const",
    "INC_income_per_hh_2024",
    "DEM_households_single_duplex_2024",
    "DEM_units_single_duplex_2024",
    "RISK_Fault_Distance_km",
    "RISK_Flood_Level_Num",
    "RISK_StormSurge_Level_Num",
    "RISK_Landslide_Level_Num",
]

# ---------------------------------------------------------
# 2. FILTER & DERIVE FEATURES
# ---------------------------------------------------------
# Filter to chosen scenario + unit model for consistent target definition
mask = (df["Scenario"] == SCENARIO_FILTER) & (df["UnitModel"] == UNITMODEL_FILTER)
df_sub = df.loc[mask].copy()

if df_sub.empty:
    raise ValueError(f"No rows found for Scenario={SCENARIO_FILTER} & UnitModel={UNITMODEL_FILTER}")

# Keep only needed columns
needed_cols = BASE_FEATURE_COLS + [TARGET_COL]
missing = [c for c in needed_cols if c not in df_sub.columns]
if missing:
    raise KeyError(f"Missing expected columns in fact table: {missing}")

# Derive demand pressure ratio (households per unit)
df_sub["DEM_HH_to_units_ratio_2024"] = (
    df_sub["DEM_households_single_duplex_2024"] /
    df_sub["DEM_units_single_duplex_2024"].replace(0, np.nan)
)

# Final feature list (including derived ratio)
FEATURE_COLS = BASE_FEATURE_COLS + ["DEM_HH_to_units_ratio_2024"]

# Drop rows with any NA in features or target
df_sub = df_sub[FEATURE_COLS + [TARGET_COL]].dropna()

print(f"\nUsing {len(df_sub):,} rows for training (after filters & NA drop).")
print(f"Scenario filter: {SCENARIO_FILTER}, UnitModel filter: {UNITMODEL_FILTER}")
print("Feature columns:", FEATURE_COLS)
print("Target column:", TARGET_COL)

X = df_sub[FEATURE_COLS]
y = df_sub[TARGET_COL]

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. MODEL 1 – LINEAR REGRESSION (proxy-based)
# ---------------------------------------------------------
linreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
linreg_pipeline.fit(X_train, y_train)

y_pred_lin = linreg_pipeline.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = mse_lin ** 0.5

print("\n=== PROXY LINEAR REGRESSION RESULTS ===")
print(f"R²:   {r2_lin:.4f}")
print(f"MAE:  {mae_lin:.4f}")
print(f"RMSE: {rmse_lin:.4f}")

linreg_model = linreg_pipeline.named_steps["model"]
coef = linreg_model.coef_
print("\nFeature Coefficients (standardized inputs):")
for name, c in zip(FEATURE_COLS, coef):
    direction = "↑ increases profitability" if c > 0 else "↓ decreases profitability"
    print(f"  {name}: {c:.3f} ({direction})")

# ---------------------------------------------------------
# 5. MODEL 2 – RANDOM FOREST REGRESSOR (proxy-based)
# ---------------------------------------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5

print("\n=== PROXY RANDOM FOREST RESULTS ===")
print(f"R²:   {r2_rf:.4f}")
print(f"MAE:  {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")

importances = rf_model.feature_importances_
print("\nFeature Importances (proxy model):")
for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# ---------------------------------------------------------
# 6. WHAT-IF SENSITIVITY TEST (e.g., improve income or GRDP)
# ---------------------------------------------------------
print("\n=== WHAT-IF SENSITIVITY TEST (INCOME) ===")
sample = X.iloc[[0]].copy()
print("Sample city baseline:")
print(sample)

base_pred_lin = linreg_pipeline.predict(sample)[0]
base_pred_rf = rf_model.predict(sample)[0]

# Simulate a 10% higher household income
whatif_income = sample.copy()
whatif_income["INC_income_per_hh_2024"] *= 1.10

whatif_pred_lin = linreg_pipeline.predict(whatif_income)[0]
whatif_pred_rf = rf_model.predict(whatif_income)[0]

print(f"\nIf income per HH increases by 10%:")
print(f"  Linear Regression → ΔProfitability: {whatif_pred_lin - base_pred_lin:.4f}")
print(f"  Random Forest     → ΔProfitability: {whatif_pred_rf - base_pred_rf:.4f}")

# ---------------------------------------------------------
# 7. SAVE MODELS (proxy-based)
# ---------------------------------------------------------
lin_model_path = OUTPUT_DIR / "profitability_proxy_linear_model.pkl"
rf_model_path = OUTPUT_DIR / "profitability_proxy_randomforest_model.pkl"

joblib.dump(linreg_pipeline, lin_model_path)
joblib.dump(rf_model, rf_model_path)

print(f"\n💾 Proxy Linear model saved to: {lin_model_path}")
print(f"💾 Proxy Random Forest model saved to: {rf_model_path}")

# ---------------------------------------------------------
# 8. NEW CITY PREDICTION TEST (ONLY RAW FEATURES)
# ---------------------------------------------------------
print("\n=== NEW CITY PREDICTION TEST (PROXY MODEL) ===")

# Example: completely new city with only raw data known
new_city = pd.DataFrame({
    "GRDP_grdp_pc_2024_const": [150000],       # example value
    "INC_income_per_hh_2024": [350000],        # example value
    "DEM_households_single_duplex_2024": [50000],
    "DEM_units_single_duplex_2024": [30000],
    "RISK_Fault_Distance_km": [12.5],
    "RISK_Flood_Level_Num": [2],
    "RISK_StormSurge_Level_Num": [1],
    "RISK_Landslide_Level_Num": [1],
})

# Derive ratio for new city
new_city["DEM_HH_to_units_ratio_2024"] = (
    new_city["DEM_households_single_duplex_2024"] /
    new_city["DEM_units_single_duplex_2024"].replace(0, np.nan)
)

print("\nIncoming NEW CITY raw data:")
print(new_city)

pred_lin_new = linreg_pipeline.predict(new_city)[0]
pred_rf_new = rf_model.predict(new_city)[0]

print("\nPredicted ProfitabilityScore_scn for NEW CITY:")
print(f"  Proxy Linear Regression: {pred_lin_new:.4f}")
print(f"  Proxy Random Forest:     {pred_rf_new:.4f}")

# ---------------------------------------------------------
# 9. INTERPRETATION SUMMARY
# ---------------------------------------------------------
print("\n=== INTERPRETATION SUMMARY (PROXY MODEL) ===")
main_driver = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[0][0]
print(f"Strongest profitability driver (proxy model): {main_driver}")
print("This proxy-based model uses raw economic, demand, and hazard variables")
print("so that we can predict profitability and potential revenue even for")
print("entirely new imported cities that do not yet have pillar scores.")
print("\n✅ End of script. Proxy-based models are trained, saved, and ready for new cities.")
