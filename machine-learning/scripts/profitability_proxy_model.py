# ---------------------------------------------------------
# CP2 Construction Site Selection – Proxy-Based ML Model
# ---------------------------------------------------------
# Objective:
# Predict Potential Revenue (in PHP) of site locations using
# validated profitability scores plus raw economic, demand,
# and hazard indicators.
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
    "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_DYNAMIC_TEXT_FIXED_RANK.csv"
)

df = pd.read_csv(data_path)

SCENARIO_FILTER = "BASE"
UNITMODEL_FILTER = "MARKET_MEDIAN"   # adjust if needed

OUTPUT_DIR = Path(project_root) / "machine-learning" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. FILTER & FEATURE PREPARATION
# ---------------------------------------------------------
mask = (df["Scenario"] == SCENARIO_FILTER) & (df["UnitModel"] == UNITMODEL_FILTER)
df_sub = df.loc[mask].copy()

if df_sub.empty:
    raise ValueError("No data found for specified scenario/unit model filters.")

# --- Core raw drivers and validated profitability predictor ---
BASE_FEATURE_COLS = [
    "ProfitabilityScore_scn",          # validated company metric
    "GRDP_grdp_pc_2024_const",
    "INC_income_per_hh_2024",
    "DEM_households_single_duplex_2024",
    "DEM_units_single_duplex_2024",
    "RISK_Fault_Distance_km",
    "RISK_Flood_Level_Num",
    "RISK_StormSurge_Level_Num",
    "RISK_Landslide_Level_Num",
]

# --- Derive ratio feature: demand pressure ---
df_sub["DEM_HH_to_units_ratio_2024"] = (
    df_sub["DEM_households_single_duplex_2024"] /
    df_sub["DEM_units_single_duplex_2024"].replace(0, np.nan)
)

# --- Compute Potential Revenue Proxy ---
# Using TCP (unit price) × number of single/duplex units × assumed capture rate (7%)
# Adjust the column name for TCP if needed (e.g., UnitModel_TCP or Price_Median)
tcp_col_candidates = [c for c in df_sub.columns if "TCP" in c or "price" in c.lower()]
if not tcp_col_candidates:
    raise KeyError("No TCP/price column found; please verify your fact table column name.")
TCP_COL = tcp_col_candidates[0]

CAPTURE_RATE = 0.07
df_sub["PotentialRevenueProxy"] = (
    df_sub["DEM_units_single_duplex_2024"] *
    df_sub[TCP_COL] *
    CAPTURE_RATE
)

# --- Final feature list ---
FEATURE_COLS = BASE_FEATURE_COLS + ["DEM_HH_to_units_ratio_2024"]

TARGET_COL = "PotentialRevenueProxy"

# --- Drop NA rows ---
df_sub = df_sub[FEATURE_COLS + [TARGET_COL]].dropna()

print(f"\nTraining rows: {len(df_sub):,}")
print(f"Scenario: {SCENARIO_FILTER} | Unit Model: {UNITMODEL_FILTER}")
print("Feature columns:", FEATURE_COLS)
print("Target:", TARGET_COL, "(proxy for potential revenue)")

X = df_sub[FEATURE_COLS]
y = df_sub[TARGET_COL]

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. LINEAR REGRESSION MODEL
# ---------------------------------------------------------
lin_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lin_pipe.fit(X_train, y_train)

y_pred_lin = lin_pipe.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
rmse_lin = mean_squared_error(y_test, y_pred_lin) ** 0.5

print("\n=== LINEAR REGRESSION (Potential Revenue) ===")
print(f"R²:   {r2_lin:.4f}")
print(f"MAE:  {mae_lin:,.0f}")
print(f"RMSE: {rmse_lin:,.0f}")

lin_model = lin_pipe.named_steps["model"]
coef = lin_model.coef_
print("\nFeature Coefficients (standardized inputs):")
for n, c in zip(FEATURE_COLS, coef):
    print(f"  {n}: {c:.3f}")

# ---------------------------------------------------------
# 5. RANDOM FOREST REGRESSOR
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
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5

print("\n=== RANDOM FOREST (Potential Revenue) ===")
print(f"R²:   {r2_rf:.4f}")
print(f"MAE:  {mae_rf:,.0f}")
print(f"RMSE: {rmse_rf:,.0f}")

importances = rf_model.feature_importances_
print("\nFeature Importances:")
for n, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"  {n}: {imp:.3f}")

# ---------------------------------------------------------
# 6. WHAT-IF SENSITIVITY (increase income by 10%)
# ---------------------------------------------------------
print("\n=== WHAT-IF SENSITIVITY TEST (Income +10%) ===")
sample = X.iloc[[0]].copy()
base_lin = lin_pipe.predict(sample)[0]
base_rf = rf_model.predict(sample)[0]

whatif = sample.copy()
whatif["INC_income_per_hh_2024"] *= 1.10

whatif_lin = lin_pipe.predict(whatif)[0]
whatif_rf = rf_model.predict(whatif)[0]

print(f"Linear Regression → ΔRevenue: {whatif_lin - base_lin:,.0f} PHP")
print(f"Random Forest     → ΔRevenue: {whatif_rf - base_rf:,.0f} PHP")

# ---------------------------------------------------------
# 7. SAVE MODELS
# ---------------------------------------------------------
lin_path = OUTPUT_DIR / "potential_revenue_linear_model.pkl"
rf_path = OUTPUT_DIR / "potential_revenue_randomforest_model.pkl"

joblib.dump(lin_pipe, lin_path)
joblib.dump(rf_model, rf_path)

print(f"\n💾 Linear model saved to: {lin_path}")
print(f"💾 Random Forest model saved to: {rf_path}")

# ---------------------------------------------------------
# 8. NEW CITY PREDICTION TEST
# ---------------------------------------------------------
print("\n=== NEW CITY PREDICTION TEST ===")

new_city = pd.DataFrame({
    "ProfitabilityScore_scn": [0.65],
    "GRDP_grdp_pc_2024_const": [150000],
    "INC_income_per_hh_2024": [350000],
    "DEM_households_single_duplex_2024": [50000],
    "DEM_units_single_duplex_2024": [30000],
    "RISK_Fault_Distance_km": [12.0],
    "RISK_Flood_Level_Num": [2],
    "RISK_StormSurge_Level_Num": [1],
    "RISK_Landslide_Level_Num": [1],
})
new_city["DEM_HH_to_units_ratio_2024"] = (
    new_city["DEM_households_single_duplex_2024"] /
    new_city["DEM_units_single_duplex_2024"]
)

print("New city raw data:")
print(new_city)

pred_lin_new = lin_pipe.predict(new_city)[0]
pred_rf_new = rf_model.predict(new_city)[0]

print(f"\nPredicted Potential Revenue (PHP):")
print(f"  Linear Regression: {pred_lin_new:,.0f}")
print(f"  Random Forest:     {pred_rf_new:,.0f}")

# ---------------------------------------------------------
# 9. INTERPRETATION SUMMARY
# ---------------------------------------------------------
print("\n=== INTERPRETATION SUMMARY ===")
main_driver = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[0][0]
print(f"Strongest driver of potential revenue: {main_driver}")
print("This ML model extends the validated profitability logic by forecasting")
print("expected revenue outcomes (in pesos) using economic, demand, and hazard variables.")
print("It is therefore capable of predicting the profitability and potential revenue")
print("of entirely new city data without requiring pillar scores.")
print("\n✅ End of script. Models trained, saved, and ready for new-city predictions.")