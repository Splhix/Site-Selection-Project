"""
Updated Revenue Model — now saves comparison CSV outputs
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib, pickle

# -------------------------------------------------------------------
# DIRECTORIES
# -------------------------------------------------------------------
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORT_DIR = project_root / "machine-learning" / "reports" / "revenue"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = project_root / "data" / "curated" / "with scores" / "app-ready" / \
    "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv"

SCENARIO = "BASE"
UNITMODEL = "MARKET_MEDIAN"
CITY_GROUP = "City"
RANDOM_SEED = 42

LOWER_MULT = 0.85
UPPER_MULT = 1.10
NOISE_SD   = 0.01

CAPTURE_RATE = 0.07

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT_JOBLIB = OUTPUT_DIR / "potential_revenue_randomforest_model.joblib"
MODEL_OUT_PKL    = OUTPUT_DIR / "potential_revenue_randomforest_model.pkl"
META_OUT         = OUTPUT_DIR / "potential_revenue_meta.pkl"

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df[(df["Scenario"] == SCENARIO) & (df["UnitModel"] == UNITMODEL)].copy()
if df.empty:
    raise ValueError("No rows found for BASE + MARKET_MEDIAN.")

if "TCP_Model" not in df.columns:
    if "TCP_Model_scn" in df.columns:
        df["TCP_Model"] = df["TCP_Model_scn"]
    else:
        raise KeyError("TCP_Model column missing.")

if "HazardRisk_NoFault" not in df.columns and "HazardSafety_NoFault" in df.columns:
    df["HazardRisk_NoFault"] = (1.0 - df["HazardSafety_NoFault"]).clip(0,1)

FEATURE_COLS = [
    "FeasibilityScore_scn","EconomyScore","DemandScore","HazardSafety_NoFault","ProfitabilityScore_scn",
    "GRDP_grdp_pc_2024_const","INC_income_per_hh_2024","EMP_w",
    "DEM_households_single_duplex_2024","DEM_units_single_duplex_2024",
    "RISK_Fault_Distance_km","RISK_Flood_Level_Num","RISK_StormSurge_Level_Num","RISK_Landslide_Level_Num",
    "TCP_Model",
]

for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0

X = df[FEATURE_COLS].astype(float).reset_index(drop=True)
groups = df[CITY_GROUP].astype(str).reset_index(drop=True).to_numpy()
tcp = df["TCP_Model"].astype(float).reset_index(drop=True).to_numpy()

print(f"Loaded {len(X)} rows for training revenue model.")

# -------------------------------------------------------------------
# LABEL CREATION
# -------------------------------------------------------------------
base = 0.95
m = (
    base
    + 0.08 * df["FeasibilityScore_scn"].astype(float).to_numpy()
    + 0.05 * df["EconomyScore"].astype(float).to_numpy()
    + 0.03 * df["DemandScore"].astype(float).to_numpy()
    + 0.02 * df["HazardSafety_NoFault"].astype(float).to_numpy()
    - 0.03 * df["HazardRisk_NoFault"].astype(float).to_numpy()
)

rng = np.random.default_rng(123)
m = m * rng.normal(1.0, NOISE_SD, len(m))
y_mult = np.clip(m, LOWER_MULT, UPPER_MULT)

# -------------------------------------------------------------------
# MODEL PIPELINE
# -------------------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rfr", RandomForestRegressor(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1))
])

# -------------------------------------------------------------------
# CROSS VALIDATION + NEW: COMPARISON CSV GENERATION
# -------------------------------------------------------------------
unique_groups = np.unique(groups)
n_splits = min(5, len(unique_groups)) if len(unique_groups) >= 2 else 1

preds = np.zeros(len(X), dtype=float)

# We track each fold’s predictions for comparison
comparison_rows = []

if n_splits == 1:
    pipe.fit(X, y_mult)
    preds[:] = pipe.predict(X)
else:
    gkf = GroupKFold(n_splits=n_splits)
    fold = 1
    for tr, te in gkf.split(X, y_mult, groups):
        pipe.fit(X.iloc[tr], y_mult[tr])
        preds[te] = pipe.predict(X.iloc[te])

        # Save fold comparison
        for idx in te:
            comparison_rows.append({
                "City": df.loc[idx, CITY_GROUP],
                "TCP_Model": tcp[idx],
                "Multiplier_true": y_mult[idx],
                "Multiplier_pred": np.clip(preds[idx], LOWER_MULT, UPPER_MULT),
                "Price_true": tcp[idx] * y_mult[idx],
                "Price_pred": tcp[idx] * np.clip(preds[idx], LOWER_MULT, UPPER_MULT),
            })

# -------------------------------------------------------------------
# BUILD FINAL COMPARISON CSVs
# -------------------------------------------------------------------
comp_df = pd.DataFrame(comparison_rows)
comp_df["Residual"] = comp_df["Price_true"] - comp_df["Price_pred"]

sample_path = REPORT_DIR / "revenue_sample_comparison.csv"
full_path   = REPORT_DIR / "revenue_full_test_comparison.csv"

comp_df.head(200).to_csv(sample_path, index=False)
comp_df.to_csv(full_path, index=False)

print(f"✔ Saved revenue comparison CSVs:\n{sample_path}\n{full_path}")

# -------------------------------------------------------------------
# FINAL TRAIN + SAVE MODEL
# -------------------------------------------------------------------
pipe.fit(X, y_mult)

artifact = {
    "pipeline": pipe,
    "feature_order": FEATURE_COLS,
    "predicts_multiplier": True,
    "multiplier_bounds": [LOWER_MULT, UPPER_MULT],
    "capture_rate": CAPTURE_RATE,
    "trained_on": {"scenario": SCENARIO, "unitmodel": UNITMODEL}
}

joblib.dump(artifact, MODEL_OUT_JOBLIB, compress=3)

with open(MODEL_OUT_PKL, "wb") as f:
    pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

joblib.dump(
    {"feature_order": FEATURE_COLS, "multiplier_bounds": [LOWER_MULT, UPPER_MULT]},
    META_OUT,
    compress=3
)

print("Revenue model saved successfully.")
