"""
CP2 – Potential Revenue RandomForest Model (New Artifact)
This script trains a RandomForestRegressor on the same pseudo-label
multiplier logic used for GradientBoost, performs GroupKFold CV,
and exports ETL-ready model artifacts.

Artifacts:
    potential_revenue_randomforest_model.joblib
    potential_revenue_randomforest_model.pkl
    potential_revenue_randomforest_meta.pkl
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

import joblib
import pickle

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(
    project_root,
    "data", "curated", "with scores", "app-ready",
    "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv",
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# New artifact names (per user instruction)
MODEL_OUT_JOBLIB = OUTPUT_DIR / "potential_revenue_randomforest_model.joblib"
MODEL_OUT_PKL    = OUTPUT_DIR / "potential_revenue_randomforest_model.pkl"
META_OUT         = OUTPUT_DIR / "potential_revenue_randomforest_meta.pkl"

SCENARIO    = "BASE"
UNITMODEL   = "MARKET_MEDIAN"
CITY_GROUP  = "City"
RANDOM_SEED = 42

LOWER_MULT = 0.85
UPPER_MULT = 1.10
NOISE_SD   = 0.01     # tiny noise for de-collinearity
CAPTURE_RATE = 0.07   # defensible default for ETL revenue proxy

# -----------------------------------------------------------
# LOAD AND FILTER
# -----------------------------------------------------------
df = pd.read_csv(DATA_PATH)
mask = (df["Scenario"] == SCENARIO) & (df["UnitModel"] == UNITMODEL)
df = df.loc[mask].copy()
if df.empty:
    raise ValueError("No rows found for BASE + MARKET_MEDIAN.")

# Ensure TCP column
if "TCP_Model" not in df.columns:
    if "TCP_Model_scn" in df.columns:
        df["TCP_Model"] = df["TCP_Model_scn"]
    else:
        raise KeyError("TCP_Model column missing.")

# Build HazardRisk_NoFault if absent
if "HazardRisk_NoFault" not in df.columns and "HazardSafety_NoFault" in df.columns:
    df["HazardRisk_NoFault"] = (1.0 - df["HazardSafety_NoFault"]).clip(0, 1)

# -----------------------------------------------------------
# FEATURES (same as GB version)
# -----------------------------------------------------------
FEATURE_COLS = [
    # pillar summaries
    "FeasibilityScore_scn","EconomyScore","DemandScore","HazardSafety_NoFault","ProfitabilityScore_scn",
    # fundamentals
    "GRDP_grdp_pc_2024_const","INC_income_per_hh_2024","EMP_w",
    "DEM_households_single_duplex_2024","DEM_units_single_duplex_2024",
    # risk details
    "RISK_Fault_Distance_km","RISK_Flood_Level_Num","RISK_StormSurge_Level_Num","RISK_Landslide_Level_Num",
    # pricing context
    "TCP_Model",
]
if "TCP_MarketMedian" in df.columns:
    FEATURE_COLS.append("TCP_MarketMedian")

# Fill any missing feature columns with neutral defaults
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0

X = df[FEATURE_COLS].astype(float).reset_index(drop=True)
groups = df[CITY_GROUP].astype(str).reset_index(drop=True).to_numpy()
tcp = df["TCP_Model"].astype(float).reset_index(drop=True).to_numpy()

print(f"✅ Loaded {len(X)} training rows for BASE, MARKET_MEDIAN")

# -----------------------------------------------------------
# DEFENSIBLE PSEUDO LABEL (copied exactly from GB script)
# -----------------------------------------------------------
base = 0.95
m = (
    base
    + 0.08 * df["FeasibilityScore_scn"].astype(float).to_numpy()
    + 0.05 * df["EconomyScore"].astype(float).to_numpy()
    + 0.03 * df["DemandScore"].astype(float).to_numpy()
    + 0.02 * df["HazardSafety_NoFault"].astype(float).to_numpy()
    - 0.03 * df["HazardRisk_NoFault"].astype(float).to_numpy()
)

# Noise
rng = np.random.default_rng(123)
m = m * rng.normal(1.0, NOISE_SD, size=len(m))

# Clip
y_mult = np.clip(m, LOWER_MULT, UPPER_MULT)

# -----------------------------------------------------------
# RANDOM FOREST MODEL
# -----------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("rf", RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        bootstrap=True,
        n_jobs=-1,
    ))
])

# -----------------------------------------------------------
# GROUPED CROSS VALIDATION
# -----------------------------------------------------------
unique_groups = np.unique(groups)
n_splits = min(5, len(unique_groups)) if len(unique_groups) >= 2 else 1
preds_mult = np.zeros(len(X), dtype=float)

if n_splits == 1:
    pipe.fit(X, y_mult)
    preds_mult[:] = pipe.predict(X)

else:
    gkf = GroupKFold(n_splits=n_splits)
    fold = 1
    for tr, te in gkf.split(X, y_mult, groups=groups):
        pipe.fit(X.iloc[tr], y_mult[tr])
        preds_mult[te] = pipe.predict(X.iloc[te])

        # evaluate multiplier
        m_true = y_mult[te]
        m_pred = np.clip(preds_mult[te], LOWER_MULT, UPPER_MULT)

        mae_m = mean_absolute_error(m_true, m_pred)
        rmse_m = mean_squared_error(m_true, m_pred)**0.5
        r2_m = r2_score(m_true, m_pred)

        # evaluate price (TCP * m)
        y_true_p = tcp[te] * m_true
        y_pred_p = tcp[te] * m_pred
        mae_p = mean_absolute_error(y_true_p, y_pred_p)
        rmse_p = mean_squared_error(y_true_p, y_pred_p)**0.5
        r2_p = r2_score(y_true_p, y_pred_p)

        print(
            f"Fold {fold}: [mult] R²={r2_m:.4f}, MAE={mae_m:.4f}, RMSE={rmse_m:.4f} "
            f"[price] R²={r2_p:.4f}, MAE={mae_p:,.0f}, RMSE={rmse_p:,.0f}"
        )
        fold += 1

# Overall CV summary
m_true_all = y_mult
m_pred_all = np.clip(preds_mult, LOWER_MULT, UPPER_MULT)
mae_m_all = mean_absolute_error(m_true_all, m_pred_all)
rmse_m_all = mean_squared_error(m_true_all, m_pred_all)**0.5
r2_m_all = r2_score(m_true_all, m_pred_all)

y_true_p_all = tcp * m_true_all
y_pred_p_all = tcp * m_pred_all
mae_p_all = mean_absolute_error(y_true_p_all, y_pred_p_all)
rmse_p_all = mean_squared_error(y_true_p_all, y_pred_p_all)**0.5
r2_p_all = r2_score(y_true_p_all, y_pred_p_all)

print("\n=== CV Summary ===")
print(f"[mult]  R²={r2_m_all:.4f}, MAE={mae_m_all:.4f}, RMSE={rmse_m_all:.4f}")
print(f"[price] R²={r2_p_all:.4f}, MAE={mae_p_all:,.0f}, RMSE={rmse_p_all:,.0f}")

# -----------------------------------------------------------
# FIT FINAL MODEL ON FULL DATA
# -----------------------------------------------------------
pipe.fit(X, y_mult)

artifact = {
    "pipeline": pipe,
    "feature_order": FEATURE_COLS,
    "predicts_multiplier": True,
    "multiplier_bounds": [LOWER_MULT, UPPER_MULT],
    "capture_rate": CAPTURE_RATE,
    "label_type": "multiplier",
    "trained_on": {"scenario": SCENARIO, "unitmodel": UNITMODEL},
    "model_type": "RandomForestRegressor"
}

# -----------------------------------------------------------
# SAVE ARTIFACTS
# -----------------------------------------------------------
joblib.dump(artifact, MODEL_OUT_JOBLIB, compress=3)

with open(MODEL_OUT_PKL, "wb") as f:
    pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

joblib.dump(
    {"feature_order": FEATURE_COLS, "multiplier_bounds": [LOWER_MULT, UPPER_MULT], "capture_rate": CAPTURE_RATE},
    META_OUT,
    compress=3
)

print(f"\n✅ Saved model → {MODEL_OUT_JOBLIB.name}")
print(f"✅ Saved model → {MODEL_OUT_PKL.name}")
print(f"✅ Saved meta  → {META_OUT.name}")

# -----------------------------------------------------------
# SENSITIVITY CHECK (same as original)
# -----------------------------------------------------------
X_sens = X.copy()
X_sens["FeasibilityScore_scn"] = np.clip(X_sens["FeasibilityScore_scn"] + 0.10, 0, 1)
m0 = np.clip(pipe.predict(X), LOWER_MULT, UPPER_MULT)
m1 = np.clip(pipe.predict(X_sens), LOWER_MULT, UPPER_MULT)
p0 = tcp * m0
p1 = tcp * m1
print("\n=== Sensitivity check ===")
print(f"Avg Δmultiplier when Feasibility +0.10: {float(np.mean(m1 - m0)):.4f}")
print(f"Avg Δprice      when Feasibility +0.10: {float(np.mean(p1 - p0)):.0f} PHP")
