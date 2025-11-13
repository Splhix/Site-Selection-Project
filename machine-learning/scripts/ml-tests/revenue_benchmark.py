"""
CP2 – Potential Revenue ML Benchmark Script
Location: scripts/ml_tests/revenue_benchmark.py

Compares multiple regressors for the per-unit price multiplier (and implied price)
and saves metrics and plots for documentation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional models (skipped gracefully if not installed)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# -----------------------------
# PATHS / CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "curated"
    / "with scores"
    / "app-ready"
    / "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv"
)

REPORT_DIR = PROJECT_ROOT / "machine-learning" / "reports" / "revenue"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO = "BASE"
UNITMODEL = "MARKET_MEDIAN"
LOWER_MULT = 0.85
UPPER_MULT = 1.10
NOISE_SD = 0.01  # tiny noise for de-collinearity


# -----------------------------
# UTILITIES
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))


# -----------------------------
# DATA PREPARATION (aligned with prod script)
# -----------------------------
def load_and_prepare_data():
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

    feature_cols = [
        # pillar summaries
        "FeasibilityScore_scn",
        "EconomyScore",
        "DemandScore",
        "HazardSafety_NoFault",
        "ProfitabilityScore_scn",
        # fundamentals
        "GRDP_grdp_pc_2024_const",
        "INC_income_per_hh_2024",
        "EMP_w",
        "DEM_households_single_duplex_2024",
        "DEM_units_single_duplex_2024",
        # risk details
        "RISK_Fault_Distance_km",
        "RISK_Flood_Level_Num",
        "RISK_StormSurge_Level_Num",
        "RISK_Landslide_Level_Num",
        # pricing context
        "TCP_Model",
    ]
    if "TCP_MarketMedian" in df.columns:
        feature_cols.append("TCP_MarketMedian")

    # Fill missing feature columns with neutral defaults
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].astype(float).reset_index(drop=True)
    tcp = df["TCP_Model"].astype(float).reset_index(drop=True).to_numpy()

    # Pseudo-label multiplier logic (aligned with prod model)
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
    m = m * rng.normal(1.0, NOISE_SD, size=len(m))
    y_mult = np.clip(m, LOWER_MULT, UPPER_MULT)

    return X, y_mult, tcp


# -----------------------------
# MODEL BENCHMARKING
# -----------------------------
def benchmark_models():
    X, y_mult, tcp = load_and_prepare_data()

    X_train, X_test, y_train, y_test, tcp_train, tcp_test = train_test_split(
        X,
        y_mult,
        tcp,
        test_size=0.2,
        random_state=42,
    )

    models = {}

    models["LinearRegression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )

    models["RandomForestRegressor"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    models["GradientBoostingRegressor"] = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )

    models["SVR"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="rbf")),
        ]
    )

    if HAS_XGB:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        print("⚠️ xgboost not installed, skipping XGBRegressor.")

    if HAS_LGBM:
        models["LGBMRegressor"] = LGBMRegressor(
            random_state=42,
        )
    else:
        print("⚠️ lightgbm not installed, skipping LGBMRegressor.")

    rows = []
    best_model_name = None
    best_r2_price = -np.inf
    best_y_price_pred = None
    best_y_price_true = None

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred_mult = model.predict(X_test)
        y_pred_mult = np.clip(y_pred_mult, LOWER_MULT, UPPER_MULT)

        # Metrics on multiplier
        mae_m = mean_absolute_error(y_test, y_pred_mult)
        rmse_m = mean_squared_error(y_test, y_pred_mult) ** 0.5
        r2_m = r2_score(y_test, y_pred_mult)

        # Implied price metrics
        y_true_price = tcp_test * y_test
        y_pred_price = tcp_test * y_pred_mult

        mae_p = mean_absolute_error(y_true_price, y_pred_price)
        rmse_p = mean_squared_error(y_true_price, y_pred_price) ** 0.5
        r2_p = r2_score(y_true_price, y_pred_price)
        mape_p = mape(y_true_price, y_pred_price)

        rows.append(
            {
                "Model": name,
                "MAE_mult": mae_m,
                "RMSE_mult": rmse_m,
                "R2_mult": r2_m,
                "MAE_price": mae_p,
                "RMSE_price": rmse_p,
                "MAPE_price": mape_p,
                "R2_price": r2_p,
            }
        )

        if r2_p > best_r2_price:
            best_r2_price = r2_p
            best_model_name = name
            best_y_price_pred = y_pred_price
            best_y_price_true = y_true_price

    # Save comparison table
    results_df = pd.DataFrame(rows).sort_values(
        by="R2_price", ascending=False
    )
    csv_path = REPORT_DIR / "revenue_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved model comparison table → {csv_path}")

    # Plots for best model
    if best_model_name is not None and best_y_price_pred is not None:
        print(f"\nBest model based on R2_price: {best_model_name}")

        # Actual vs predicted
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(best_y_price_true, best_y_price_pred, s=10)
        min_val = min(best_y_price_true.min(), best_y_price_pred.min())
        max_val = max(best_y_price_true.max(), best_y_price_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val])
        ax1.set_xlabel("Actual Price")
        ax1.set_ylabel("Predicted Price")
        ax1.set_title(f"Actual vs Predicted Price – {best_model_name}")
        fig1.tight_layout()
        avp_path = REPORT_DIR / "revenue_best_model_actual_vs_predicted.png"
        fig1.savefig(avp_path, dpi=150)
        plt.close(fig1)
        print(f"✅ Saved actual vs predicted plot → {avp_path}")

        # Residuals histogram
        residuals = best_y_price_pred - best_y_price_true
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.hist(residuals, bins=30)
        ax2.set_xlabel("Residual (Predicted − Actual)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Residual Distribution – {best_model_name}")
        fig2.tight_layout()
        resid_path = REPORT_DIR / "revenue_best_model_residuals.png"
        fig2.savefig(resid_path, dpi=150)
        plt.close(fig2)
        print(f"✅ Saved residuals plot → {resid_path}")


if __name__ == "__main__":
    print("Running potential revenue ML benchmarking...")
    benchmark_models()
    print("\nDone.")
