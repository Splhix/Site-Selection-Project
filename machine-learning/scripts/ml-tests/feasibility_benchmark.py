"""
CP2 – Feasibility ML Benchmark Script
Location: scripts/ml_tests/feasibility_benchmark.py

Compares multiple classifiers for the comprehensive feasibility label and
saves metrics and plots for documentation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Optional models (skipped gracefully if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
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

REPORT_DIR = PROJECT_ROOT / "machine-learning" / "reports" / "feasibility"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LABEL LOGIC (same as prod)
# -----------------------------
def create_comprehensive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Affordability via IPR
    affordability_score = pd.cut(
        df["IPR_20yr"],
        bins=[-np.inf, 0.8, 1.0, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    # 2) Risk component, average of *_Level_Num
    risk_columns = [
        c for c in df.columns if c.startswith("RISK_") and c.endswith("_Level_Num")
    ]
    avg_risk = df[risk_columns].mean(axis=1)
    risk_score = pd.cut(
        avg_risk,
        bins=[-np.inf, 1.0, 1.5, np.inf],
        labels=[2, 1, 0],  # higher risk, lower score
    ).astype(int)

    # 3) Market
    demand_score = pd.cut(
        df.get("DemandScore", pd.Series(0.5, index=df.index)),
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    # 4) Economy
    economy_score = pd.cut(
        df.get("EconomyScore", pd.Series(0.5, index=df.index)),
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    weights = {
        "affordability": 0.4,
        "risk": 0.3,
        "demand": 0.15,
        "economy": 0.15,
    }
    weighted_score = (
        affordability_score * weights["affordability"]
        + risk_score * weights["risk"]
        + demand_score * weights["demand"]
        + economy_score * weights["economy"]
    )

    feasibility_class = pd.cut(
        weighted_score,
        bins=[-np.inf, 0.7, 1.3, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    df["ComprehensiveFeasibilityClass"] = feasibility_class
    return df


# -----------------------------
# DATA PREPARATION
# -----------------------------
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Build labels
    df = create_comprehensive_labels(df)

    # Numeric feature sets (same spirit as prod feasibility model)
    economic_features = [
        "INC_income_per_hh_2024",
        "GRDP_grdp_pc_2024_const",
        "EMP_w",
        "EconomyScore",
    ]
    demand_features = [
        "DEM_households_single_duplex_2024",
        "DEM_units_single_duplex_2024",
        "DemandScore",
    ]
    risk_features = [
        "RISK_Flood_Level_Num",
        "RISK_StormSurge_Level_Num",
        "RISK_Landslide_Level_Num",
    ]
    payment_features = [
        "TCP_Model",
        "MonthlyPayment_Model",
    ]

    numeric_features = [
        f
        for f in (
            economic_features
            + demand_features
            + risk_features
            + payment_features
        )
        if f in df.columns
    ]

    categorical_features = [
        c
        for c in ["Region", "Province", "Scenario", "UnitModel"]
        if c in df.columns
    ]

    # Build X, y
    X = df[numeric_features + categorical_features].copy()
    y = df["ComprehensiveFeasibilityClass"].copy()

    # Basic preprocessing: fill and encode
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")
            X[col] = pd.Categorical(X[col]).codes

    return X, y


# -----------------------------
# MODEL BENCHMARKING
# -----------------------------
def benchmark_models():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            multi_class="auto",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42
        ),
        "SVC": SVC(
            probability=True,
            random_state=42,
        ),
    }

    if HAS_XGB:
        models["XGBClassifier"] = XGBClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        print("⚠️ xgboost not installed, skipping XGBClassifier.")

    if HAS_LGBM:
        models["LGBMClassifier"] = LGBMClassifier(
            random_state=42,
        )
    else:
        print("⚠️ lightgbm not installed, skipping LGBMClassifier.")

    rows = []
    best_model_name = None
    best_macro_f1 = -np.inf
    best_y_pred = None

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision_macro": prec_macro,
                "Recall_macro": rec_macro,
                "F1_macro": f1_macro,
                "F1_weighted": f1_weighted,
            }
        )

        if f1_macro > best_macro_f1:
            best_macro_f1 = f1_macro
            best_model_name = name
            best_y_pred = y_pred

    # Save comparison table
    results_df = pd.DataFrame(rows).sort_values(
        by="F1_macro", ascending=False
    )
    csv_path = REPORT_DIR / "feasibility_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved model comparison table → {csv_path}")

    # Confusion matrix for best model
    if best_model_name is not None and best_y_pred is not None:
        print(f"\nBest model based on F1_macro: {best_model_name}")
        cm = confusion_matrix(y_test, best_y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm)
        ax.set_title(f"Confusion Matrix – {best_model_name}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()
        cm_path = REPORT_DIR / "feasibility_best_model_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved confusion matrix plot → {cm_path}")

        # Classification report
        report = classification_report(y_test, best_y_pred, digits=3)
        report_path = (
            REPORT_DIR / "feasibility_best_model_classification_report.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Best model: {best_model_name}\n\n")
            f.write(report)
        print(f"✅ Saved classification report → {report_path}")


if __name__ == "__main__":
    print("Running feasibility ML benchmarking...")
    benchmark_models()
    print("\nDone.")
