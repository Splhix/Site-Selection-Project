# ==============================================================
# Comprehensive Feasibility Model — updated with comparison CSV outputs
# ==============================================================

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# -------------------------------------------------------------------
# NEW: Define report directory for outputs
# -------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORT_DIR = PROJECT_ROOT / "machine-learning" / "reports" / "feasibility"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def create_comprehensive_labels(df):
    df = df.copy()

    affordability_score = pd.cut(
        df['IPR_20yr'],
        bins=[-np.inf, 0.8, 1.0, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    risk_columns = [c for c in df.columns if c.startswith('RISK_') and c.endswith('_Level_Num')]
    avg_risk = df[risk_columns].mean(axis=1)
    risk_score = pd.cut(
        avg_risk,
        bins=[-np.inf, 1.0, 1.5, np.inf],
        labels=[2, 1, 0]
    ).astype(int)

    demand_score = pd.cut(
        df['DemandScore'],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    economy_score = pd.cut(
        df['EconomyScore'],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    weights = {'affordability': 0.4, 'risk': 0.3, 'demand': 0.15, 'economy': 0.15}

    weighted_score = (
        affordability_score * weights['affordability']
        + risk_score * weights['risk']
        + demand_score * weights['demand']
        + economy_score * weights['economy']
    )

    feasibility_class = pd.cut(
        weighted_score,
        bins=[-np.inf, 0.7, 1.3, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    df["ComprehensiveFeasibilityClass"] = feasibility_class
    return df


def load_and_prepare_data():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(
        project_root, "data", "curated", "with scores", "app-ready",
        "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv"
    )

    df = pd.read_csv(data_path)

    df = create_comprehensive_labels(df)

    econ = ['INC_income_per_hh_2024','GRDP_grdp_pc_2024_const','EMP_w','EconomyScore']
    dem =  ['DEM_households_single_duplex_2024','DEM_units_single_duplex_2024','DemandScore']
    risk = ['RISK_Flood_Level_Num','RISK_StormSurge_Level_Num','RISK_Landslide_Level_Num']
    pay  = ['TCP_Model','MonthlyPayment_Model']

    numeric_features = [c for c in (econ + dem + risk + pay) if c in df.columns]

    categorical_features = [c for c in ['Region','Province','Scenario','UnitModel'] if c in df.columns]

    return df, numeric_features, categorical_features


def train_and_evaluate_model(df, numeric_features, categorical_features):

    X = df[numeric_features + categorical_features].copy()
    y = df["ComprehensiveFeasibilityClass"].copy()

    # PREPROCESS
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")
            X[col] = pd.Categorical(X[col]).codes

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # -------------------------------------------------------------------
    # NEW: Generate comparison CSVs
    # -------------------------------------------------------------------
    comp = df.loc[idx_test].copy().reset_index(drop=True)
    comp["Feasibility_true"] = y_test.reset_index(drop=True)
    comp["Feasibility_pred"] = y_pred
    comp["correct"] = comp["Feasibility_true"] == comp["Feasibility_pred"]

    # Add probabilities
    proba_df = pd.DataFrame(
        y_proba,
        columns=[f"prob_class_{c}" for c in clf.classes_]
    )
    comp = pd.concat([comp, proba_df], axis=1)

    # Save sample and full test outputs
    sample_path = REPORT_DIR / "feasibility_sample_comparison.csv"
    full_path   = REPORT_DIR / "feasibility_full_test_comparison.csv"

    comp.head(200).to_csv(sample_path, index=False)
    comp.to_csv(full_path, index=False)

    print(f"✔ Saved feasibility comparison CSVs to:\n{sample_path}\n{full_path}")

    # -------------------------------------------------------------------
    # Save model artifacts
    # -------------------------------------------------------------------
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, out_dir / "comprehensive_feasibility_model.joblib", compress=3)
    with open(out_dir / "comprehensive_feasibility_model.pkl", "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

    return clf


if __name__ == "__main__":
    print("Loading feasibility data...")
    df, num_feats, cat_feats = load_and_prepare_data()
    print("Training feasibility model...")
    _ = train_and_evaluate_model(df, num_feats, cat_feats)
