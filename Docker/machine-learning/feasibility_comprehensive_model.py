# ==============================================================
# Comprehensive Feasibility Model, portable artifacts
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

def create_comprehensive_labels(df):
    df = df.copy()

    # 1) Affordability via IPR
    affordability_score = pd.cut(
        df['IPR_20yr'],
        bins=[-np.inf, 0.8, 1.0, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # 2) Risk component, average of *_Level_Num
    risk_columns = [c for c in df.columns if c.startswith('RISK_') and c.endswith('_Level_Num')]
    avg_risk = df[risk_columns].mean(axis=1)
    risk_score = pd.cut(
        avg_risk,
        bins=[-np.inf, 1.0, 1.5, np.inf],
        labels=[2, 1, 0]  # higher risk, lower score
    ).astype(int)

    # 3) Market
    demand_score = pd.cut(
        df.get('DemandScore', pd.Series(0.5, index=df.index)),
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # 4) Economy
    economy_score = pd.cut(
        df.get('EconomyScore', pd.Series(0.5, index=df.index)),
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

    df['AffordabilityScore'] = affordability_score
    df['RiskScore'] = risk_score
    df['MarketScore'] = demand_score
    df['EconomicScore'] = economy_score
    df['WeightedFeasibilityScore'] = weighted_score
    df['ComprehensiveFeasibilityClass'] = feasibility_class

    return df

def load_and_prepare_data():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(
        project_root, "downloads", "cp2_import_etl", "data", "curated", "with scores", "app-ready",
        "fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv"
    )
    df = pd.read_csv(data_path)

    df = create_comprehensive_labels(df)

    # numeric features
    economic_features = ['INC_income_per_hh_2024','GRDP_grdp_pc_2024_const','EMP_w','EconomyScore']
    demand_features   = ['DEM_households_single_duplex_2024','DEM_units_single_duplex_2024','DemandScore']
    risk_features     = ['RISK_Flood_Level_Num','RISK_StormSurge_Level_Num','RISK_Landslide_Level_Num']
    payment_features  = ['TCP_Model','MonthlyPayment_Model']

    numeric_features = [f for f in (economic_features + demand_features + risk_features + payment_features)
                        if f in df.columns]

    # simple categorical set
    categorical_features = [c for c in ['Region','Province','Scenario','UnitModel'] if c in df.columns]

    print("\nFeature Groups:")
    print("Economic:", [f for f in economic_features if f in numeric_features])
    print("Demand:",   [f for f in demand_features   if f in numeric_features])
    print("Risk:",     [f for f in risk_features     if f in numeric_features])
    print("Payment:",  [f for f in payment_features  if f in numeric_features])
    print("Categorical:", categorical_features)

    return df, numeric_features, categorical_features

def train_and_evaluate_model(df, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features].copy()
    y = df['ComprehensiveFeasibilityClass'].copy()

    # minimal preprocessing for training
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")
            X[col] = pd.Categorical(X[col]).codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    print("\nClassification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, digits=3))

    # Save portable artifacts to models/
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_joblib = out_dir / "comprehensive_feasibility_model.joblib"
    model_pkl    = out_dir / "comprehensive_feasibility_model.pkl"
    feat_info    = out_dir / "feasibility_feature_info.joblib"

    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_order': X.columns.tolist()
    }

    joblib.dump(clf, model_joblib, compress=3)
    with open(model_pkl, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    joblib.dump(feature_info, feat_info, compress=3)

    print(f"\n✅ Saved feasibility model → {model_joblib.name} and {model_pkl.name}")
    print(f"✅ Saved feature info      → {feat_info.name}")

    return clf

if __name__ == "__main__":
    print("Loading data and building labels...")
    df, num_feats, cat_feats = load_and_prepare_data()
    print("Training model...")
    _ = train_and_evaluate_model(df, num_feats, cat_feats)
