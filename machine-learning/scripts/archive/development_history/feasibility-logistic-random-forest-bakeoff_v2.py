# ==============================================================
# CP2 - Feasibility Classifier Comparison (Multiple Scenarios)
# ==============================================================
# Trains multiple models to handle different missing data scenarios:
# 1. Full model (all features available)
# 2. No TCP model (when TCP is missing)
# 3. No Payment model (when monthly payment is missing)
# 4. Minimal model (when all payment/price info is missing)
# ==============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from sklearn.inspection import permutation_importance

# === LOAD DATA ===
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
df = pd.read_csv(data_path)

# === DEFINE TARGET (rule-based from IPR) ===
def label_feasibility(ipr, t_border=0.8, t_aff=1.0):
    if ipr >= t_aff:
        return 2  # Affordable
    elif ipr >= t_border:
        return 1  # Borderline
    else:
        return 0  # Hard to afford

df = df.copy()
if "FeasibilityClass" not in df.columns:
    df["FeasibilityClass"] = df["IPR_20yr"].apply(label_feasibility)

# === DEFINE FEATURE GROUPS FOR DIFFERENT MISSING DATA SCENARIOS ===
# Always excluded (derived/leaky)
always_excluded = {"IPR_20yr", "FeasibilityScore_scn", "FinalCityScore_scn", "FeasibilityClass"}

# Base features (always available)
base_numeric = [
    "INC_income_per_hh_2024",
    "GRDP_grdp_pc_2024_const",
    "DEM_households_single_duplex_2024",
    "DEM_units_single_duplex_2024",
    "EconomyScore",
    "DemandScore",
    "RISK_Flood_Level_Num",
    "RISK_StormSurge_Level_Num",
    "RISK_Landslide_Level_Num",
]

# Payment/price features that might be missing
payment_features = {
    "full": ["TCP_Model", "MonthlyPayment_Model"],  # All available
    "no_tcp": ["MonthlyPayment_Model"],  # TCP missing
    "no_payment": ["TCP_Model"],  # Monthly payment missing
    "minimal": []  # No payment info (current case)
}

# Categorical features (available in all scenarios)
categorical_features = ["Region", "Province", "Scenario", "UnitModel"]

# Function to get features for a scenario
def get_features_for_scenario(scenario):
    all_cols = set(df.columns.tolist())
    excluded = always_excluded.union(set(payment_features["full"]) - set(payment_features[scenario]))
    
    # Filter numeric candidates that exist and aren't excluded
    numeric_features = [c for c in base_numeric + payment_features[scenario] 
                       if c in all_cols and c not in excluded]
    
    print(f"\nScenario '{scenario}' uses numeric features:", numeric_features)
    return numeric_features

def run_cv_and_eval(model, X_train, y_train, X_test, y_test, preprocessor, model_name="model"):
    # Fit pipeline on training data
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])

    # 5-fold stratified CV - get OOF predicted probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")
    
    # Convert OOF probs to predicted classes
    oof_pred = np.argmax(oof_proba, axis=1)

    # Metrics on OOF (train CV)
    oof_acc = accuracy_score(y_train, oof_pred)
    oof_f1 = f1_score(y_train, oof_pred, average="macro")
    try:
        oof_logloss = log_loss(y_train, oof_proba)
    except Exception:
        oof_logloss = np.nan

    print(f"\n[{model_name}] CV OOF results:")
    print(f"Accuracy={oof_acc:.3f}, macro-F1={oof_f1:.3f}, log-loss={oof_logloss:.3f}")

    # Fit final pipeline on full training set and evaluate on held-out test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_logloss = log_loss(y_test, y_proba)

    print(f"\n[{model_name}] Test results:")
    print(f"Accuracy={test_acc:.3f}, macro-F1={test_f1:.3f}, log-loss={test_logloss:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Permutation importance on test set
    X_test_trans = pipe.named_steps["preprocessor"].transform(X_test)
    try:
        r = permutation_importance(pipe.named_steps["clf"], X_test_trans, y_test, 
                                 n_repeats=20, random_state=42, n_jobs=-1)
        feature_names = (scenario_numeric + 
                        list(pipe.named_steps["preprocessor"]
                        .named_transformers_["cat"]
                        .get_feature_names_out(categorical_features)))
        
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        })
        imp_df = imp_df.sort_values(by="importance_mean", ascending=False)
        print(f"\nTop 10 features for {model_name}:")
        print(imp_df.head(10).to_string(index=False))
    except Exception as e:
        print("Permutation importance failed:", e)

    return pipe, {
        "oof_acc": oof_acc,
        "oof_f1": oof_f1,
        "oof_logloss": oof_logloss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_logloss": test_logloss,
    }

# ======================================================
# Run experiments for each missing data scenario
# ======================================================

# Print target distribution once
y = df["FeasibilityClass"].astype(int)
print("\nTarget distribution:")
print(y.value_counts(normalize=False).sort_index())

results = {}
scenarios = ["full", "no_tcp", "no_payment", "minimal"]

for scenario in scenarios:
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario}")
    print(f"{'='*50}")
    
    # Get features for this scenario
    scenario_numeric = get_features_for_scenario(scenario)
    
    # Build X, y for this scenario
    X_scenario = df[scenario_numeric + categorical_features]
    y = df["FeasibilityClass"].astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scenario, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing for this scenario
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), scenario_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    
    # Train and evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    print(f"\nTraining Random Forest for {scenario} scenario")
    rf_pipe, rf_metrics = run_cv_and_eval(
        rf, X_train, y_train, X_test, y_test, preprocessor,
        model_name=f"RandomForest_{scenario}"
    )
    
    # Train and evaluate Logistic Regression
    log = LogisticRegression(max_iter=2000)
    print(f"\nTraining Logistic Regression for {scenario} scenario")
    log_pipe, log_metrics = run_cv_and_eval(
        log, X_train, y_train, X_test, y_test, preprocessor,
        model_name=f"LogisticRegression_{scenario}"
    )
    
    # Store results
    results[scenario] = {
        "rf": rf_metrics,
        "log": log_metrics,
        "rf_pipe": rf_pipe,
        "log_pipe": log_pipe,
        "features": scenario_numeric + categorical_features
    }

# === SUMMARY ACROSS SCENARIOS ===
print("\n=== SUMMARY ACROSS SCENARIOS ===")
for scenario in scenarios:
    print(f"\nScenario: {scenario}")
    print("Features used:", len(results[scenario]['features']))
    print(f"RandomForest - Test: accuracy={results[scenario]['rf']['test_acc']:.3f}, macro-F1={results[scenario]['rf']['test_f1']:.3f}")
    print(f"LogisticReg - Test: accuracy={results[scenario]['log']['test_acc']:.3f}, macro-F1={results[scenario]['log']['test_f1']:.3f}")

print("\nThis experiment shows how well the models perform with different levels of available data.")
print("If the models achieve high fidelity even with missing features, they can be used as reliable surrogates.")