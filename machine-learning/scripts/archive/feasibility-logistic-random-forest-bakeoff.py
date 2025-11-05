# ==============================================================
# CP2 - Feasibility Classifier Comparison
# ==============================================================
# Predicts feasibility classes (0=Hard to Afford, 1=Borderline, 2=Affordable)
# using logistic regression (interpretable) and random forest (stronger learner)
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import cross_val_score
from sklearn.utils import compute_sample_weight

# ======================================================================
# Defensible ML experiment: reproduce rule-based feasibility label when
# payment/TCP/IPR are unavailable (simulate missing payment info).
# Steps:
#  - Load final fact table
#  - Define rule-based target from IPR
#  - Build feature set that EXCLUDES TCP_Model, MonthlyPayment_Model, IPR_20yr,
#    and derived final scores to avoid leakage
#  - Train classifiers with 5-fold stratified CV (OOF probabilities)
#  - Evaluate accuracy, macro-F1, log-loss, and print confusion matrix
#  - Compute permutation importances on held-out test fold
# ======================================================================

# === LOAD DATA ===
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
df = pd.read_csv(data_path)

# === DEFINE TARGET (rule-based from IPR) ===
def label_feasibility(ipr, t_border=0.8, t_aff=1.0):
    if ipr >= t_aff:
        return 2
    elif ipr >= t_border:
        return 1
    else:
        return 0

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

# Function to get features for a scenario
def get_features_for_scenario(scenario):
    all_cols = set(df.columns.tolist())
    excluded = always_excluded.union(set(payment_features["full"]) - set(payment_features[scenario]))
    
    # Filter numeric candidates that exist and aren't excluded
    numeric_features = [c for c in base_numeric + payment_features[scenario] 
                       if c in all_cols and c not in excluded]
    
    print(f"\nScenario '{scenario}' uses features:", numeric_features)
    return numeric_features

# Categorical features (available in all scenarios)
categorical_features = ["Region", "Province", "Scenario", "UnitModel"]

# === SELECT FEATURES FOR EACH SCENARIO ===
scenarios = ["full", "no_tcp", "no_payment", "minimal"]
print("\nFeatures available for each scenario:")
for scenario in scenarios:
    print(f"\n{scenario}:")
    _ = get_features_for_scenario(scenario)

numeric_features = get_features_for_scenario("minimal")  # Start with minimal case

# Filter numeric candidates that actually exist in the table
numeric_features = [c for c in numeric_candidates if c in all_cols and c not in excluded]

# Categorical features to include
categorical_candidates = ["Region", "Province", "Scenario", "UnitModel"]
categorical_features = [c for c in categorical_candidates if c in all_cols and c not in excluded]

print("Using numeric features:", numeric_features)
print("Using categorical features:", categorical_features)

# Print target distribution once
y = df["FeasibilityClass"].astype(int)
print("\nTarget distribution:\n", y.value_counts(normalize=False).sort_index())

# Utility: get feature names after preprocessing
def get_feature_names(preprocessor):
    names = []
    # numeric
    if hasattr(preprocessor, "transformers_"):
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                # OneHotEncoder inside
                ohe = trans
                try:
                    ohe_cols = list(ohe.get_feature_names_out(cols))
                except Exception:
                    ohe_cols = []
                names.extend(ohe_cols)
    return names

# Models to evaluate
rf = RandomForestClassifier(n_estimators=200, random_state=42)
log = LogisticRegression(max_iter=2000)

def run_cv_and_eval(model, X_train, y_train, X_test, y_test, preprocessor, model_name="model"):
    # Fit pipeline on training data
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])

    # 5-fold stratified CV - get OOF predicted probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")
    # For multi-class, cross_val_predict with predict_proba returns shape (n_samples, n_classes)

    # Convert OOF probs to predicted classes
    oof_pred = np.argmax(oof_proba, axis=1)

    # Metrics on OOF (train CV)
    oof_acc = accuracy_score(y_train, oof_pred)
    oof_f1 = f1_score(y_train, oof_pred, average="macro")
    try:
        oof_logloss = log_loss(y_train, oof_proba)
    except Exception:
        oof_logloss = np.nan

    print(f"\\n[{model_name}] CV OOF results: accuracy={oof_acc:.3f}, macro-F1={oof_f1:.3f}, log-loss={oof_logloss:.3f}")

    # Fit final pipeline on full training set and evaluate on held-out test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_logloss = log_loss(y_test, y_proba)

    print(f"[{model_name}] Test results: accuracy={test_acc:.3f}, macro-F1={test_f1:.3f}, log-loss={test_logloss:.3f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Permutation importance on test set (use fitted model)
    # Need to extract feature matrix after preprocessing
    X_test_trans = pipe.named_steps["preprocessor"].transform(X_test)
    feature_names = get_feature_names(pipe.named_steps["preprocessor"])

    try:
        r = permutation_importance(pipe.named_steps["clf"], X_test_trans, y_test, n_repeats=20, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feature_names, "importance_mean": r.importances_mean})
        imp_df = imp_df.sort_values(by="importance_mean", ascending=False)
        print(f"\\nTop permutation importances for {model_name}:\n", imp_df.head(15).to_string(index=False))
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
results = {}

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
    print(f"\nTraining Random Forest for {scenario} scenario")
    rf_pipe, rf_metrics = run_cv_and_eval(
        rf, X_train, y_train, X_test, y_test, preprocessor,
        model_name=f"RandomForest_{scenario}"
    )
    
    # Train and evaluate Logistic Regression
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
    print(f"Features used: {results[scenario]['features']}")
    print(f"RandomForest - Test: accuracy={results[scenario]['rf']['test_acc']:.3f}, macro-F1={results[scenario]['rf']['test_f1']:.3f}")
    print(f"LogisticReg - Test: accuracy={results[scenario]['log']['test_acc']:.3f}, macro-F1={results[scenario]['log']['test_f1']:.3f}")

print("\\nIf these models achieve high fidelity to the rule-based label (e.g., macro-F1 >= 0.9),\nwe can justify ML as a useful surrogate when payment/TCP data are unavailable.\nNext steps: calibration plots, SHAP explanations, saving models, and scenario holdout tests.")
# End of experiment script
