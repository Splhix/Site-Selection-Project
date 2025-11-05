import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load your fact table
# Try multiple possible file locations
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

possible_paths = [
    "fact_table_app_READY_WITH_SCENARIOS.csv",  # Same directory as script
    project_root / "data" / "curated" / "with scores" / "app-ready" / "fact_table_app_READY_WITH_SCENARIOS.csv",
    project_root / "data" / "curated" / "with scores" / "app-ready" / "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv",
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    raise FileNotFoundError(
        f"Could not find fact table CSV file. Tried:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths)
    )

# 2. Filter to BASE, 2024 only
print(f"Total records: {len(df)}")
if "Scenario" not in df.columns:
    raise ValueError("Column 'Scenario' not found in data. Available columns: " + ", ".join(df.columns))
if "year" not in df.columns:
    raise ValueError("Column 'year' not found in data. Available columns: " + ", ".join(df.columns))

df_base = df[(df["Scenario"] == "BASE") & (df["year"] == 2024)].copy()
print(f"Records after filtering (BASE, 2024): {len(df_base)}")

if len(df_base) == 0:
    raise ValueError("No records found with Scenario='BASE' and year=2024")

# 3. Select features and target
feature_cols_num = [
    "RISK_Fault_Distance_km",
    "RISK_Flood_Level_Num",
    "RISK_StormSurge_Level_Num",
    "RISK_Landslide_Level_Num",
]
feature_cols_cat = ["Region"]

# Check that all required columns exist
missing_cols = [col for col in feature_cols_num + feature_cols_cat + ["RISK_Risk_Gate"] if col not in df_base.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df_base.columns)}")

X = df_base[feature_cols_num + feature_cols_cat]
y = df_base["RISK_Risk_Gate"]  # e.g. "PASS", "REVIEW", "NO_BUILD"

print(f"Features: {feature_cols_num + feature_cols_cat}")
print(f"Target distribution:\n{y.value_counts()}")

# 4. Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", feature_cols_num),
        ("cat", OneHotEncoder(drop="first"), feature_cols_cat),
    ]
)

# 5. Model
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", tree),
])

# 6. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy:", accuracy.mean(), "+/-", accuracy.std())

# 7. Fit on full data to extract rules
clf.fit(X, y)

# Extract the tree from pipeline
tree_trained = clf.named_steps["model"]
feature_names_out = (
    feature_cols_num +
    list(clf.named_steps["preprocess"]
         .named_transformers_["cat"]
         .get_feature_names_out(feature_cols_cat))
)

tree_text = export_text(tree_trained, feature_names=feature_names_out)
print(tree_text)

# 8. Save the model
models_dir = script_dir.parent / "models"
models_dir.mkdir(exist_ok=True)
print(f"\n📁 Saving model to: {models_dir}", flush=True)

# Create timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"risk_model_{timestamp}.pkl"
model_path = models_dir / model_filename

# Save the trained pipeline
try:
    joblib.dump(clf, model_path)
    print(f"\n✅ Model saved to: {model_path}", flush=True)
except Exception as e:
    print(f"❌ Error saving model: {e}", flush=True)
    raise

# Save metadata
metadata = {
    "model_type": "DecisionTreeClassifier",
    "timestamp": timestamp,
    "cv_accuracy_mean": float(accuracy.mean()),
    "cv_accuracy_std": float(accuracy.std()),
    "n_samples": len(df_base),
    "features_numeric": feature_cols_num,
    "features_categorical": feature_cols_cat,
    "target_distribution": y.value_counts().to_dict(),
    "tree_text": tree_text,
    "model_params": {
        "max_depth": 3,
        "min_samples_leaf": 2,
        "random_state": 42
    }
}

metadata_path = models_dir / f"risk_model_{timestamp}_metadata.json"
try:
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}", flush=True)
except Exception as e:
    print(f"❌ Error saving metadata: {e}", flush=True)
    raise
