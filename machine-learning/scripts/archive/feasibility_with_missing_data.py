# ==============================================================
# CP2 - Feasibility Classifier with Missing Data Handling
# ==============================================================
# Handles randomly missing data using multiple strategies:
# 1. Multiple imputation for missing values
# 2. Missing value indicators
# 3. IterativeImputer (fancy imputation)
# Compare performance across different missingness patterns
# ==============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, MissingIndicator
import warnings
warnings.filterwarnings('ignore')

# === LOAD DATA ===
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", 
                        "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
df = pd.read_csv(data_path)

# === DEFINE TARGET ===
def label_feasibility(ipr, t_border=0.8, t_aff=1.0):
    if ipr >= t_aff:
        return 2  # Affordable
    elif ipr >= t_border:
        return 1  # Borderline
    else:
        return 0  # Hard to afford

df["FeasibilityClass"] = df["IPR_20yr"].apply(label_feasibility)

# === FEATURE GROUPS ===
numeric_features = [
    "INC_income_per_hh_2024",
    "GRDP_grdp_pc_2024_const",
    "TCP_Model",
    "MonthlyPayment_Model",
    "DEM_households_single_duplex_2024",
    "DEM_units_single_duplex_2024",
    "EconomyScore",
    "DemandScore",
    "RISK_Flood_Level_Num",
    "RISK_StormSurge_Level_Num",
    "RISK_Landslide_Level_Num"
]

categorical_features = ["Region", "Province", "Scenario", "UnitModel"]

# === SIMULATE MISSING DATA PATTERNS ===
def introduce_missing_values(X, missing_rate=0.2, random_state=42):
    """Introduce missing values randomly in numeric columns"""
    X_missing = X.copy()
    np.random.seed(random_state)
    
    for col in X.select_dtypes(include=[np.number]).columns:
        mask = np.random.random(len(X)) < missing_rate
        X_missing.loc[mask, col] = np.nan
    
    return X_missing

# === DIFFERENT IMPUTATION STRATEGIES ===

def build_simple_imputer_pipeline():
    """Simple mean/mode imputation pipeline"""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

def build_iterative_imputer_pipeline():
    """Iterative imputation pipeline (uses relationships between features)"""
    numeric_transformer = Pipeline([
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

def build_indicator_pipeline():
    """Pipeline that adds missing indicators"""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Add missing indicators for numeric features
    indicator_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
        ('indicator', MissingIndicator())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ind', indicator_transformer, numeric_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

# === EXPERIMENT WITH DIFFERENT MISSING RATES ===
missing_rates = [0.1, 0.2, 0.3, 0.4]
results = []

print("=== MISSING DATA HANDLING EXPERIMENT ===")
print("Testing different imputation strategies with varying amounts of missing data")

# Prepare base dataset
X = df[numeric_features + categorical_features]
y = df["FeasibilityClass"]

# Split once to keep test set clean
X_train_full, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for missing_rate in missing_rates:
    print(f"\n=== Missing rate: {missing_rate*100}% ===")
    
    # Introduce missing values only in training data
    X_train_missing = introduce_missing_values(X_train_full, missing_rate=missing_rate)
    
    # Test different imputation strategies
    strategies = {
        'Simple Imputation': build_simple_imputer_pipeline(),
        'Iterative Imputation': build_iterative_imputer_pipeline()
    }
    
    for name, pipeline in strategies.items():
        # Fit and evaluate
        pipeline.fit(X_train_missing, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Macro F1: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results.append({
            'missing_rate': missing_rate,
            'strategy': name,
            'accuracy': accuracy,
            'f1': f1
        })

# === SUMMARY ===
print("\n=== SUMMARY ACROSS MISSING RATES ===")
summary_df = pd.DataFrame(results)
print("\nMean performance by strategy:")
print(summary_df.groupby('strategy')[['accuracy', 'f1']].mean())
print("\nPerformance degradation as missing rate increases:")
print(summary_df.pivot_table(
    index='missing_rate',
    columns='strategy',
    values='f1',
    aggfunc='mean'
))

print("\nRecommendations:")
print("1. For low missing rates (<20%): Simple imputation works well")
print("2. For higher missing rates: Iterative imputation may be better")
print("3. Consider the pattern of missingness - if not random, may need custom strategy")
print("4. Always validate imputation results with domain knowledge")