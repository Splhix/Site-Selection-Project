import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance

# === LOAD DATA ===
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")

# === DEFINE TARGET (rule-based from IPR) ===
def label_feasibility(ipr, t_border=0.8, t_aff=1.0):
    if ipr >= t_aff:
        return 2  # Affordable
    elif ipr >= t_border:
        return 1  # Borderline
    else:
        return 0  # Hard to Afford

def load_and_prepare_data():
    """Load and prepare the dataset with all available features"""
    df = pd.read_csv(data_path)
    df = df.copy()
    
    if "FeasibilityClass" not in df.columns:
        df["FeasibilityClass"] = df["IPR_20yr"].apply(label_feasibility)
    
    # Define feature groups
    excluded_cols = {"IPR_20yr", "FeasibilityScore_scn", "FinalCityScore_scn", "FeasibilityClass"}
    
    # All numeric features (including those that might be missing in production)
    numeric_features = [
        "INC_income_per_hh_2024",
        "GRDP_grdp_pc_2024_const",
        "DEM_households_single_duplex_2024",
        "DEM_units_single_duplex_2024",
        "EconomyScore",
        "DemandScore",
        "RISK_Flood_Level_Num",
        "RISK_StormSurge_Level_Num",
        "RISK_Landslide_Level_Num",
        "TCP_Model",
        "MonthlyPayment_Model"
    ]
    
    # Categorical features
    categorical_features = ["Region", "Province", "Scenario", "UnitModel"]
    
    # Filter features that exist in the dataset
    numeric_features = [c for c in numeric_features if c in df.columns and c not in excluded_cols]
    categorical_features = [c for c in categorical_features if c in df.columns and c not in excluded_cols]
    
    return df, numeric_features, categorical_features

def create_pipeline(numeric_features, categorical_features):
    """Create a pipeline with proper imputation and preprocessing"""
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    return pipeline

def evaluate_model(pipeline, X, y, feature_names):
    """Evaluate model with cross-validation and feature importance"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate feature importance
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Get feature names after preprocessing
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_features = numeric_features + list(cat_features)
    
    # Calculate permutation importance
    X_test_transformed = preprocessor.transform(X_test)
    perm_importance = permutation_importance(classifier, X_test_transformed, y_test, n_repeats=10, random_state=42)
    
    # Sort and print feature importances
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print("-" * 50)
    print(feature_importance.head(10))
    
    return pipeline, feature_importance

def save_model(pipeline, feature_names, output_dir):
    """Save the model and feature names"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'feasibility_model.joblib')
    feature_path = os.path.join(output_dir, 'feature_names.joblib')
    
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_names, feature_path)
    print(f"\nModel saved to {model_path}")
    print(f"Feature names saved to {feature_path}")

def predict_with_missing_data(pipeline, data, numeric_features, categorical_features):
    """Make predictions on new data that might have missing values"""
    # Ensure all expected columns exist (add missing ones)
    for col in numeric_features + categorical_features:
        if col not in data.columns:
            data[col] = np.nan
    
    # Make prediction using pipeline (which includes imputation)
    predictions = pipeline.predict(data)
    probabilities = pipeline.predict_proba(data)
    
    return predictions, probabilities

if __name__ == "__main__":
    print("Loading and preparing data...")
    df, numeric_features, categorical_features = load_and_prepare_data()
    
    print("\nFeatures being used:")
    print("Numeric:", numeric_features)
    print("Categorical:", categorical_features)
    
    # Prepare X and y
    X = df[numeric_features + categorical_features]
    y = df['FeasibilityClass']
    
    # Create and evaluate pipeline
    pipeline = create_pipeline(numeric_features, categorical_features)
    pipeline, feature_importance = evaluate_model(pipeline, X, y, numeric_features + categorical_features)
    
    # Save model
    output_dir = os.path.join(project_root, "machine-learning", "models")
    save_model(pipeline, numeric_features + categorical_features, output_dir)
    
    print("\nExample usage for new data with missing values:")
    print("-" * 50)
    print("To use the model:")
    print("1. Load the saved model:")
    print("   pipeline = joblib.load('models/feasibility_model.joblib')")
    print("   feature_names = joblib.load('models/feature_names.joblib')")
    print("2. Make predictions:")
    print("   predictions, probabilities = predict_with_missing_data(pipeline, new_data, numeric_features, categorical_features)")