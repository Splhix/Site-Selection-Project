import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the original data to create test scenarios
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
df = pd.read_csv(data_path)

# Load the trained model and feature names
model_dir = os.path.join(project_root, "machine-learning", "models")
pipeline = joblib.load(os.path.join(model_dir, 'feasibility_model.joblib'))
feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))

def create_test_scenario(df, scenario_name, columns_to_drop=None, mask_ratio=None):
    """Create a test scenario by dropping columns or masking values"""
    test_df = df.copy()
    
    if "FeasibilityClass" not in test_df.columns:
        test_df["FeasibilityClass"] = test_df["IPR_20yr"].apply(
            lambda x: 2 if x >= 1.0 else (1 if x >= 0.8 else 0)
        )
    
    # Store true labels
    y_true = test_df["FeasibilityClass"].copy()
    
    # Drop specified columns
    if columns_to_drop:
        for col in columns_to_drop:
            if col in test_df.columns:
                test_df[col] = np.nan
    
    # Randomly mask values if specified
    if mask_ratio:
        for col in test_df.columns:
            if col not in ["FeasibilityClass", "IPR_20yr"]:
                mask = np.random.random(len(test_df)) < mask_ratio
                test_df.loc[mask, col] = np.nan
    
    print(f"\nScenario: {scenario_name}")
    print("-" * 50)
    if columns_to_drop:
        print(f"Missing columns: {columns_to_drop}")
    if mask_ratio:
        print(f"Random missing ratio: {mask_ratio}")
    
    # Get features for prediction
    X = test_df[feature_names]
    
    # Make predictions
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Print confidence statistics
    confidence = np.max(y_prob, axis=1)
    print("\nPrediction Confidence:")
    print(f"Mean confidence: {confidence.mean():.3f}")
    print(f"Min confidence: {confidence.min():.3f}")
    print(f"Max confidence: {confidence.max():.3f}")
    
    return y_pred, y_prob, confidence

# Test Scenario 1: Missing single important feature
print("\nTesting missing single important feature...")
create_test_scenario(
    df,
    "Missing Monthly Payment",
    columns_to_drop=["MonthlyPayment_Model"]
)

# Test Scenario 2: Missing multiple features
print("\nTesting missing multiple features...")
create_test_scenario(
    df,
    "Missing Economic Indicators",
    columns_to_drop=["GRDP_grdp_pc_2024_const", "EconomyScore"]
)

# Test Scenario 3: Missing all risk scores
print("\nTesting missing risk category...")
create_test_scenario(
    df,
    "Missing Risk Scores",
    columns_to_drop=[
        "RISK_Flood_Level_Num",
        "RISK_StormSurge_Level_Num",
        "RISK_Landslide_Level_Num"
    ]
)

# Test Scenario 4: New city with minimal data
print("\nTesting new city with minimal data...")
minimal_scenario = create_test_scenario(
    df,
    "New City (Minimal Data)",
    columns_to_drop=[
        "MonthlyPayment_Model",
        "TCP_Model",
        "RISK_Flood_Level_Num",
        "RISK_StormSurge_Level_Num",
        "RISK_Landslide_Level_Num"
    ]
)

# Test Scenario 5: Random missing values
print("\nTesting random missing values...")
random_scenario = create_test_scenario(
    df,
    "Random Missing Values",
    mask_ratio=0.3
)

# Create example of new city with minimal data
print("\nExample prediction for a new city:")
print("-" * 50)
example_city = pd.DataFrame({
    'INC_income_per_hh_2024': [50000],  # Only income known
    'Region': ['REGION IV-A'],
    'Province': ['CAVITE'],
    'Scenario': ['BASE'],
    'UnitModel': ['MARKET_MEDIAN']
})

# Add missing columns
for col in feature_names:
    if col not in example_city.columns:
        example_city[col] = np.nan

# Make prediction
pred = pipeline.predict(example_city)
prob = pipeline.predict_proba(example_city)

print("\nPrediction:", ["Hard to Afford", "Borderline", "Affordable"][pred[0]])
print("Probabilities:", {i: f"{p:.3f}" for i, p in enumerate(prob[0])})