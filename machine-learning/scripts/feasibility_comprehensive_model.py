# ==============================================================
# Comprehensive Feasibility Model
# ==============================================================
# This model uses a more comprehensive labeling approach that considers:
# 1. Affordability (IPR)
# 2. Risk factors (flood, earthquake, etc.)
# 3. Market conditions (demand vs supply)
# 4. Economic indicators
# ==============================================================

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

def create_comprehensive_labels(df):
    """
    Create comprehensive feasibility labels considering multiple factors:
    0: Not Feasible
    1: Potentially Feasible
    2: Highly Feasible
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # 1. Affordability Component (IPR threshold)
    affordability_score = pd.cut(
        df['IPR_20yr'],
        bins=[-np.inf, 0.8, 1.0, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    
    # 2. Risk Component
    risk_columns = [col for col in df.columns if col.startswith('RISK_') and col.endswith('_Level_Num')]
    avg_risk = df[risk_columns].mean(axis=1)
    risk_score = pd.cut(
        avg_risk,
        bins=[-np.inf, 1.0, 1.5, np.inf],
        labels=[2, 1, 0]  # Inverse scale: higher risk = lower score
    ).astype(int)
    
    # 3. Market Conditions
    if 'DemandScore' in df.columns:
        demand_score = pd.cut(
            df['DemandScore'],
            bins=[-np.inf, 0.4, 0.7, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        demand_score = pd.Series(1, index=df.index)  # Neutral if not available
    
    # 4. Economic Component
    if 'EconomyScore' in df.columns:
        economy_score = pd.cut(
            df['EconomyScore'],
            bins=[-np.inf, 0.4, 0.7, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        economy_score = pd.Series(1, index=df.index)  # Neutral if not available
    
    # Combine scores with weights
    weights = {
        'affordability': 0.4,  # Affordability still important but not dominant
        'risk': 0.3,          # Risk is a major factor
        'demand': 0.15,       # Market conditions
        'economy': 0.15       # Economic conditions
    }
    
    weighted_score = (
        affordability_score * weights['affordability'] +
        risk_score * weights['risk'] +
        demand_score * weights['demand'] +
        economy_score * weights['economy']
    )
    
    # Convert to final labels
    feasibility_class = pd.cut(
        weighted_score,
        bins=[-np.inf, 0.7, 1.3, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    
    # Add component scores for analysis
    df['AffordabilityScore'] = affordability_score
    df['RiskScore'] = risk_score
    df['MarketScore'] = demand_score
    df['EconomicScore'] = economy_score
    df['WeightedFeasibilityScore'] = weighted_score
    df['ComprehensiveFeasibilityClass'] = feasibility_class
    
    return df

def load_and_prepare_data():
    """Load and prepare the dataset with comprehensive labels"""
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
    df = pd.read_csv(data_path)
    
    # Create comprehensive labels
    df = create_comprehensive_labels(df)
    
    # Define feature groups
    excluded_cols = {
        'IPR_20yr', 'FeasibilityScore_scn', 'FinalCityScore_scn', 
        'FeasibilityClass', 'ComprehensiveFeasibilityClass',
        'AffordabilityScore', 'RiskScore', 'MarketScore', 'EconomicScore',
        'WeightedFeasibilityScore'
    }
    
    # Features by category
    economic_features = [
        'INC_income_per_hh_2024',
        'GRDP_grdp_pc_2024_const',
        'EconomyScore'
    ]
    
    demand_features = [
        'DEM_households_single_duplex_2024',
        'DEM_units_single_duplex_2024',
        'DemandScore'
    ]
    
    risk_features = [
        'RISK_Flood_Level_Num',
        'RISK_StormSurge_Level_Num',
        'RISK_Landslide_Level_Num',
        'RISK_GroundShaking_Level_Num',
        'RISK_Liquefaction_Level_Num',
        'RISK_Tsunami_Level_Num',
        'RiskScore'
    ]
    
    payment_features = [
        'TCP_Model',
        'MonthlyPayment_Model'
    ]
    
    # Combine all numeric features
    numeric_features = [f for f in (economic_features + demand_features + 
                                  risk_features + payment_features)
                       if f in df.columns and f not in excluded_cols]
    
    # Categorical features
    categorical_features = ['Region', 'Province', 'Scenario', 'UnitModel']
    categorical_features = [f for f in categorical_features 
                          if f in df.columns and f not in excluded_cols]
    
    print("\nFeature Groups:")
    print("Economic:", [f for f in economic_features if f in numeric_features])
    print("Demand:", [f for f in demand_features if f in numeric_features])
    print("Risk:", [f for f in risk_features if f in numeric_features])
    print("Payment:", [f for f in payment_features if f in numeric_features])
    print("Categorical:", categorical_features)
    
    return df, numeric_features, categorical_features

def analyze_labels(df):
    """Analyze the comprehensive labels and their components"""
    print("\nLabel Distribution Analysis:")
    print("-" * 50)
    
    # Overall distribution
    print("\nComprehensive Feasibility Class Distribution:")
    print(df['ComprehensiveFeasibilityClass'].value_counts(normalize=True).round(3))
    
    # Component analysis
    print("\nComponent Score Correlations with Final Class:")
    correlations = df[[
        'ComprehensiveFeasibilityClass',
        'AffordabilityScore',
        'RiskScore',
        'MarketScore',
        'EconomicScore'
    ]].corr()['ComprehensiveFeasibilityClass'].sort_values(ascending=False)
    print(correlations)
    
    # Cross-tabulation with original feasibility
    if 'FeasibilityClass' in df.columns:
        print("\nComparison with Original Feasibility Class:")
        print(pd.crosstab(
            df['ComprehensiveFeasibilityClass'],
            df['FeasibilityClass'],
            normalize='index'
        ).round(3))

def train_and_evaluate_model(df, numeric_features, categorical_features):
    """Train and evaluate the comprehensive feasibility model"""
    # Prepare features and target
    X = df[numeric_features + categorical_features]
    y = df['ComprehensiveFeasibilityClass']
    
    # Create the classifier (preprocessing will be handled in ETL)
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Preprocess data for training (this logic will move to ETL)
    # For now, we'll do simple preprocessing just for training
    X = X.copy()
    
    # Simple imputation for training
    for col in X.columns:
        if X[col].dtype.name in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna('missing')
    
    # Basic encoding for categorical variables
    for col in categorical_features:
        X[col] = pd.Categorical(X[col]).codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print("\nModel Evaluation:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'comprehensive_feasibility_model.pkl')
    
    # Save feature information separately for ETL reference
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_order': X.columns.tolist()
    }
    feature_info_path = os.path.join(output_dir, 'feasibility_feature_info.pkl')
    
    # Save both files
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    with open(feature_info_path, 'wb') as f:
        pickle.dump(feature_info, f)
        
    print(f"\nModel saved to {model_path}")
    print(f"Feature info saved to {feature_info_path}")
    
    return classifier

if __name__ == "__main__":
    print("Loading and preparing data with comprehensive feasibility labels...")
    df, numeric_features, categorical_features = load_and_prepare_data()
    
    print("\nAnalyzing comprehensive labels...")
    analyze_labels(df)
    
    print("\nTraining and evaluating model...")
    pipeline = train_and_evaluate_model(df, numeric_features, categorical_features)