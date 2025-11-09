import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_data_and_model():
    """Load the data and trained model"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "curated", "with scores", "app-ready", "fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv")
    model_dir = os.path.join(project_root, "machine-learning", "models")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Load model and features
    pipeline = joblib.load(os.path.join(model_dir, 'feasibility_model.joblib'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    
    return df, pipeline, feature_names

def analyze_risk_impact():
    """Analyze how risk factors impact feasibility predictions"""
    df, pipeline, feature_names = load_data_and_model()
    
    # Define risk levels
    risk_columns = [col for col in df.columns if col.startswith('RISK_') and col.endswith('_Level_Num')]
    
    # Calculate average risk level
    df['avg_risk_level'] = df[risk_columns].mean(axis=1)
    
    # Create risk categories based on average risk level
    df['risk_category'] = pd.cut(
        df['avg_risk_level'],
        bins=[0, 1.0, 1.5, float('inf')],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    
    # Get predictions
    X = df[feature_names]
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    
    # Add predictions to dataframe
    df['predicted_feasibility'] = predictions
    df['prediction_confidence'] = np.max(probabilities, axis=1)
    
    # Analysis by risk category
    print("\nFeasibility Analysis by Risk Level:")
    print("=" * 50)
    
    for risk_level in ['Low Risk', 'Medium Risk', 'High Risk']:
        mask = df['risk_category'] == risk_level
        subset = df[mask]
        
        print(f"\n{risk_level} Areas:")
        print("-" * 30)
        print("Distribution of Feasibility Predictions:")
        print(subset['predicted_feasibility'].value_counts(normalize=True).round(3))
        print("\nAverage Prediction Confidence:", subset['prediction_confidence'].mean().round(3))
        print("\nClassification Report:")
        if 'FeasibilityClass' in subset.columns:
            print(classification_report(subset['FeasibilityClass'], subset['predicted_feasibility']))
    
    # Analyze feature importance for different risk levels
    print("\nRisk Factor Analysis:")
    print("=" * 50)
    
    # Get preprocessed feature matrix
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Get feature names after preprocessing
    numeric_feature_names = [f for f in feature_names if f in df.columns and not pd.api.types.is_categorical_dtype(df[f])]
    categorical_features = [f for f in feature_names if f in df.columns and pd.api.types.is_categorical_dtype(df[f])]
    
    if categorical_features:
        cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    else:
        cat_feature_names = []
    
    all_feature_names = numeric_feature_names + list(cat_feature_names)
    
    # Calculate feature importances
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Filter for risk-related features
    risk_features = feature_importance[feature_importance['feature'].str.contains('RISK', case=False, na=False)]
    print("\nImportance of Risk Features:")
    print(risk_features)
    
    # Visualize risk feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(risk_features)), risk_features['importance'])
    plt.xticks(range(len(risk_features)), risk_features['feature'], rotation=45, ha='right')
    plt.title('Importance of Risk Features in Feasibility Prediction')
    plt.xlabel('Risk Features')
    plt.ylabel('Feature Importance')
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'risk_feature_importance.png'))
    plt.close()
    
    # Analyze interaction between risk and other factors
    print("\nCorrelation between Risk and Other Factors:")
    correlation_cols = risk_columns + ['INC_income_per_hh_2024', 'TCP_Model', 'MonthlyPayment_Model']
    correlation_cols = [col for col in correlation_cols if col in df.columns]
    
    correlations = df[correlation_cols].corr()
    print("\nCorrelations with Risk Factors:")
    print(correlations['avg_risk_level'].sort_values(ascending=False))
    
    # Visualize feasibility distribution by risk category
    plt.figure(figsize=(12, 6))
    
    # Calculate proportions
    risk_feasibility_dist = df.groupby('risk_category')['predicted_feasibility'].value_counts(normalize=True).unstack()
    
    # Create stacked bar chart
    risk_feasibility_dist.plot(kind='bar', stacked=True)
    plt.title('Distribution of Feasibility Classes by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Proportion')
    plt.legend(title='Feasibility Class', labels=['Hard to Afford', 'Borderline', 'Affordable'])
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, 'feasibility_by_risk_level.png'))
    plt.close()
    
    # Create scatter plot of risk level vs feasibility confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(df['avg_risk_level'], df['prediction_confidence'], alpha=0.5)
    plt.title('Risk Level vs Prediction Confidence')
    plt.xlabel('Average Risk Level')
    plt.ylabel('Model Confidence')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, 'risk_vs_confidence.png'))
    plt.close()
    
    # Additional insights
    print("\nKey Insights:")
    print("=" * 50)
    
    # Analyze high-risk but feasible areas
    high_risk_feasible = df[
        (df['risk_category'] == 'High Risk') & 
        (df['predicted_feasibility'] == 2)
    ]
    
    print("\nCharacteristics of High-Risk but Feasible Areas:")
    if not high_risk_feasible.empty:
        for col in ['INC_income_per_hh_2024', 'TCP_Model', 'MonthlyPayment_Model']:
            if col in high_risk_feasible.columns:
                print(f"\nAverage {col}:")
                print(f"- High Risk & Feasible: {high_risk_feasible[col].mean():.2f}")
                print(f"- Overall Average: {df[col].mean():.2f}")

if __name__ == "__main__":
    print("Analyzing Risk Impact on Feasibility Predictions...")
    analyze_risk_impact()