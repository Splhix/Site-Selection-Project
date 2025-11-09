# Machine Learning Models

This directory contains the trained models for the Site Selection Project.

## Active Models

### Feasibility Model
- `comprehensive_feasibility_model.pkl`
  - Current production model
  - Integrated approach including risk, economic, and market factors
  - Trained using `feasibility_comprehensive_model.py`
  - Features:
    - Comprehensive feasibility scoring
    - Pure classifier without preprocessing
    - Multi-factor decision making

- `feasibility_feature_info.pkl`
  - Feature metadata for ETL pipeline
  - Contains:
    - List of numeric features
    - List of categorical features
    - Expected feature order

### Profitability Models
- `profitability_randomforest_model.pkl`
  - Random Forest model for profitability prediction
  - Full feature set version

- `profitability_linear_model.pkl`
  - Linear regression model for profitability
  - Used for baseline comparisons

- `profitability_proxy_randomforest_model.pkl`
  - Proxy model for limited data scenarios
  - Random Forest implementation

- `profitability_proxy_linear_model.pkl`
  - Linear version of proxy model
  - Used for simpler scenarios

## Model Usage

The comprehensive feasibility model can be loaded and used as follows:

```python
import pickle

# Load model and feature info
with open('comprehensive_feasibility_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feasibility_feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

# Preprocess data according to feature_info (handled in ETL)
# ...

# Make predictions
predictions = model.predict(processed_data)
probabilities = model.predict_proba(processed_data)
```

## Archive

Old models are stored in the `archive/` directory for reference:
- Previous feasibility models
- Separate risk models (now integrated into comprehensive model)
- Historical feature configurations