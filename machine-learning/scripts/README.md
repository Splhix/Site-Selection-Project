# Machine Learning Scripts

This directory contains the main machine learning implementation for the Site Selection Project.

## Active Scripts

### Feasibility Models
1. `feasibility_comprehensive_model.py`
   - Main production model for site feasibility prediction
   - Uses a comprehensive approach considering multiple factors:
     - Affordability (40% weight)
     - Risk Factors (30% weight)
     - Market Demand (15% weight)
     - Economic Conditions (15% weight)
   - Implements robust missing data handling
   - Features full pipeline with preprocessing and model persistence

2. `analyze_risk_impact.py`
   - Analysis tool for understanding risk factor relationships
   - Visualizes risk impact on feasibility
   - Helps validate model decisions

### Profitability Models
3. `profitability_regression_model.py`
   - Regression model for predicting project profitability
   - Used for financial forecasting and ROI analysis

4. `profitability_proxy_model.py`
   - Proxy model for profitability estimation
   - Handles scenarios with limited financial data

## Model Performance

The comprehensive feasibility model achieves:
- 97% overall accuracy
- Balanced performance across all feasibility classes
- Robust handling of missing data
- Interpretable predictions based on multiple factors

## Usage

1. Training the model:
```bash
python feasibility_comprehensive_model.py
```

2. Analyzing risk impact:
```bash
python analyze_risk_impact.py
```

## Archive

Historical development scripts are archived in:
- `archive/development_history/`: Development iterations and experiments
- `archive/`: Previous production versions

## Model Location

Trained models are saved in:
- `../models/comprehensive_feasibility_model.joblib`