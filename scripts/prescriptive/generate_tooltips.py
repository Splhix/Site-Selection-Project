#!/usr/bin/env python3
"""
Generate standardized tooltips for the app interface.
"""

def generate_tooltips(df):
    """Add tooltip columns to the dataframe."""
    
    # Standard tooltips
    tooltips = {
        'IPR': "IPR shows how many times a household's monthly income covers the mortgage payment; higher is better. Around 3× is typically affordable.",
        'Economy': "Economy score reflects jobs and output (employment rate and GRDP per capita). Higher = stronger.",
        'Demand': "Demand score reflects structural housing need based on occupied units. Higher = more demand.",
        'Safety': "Safety combines flood, storm surge, landslide, and earthquake risks into one index; higher = safer.",
        'FinalScore': "Final score balances Profitability and Safety (50/50). Higher = better overall site.",
        'UnitModel': "Uses client unit pricing (TCP) for Andrew, Bernie, and Nathan; market median uses listings. Payments use scenario rate and a 20-year tenor."
    }
    
    # Add tooltip columns
    for key, text in tooltips.items():
        df[f'Tooltip_{key}'] = text
    
    return df


def generate_recommendation_text(row):
    """Generate the recommendation text for a city based on its scores."""
    
    # Format scores for text
    scores = {
        'final': int(row['FinalCityScore_scn'] * 100),
        'safety': int(row['HazardSafety_NoFault'] * 100),
        'feasibility': int(row['FeasibilityScore_scn'] * 100),
        'economy': int(row['EconomyScore'] * 100),
        'demand': int(row['DemandScore'] * 100),
    }
    
    # Get unit model specific text
    if row['UnitModel'] != 'MARKET_MEDIAN':
        price_text = f"using the {row['UnitModel']} Model (TCP ≈ ₱{row['TCP_Model']:,.2f}, est. monthly ≈ ₱{row['MonthlyPayment_Model']:,.2f})"
    else:
        price_text = "using the market median price"
    
    # Base recommendation on final score
    if scores['final'] >= 70:
        recommendation = "Recommended"
    elif scores['final'] >= 50:
        recommendation = "Review Further"
    else:
        recommendation = "Not Recommended"
    
    # Generate risk action text based on risk levels
    risk_actions = []
    if row['RISK_Flood_Level_Num'] > 0:
        risk_actions.append("maintain flood safeguards")
    if row['RISK_Fault_Distance_km'] < 5:
        risk_actions.append("follow seismic codes")
    if row['RISK_StormSurge_Level_Num'] > 0:
        risk_actions.append("implement storm protections")
    if not risk_actions:
        risk_actions = ["maintain standard safeguards"]
    
    # Construct full recommendation text
    text = (
        f"{recommendation}. {row['City']} ({row['Scenario']}) scores {scores['final']}/100 "
        f"with safety {scores['safety']}/100. Affordability is {scores['feasibility']}/100 — "
        f"{price_text}, households earn about {row['IPR_20yr']:.2f}× the monthly payment "
        f"(higher is easier). Economy is {scores['economy']}/100; demand is {scores['demand']}/100. "
        f"Risk actions: {', '.join(risk_actions)}."
    )
    
    return recommendation, text


def add_recommendations(df):
    """Add recommendation labels and text to the dataframe."""
    
    # Apply recommendation generation to each row
    recommendations = df.apply(generate_recommendation_text, axis=1)
    df['Site_Recommendation_Label_Client'] = [r[0] for r in recommendations]
    df['Site_Recommendation_Text_Client'] = [r[1] for r in recommendations]
    
    return df


def main():
    # Input and output paths
    IN_PATH = "data/curated/with scores/fact_table_with_unit_models.csv"
    OUT_PATH = "data/curated/with scores/app-ready/fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv"
    
    print("Loading scored fact table...")
    df = pd.read_csv(IN_PATH)
    
    print("Adding tooltips and recommendations...")
    df = generate_tooltips(df)
    df = add_recommendations(df)
    
    print("Saving final app-ready fact table...")
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved to: {OUT_PATH}")
    
    # Print sample recommendations
    print("\nSample Recommendations:")
    sample = df.groupby('Site_Recommendation_Label_Client').first()
    for label, row in sample.iterrows():
        print(f"\n{label}:")
        print(row['Site_Recommendation_Text_Client'])


if __name__ == "__main__":
    import pandas as pd
    main()