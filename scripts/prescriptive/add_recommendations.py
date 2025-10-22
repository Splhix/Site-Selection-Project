#!/usr/bin/env python3
"""
Add prescriptive recommendations to the fact table.
Based on CP2 methods guide section 7.
"""

import pandas as pd
import numpy as np
import os


def pct01(x):
    """Convert to percentage (0-100) with error handling."""
    try:
        return int(round(float(x) * 100))
    except:
        return None


def label_client(final, safety):
    """Generate client recommendation label based on scores."""
    if pd.isna(final) or pd.isna(safety):
        return "Review Further"
    if final >= 0.70 and safety >= 0.70:
        return "Recommended"
    if final >= 0.55 and safety >= 0.55:
        return "Review Further"
    return "Not Recommended"


def risk_blurb(row):
    """Generate risk summary text."""
    parts = []
    risk_mappings = [
        ("FloodRisk", "Flood"),
        ("StormSurgeRisk", "Storm Surge"),
        ("LandslideRisk", "Landslide"),
        ("EarthquakeRisk", "Earthquake")
    ]
    
    for k, name in risk_mappings:
        if k in row and pd.notna(row[k]):
            parts.append(f"{name} {pct01(row[k])}/100")
    
    return "; ".join(parts) if parts else "Risks within expected range"


def main():
    # Input and output file paths
    IN = "data/curated/with scores/app-ready/fact_table_app_READY_WITH_SCENARIOS.csv"
    OUT = "data/curated/with scores/app-ready/fact_table_app_READY_WITH_RECS.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    
    print("Loading scenario data...")
    
    # Load scenario data
    try:
        df = pd.read_csv(IN)
        print(f"✅ Loaded scenario data: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading scenario data: {e}")
        return
    
    print("Generating recommendations...")
    
    # Generate labels and recommendation texts
    labels, texts = [], []
    
    for _, r in df.iterrows():
        # Generate recommendation label
        lab = label_client(r.get("FinalCityScore_scn"), r.get("HazardSafety_NoFault"))
        labels.append(lab)
        
        # Generate recommendation text
        city = r.get('City', '(City)')
        scenario = r.get('Scenario', 'BASE')
        final_score = pct01(r.get('FinalCityScore_scn'))
        safety_score = pct01(r.get('HazardSafety_NoFault'))
        feasibility_score = pct01(r.get('FeasibilityScore_scn'))
        economy_score = pct01(r.get('EconomyScore'))
        demand_score = pct01(r.get('DemandScore'))
        ipr = r.get('IPR_20yr', np.nan)
        
        txt = (
            f"{lab}: {city} ({scenario}) scores {final_score}/100 overall "
            f"with Safety {safety_score}/100. "
            f"Feasibility {feasibility_score}/100 (IPR≈{ipr:.2f}×), "
            f"Economy {economy_score}/100, Demand {demand_score}/100. "
            f"Risk — {risk_blurb(r)}."
        )
        texts.append(txt)
    
    # Add recommendation columns
    df["Site_Recommendation_Label_Client"] = labels
    df["Site_Recommendation_Text_Client"] = texts
    
    # Clean up legacy columns if they exist
    legacy_cols = ["AffordabilityScore", "ProfitabilityScore", "FinalCityScore"]
    for col in legacy_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Sort by scenario and final score
    df = df.sort_values(['Scenario', 'FinalCityScore_scn'], ascending=[True, False]).reset_index(drop=True)
    
    # Save with recommendations
    df.to_csv(OUT, index=False)
    print(f"✅ Recommendations added and saved to: {OUT}")
    print(f"   - {len(df)} records processed")
    
    # Print recommendation summary
    print("\n📋 Recommendation Summary:")
    rec_counts = df['Site_Recommendation_Label_Client'].value_counts()
    for label, count in rec_counts.items():
        print(f"   - {label}: {count} cities")
    
    # Print top recommendations by scenario
    print("\n🏆 Top Recommendations by Scenario:")
    for scenario in df['Scenario'].unique():
        scenario_data = df[df['Scenario'] == scenario].head(3)
        print(f"\n   {scenario}:")
        for _, row in scenario_data.iterrows():
            print(f"     • {row['City']} - {row['Site_Recommendation_Label_Client']} (Score: {pct01(row['FinalCityScore_scn'])}/100)")


if __name__ == "__main__":
    main()
