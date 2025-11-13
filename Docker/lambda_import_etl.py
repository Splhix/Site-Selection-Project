"""
CP2 Construction Site Selection, ETL full version with ML inference
Computes all scores, tips, reasons, labels, recommendation text, ranks, and deltas
Expands to 4 TCP unit models and 7 scenarios, no extra columns beyond your fact table schema
Appends ML predictions at the end, feasibility class and revenue multiplier and revenue

Output columns computed for new rows match your current fact table headers:
Region, Province, City, Scenario, UnitModel, IPR_20yr, FeasibilityScore_scn, ProfitabilityScore_scn, FinalCityScore_scn,
Rank_in_Scenario, Delta_Final_vs_BASE, year, EconomyScore, DemandScore, HazardSafety_NoFault, GRDP_grdp_pc_2024_const,
EMP_w, INC_income_per_hh_2024, DEM_households_single_duplex_2024, DEM_units_single_duplex_2024, RISK_Risk_Gate,
RISK_Fault_Distance_km, RISK_Nearest_Fault_Name, RISK_Flood_Level_Num, RISK_StormSurge_Level_Num, RISK_Landslide_Level_Num,
RISK_Flood_Level_Tip, RISK_StormSurge_Level_Tip, RISK_Landslide_Level_Tip, RISK_Earthquake_Level_Tip, RISK_Risk_Gate_Reasons,
RISK_events_50km, RISK_m5plus_50km, RISK_avg_mag_50km, RISK_events_50km_per_year, RISK_avg_depth_km_50km,
FloodRisk, StormSurgeRisk, LandslideRisk, RISK_HydroRisk, EarthquakeRisk, HazardRisk_NoFault, TCP_Model,
price_source_flag, MonthlyPayment_Model, rate_snapshot_date, Site_Recommendation_Label_Client, Site_Recommendation_Text_Client,
Tooltip_IPR, Tooltip_Economy, Tooltip_Demand, Tooltip_Safety, Tooltip_FinalScore, Tooltip_UnitModel,
Predicted_Multiplier, Expected_PerUnitPrice, PotentialRevenue_pred
"""

import json
import boto3
import csv
import pandas as pd
import numpy as np
import io
import datetime
import traceback
import logging
import joblib
from pathlib import Path

# -----------------------------
# AWS and constants
# -----------------------------
s3 = boto3.client("s3")
logging.getLogger().setLevel(logging.INFO)

BUCKET_NAME = "site-selection-project-data"
def get_latest_fact_table(bucket, prefix="curated/app-ready/"):
    """Find the latest processed fact table in the specified S3 prefix."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [obj for obj in response.get("Contents", []) if obj["Key"].endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in s3://{bucket}/{prefix}")
    # Sort by last modified date, newest first
    files.sort(key=lambda x: x["LastModified"], reverse=True)
    latest_key = files[0]["Key"]
    logging.info(f"📂 Latest fact table found: {latest_key}")
    return latest_key

# Dynamically resolve the latest base fact table
BASE_PATH = get_latest_fact_table(BUCKET_NAME)

OUTPUT_PREFIX = "curated/app-ready/"

DEFAULT_YEAR = 2024
BASE_RATE = 0.085  # 8.5 percent nominal
TCP_ANDREW = 5_783_032.65
TCP_BERNIE = 5_171_067.08
TCP_NATHAN = 4_376_525.02

SCENARIOS = [
    "BASE",
    "RATE_+25bp",
    "RATE_+100bp",
    "PRICE_+10",
    "PRICE_-10",
    "RATE_+25bp__PRICE_+10",
    "RATE_+100bp__PRICE_+10",
]

UNIT_MODELS = ["MARKET_MEDIAN", "ANDREW", "BERNIE", "NATHAN"]

# -----------------------------
# Model paths
# -----------------------------
MODEL_DIR = Path("/var/task/models") if Path("/var/task/models").exists() else Path("models")
FEAS_MODEL_PATHS = [
    MODEL_DIR / "comprehensive_feasibility_model.joblib",
    MODEL_DIR / "comprehensive_feasibility_model.pkl",
]
FEAS_FEATURE_INFO_PATHS = [
    MODEL_DIR / "feasibility_feature_info.joblib",
    MODEL_DIR / "feasibility_feature_info.pkl",
]
REV_MODEL_PATHS = [
    MODEL_DIR / "potential_revenue_randomforest_model.joblib",
    MODEL_DIR / "potential_revenue_randomforest_model.pkl",
]

# -----------------------------
# Helpers
# -----------------------------
def load_model_first_available(paths):
    import pickle
    errors = []
    for p in paths:
        if not p.exists():
            errors.append(f"missing {p}")
            continue
        try:
            return joblib.load(p)
        except Exception as e:
            errors.append(f"joblib error {p}: {e}")
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e2:
                errors.append(f"pickle error {p}: {e2}")
    raise RuntimeError(f"none loadable: {errors}")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def clamp01(x):
    x = safe_float(x)
    if pd.isna(x):
        return 0.0
    return float(min(max(x, 0.0), 1.0))

def compute_amortization(P, annual_rate, years=20):
    P = safe_float(P)
    if pd.isna(P) or P <= 0:
        return np.nan
    r = annual_rate / 12.0
    n = years * 12
    return P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def feasibility_from_ipr(ipr):
    ipr = safe_float(ipr)
    if pd.isna(ipr) or ipr <= 0:
        return 0.0
    a, b = 2.5, 1.16
    return clamp01(1.0 / (1.0 + np.exp(-a * (ipr - b))))

def ordinal_suffix(n):
    try:
        n = int(n)
    except Exception:
        n = 0
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return suffix

def rec_label(x):
    if x >= 0.67:
        return "Recommended"
    if x >= 0.34:
        return "Review Further"
    return "Not Recommended"

def gen_reco(row):
    city = row.get("City", "This city")
    score = clamp01(row.get("FinalCityScore_scn", 0)) * 100
    rank = int(safe_float(row.get("Rank_in_Scenario", 0)) or 0)
    feas = clamp01(row.get("FeasibilityScore_scn", 0)) * 100
    econ = clamp01(row.get("EconomyScore", 0)) * 100
    dem = clamp01(row.get("DemandScore", 0)) * 100
    safe = clamp01(row.get("HazardSafety_NoFault", 0)) * 100
    gate = row.get("RISK_Risk_Gate", "")
    label = row.get("Site_Recommendation_Label_Client", "")
    txt = f"{city} scored {score:.0f} out of 100 and ranked {rank}{ordinal_suffix(rank)} overall. "
    txt += "Affordability is limited, " if feas < 30 else "Affordability is moderate, " if feas < 60 else "Affordability is favorable, "
    txt += "the economy is weak " if econ < 40 else "the economy is steady " if econ < 70 else "the economy is strong "
    txt += "with low housing demand. " if dem < 40 else "with fair housing demand. " if dem < 70 else "with strong and consistent housing demand. "
    txt += "Safety levels are low, " if safe < 40 else "Safety is moderate, " if safe < 70 else "Safety levels are strong, "
    txt += "and further review is advised due to nearby hazard exposures. " if gate == "REVIEW" else "and no major hazard triggers detected. "
    if label == "Recommended":
        txt += "Overall, it is a recommended site for future housing development."
    elif label == "Review Further":
        txt += "Overall, it is advisable to review this site further before investment."
    else:
        txt += "Overall, it is not recommended for housing development."
    return txt

def flood_tip(level_num):
    lv = safe_float(level_num)
    if pd.isna(lv) or lv < 1.5:
        return "This area has a low chance of flooding and is generally safe from water accumulation."
    if lv < 2.5:
        return "This area is in a moderate flood zone, occasional flooding can occur during heavy rains."
    return "This area is highly prone to flooding, extra caution is advised during strong rain and typhoons."

def surge_tip(level_num):
    lv = safe_float(level_num)
    if pd.isna(lv) or lv < 1.5:
        return "Storm surge exposure is low, coastal flooding is unlikely in typical conditions."
    if lv < 2.5:
        return "There is a moderate risk of storm surges during severe typhoons, especially near the coast."
    return "Storm surge exposure is high, coastal flooding is likely in severe weather."

def slide_tip(level_num):
    lv = safe_float(level_num)
    if pd.isna(lv) or lv < 1.5:
        return "Ground movement risk is low, terrain is generally stable."
    if lv < 2.5:
        return "There is a moderate chance of landslides in sloped or hilly areas."
    return "The area is highly susceptible to landslides, especially after prolonged rainfall."

def classify_risk_gate(row):
    fd = safe_float(row.get("RISK_Fault_Distance_km", np.nan))
    eq = safe_float(row.get("EarthquakeRisk", np.nan))
    flood = safe_float(row.get("RISK_Flood_Level_Num", np.nan))
    surge = safe_float(row.get("RISK_StormSurge_Level_Num", np.nan))
    slide = safe_float(row.get("RISK_Landslide_Level_Num", np.nan))
    if (not pd.isna(fd) and fd <= 5) or (not pd.isna(eq) and eq > 0.4) \
       or (not pd.isna(flood) and flood >= 2) \
       or (not pd.isna(surge) and surge >= 2) \
       or (not pd.isna(slide) and slide >= 2):
        return "REVIEW"
    return "PASS"

def risk_gate_reasons(row):
    reasons = []
    if not pd.isna(row.get("RISK_Fault_Distance_km")) and safe_float(row["RISK_Fault_Distance_km"]) <= 5:
        reasons.append("it is close to an active fault")
    if not pd.isna(row.get("RISK_Flood_Level_Num")) and safe_float(row["RISK_Flood_Level_Num"]) >= 2:
        reasons.append("there is noticeable flood exposure")
    if not pd.isna(row.get("RISK_StormSurge_Level_Num")) and safe_float(row["RISK_StormSurge_Level_Num"]) >= 2:
        reasons.append("storm surges are a potential concern")
    if not pd.isna(row.get("RISK_Landslide_Level_Num")) and safe_float(row["RISK_Landslide_Level_Num"]) >= 2:
        reasons.append("landslide prone terrain is present")
    if not reasons:
        return "No major hazard triggers were detected. The area shows low exposure to major natural risks."
    joined = ", and ".join(reasons)
    return f"Further review is advised because {joined}."

def eq_tip(row):
    fd = safe_float(row.get("RISK_Fault_Distance_km", np.nan))
    if pd.isna(fd):
        return "Fault distance is not available, please review local seismic studies."
    if fd <= 1:
        return "Very near an active fault, seismic exposure is very high."
    if fd <= 5:
        return "Within 5 km of an active fault, seismic risk is high."
    if fd <= 10:
        return "About 5 to 10 km from an active fault, seismic exposure is moderate."
    return "More than 10 km from an active fault, direct fault risk is low."

def compute_eq_risk(row):
    fd = safe_float(row.get("RISK_Fault_Distance_km", np.nan))
    events = safe_float(row.get("RISK_events_50km", np.nan))
    if pd.isna(fd):
        prox_idx = 0.5
    elif fd <= 1:
        prox_idx = 1.0
    elif fd <= 5:
        prox_idx = 0.8
    elif fd <= 10:
        prox_idx = 0.5
    else:
        prox_idx = 0.2
    eq_idx = min(max(events, 0) / 20.0, 1.0) if not pd.isna(events) else 0.5
    return round((0.4 * prox_idx + 0.6 * eq_idx), 3)

def today_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# -----------------------------
# Main handler
# -----------------------------
def lambda_handler(event, context):
    try:
        logging.info(f"[STEP 1] Event: {json.dumps(event)}")

        # Load uploaded new dataset from triggering bucket
        record = event.get("Records", [{}])[0]
        bucket = record.get("s3", {}).get("bucket", {}).get("name")
        key = record.get("s3", {}).get("object", {}).get("key")
        if not bucket or not key:
            return {"statusCode": 500, "body": json.dumps({"error": "No S3 object key in event."})}

        obj = s3.get_object(Bucket=bucket, Key=key)
        incoming = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logging.info(f"[STEP 2] New dataset rows: {len(incoming)}")

        # Load base fact table
        base_obj = s3.get_object(Bucket=BUCKET_NAME, Key=BASE_PATH)
        base_df = pd.read_csv(io.BytesIO(base_obj["Body"].read()))
        logging.info(f"[STEP 3] Base fact rows: {len(base_df)}")

        # Required columns from upload
        required = [
            "Region","Province","City",
            "GRDP_grdp_pc_2024_const","EMP_w","INC_income_per_hh_2024",
            "DEM_households_single_duplex_2024","DEM_units_single_duplex_2024",
            "RISK_Fault_Distance_km","RISK_Nearest_Fault_Name",
            "RISK_Flood_Level_Num","RISK_StormSurge_Level_Num","RISK_Landslide_Level_Num",
            "RISK_events_50km","RISK_m5plus_50km","RISK_avg_mag_50km","RISK_avg_depth_km_50km"
        ]
        missing = [c for c in required if c not in incoming.columns]
        if missing:
            return {"statusCode": 400, "body": json.dumps({"error": f"Missing columns in upload: {missing}"})}

        # Create Market Median TCP if not provided, use 9x income per HH as a simple anchor
        if "TCP_MarketMedian" not in incoming.columns:
            incoming["TCP_MarketMedian"] = incoming["INC_income_per_hh_2024"].astype(float) * 9.0

        # Anchor year
        incoming["year"] = DEFAULT_YEAR

        # Build per unit model copies with TCP_Model
        mm = incoming.copy()
        mm["UnitModel"] = "MARKET_MEDIAN"
        mm["TCP_Model"] = mm["TCP_MarketMedian"]
        mm["price_source_flag"] = "MARKET_MEDIAN"

        andrew = incoming.copy()
        andrew["UnitModel"] = "ANDREW"
        andrew["TCP_Model"] = TCP_ANDREW
        andrew["price_source_flag"] = "CLIENT_TCP"

        bernie = incoming.copy()
        bernie["UnitModel"] = "BERNIE"
        bernie["TCP_Model"] = TCP_BERNIE
        bernie["price_source_flag"] = "CLIENT_TCP"

        nathan = incoming.copy()
        nathan["UnitModel"] = "NATHAN"
        nathan["TCP_Model"] = TCP_NATHAN
        nathan["price_source_flag"] = "CLIENT_TCP"

        new_df = pd.concat([mm, andrew, bernie, nathan], ignore_index=True)

        # Scenario expansion
        expanded = []
        for scn in SCENARIOS:
            tmp = new_df.copy()
            tmp["Scenario"] = scn

            # Adjust TCP and rate internally, then write adjusted TCP back to TCP_Model
            adj_tcp = tmp["TCP_Model"].astype(float).to_numpy()
            rate = BASE_RATE

            if "RATE_+25bp" in scn:
                rate += 0.0025
            if "RATE_+100bp" in scn:
                rate += 0.0100
            if "PRICE_+10" in scn:
                adj_tcp = adj_tcp * 1.10
            if "PRICE_-10" in scn:
                adj_tcp = adj_tcp * 0.90

            tmp["TCP_Model"] = adj_tcp
            tmp["_rate_used"] = rate  # internal only
            expanded.append(tmp)

        new_df = pd.concat(expanded, ignore_index=True)

        # Compute IPR and feasibility
        new_df["MonthlyPayment_Model"] = new_df.apply(
            lambda r: compute_amortization(r["TCP_Model"], annual_rate=r["_rate_used"], years=20), axis=1
        )
        new_df["IPR_20yr"] = new_df.apply(
            lambda r: (safe_float(r["INC_income_per_hh_2024"]) / 12.0) / r["MonthlyPayment_Model"]
            if safe_float(r["MonthlyPayment_Model"]) > 0 else np.nan,
            axis=1
        )
        new_df["FeasibilityScore_scn"] = new_df["IPR_20yr"].apply(feasibility_from_ipr)

        # Hazard risk components and tips
        new_df["FloodRisk"] = ((new_df["RISK_Flood_Level_Num"].astype(float) - 1.0) / 2.0).clip(0, 1)
        new_df["StormSurgeRisk"] = ((new_df["RISK_StormSurge_Level_Num"].astype(float) - 1.0) / 2.0).clip(0, 1)
        new_df["LandslideRisk"] = ((new_df["RISK_Landslide_Level_Num"].astype(float) - 1.0) / 2.0).clip(0, 1)
        new_df["EarthquakeRisk"] = new_df.apply(compute_eq_risk, axis=1)
        new_df["RISK_HydroRisk"] = new_df[["FloodRisk", "StormSurgeRisk", "LandslideRisk"]].mean(axis=1)
        new_df["HazardSafety_NoFault"] = (
            1 - (0.4 * new_df["EarthquakeRisk"] + 0.4 * new_df["RISK_HydroRisk"] + 0.2 * new_df["LandslideRisk"])
        ).clip(0, 1)
        new_df["HazardRisk_NoFault"] = (1 - new_df["HazardSafety_NoFault"]).clip(0, 1)

        new_df["RISK_Flood_Level_Tip"] = new_df["RISK_Flood_Level_Num"].apply(flood_tip)
        new_df["RISK_StormSurge_Level_Tip"] = new_df["RISK_StormSurge_Level_Num"].apply(surge_tip)
        new_df["RISK_Landslide_Level_Tip"] = new_df["RISK_Landslide_Level_Num"].apply(slide_tip)
        new_df["RISK_Earthquake_Level_Tip"] = new_df.apply(eq_tip, axis=1)
        new_df["RISK_Risk_Gate"] = new_df.apply(classify_risk_gate, axis=1)
        new_df["RISK_Risk_Gate_Reasons"] = new_df.apply(risk_gate_reasons, axis=1)

        # Economy and demand
        grdp_max = base_df["GRDP_grdp_pc_2024_const"].astype(float).max()
        emp_max = base_df["EMP_w"].astype(float).max()
        hh_max = base_df["DEM_households_single_duplex_2024"].astype(float).max()

        new_df["EconomyScore"] = (
            0.6 * (new_df["EMP_w"].astype(float) / emp_max if emp_max > 0 else 0.0)
            + 0.4 * (new_df["GRDP_grdp_pc_2024_const"].astype(float) / grdp_max if grdp_max > 0 else 0.0)
        ).clip(0, 1)

        new_df["DemandScore"] = (
            new_df["DEM_households_single_duplex_2024"].astype(float) / hh_max if hh_max > 0 else 0.0
        )
        new_df["DemandScore"] = new_df["DemandScore"].fillna(0.0).clip(0, 1)

        # Profitability and final scores
        new_df["ProfitabilityScore_scn"] = (
            0.4 * new_df["FeasibilityScore_scn"] + 0.4 * new_df["EconomyScore"] + 0.2 * new_df["DemandScore"]
        ).clip(0, 1)
        new_df["FinalCityScore_scn"] = (
            0.5 * new_df["ProfitabilityScore_scn"] + 0.5 * new_df["HazardSafety_NoFault"]
        ).clip(0, 1)

        new_df["rate_snapshot_date"] = today_date_str()

        # Append to base and compute ranks and deltas
        combined_df = pd.concat([base_df, new_df], ignore_index=True)

        # -------------------------------------------------------
        # DEDUPLICATION AND UPDATE HANDLING (Corrected)
        # -------------------------------------------------------
        logging.info("[STEP X] Deduplicating and handling updates (safe city-year-level)")

        # Define identifying keys — include 'year' to preserve all years/scenarios/models
        dedup_keys = ["Region", "Province", "City", "Scenario", "UnitModel", "year"]

        # Add timestamp for traceability
        now_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        new_df["last_update"] = now_ts
        if "last_update" not in base_df.columns:
            base_df["last_update"] = "2024-01-01 00:00:00"

        # Combine datasets
        combined_df = pd.concat([base_df, new_df], ignore_index=True)

        # Sort so the most recent upload comes last
        combined_df = combined_df.sort_values(by=["last_update"], ascending=True)

        # Drop duplicates while keeping the latest version per city-year-scenario-unitmodel
        combined_df = combined_df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)

        logging.info(f"[STEP X] Deduplication complete. Rows after cleanup: {len(combined_df)}")

        combined_df["Rank_in_Scenario"] = (
            combined_df.groupby(["Scenario", "UnitModel", "year"])["FinalCityScore_scn"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        base_scores = (
            combined_df[combined_df["Scenario"] == "BASE"]
            [["Region", "Province", "City", "UnitModel", "year", "FinalCityScore_scn"]]
            .rename(columns={"FinalCityScore_scn": "FinalCityScore_BASE"})
        )
        combined_df = combined_df.merge(
            base_scores,
            on=["Region", "Province", "City", "UnitModel", "year"],
            how="left",
        )
        combined_df["Delta_Final_vs_BASE"] = combined_df["FinalCityScore_scn"] - combined_df["FinalCityScore_BASE"]
        combined_df.loc[combined_df["Scenario"] == "BASE", "Delta_Final_vs_BASE"] = 0.0
        combined_df.drop(columns=["FinalCityScore_BASE"], inplace=True)

        # Labels and recommendation text
        combined_df["Site_Recommendation_Label_Client"] = combined_df["FinalCityScore_scn"].apply(rec_label)
        combined_df["Site_Recommendation_Text_Client"] = combined_df.apply(gen_reco, axis=1)

        # Tooltips
        combined_df["Tooltip_IPR"] = combined_df["IPR_20yr"].apply(
            lambda x: "Higher IPR means income can more easily cover monthly payments."
        )
        combined_df["Tooltip_Economy"] = "Economy blends employment and per capita GRDP, normalized per year."
        combined_df["Tooltip_Demand"] = "Demand uses households relative to max across cities for the anchor year."
        combined_df["Tooltip_Safety"] = "Safety is one minus the combined hazard index, earthquake and hydro risk."
        combined_df["Tooltip_FinalScore"] = "Final score averages profitability and safety, equal weights."
        combined_df["Tooltip_UnitModel"] = "Unit model uses either market median TCP or client TCP presets."

        # -----------------------------
        # ML inference at the end
        # -----------------------------
        logging.info("[STEP 11] Loading ML models")

        feas_model = load_model_first_available(FEAS_MODEL_PATHS)
        feas_info = load_model_first_available(FEAS_FEATURE_INFO_PATHS)
        rev_model = load_model_first_available(REV_MODEL_PATHS)

        if not isinstance(rev_model, dict):
            return {"statusCode": 500, "body": json.dumps({"error": "Revenue model artifact must be a dict with pipeline and feature_order"})}
        if "pipeline" not in rev_model or "feature_order" not in rev_model:
            return {"statusCode": 500, "body": json.dumps({"error": "Revenue model missing pipeline or feature_order"})}

        feas_feats = feas_info.get("feature_order", [])
        rev_feats = rev_model["feature_order"]
        rev_pipe = rev_model["pipeline"]
        bounds = rev_model.get("multiplier_bounds", [0.85, 1.10])
        capture_rate = rev_model.get("capture_rate", 0.07)

        # Only predict for the newly appended slice
        new_slice_mask = combined_df.index >= (len(combined_df) - len(new_df))
        new_slice = combined_df.loc[new_slice_mask].copy()

        # Feasibility classifier
        fdf = new_slice.copy()
        for col in feas_feats:
            if col not in fdf.columns:
                fdf[col] = np.nan
            if fdf[col].dtype.kind in "biufc":
                fdf[col] = fdf[col].fillna(fdf[col].median())
            else:
                fdf[col] = pd.Categorical(fdf[col]).codes

        try:
            feas_pred = feas_model.predict(fdf[feas_feats])
        except Exception:
            # fall back to zeros if mismatch
            feas_pred = np.zeros(len(fdf), dtype=int)

        new_slice["FeasibilityClass_pred"] = feas_pred
        new_slice["FeasibilityClass_Label"] = new_slice["FeasibilityClass_pred"].map(
            {0: "Not Feasible", 1: "Potentially Feasible", 2: "Highly Feasible"}
        )

        # Revenue multiplier model
        for c in rev_feats:
            if c not in new_slice.columns:
                new_slice[c] = 0.0
        X_rev = new_slice[rev_feats].fillna(0.0)

        try:
            pred_mult = np.clip(rev_pipe.predict(X_rev), bounds[0], bounds[1])
        except Exception:
            pred_mult = np.ones(len(X_rev))  # safe default

        new_slice["Predicted_Multiplier"] = pred_mult
        new_slice["Expected_PerUnitPrice"] = new_slice["TCP_Model"].astype(float) * new_slice["Predicted_Multiplier"].astype(float)
        new_slice["PotentialRevenue_pred"] = (
            new_slice["DEM_units_single_duplex_2024"].astype(float)
            * new_slice["Expected_PerUnitPrice"].astype(float)
            * capture_rate
        )

        # Merge back into combined frame
        for col in ["FeasibilityClass_pred","FeasibilityClass_Label","Predicted_Multiplier","Expected_PerUnitPrice","PotentialRevenue_pred"]:
            combined_df.loc[new_slice_mask, col] = new_slice[col].values

        combined_df["rate_snapshot_date"] = combined_df["rate_snapshot_date"].fillna(today_date_str())

        # Remove internal helper column
        if "_rate_used" in combined_df.columns:
            combined_df = combined_df.drop(columns=["_rate_used"])

        # -----------------------------
        # Write to S3
        # -----------------------------
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_key = f"{OUTPUT_PREFIX}fact_table_app_READY_WITH_CLIENT_DATA_{ts}.csv"
        out_buf = io.StringIO()
        combined_df.to_csv(
            out_buf,
            index=False,
            quoting=csv.QUOTE_ALL,   # ✅ wrap all text fields in double quotes
            escapechar='\\',         # ✅ handle embedded quotes cleanly
        )
        s3.put_object(Bucket=BUCKET_NAME, Key=out_key, Body=out_buf.getvalue().encode("utf-8"))
        logging.info(f"[DONE] Wrote output to s3://{BUCKET_NAME}/{out_key}")

        return {"statusCode": 200, "body": json.dumps({"message": "ETL plus ML complete", "rows": len(combined_df), "output_key": out_key})}

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
