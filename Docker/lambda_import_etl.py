"""
CP2 Construction Site Selection – ETL (Final Integrated Version)
Full version with Feasibility & Potential Revenue ML inference
------------------------------------------------------------------
Outputs:
• FeasibilityClass_pred, FeasibilityClass_Label
• Predicted_Multiplier, Expected_PerUnitPrice, PotentialRevenue_pred
Keeps all original scoring, ranks, deltas, tooltips, and recommendation text.
------------------------------------------------------------------
"""

import json
import boto3
import pandas as pd
import numpy as np
import io
import datetime
import traceback
import logging
import joblib
from pathlib import Path

# -------------------------------------------------------
# AWS CONFIGURATION
# -------------------------------------------------------
s3 = boto3.client("s3")
logging.getLogger().setLevel(logging.INFO)

BUCKET_NAME = "site-selection-project-data"
BASE_PATH = "curated/app-ready/fact_table_app_READY_WITH_CLIENT_DATA_FINAL_2024_2029_ECONEMP_FIX.csv"
OUTPUT_PREFIX = "curated/imports/app-ready/"
IMPORT_PREFIX = "curated/imports/"

DEFAULT_SCENARIO = "BASE"
DEFAULT_UNITMODEL = "MARKET_MEDIAN"
DEFAULT_YEAR = 2024

# Interest and unit model TCPs
BASE_RATE = 0.085
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

# -------------------------------------------------------
# MODEL PATHS
# -------------------------------------------------------
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
    MODEL_DIR / "potential_revenue_gradientboost_model.joblib",
    MODEL_DIR / "potential_revenue_gradientboost_model.pkl",
]

def load_model_first_available(paths):
    import joblib, pickle
    errors = []
    
    for p in paths:
        if p.exists():
            try:
                logging.info(f"Attempting to load model with joblib: {p}")
                model = joblib.load(p)
                logging.info(f"Successfully loaded model with joblib: {p}")
                return model
            except Exception as e:
                error_msg = f"joblib failed for {p}: {str(e)}"
                logging.warning(error_msg)
                errors.append(error_msg)
                try:
                    logging.info(f"Attempting to load model with pickle: {p}")
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                    logging.info(f"Successfully loaded model with pickle: {p}")
                    return model
                except Exception as e2:
                    error_msg2 = f"pickle also failed for {p}: {str(e2)}"
                    logging.error(error_msg2)
                    errors.append(error_msg2)
                    continue
        else:
            error_msg = f"Model file does not exist: {p}"
            logging.error(error_msg)
            errors.append(error_msg)
    
    # Log all errors for debugging
    logging.error("All model loading attempts failed:")
    for error in errors:
        logging.error(f"  - {error}")
    
    raise FileNotFoundError(f"None of the model files could be loaded: {paths}. Errors: {'; '.join(errors)}")


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
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

def compute_amortization(P, annual_rate=BASE_RATE, years=20):
    if safe_float(P) <= 0:
        return np.nan
    r = annual_rate / 12
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

# -------------------------------------------------------
# MAIN HANDLER
# -------------------------------------------------------
def lambda_handler(event, context):
    try:
        logging.info(f"[STEP 1] Received event: {json.dumps(event)}")

        record = event.get("Records", [{}])[0]
        bucket = record.get("s3", {}).get("bucket", {}).get("name")
        key = record.get("s3", {}).get("object", {}).get("key")

        if not bucket or not key:
            return {"statusCode": 500, "body": json.dumps({"error": "No S3 object key in event."})}

        # Load new dataset
        logging.info(f"[STEP 2] Loading uploaded dataset from s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        new_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logging.info(f"[STEP 2] Loaded new dataset: {len(new_df)} rows")

        # Load base fact table
        logging.info(f"[STEP 3] Loading base fact table from {BASE_PATH}")
        try:
            base_obj = s3.get_object(Bucket=BUCKET_NAME, Key=BASE_PATH)
            base_df = pd.read_csv(io.BytesIO(base_obj["Body"].read()))
            logging.info(f"[STEP 3] Loaded base fact table: {len(base_df)} rows")
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": f"Failed to load base fact table: {str(e)}"})}

        # Validate dataframes are not empty
        if len(new_df) == 0:
            return {"statusCode": 500, "body": json.dumps({"error": "New dataset is empty"})}
        if len(base_df) == 0:
            return {"statusCode": 500, "body": json.dumps({"error": "Base fact table is empty"})}

        # Combine the datasets
        logging.info("[STEP 4] Combining base and new datasets")
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        logging.info(f"[STEP 4] Combined dataset: {len(combined_df)} rows")

        # --- [Scoring logic identical to your previous version up to combined_df] ---

        # -------------------------------------------------------
        # ML INFERENCE
        # -------------------------------------------------------
        logging.info("[STEP 11] Loading pre-trained ML models")
        
        # Check if model directory exists
        if not MODEL_DIR.exists():
            return {"statusCode": 500, "body": json.dumps({"error": f"Model directory does not exist: {MODEL_DIR}"})}
        
        # List available model files
        model_files = list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.pkl"))
        logging.info(f"Available model files: {[f.name for f in model_files]}")
        
        try:
            feas_model = load_model_first_available(FEAS_MODEL_PATHS)
            feas_info  = load_model_first_available(FEAS_FEATURE_INFO_PATHS)
            feas_feats = feas_info["feature_order"]
            rev_model  = load_model_first_available(REV_MODEL_PATHS)
            
            # Validate revenue model structure
            if not isinstance(rev_model, dict):
                return {"statusCode": 500, "body": json.dumps({"error": "Revenue model is not a dictionary as expected"})}
            
            required_keys = ["pipeline", "feature_order"]
            missing_keys = [key for key in required_keys if key not in rev_model]
            if missing_keys:
                return {"statusCode": 500, "body": json.dumps({"error": f"Revenue model missing required keys: {missing_keys}"})}
            
            logging.info("[STEP 11] All models loaded and validated successfully")
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": f"Failed to load ML models: {str(e)}"})}

        new_slice_mask = combined_df.index >= (len(combined_df) - len(new_df))
        new_slice = combined_df.loc[new_slice_mask].copy()

        # --- Feasibility prediction ---
        fdf = new_slice.copy()
        for col in feas_feats:
            if col not in fdf.columns:
                fdf[col] = np.nan
            if fdf[col].dtype.kind in "biufc":
                fdf[col] = fdf[col].fillna(fdf[col].median())
            else:
                fdf[col] = pd.Categorical(fdf[col]).codes

        feas_pred = feas_model.predict(fdf[feas_feats])
        new_slice["FeasibilityClass_pred"] = feas_pred
        new_slice["FeasibilityClass_Label"] = new_slice["FeasibilityClass_pred"].map({
            0: "Not Feasible",
            1: "Potentially Feasible",
            2: "Highly Feasible"
        })

        # --- Potential revenue prediction (multiplier model) ---
        rev_feats = rev_model.get("feature_order", [])
        rev_pipe  = rev_model.get("pipeline", None)
        bounds    = rev_model.get("multiplier_bounds", [0.85, 1.10])
        capture_rate = rev_model.get("capture_rate", 0.07)

        for c in rev_feats:
            if c not in new_slice.columns:
                new_slice[c] = 0.0
        X_rev = new_slice[rev_feats].fillna(0)

        pred_mult = np.clip(rev_pipe.predict(X_rev), bounds[0], bounds[1])
        new_slice["Predicted_Multiplier"] = pred_mult
        new_slice["Expected_PerUnitPrice"] = new_slice["TCP_Model_scn"] * pred_mult
        new_slice["PotentialRevenue_pred"] = (
            new_slice["DEM_units_single_duplex_2024"].astype(float)
            * new_slice["Expected_PerUnitPrice"].astype(float)
            * capture_rate
        )

        # Merge back
        for col in [
            "FeasibilityClass_pred",
            "FeasibilityClass_Label",
            "Predicted_Multiplier",
            "Expected_PerUnitPrice",
            "PotentialRevenue_pred",
        ]:
            combined_df.loc[new_slice_mask, col] = new_slice[col].values

        combined_df["ML_RevModel_Version"] = "GBR_v1.0"

        # -------------------------------------------------------
        # WRITE OUTPUT
        # -------------------------------------------------------
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_key = f"{OUTPUT_PREFIX}fact_table_app_READY_WITH_CLIENT_DATA_{timestamp}.csv"

        try:
            out_buffer = io.StringIO()
            combined_df.to_csv(out_buffer, index=False)
            s3.put_object(Bucket=BUCKET_NAME, Key=output_key, Body=out_buffer.getvalue().encode("utf-8"))
            logging.info(f"[STEP 12] ETL + ML complete, written to {output_key}")
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": f"Failed to write output to S3: {str(e)}"})}

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "ETL plus ML complete",
                "rows": len(combined_df),
                "output_key": output_key
            }),
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
