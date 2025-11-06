#!/usr/bin/env python3
"""
ETL: Score comprehensive feasibility on the app-ready fact table using the
saved model from machine-learning/models, and optionally upload the result to S3.

Usage:
  python scripts/scoring/comprehensive_feasibility_etl.py \
    --in data/curated/with scores/app-ready/fact_table_app_READY_WITH_CLIENT_DATA_FINAL.csv \
    --out data/curated/with scores/app-ready/fact_table_app_READY_WITH_COMPREHENSIVE_FEAS.csv \
    --s3-bucket my-bucket --s3-key analytics/app/fact_table_app_READY_WITH_COMPREHENSIVE_FEAS.csv \
    [--aws-profile default] [--aws-region ap-southeast-1]

Notes:
  - This mirrors preprocessing choices used during training in
    machine-learning/scripts/feasibility_comprehensive_model.py
  - It expects the following model artifacts to exist:
      machine-learning/models/comprehensive_feasibility_model.pkl
      machine-learning/models/feasibility_feature_info.pkl
"""

import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

# Optional dependency for S3 upload
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None
    BotoCoreError = ClientError = Exception

# Optional dependency for HTTP upload to AWS Lambda URL (frontend pattern)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


def load_artifacts(project_root: str) -> Dict[str, object]:
    models_dir = os.path.join(project_root, "machine-learning", "models")
    model_path = os.path.join(models_dir, "comprehensive_feasibility_model.pkl")
    feature_info_path = os.path.join(models_dir, "feasibility_feature_info.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_info_path, "rb") as f:
        feature_info = pickle.load(f)

    return {
        "model": model,
        "numeric_features": feature_info.get("numeric_features", []),
        "categorical_features": feature_info.get("categorical_features", []),
        "feature_order": feature_info.get("feature_order", []),
    }


def prepare_features(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str], feature_order: List[str]) -> pd.DataFrame:
    X = df.copy()

    # Ensure missing expected columns exist
    for col in numeric_features:
        if col not in X.columns:
            X[col] = np.nan
    for col in categorical_features:
        if col not in X.columns:
            X[col] = "missing"

    # Select only the features we care about
    X = X[numeric_features + categorical_features]

    # Impute numerics with column median, categoricals with 'missing'
    for col in numeric_features:
        median_val = X[col].median(skipna=True)
        X[col] = X[col].fillna(median_val)
    for col in categorical_features:
        X[col] = X[col].astype("object").fillna("missing")

    # Encode categoricals as category codes (replicates training approach)
    for col in categorical_features:
        X[col] = pd.Categorical(X[col]).codes

    # Reorder columns to match training feature order where possible
    if feature_order:
        keep = [c for c in feature_order if c in X.columns]
        # Append any new columns that were not in training to the end
        rest = [c for c in X.columns if c not in keep]
        X = X[keep + rest]

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path (app-ready fact table)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path with model scores")
    ap.add_argument("--s3-bucket", dest="s3_bucket", default=None, help="Optional S3 bucket to upload the output")
    ap.add_argument("--s3-key", dest="s3_key", default=None, help="Optional S3 key (object path) for the uploaded file")
    ap.add_argument("--aws-profile", dest="aws_profile", default=None, help="Optional AWS profile name")
    ap.add_argument("--aws-region", dest="aws_region", default=None, help="Optional AWS region name")
    ap.add_argument("--lambda-url", dest="lambda_url", default=None, help="Optional AWS Lambda Function URL to POST the CSV (frontend-style)")
    ap.add_argument("--lambda-header", dest="lambda_header", action="append", default=None, help="Optional custom header(s) for Lambda POST, format Key:Value. Can be passed multiple times.")
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    print("Loading artifacts...")
    artifacts = load_artifacts(project_root)
    model = artifacts["model"]
    numeric_features = artifacts["numeric_features"]
    categorical_features = artifacts["categorical_features"]
    feature_order = artifacts["feature_order"]

    print(f"Reading input: {args.in_path}")
    df = pd.read_csv(args.in_path)

    print("Preparing features for inference...")
    X = prepare_features(df, numeric_features, categorical_features, feature_order)

    print("Scoring with comprehensive feasibility model...")
    y_pred = model.predict(X)

    # Try probabilities if supported
    proba_cols = []
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        # Build deterministic class order
        classes_ = list(getattr(model, "classes_", list(range(y_proba.shape[1]))))
        for i, cls in enumerate(classes_):
            col_name = f"ComprehensiveFeasibilityProb_{cls}"
            df[col_name] = y_proba[:, i]
            proba_cols.append(col_name)

    # Attach predictions
    df["ComprehensiveFeasibilityClass_Pred"] = y_pred

    # Optional: derive a 0–1 score proxy from expected value of class
    if proba_cols:
        # Expected class scaled to [0,1] by dividing by max class (2)
        max_class = max(getattr(model, "classes_", [2])) or 2
        expected = np.zeros(len(df), dtype=float)
        for col in proba_cols:
            cls_val = int(col.rsplit("_", 1)[-1])
            expected += df[col].values * cls_val
        df["ComprehensiveFeasibilityScore_Model"] = (expected / float(max_class)).clip(0.0, 1.0)

    # Write output
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_path, index=False)

    print(f"Saved scored table to: {args.out_path}")

    # Optional S3 upload
    if args.s3_bucket:
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 upload. Please install boto3 and retry.")

        key = args.s3_key or os.path.basename(args.out_path)
        print(f"Uploading to s3://{args.s3_bucket}/{key} ...")

        try:
            if args.aws_profile:
                session = boto3.Session(profile_name=args.aws_profile, region_name=args.aws_region)
            else:
                session = boto3.Session(region_name=args.aws_region)
            s3 = session.client("s3")
            s3.upload_file(args.out_path, args.s3_bucket, key)
            print(f"Uploaded to s3://{args.s3_bucket}/{key}")
        except (BotoCoreError, ClientError) as e:  # pragma: no cover
            raise RuntimeError(f"Failed to upload to S3: {e}")

    # Optional HTTP POST to Lambda Function URL (mirrors frontend AWS pattern)
    if args.lambda_url:
        if requests is None:
            raise RuntimeError("requests is required for Lambda upload. Please install requests and retry.")

        headers = {"Content-Type": "text/csv", "X-Filename": os.path.basename(args.out_path)}
        if args.lambda_header:
            for h in args.lambda_header:
                if ":" in h:
                    k, v = h.split(":", 1)
                    headers[k.strip()] = v.strip()

        print(f"POSTing output CSV to Lambda URL: {args.lambda_url}")
        with open(args.out_path, "rb") as f:
            resp = requests.post(args.lambda_url, data=f.read(), headers=headers, timeout=60)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(f"Lambda upload failed: {resp.status_code} {resp.text[:500]}")
        print("Lambda upload succeeded.")


if __name__ == "__main__":
    main()


