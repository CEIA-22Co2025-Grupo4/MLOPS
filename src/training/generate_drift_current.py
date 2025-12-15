"""
Generate current_{date}.csv for drift monitoring testing.

This script:
1. Loads test data from MinIO
2. Sends predictions to FastAPI
3. Saves results as current_{date}.csv for drift DAG

Usage:
    docker-compose run --rm trainer python /trainer/generate_drift_current.py

Or locally:
    python src/testing/generate_drift_current.py
"""

import os
import sys
import re
from datetime import datetime

import pandas as pd
import numpy as np
import requests
import s3fs

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8800")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://s3:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "data")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")


def get_s3_fs():
    """Create s3fs client for MinIO."""
    return s3fs.S3FileSystem(
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )


def load_test_data():
    """Load latest test data from MinIO."""
    print("[*] Loading test data from MinIO...")

    fs = get_s3_fs()
    pattern = re.compile(r"test_(\d{4}-\d{2}-\d{2})\.csv$")

    latest_date = None
    latest_file = None

    for obj in fs.glob(f"{MINIO_BUCKET}/ml-ready-data/*"):
        match = pattern.search(obj)
        if not match:
            continue
        date_str = match.group(1)
        try:
            run_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        if latest_date is None or run_date > latest_date:
            latest_date = run_date
            latest_file = obj

    if not latest_file:
        raise FileNotFoundError("No test data found in MinIO")

    print(f"  [OK] Found: {latest_file}")

    with fs.open(latest_file, "rb") as f:
        df = pd.read_csv(f)

    print(f"  [OK] Loaded {len(df)} rows")
    return df


def add_simulated_drift(df, drift_percentage=0.1):
    """
    Optionally add simulated drift to test data.
    This helps test if drift detection is working.
    """
    print(f"[*] Adding simulated drift ({drift_percentage*100:.0f}% of data)...")

    df_drifted = df.copy()
    n_drift = int(len(df) * drift_percentage)
    drift_indices = np.random.choice(df.index, size=n_drift, replace=False)

    # Add noise to numeric columns (simulates feature drift)
    numeric_cols = df_drifted.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'arrest']

    for col in numeric_cols[:3]:  # Drift first 3 numeric features
        std = df_drifted[col].std()
        df_drifted.loc[drift_indices, col] += np.random.normal(0, std * 0.5, n_drift)

    print(f"  [OK] Added drift to {n_drift} rows")
    return df_drifted


def get_predictions_from_api(df, batch_size=100):
    """Send data to FastAPI and get predictions."""
    print(f"[*] Getting predictions from API ({FASTAPI_URL})...")

    # Check API health first
    try:
        health = requests.get(f"{FASTAPI_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"  [OK] API is healthy")
    except Exception as e:
        print(f"  [ERROR] API not available: {e}")
        print("  [INFO] Falling back to local prediction simulation...")
        return simulate_predictions(df)

    # Prepare features (exclude target)
    feature_cols = [c for c in df.columns if c != 'arrest']
    predictions = []

    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_num = i // batch_size + 1

        try:
            # Send batch prediction request
            payload = {"instances": batch[feature_cols].to_dict(orient='records')}
            response = requests.post(
                f"{FASTAPI_URL}/predict/batch",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            batch_preds = response.json()["predictions"]
            predictions.extend(batch_preds)

            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"  [OK] Processed batch {batch_num}/{total_batches}")

        except Exception as e:
            print(f"  [WARN] Batch {batch_num} failed: {e}")
            # Fallback: random predictions for this batch
            predictions.extend(np.random.randint(0, 2, len(batch)).tolist())

    print(f"  [OK] Got {len(predictions)} predictions")
    return predictions


def simulate_predictions(df):
    """Simulate predictions if API is not available."""
    print("[*] Simulating predictions (API not available)...")

    # Simple simulation: mostly correct with some errors
    true_labels = df['arrest'].values
    predictions = true_labels.copy()

    # Introduce ~15% error rate
    error_indices = np.random.choice(
        len(predictions),
        size=int(len(predictions) * 0.15),
        replace=False
    )
    predictions[error_indices] = 1 - predictions[error_indices]

    print(f"  [OK] Simulated {len(predictions)} predictions")
    return predictions.tolist()


def save_current_data(df, predictions):
    """Save current data with predictions to MinIO."""
    print("[*] Saving current data for drift monitoring...")

    fs = get_s3_fs()
    today = datetime.now().strftime("%Y-%m-%d")

    # Create current dataframe
    current_df = df.copy()
    current_df['prediction'] = predictions

    # Save to MinIO
    current_path = f"{MINIO_BUCKET}/drift/current/current_{today}.csv"

    with fs.open(current_path, "w") as f:
        current_df.to_csv(f, index=False)

    print(f"  [OK] Saved: {current_path}")
    print(f"  [OK] Shape: {current_df.shape[0]} rows, {current_df.shape[1]} columns")

    # Print summary
    print(f"\n[*] Summary:")
    print(f"    Predictions distribution:")
    print(f"      Class 0: {(current_df['prediction'] == 0).sum()}")
    print(f"      Class 1: {(current_df['prediction'] == 1).sum()}")

    if 'arrest' in current_df.columns:
        accuracy = (current_df['prediction'] == current_df['arrest']).mean()
        print(f"    Accuracy vs true labels: {accuracy:.2%}")

    return current_path


def main():
    print("\n" + "="*60)
    print("GENERATE CURRENT DATA FOR DRIFT MONITORING")
    print("="*60)

    try:
        # Load test data
        df = load_test_data()

        # Optionally add simulated drift (set to 0 for no drift)
        # This helps test if drift detection works
        drift_pct = float(os.getenv("DRIFT_PERCENTAGE", "0.1"))
        if drift_pct > 0:
            df = add_simulated_drift(df, drift_percentage=drift_pct)

        # Get predictions
        predictions = get_predictions_from_api(df)

        # Save current data
        current_path = save_current_data(df, predictions)

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"""
Now you can run the drift monitoring DAG:
  1. Go to Airflow UI: http://localhost:8080
  2. Trigger DAG: drift_with_taskflow
  3. Check results in: {MINIO_BUCKET}/drift/results/
""")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
