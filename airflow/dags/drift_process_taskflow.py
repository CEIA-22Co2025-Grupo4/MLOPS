import datetime
import logging
import os
import tempfile

import pandas as pd
import numpy as np

from airflow.decorators import dag, task
from scipy.stats import ks_2samp

from etl_helpers.minio_utils import (
    create_bucket_if_not_exists,
    download_to_dataframe,
    upload_from_dataframe,
    check_file_exists,
)

logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------

BUCKET_NAME = os.getenv("DATA_REPO_BUCKET_NAME", "data")

PREFIX_REFERENCE = "drift/reference/"
PREFIX_CURRENT = "drift/current/"
PREFIX_RESULTS = "drift/results/"

TARGET_COLUMN = "arrest"
PREDICTION_COLUMN = "prediction"   # must exist on current datasets

# Threshold
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.1
CONCEPT_DRIFT_DELTA = 0.05

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


# ---------------- HELPERS (DRIFT CALCULATIONS) ----------------

def calculate_psi(ref, cur, bins=10):
    """Population Stability Index."""
    ref_counts, _ = pd.cut(ref, bins=bins, retbins=True)
    cur_counts = pd.cut(cur, bins=ref_counts.cat.categories)

    ref_dist = ref_counts.value_counts(normalize=True)
    cur_dist = cur_counts.value_counts(normalize=True)

    psi = ((cur_dist - ref_dist) * np.log((cur_dist + 1e-8) / (ref_dist + 1e-8))).sum()
    return psi


def ks_test(ref, cur):
    """KS statistic for drift."""
    return ks_2samp(ref, cur).statistic


def compute_feature_drift(ref_df, cur_df):
    drift_rows = []

    for col in ref_df.columns:
        if col == TARGET_COLUMN:
            continue

        if col not in cur_df.columns:
            continue

        try:
            ref = ref_df[col].dropna()
            cur = cur_df[col].dropna()

            if ref.dtype == "object":
                ref = ref.astype(str)
                cur = cur.astype(str)

            psi = calculate_psi(ref, cur)
            ks = ks_test(ref, cur)

            drift_rows.append({
                "feature": col,
                "psi": psi,
                "ks": ks,
                "psi_flag": psi > PSI_THRESHOLD,
                "ks_flag": ks > KS_THRESHOLD,
            })
        except Exception as e:
            logger.error(f"Error processing feature {col}: {e}")

    return pd.DataFrame(drift_rows)


def compute_prediction_drift(ref_df, cur_df):
    ref = ref_df[PREDICTION_COLUMN]
    cur = cur_df[PREDICTION_COLUMN]

    psi = calculate_psi(ref, cur)
    ks = ks_test(ref, cur)

    return pd.DataFrame([{
        "psi": psi,
        "ks": ks,
        "psi_flag": psi > PSI_THRESHOLD,
        "ks_flag": ks > KS_THRESHOLD,
    }])


def compute_concept_drift(ref_df, cur_df):
    """Compare model accuracy across periods (if labels exist)."""
    if TARGET_COLUMN not in cur_df.columns:
        return pd.DataFrame([{
            "concept_drift": None,
            "flag": False,
            "note": "No labels available"
        }])

    ref_acc = (ref_df[PREDICTION_COLUMN] == ref_df[TARGET_COLUMN]).mean()
    cur_acc = (cur_df[PREDICTION_COLUMN] == cur_df[TARGET_COLUMN]).mean()

    delta = ref_acc - cur_acc

    return pd.DataFrame([{
        "ref_acc": ref_acc,
        "cur_acc": cur_acc,
        "delta": delta,
        "flag": delta > CONCEPT_DRIFT_DELTA,
    }])


# ---------------- DAG ----------------

@dag(
    dag_id="drift_with_taskflow",
    description="Drift monitoring pipeline",
    default_args=default_args,
    schedule="@monthly",
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["Drift", "Monitoring"],
)
def drift_monitoring():

    @task.python
    def setup_bucket():
        create_bucket_if_not_exists(BUCKET_NAME)

    @task.python
    def load_reference():
        """Load reference dataset."""
        key = f"{PREFIX_REFERENCE}reference.csv"

        if not check_file_exists(BUCKET_NAME, key):
            raise FileNotFoundError("Reference dataset missing.")

        df = download_to_dataframe(BUCKET_NAME, key)
        logger.info(f"Loaded reference dataset with {len(df)} rows")
        return {"key": key}

    @task.python
    def load_current(**context):
        """Load latest production batch."""
        run_date = context["ds"]
        key = f"{PREFIX_CURRENT}current_{run_date}.csv"

        if not check_file_exists(BUCKET_NAME, key):
            raise FileNotFoundError("Current dataset missing.")

        df = download_to_dataframe(BUCKET_NAME, key)
        logger.info(f"Loaded current dataset with {len(df)} rows")
        return {"key": key, "run_date": run_date}

    @task.python
    def calculate_drift(ref_info, cur_info):
        """Main drift calculation."""
        ref_df = download_to_dataframe(BUCKET_NAME, ref_info["key"])
        cur_df = download_to_dataframe(BUCKET_NAME, cur_info["key"])

        feature_drift = compute_feature_drift(ref_df, cur_df)
        pred_drift = compute_prediction_drift(ref_df, cur_df)
        concept_drift = compute_concept_drift(ref_df, cur_df)

        run_date = cur_info["run_date"]
        drift_key = f"{PREFIX_RESULTS}drift_{run_date}.csv"

        result = pd.concat(
            [
                feature_drift.assign(type="feature"),
                pred_drift.assign(type="prediction"),
                concept_drift.assign(type="concept"),
            ],
            ignore_index=True,
        )

        upload_from_dataframe(result, BUCKET_NAME, drift_key)
        logger.info(f"Drift metrics stored in {drift_key}")

        return {"drift_file": drift_key}

    @task.python
    def alert_if_drift(drift_info):
        df = download_to_dataframe(BUCKET_NAME, drift_info["drift_file"])

        flags = df.get("psi_flag", [] ).sum() + df.get("ks_flag", [] ).sum()

        concept_flag = False
        if "flag" in df.columns:
            concept_flag = df["flag"].any()

        if flags > 0 or concept_flag:
            logger.warning("Drift detected")
        else:
            logger.info("Drift OK")

    setup = setup_bucket()
    ref = load_reference()
    cur = load_current()
    drift = calculate_drift(ref, cur)
    alert = alert_if_drift(drift)

    setup >> ref >> cur >> drift >> alert


dag = drift_monitoring()
