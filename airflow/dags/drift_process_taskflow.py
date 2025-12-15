"""
Drift Monitoring DAG

Downloads fresh crime data, gets predictions from the model API,
and calculates drift metrics comparing against reference data.

On first run (no reference exists), creates a baseline reference dataset.
Subsequent runs compare current data against the reference.

DAG Parameters:
- test_mode: Use minimal data delay for testing (default: False)
- data_delay_days: Override data delay (default: from config)
"""

import datetime
import logging

import pandas as pd

from airflow.decorators import dag, task
from airflow.models.param import Param

from drift_config import drift_config
from etl_helpers.minio import (
    create_bucket_if_not_exists,
    download_to_dataframe,
    upload_from_dataframe,
    list_objects,
)
from etl_helpers.data_loader import download_crimes_incremental
from etl_helpers.inference_preprocessing import preprocess_raw_crimes
from etl_helpers.drift_helpers import (
    compute_feature_drift,
    compute_prediction_drift,
    compute_concept_drift,
    call_batch_prediction_api,
)

logger = logging.getLogger(__name__)

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="drift_with_taskflow",
    description="Drift monitoring pipeline - downloads fresh data and analyzes drift",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["Drift", "Monitoring"],
    params={
        "test_mode": Param(
            default=False,
            type="boolean",
            description="Use minimal data delay for testing",
        ),
        "data_delay_days": Param(
            default=None,
            type=["null", "integer"],
            description="Override data delay days (null = use config default)",
        ),
    },
)
def drift_monitoring():
    @task.python
    def setup_bucket():
        """Create bucket and drift prefixes if they don't exist."""
        create_bucket_if_not_exists(drift_config.bucket_name)
        logger.info(f"Bucket {drift_config.bucket_name} ready for drift monitoring")

    @task.python
    def load_reference():
        """
        Load latest reference dataset from training.

        Returns None if no reference exists (first run scenario).
        """
        ref_files = list_objects(
            drift_config.bucket_name, prefix=drift_config.PREFIX_REFERENCE
        )
        ref_files = [f for f in ref_files if f.endswith(".csv")]

        if not ref_files:
            logger.warning(
                f"No reference dataset found in {drift_config.PREFIX_REFERENCE}. "
                "This appears to be the first run - will create baseline."
            )
            return None

        # Get the latest reference file (sorted by name = sorted by date)
        latest_ref = sorted(ref_files)[-1]
        logger.info(
            f"Found {len(ref_files)} reference file(s), using latest: {latest_ref}"
        )

        # Extract date from filename for logging
        ref_date = (
            latest_ref.split("reference_")[-1].replace(".csv", "")
            if "reference_" in latest_ref
            else "unknown"
        )
        logger.info(f"Reference date: {ref_date}")

        df = download_to_dataframe(drift_config.bucket_name, latest_ref)
        logger.info(f"Loaded reference dataset with {len(df)} rows")

        return {"key": latest_ref, "rows": len(df), "ref_date": ref_date}

    @task.python
    def fetch_and_preprocess_current(**context):
        """
        Download recent crime data, preprocess, get predictions, and save.

        Uses test_mode param to reduce data delay for testing.
        """
        run_date = context["ds"]
        params = context["params"]

        # Determine data delay
        test_mode = params.get("test_mode", False)
        custom_delay = params.get("data_delay_days")

        if custom_delay is not None:
            data_delay = custom_delay
        elif test_mode:
            data_delay = drift_config.MIN_DATA_DELAY_DAYS
        else:
            data_delay = drift_config.DATA_DELAY_DAYS

        # Calculate date range
        run_datetime = datetime.datetime.strptime(run_date, "%Y-%m-%d")
        end_date = run_datetime - datetime.timedelta(days=data_delay)
        start_date = end_date - datetime.timedelta(days=drift_config.DRIFT_WINDOW_DAYS)

        logger.info(
            f"Fetching crime data from {start_date.date()} to {end_date.date()} "
            f"(delay: {data_delay} days, test_mode: {test_mode})"
        )

        # 1. Download raw crime data
        crimes_df = download_crimes_incremental(start_date, end_date)
        if len(crimes_df) == 0:
            raise ValueError(
                f"No crime data available for period {start_date} to {end_date}. "
                f"Try increasing data_delay_days (current: {data_delay})"
            )

        logger.info(f"Downloaded {len(crimes_df)} crime records")

        # 2. Preprocess raw data (enrich + encode + scale)
        features_df, enriched_df = preprocess_raw_crimes(crimes_df)

        # 3. Get predictions from API
        predictions = call_batch_prediction_api(features_df)

        # 4. Build current dataset
        current_df = features_df.copy()
        current_df[drift_config.PREDICTION_COLUMN] = predictions

        # Include ground truth if available (arrest column)
        if drift_config.TARGET_COLUMN in enriched_df.columns:
            current_df[drift_config.TARGET_COLUMN] = enriched_df[
                drift_config.TARGET_COLUMN
            ].values[: len(current_df)]

        # 5. Save to MinIO
        key = f"{drift_config.PREFIX_CURRENT}current_{run_date}.csv"
        upload_from_dataframe(current_df, drift_config.bucket_name, key)
        logger.info(f"Saved current dataset to {key} with {len(current_df)} rows")

        return {"key": key, "run_date": run_date, "rows": len(current_df)}

    @task.python
    def calculate_drift(ref_info, cur_info):
        """
        Calculate drift metrics.

        If ref_info is None (no reference), skips drift calculation
        and returns a warning to run training first.
        """
        run_date = cur_info["run_date"]

        # No reference: warn user to run training first
        if ref_info is None:
            logger.warning("=" * 50)
            logger.warning("NO REFERENCE DATA AVAILABLE")
            logger.warning("=" * 50)
            logger.warning(
                "Drift monitoring requires a reference dataset from training. "
                "Please run 'make train' to create the reference data."
            )
            return {
                "drift_file": None,
                "no_reference": True,
            }

        # Normal run: calculate drift
        cur_df = download_to_dataframe(drift_config.bucket_name, cur_info["key"])
        ref_df = download_to_dataframe(drift_config.bucket_name, ref_info["key"])

        ref_date = ref_info.get("ref_date", "unknown")
        cur_date = cur_info.get("run_date", "unknown")

        logger.info(
            f"Comparing reference ({ref_date}, {len(ref_df)} rows) "
            f"vs current ({cur_date}, {len(cur_df)} rows)"
        )

        feature_drift = compute_feature_drift(ref_df, cur_df)
        pred_drift = compute_prediction_drift(ref_df, cur_df)
        concept_drift = compute_concept_drift(ref_df, cur_df)

        drift_key = f"{drift_config.PREFIX_RESULTS}drift_{run_date}.csv"

        result = pd.concat(
            [
                feature_drift.assign(type="feature"),
                pred_drift.assign(type="prediction"),
                concept_drift.assign(type="concept"),
            ],
            ignore_index=True,
        )

        upload_from_dataframe(result, drift_config.bucket_name, drift_key)
        logger.info(f"Drift metrics stored in {drift_key}")

        return {"drift_file": drift_key}

    @task.python
    def alert_if_drift(drift_info):
        """Check drift results and log alerts."""
        # Handle no reference case
        if drift_info.get("no_reference"):
            logger.warning("=" * 50)
            logger.warning("DRIFT CHECK SKIPPED - NO REFERENCE DATA")
            logger.warning("=" * 50)
            logger.warning(
                "Run 'make train' to create reference data for drift monitoring."
            )
            return {"status": "skipped", "action": "run_training"}

        df = download_to_dataframe(drift_config.bucket_name, drift_info["drift_file"])

        # Count feature drift flags
        psi_flags = df.get("psi_flag", pd.Series([False])).sum()
        ks_flags = df.get("ks_flag", pd.Series([False])).sum()

        # Check concept drift
        concept_flag = False
        if "flag" in df.columns:
            concept_flag = df[df["type"] == "concept"]["flag"].any()

        # Log summary
        logger.info("=" * 50)
        logger.info("DRIFT MONITORING SUMMARY")
        logger.info("=" * 50)

        if psi_flags > 0:
            logger.warning(f"PSI drift detected in {psi_flags} features")
        else:
            logger.info("PSI: No significant drift")

        if ks_flags > 0:
            logger.warning(f"KS drift detected in {ks_flags} features")
        else:
            logger.info("KS: No significant drift")

        if concept_flag:
            logger.warning("CONCEPT DRIFT detected - model accuracy degraded!")
        else:
            logger.info("Concept: No significant accuracy degradation")

        if psi_flags > 0 or ks_flags > 0 or concept_flag:
            logger.warning("DRIFT DETECTED - Consider retraining the model")
            return {"status": "drift_detected", "action": "consider_retraining"}
        else:
            logger.info("All drift checks passed")
            return {"status": "ok", "action": "none"}

    # Task dependencies
    setup = setup_bucket()
    ref = load_reference()
    cur = fetch_and_preprocess_current()
    drift = calculate_drift(ref, cur)
    alert = alert_if_drift(drift)

    setup >> ref >> cur >> drift >> alert


dag = drift_monitoring()
