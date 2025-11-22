import datetime
import os
import tempfile
import pandas as pd
from airflow.decorators import dag, task
from etl_helpers.minio_utils import (
    upload_to_minio,
    list_objects,
    create_bucket_if_not_exists,
    download_to_dataframe,
    upload_from_dataframe,
    check_file_exists,
    set_bucket_lifecycle_policy,
)
from etl_helpers.data_loader import (
    download_crimes_incremental,
    download_police_stations,
)
from etl_helpers.data_enrichment import enrich_crime_data
from etl_helpers.data_splitter import preprocess_for_split, split_train_test
from etl_helpers.outlier_processing import process_outliers as process_outliers_fn

# Configuration
BUCKET_NAME = os.getenv("DATA_REPO_BUCKET_NAME", "data")

# Bucket structure
PREFIX_RAW = "0-raw-data/"
PREFIX_MERGED = "1-merged-data/"
PREFIX_ENRICHED = "2-enriched-data/"
PREFIX_SPLIT = "3-split-data/"
PREFIX_OUTLIERS = "4-outliers/"

# Data processing parameters
LIFECYCLE_TTL_DAYS = 60  # All files deleted after 60 days
ROLLING_WINDOW_DAYS = 365  # Merged data contains 365 days of crimes
TARGET_COLUMN = "arrest"  # ML target variable
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
OUTLIER_STD_THRESHOLD = 3  # Number of standard deviations for outlier detection

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(
        minutes=60
    ),  # Increased for large data downloads
}


@dag(
    dag_id="etl_with_taskflow",
    description="Chicago Crime Data ETL Pipeline with TaskFlow API",
    default_args=default_args,
    schedule="@monthly",  # Run monthly for incremental updates
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["ETL", "Chicago", "Crime", "TaskFlow"],
)
def process_etl_taskflow():
    @task.python
    def setup_s3():
        """Setup MinIO bucket with TTL for all prefixes."""
        all_prefixes = [
            PREFIX_RAW,
            PREFIX_MERGED,
            PREFIX_ENRICHED,
            PREFIX_SPLIT,
            PREFIX_OUTLIERS,
        ]

        create_bucket_if_not_exists(BUCKET_NAME)
        set_bucket_lifecycle_policy(BUCKET_NAME, all_prefixes, LIFECYCLE_TTL_DAYS)

    @task.python
    def download_data(**context):
        """Download crime and police station data for the current period."""
        start_date = context["data_interval_start"]
        end_date = context["data_interval_end"]
        month_folder = start_date.strftime("%Y-%m")

        # Define output paths
        crimes_key = f"{PREFIX_RAW}{month_folder}/crimes.csv"
        stations_key = f"{PREFIX_RAW}police_stations.csv"

        # Check if already downloaded (idempotent)
        if check_file_exists(BUCKET_NAME, crimes_key):
            crimes_df = download_to_dataframe(BUCKET_NAME, crimes_key)
            return {
                "status": "success",
                "crimes_file": crimes_key,
                "stations_file": stations_key,
                "records": len(crimes_df),
            }

        # Determine download date range
        existing_merged = list_objects(
            BUCKET_NAME, prefix=f"{PREFIX_MERGED}crimes_12m_"
        )

        if len(existing_merged) == 0:
            # No existing data - download full rolling window
            download_start = end_date - datetime.timedelta(days=ROLLING_WINDOW_DAYS)
            download_end = end_date
        else:
            # Use scheduled interval
            download_start = start_date
            download_end = end_date

        # Download data to temp files
        crimes_temp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        stations_temp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )

        try:
            # Download crimes (always use incremental with appropriate date range)
            crimes_df = download_crimes_incremental(
                download_start, download_end, output_file=crimes_temp.name
            )

            if len(crimes_df) == 0:
                return {"status": "no_data", "records": 0}

            # Download police stations
            download_police_stations(output_file=stations_temp.name)

            # Upload to MinIO
            upload_to_minio(crimes_temp.name, BUCKET_NAME, crimes_key)
            upload_to_minio(stations_temp.name, BUCKET_NAME, stations_key)

            return {
                "status": "success",
                "crimes_file": crimes_key,
                "stations_file": stations_key,
                "records": len(crimes_df),
            }
        finally:
            if os.path.exists(crimes_temp.name):
                os.remove(crimes_temp.name)
            if os.path.exists(stations_temp.name):
                os.remove(stations_temp.name)

    @task.python
    def merge_data(download_result, **context):
        """Merge downloaded data into rolling 12-month window."""
        run_date = context["ds"]
        merged_key = f"{PREFIX_MERGED}crimes_12m_{run_date}.csv"

        # Check if merged file already exists (idempotent)
        if check_file_exists(BUCKET_NAME, merged_key):
            merged_df = download_to_dataframe(BUCKET_NAME, merged_key)
            return {
                "status": "success",
                "merged_file": merged_key,
                "stations_file": download_result.get("stations_file"),
                "records": len(merged_df),
            }

        # Check if upstream returned no_data
        if download_result.get("status") == "no_data":
            return {"status": "no_data"}

        # Load newly downloaded data
        df_new = download_to_dataframe(BUCKET_NAME, download_result["crimes_file"])

        # Check for existing merged data
        existing_merged = list_objects(
            BUCKET_NAME, prefix=f"{PREFIX_MERGED}crimes_12m_"
        )

        if len(existing_merged) == 0:
            # First merge - use downloaded data as-is
            merged_df = df_new
        else:
            # Merge with latest existing data
            latest_merged = sorted(existing_merged)[-1]
            df_existing = download_to_dataframe(BUCKET_NAME, latest_merged)
            merged_df = pd.concat([df_existing, df_new], ignore_index=True)

            # Apply rolling window filter
            merged_df["date"] = pd.to_datetime(merged_df["date"])
            cutoff_date = datetime.datetime.now() - datetime.timedelta(
                days=ROLLING_WINDOW_DAYS
            )
            merged_df = merged_df[merged_df["date"] >= cutoff_date]

        # Save merged data
        upload_from_dataframe(merged_df, BUCKET_NAME, merged_key)

        return {
            "status": "success",
            "merged_file": merged_key,
            "stations_file": download_result["stations_file"],
            "records": len(merged_df),
        }

    @task.python
    def enrich_data(merge_result, **context):
        """Add nearest station info and temporal features to crime data."""
        run_date = context["ds"]
        enriched_key = f"{PREFIX_ENRICHED}crimes_enriched_{run_date}.csv"

        # Check if enriched file already exists (idempotent)
        if check_file_exists(BUCKET_NAME, enriched_key):
            enriched_df = download_to_dataframe(BUCKET_NAME, enriched_key)
            return {
                "status": "success",
                "enriched_file": enriched_key,
                "records": len(enriched_df),
            }

        # Check if upstream returned no_data
        if merge_result.get("status") == "no_data":
            return {"status": "no_data"}

        # Process data
        crimes_df = download_to_dataframe(BUCKET_NAME, merge_result["merged_file"])
        stations_df = download_to_dataframe(BUCKET_NAME, merge_result["stations_file"])
        enriched_df = enrich_crime_data(crimes_df, stations_df)

        # Upload enriched data
        upload_from_dataframe(enriched_df, BUCKET_NAME, enriched_key)

        return {
            "status": "success",
            "enriched_file": enriched_key,
            "records": len(enriched_df),
        }

    @task.python
    def split_data(enrich_result, **context):
        """Split dataset into train and test sets with stratification."""
        run_date = context["ds"]
        train_key = f"{PREFIX_SPLIT}crimes_train_{run_date}.csv"
        test_key = f"{PREFIX_SPLIT}crimes_test_{run_date}.csv"

        # Check if split files already exist (idempotent)
        if check_file_exists(BUCKET_NAME, train_key) and check_file_exists(
            BUCKET_NAME, test_key
        ):
            train_df = download_to_dataframe(BUCKET_NAME, train_key)
            test_df = download_to_dataframe(BUCKET_NAME, test_key)
            return {
                "status": "success",
                "train_file": train_key,
                "test_file": test_key,
                "train_records": len(train_df),
                "test_records": len(test_df),
            }

        # Check if upstream returned no_data
        if enrich_result.get("status") == "no_data":
            return {"status": "no_data"}

        # Process data
        df_enriched = download_to_dataframe(BUCKET_NAME, enrich_result["enriched_file"])
        df_clean = preprocess_for_split(df_enriched)
        train_df, test_df = split_train_test(
            df_clean,
            test_size=SPLIT_TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify_column=TARGET_COLUMN,
        )

        # Upload train and test datasets
        upload_from_dataframe(train_df, BUCKET_NAME, train_key)
        upload_from_dataframe(test_df, BUCKET_NAME, test_key)

        return {
            "status": "success",
            "train_file": train_key,
            "test_file": test_key,
            "train_records": len(train_df),
            "test_records": len(test_df),
        }

    @task.python
    def process_outliers(split_result, **context):
        """
        Process outliers using log transformation and standard deviation method.
        Uses train statistics for both datasets to avoid data leakage.
        """
        run_date = context["ds"]
        train_key = f"{PREFIX_OUTLIERS}crimes_train_no_outliers_{run_date}.csv"
        test_key = f"{PREFIX_OUTLIERS}crimes_test_no_outliers_{run_date}.csv"

        # Check if processed files already exist (idempotent)
        if check_file_exists(BUCKET_NAME, train_key) and check_file_exists(
            BUCKET_NAME, test_key
        ):
            train_processed = download_to_dataframe(BUCKET_NAME, train_key)
            test_processed = download_to_dataframe(BUCKET_NAME, test_key)
            return {
                "status": "success",
                "train_file": train_key,
                "test_file": test_key,
                "train_records": len(train_processed),
                "test_records": len(test_processed),
            }

        # Check if upstream returned no_data
        if split_result.get("status") == "no_data":
            return {"status": "no_data"}

        # Process data
        train_df = download_to_dataframe(BUCKET_NAME, split_result["train_file"])
        test_df = download_to_dataframe(BUCKET_NAME, split_result["test_file"])
        train_processed, test_processed = process_outliers_fn(
            train_df, test_df, n_std=OUTLIER_STD_THRESHOLD
        )

        # Upload processed datasets
        upload_from_dataframe(train_processed, BUCKET_NAME, train_key)
        upload_from_dataframe(test_processed, BUCKET_NAME, test_key)

        return {
            "status": "success",
            "train_file": train_key,
            "test_file": test_key,
            "train_records": len(train_processed),
            "test_records": len(test_processed),
        }

    @task.python(multiple_outputs=True)
    def encode_data():
        """Encode categorical variables."""
        print("Applying encoding...")

    @task.python(multiple_outputs=True)
    def scale_data():
        """Scale numerical features."""
        print("Applying scaling...")

    @task.python(multiple_outputs=True)
    def balance_data():
        """Balance dataset using SMOTE and undersampling."""
        print("Balancing dataset...")

    @task.python(multiple_outputs=True)
    def extract_features():
        """Extract and select relevant features."""
        print("Extracting features...")

    # Task dependencies
    s3_setup = setup_s3()
    downloaded = download_data()
    merged = merge_data(downloaded)
    enriched = enrich_data(merged)
    split = split_data(enriched)
    outliers = process_outliers(split)
    encoded = encode_data()
    scaled = scale_data()
    balanced = balance_data()
    features = extract_features()

    (
        s3_setup
        >> downloaded
        >> merged
        >> enriched
        >> split
        >> outliers
        >> encoded
        >> scaled
        >> balanced
        >> features
    )


dag = process_etl_taskflow()
