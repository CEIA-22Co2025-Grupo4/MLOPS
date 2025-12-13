"""
ETL Pipeline Configuration

Centralized configuration for the Chicago Crime ETL pipeline.
"""

import os


BUCKET_NAME = os.getenv("DATA_REPO_BUCKET_NAME", "data")

PREFIXES = {
    "raw": "0-raw-data/",
    "merged": "1-merged-data/",
    "enriched": "2-enriched-data/",
    "split": "3-split-data/",
    "outliers": "4-outliers/",
    "encoded": "5-encoded/",
    "scaled": "6-scaled/",
    "balanced": "7-balanced/",
    "ml_ready": "ml-ready-data/",
}

LIFECYCLE_TTL_DAYS = int(os.getenv("LIFECYCLE_TTL_DAYS", "60"))
ROLLING_WINDOW_DAYS = int(os.getenv("ROLLING_WINDOW_DAYS", "365"))

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "arrest")
SPLIT_TEST_SIZE = float(os.getenv("SPLIT_TEST_SIZE", "0.2"))
SPLIT_RANDOM_STATE = int(os.getenv("SPLIT_RANDOM_STATE", "42"))
OUTLIER_STD_THRESHOLD = int(os.getenv("OUTLIER_STD_THRESHOLD", "3"))
MI_THRESHOLD = float(os.getenv("MI_THRESHOLD", "0.05"))

SMOTE_SAMPLING_STRATEGY = float(os.getenv("SMOTE_SAMPLING_STRATEGY", "0.5"))
UNDERSAMPLE_STRATEGY = float(os.getenv("UNDERSAMPLE_STRATEGY", "0.8"))
BALANCE_RANDOM_STATE = int(os.getenv("BALANCE_RANDOM_STATE", "17"))

