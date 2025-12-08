"""
Outlier Processing Functions

Functions for detecting and removing outliers from crime data.
"""

import os
import sys
import numpy as np
import logging

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl_config import config

logger = logging.getLogger(__name__)

# Column to process for outliers (from config)
DISTANCE_COLUMN = "distance_crime_to_police_station"


def apply_log_transformation(df, column=DISTANCE_COLUMN):
    """
    Apply log1p transformation to reduce skewness and outlier influence.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to transform

    Returns:
        pd.DataFrame: DataFrame with transformed column
    """
    df_transformed = df.copy()

    if column not in df_transformed.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df_transformed

    # Apply log1p transformation
    df_transformed[column] = np.log1p(df_transformed[column])
    logger.info(f"Applied log1p transformation to '{column}'")

    return df_transformed


def remove_outliers_std_method(df, column=DISTANCE_COLUMN, n_std=3):
    """
    Remove outliers using standard deviation method (mean ± n*std).

    Args:
        df (pd.DataFrame): Input dataframe (should be log-transformed)
        column (str): Column name to check for outliers
        n_std (int): Number of standard deviations for threshold (default: 3)

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df

    initial_count = len(df)

    # Calculate statistics
    mean_val = df[column].mean()
    std_val = df[column].std()

    # Define bounds
    lower_bound = mean_val - n_std * std_val
    upper_bound = mean_val + n_std * std_val

    # Filter outliers
    df_no_outliers = df[
        (df[column] >= lower_bound) & (df[column] <= upper_bound)
    ].copy()

    outliers_removed = initial_count - len(df_no_outliers)
    outlier_pct = 100 * outliers_removed / initial_count if initial_count > 0 else 0

    logger.info(f"Outlier detection using ±{n_std} std method:")
    logger.info(f"  Bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    logger.info(f"  Removed {outliers_removed} outliers ({outlier_pct:.2f}%)")
    logger.info(f"  Remaining records: {len(df_no_outliers)}")

    return df_no_outliers


def process_outliers(train_df, test_df, column=DISTANCE_COLUMN, n_std=None):
    """
    Remove outliers from both train and test datasets.
    Uses statistics from train data only to avoid data leakage.

    Steps:
    1. Check for and handle NaN values
    2. Calculate statistics from TRAIN data
    3. Remove outliers from BOTH datasets using train statistics

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        column (str): Column to process
        n_std (int): Number of standard deviations for outlier threshold

    Returns:
        tuple: (train_processed, test_processed) DataFrames
    """
    # Use config default if not provided
    if n_std is None:
        n_std = config.OUTLIER_STD_THRESHOLD

    logger.info("Starting outlier removal...")

    if column not in train_df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return train_df, test_df

    # Check for NaN values in the target column
    train_nan_count = train_df[column].isna().sum()
    test_nan_count = test_df[column].isna().sum()

    if train_nan_count > 0:
        logger.warning(f"Found {train_nan_count} NaN values in train '{column}'. These rows will be excluded.")
    if test_nan_count > 0:
        logger.warning(f"Found {test_nan_count} NaN values in test '{column}'. These rows will be excluded.")

    # Calculate statistics from TRAIN data only (avoid data leakage)
    # Note: pandas mean/std skip NaN by default
    mean_val = train_df[column].mean()
    std_val = train_df[column].std()

    lower_bound = mean_val - n_std * std_val
    upper_bound = mean_val + n_std * std_val

    logger.info(f"Train statistics: mean={mean_val:.2f}, std={std_val:.2f}")
    logger.info(
        f"Outlier bounds (±{n_std} std): {lower_bound:.2f} to {upper_bound:.2f}"
    )

    # Remove outliers from both datasets
    train_initial = len(train_df)
    test_initial = len(test_df)

    train_processed = train_df[
        (train_df[column] >= lower_bound) & (train_df[column] <= upper_bound)
    ].copy()

    test_processed = test_df[
        (test_df[column] >= lower_bound) & (test_df[column] <= upper_bound)
    ].copy()

    # Log results
    train_removed = train_initial - len(train_processed)
    test_removed = test_initial - len(test_processed)

    logger.info(
        f"Train: removed {train_removed} outliers ({100 * train_removed / train_initial:.2f}%)"
    )
    logger.info(
        f"Test: removed {test_removed} outliers ({100 * test_removed / test_initial:.2f}%)"
    )
    logger.info("Outlier removal completed")

    return train_processed, test_processed
