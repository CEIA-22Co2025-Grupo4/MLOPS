"""
Outlier Processing Functions

Functions for detecting and removing outliers from crime data.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Column to process for outliers
DISTANCE_COLUMN = 'distance_crime_to_police_station'


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
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ].copy()

    outliers_removed = initial_count - len(df_no_outliers)
    outlier_pct = 100 * outliers_removed / initial_count if initial_count > 0 else 0

    logger.info(f"Outlier detection using ±{n_std} std method:")
    logger.info(f"  Bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    logger.info(f"  Removed {outliers_removed} outliers ({outlier_pct:.2f}%)")
    logger.info(f"  Remaining records: {len(df_no_outliers)}")

    return df_no_outliers


def process_outliers(train_df, test_df, column=DISTANCE_COLUMN, n_std=3):
    """
    Process outliers for both train and test datasets.
    Uses statistics from train data only to avoid data leakage.

    Steps:
    1. Apply log transformation to both datasets
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
    logger.info("Starting outlier processing...")

    # Step 1: Apply log transformation
    logger.info("Applying log transformation...")
    train_transformed = apply_log_transformation(train_df, column)
    test_transformed = apply_log_transformation(test_df, column)

    # Step 2: Calculate statistics from TRAIN data only (avoid data leakage)
    mean_val = train_transformed[column].mean()
    std_val = train_transformed[column].std()

    lower_bound = mean_val - n_std * std_val
    upper_bound = mean_val + n_std * std_val

    logger.info(f"Train statistics: mean={mean_val:.2f}, std={std_val:.2f}")
    logger.info(f"Outlier bounds (±{n_std} std): {lower_bound:.2f} to {upper_bound:.2f}")

    # Step 3: Remove outliers from both datasets
    train_initial = len(train_transformed)
    test_initial = len(test_transformed)

    train_processed = train_transformed[
        (train_transformed[column] >= lower_bound) &
        (train_transformed[column] <= upper_bound)
    ].copy()

    test_processed = test_transformed[
        (test_transformed[column] >= lower_bound) &
        (test_transformed[column] <= upper_bound)
    ].copy()

    # Log results
    train_removed = train_initial - len(train_processed)
    test_removed = test_initial - len(test_processed)

    logger.info(f"Train: removed {train_removed} outliers ({100*train_removed/train_initial:.2f}%)")
    logger.info(f"Test: removed {test_removed} outliers ({100*test_removed/test_initial:.2f}%)")
    logger.info(f"Outlier processing completed")

    return train_processed, test_processed
