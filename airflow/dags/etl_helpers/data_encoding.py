"""
Data Encoding Functions

Functions for encoding categorical and numerical variables.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl_config import config

logger = logging.getLogger(__name__)


def apply_log_transformation(
    train_df, test_df, column="distance_crime_to_police_station"
):
    """
    Apply log1p transformation to reduce skewness.

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        column (str): Column name to transform

    Returns:
        tuple: (train_transformed, test_transformed) DataFrames
    """
    logger.info(f"Applying log1p transformation to '{column}'...")

    train_transformed = train_df.copy()
    test_transformed = test_df.copy()

    if column in train_transformed.columns:
        # Check for NaN values
        train_nan = train_transformed[column].isna().sum()
        test_nan = test_transformed[column].isna().sum()
        if train_nan > 0 or test_nan > 0:
            logger.warning(f"NaN values found before log transform: train={train_nan}, test={test_nan}")

        # Check for negative values (log1p of negative returns NaN)
        train_neg = (train_transformed[column] < 0).sum()
        test_neg = (test_transformed[column] < 0).sum()
        if train_neg > 0 or test_neg > 0:
            logger.warning(f"Negative values found (will become NaN): train={train_neg}, test={test_neg}")

        train_transformed[column] = np.log1p(train_transformed[column])
        test_transformed[column] = np.log1p(test_transformed[column])
        logger.info(f"Log transformation applied to '{column}'")
    else:
        logger.warning(f"Column '{column}' not found in DataFrame")

    return train_transformed, test_transformed


def apply_cyclic_encoding(train_df, test_df, column="day_of_week"):
    """
    Apply cyclic encoding to day of week (sine transformation).

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        column (str): Column name (0=Monday, 6=Sunday)

    Returns:
        tuple: (train_encoded, test_encoded) DataFrames
    """
    logger.info(f"Applying cyclic encoding to '{column}'...")

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    if column in train_encoded.columns:
        # Check for NaN values
        train_nan = train_encoded[column].isna().sum()
        test_nan = test_encoded[column].isna().sum()
        if train_nan > 0 or test_nan > 0:
            logger.warning(f"NaN values found in '{column}': train={train_nan}, test={test_nan}")
            # Fill NaN with mode (most common day)
            mode_val = train_encoded[column].mode()[0] if len(train_encoded[column].mode()) > 0 else 0
            train_encoded[column] = train_encoded[column].fillna(mode_val)
            test_encoded[column] = test_encoded[column].fillna(mode_val)
            logger.info(f"Filled NaN in '{column}' with mode: {mode_val}")

        # Convert to 1-7 range and apply sine transformation
        train_encoded[f"{column}_sin"] = np.sin(
            2 * np.pi * (train_encoded[column] + 1) / 7
        )
        test_encoded[f"{column}_sin"] = np.sin(
            2 * np.pi * (test_encoded[column] + 1) / 7
        )
        logger.info(f"Cyclic encoding applied: '{column}_sin' created")
    else:
        logger.warning(f"Column '{column}' not found in DataFrame")

    return train_encoded, test_encoded


def apply_onehot_encoding(train_df, test_df, columns=None):
    """
    Apply one-hot encoding to low cardinality categorical variables.

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        columns (list): Columns to encode (default: ['season', 'day_time'])

    Returns:
        tuple: (train_encoded, test_encoded, encoders) - DataFrames and fitted encoders
    """
    if columns is None:
        columns = ["season", "day_time"]

    logger.info(f"Applying one-hot encoding to: {columns}")

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}

    for col in columns:
        if col not in train_encoded.columns:
            logger.warning(f"Column '{col}' not found, skipping...")
            continue

        # Check for NaN values (OneHotEncoder doesn't handle NaN)
        train_nan = train_encoded[col].isna().sum()
        test_nan = test_encoded[col].isna().sum()
        if train_nan > 0 or test_nan > 0:
            logger.warning(f"NaN values found in '{col}': train={train_nan}, test={test_nan}")
            # Fill with most frequent value
            mode_val = train_encoded[col].mode()[0] if len(train_encoded[col].mode()) > 0 else "UNKNOWN"
            train_encoded[col] = train_encoded[col].fillna(mode_val)
            test_encoded[col] = test_encoded[col].fillna(mode_val)
            logger.info(f"Filled NaN in '{col}' with mode: {mode_val}")

        # Initialize encoder
        ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

        # Fit on train, transform both
        train_array = ohe.fit_transform(train_encoded[[col]])
        test_array = ohe.transform(test_encoded[[col]])

        # Create column names
        feature_names = ohe.get_feature_names_out([col])

        # Create DataFrames and join
        train_ohe_df = pd.DataFrame(
            train_array, columns=feature_names, index=train_encoded.index
        )
        test_ohe_df = pd.DataFrame(
            test_array, columns=feature_names, index=test_encoded.index
        )

        train_encoded = train_encoded.join(train_ohe_df)
        test_encoded = test_encoded.join(test_ohe_df)

        encoders[col] = ohe
        logger.info(f"  {col}: {len(feature_names)} features created")

    return train_encoded, test_encoded, encoders


def apply_label_encoding(train_df, test_df, columns=None):
    """
    Apply label encoding to boolean variables (True=1, False=0).

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        columns (list): Columns to encode (default: ['domestic'])

    Returns:
        tuple: (train_encoded, test_encoded) DataFrames
    """
    if columns is None:
        columns = ["domestic"]

    logger.info(f"Applying label encoding to: {columns}")

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in columns:
        if col not in train_encoded.columns:
            logger.warning(f"Column '{col}' not found, skipping...")
            continue

        # Check for NaN values (.astype(int) fails on NaN)
        train_nan = train_encoded[col].isna().sum()
        test_nan = test_encoded[col].isna().sum()
        if train_nan > 0 or test_nan > 0:
            logger.warning(f"NaN values found in '{col}': train={train_nan}, test={test_nan}")
            # Fill with False (0) for boolean columns
            train_encoded[col] = train_encoded[col].fillna(False)
            test_encoded[col] = test_encoded[col].fillna(False)
            logger.info(f"Filled NaN in '{col}' with False")

        # Create new column with _tag suffix
        new_col = f"{col}_tag"
        train_encoded[new_col] = train_encoded[col].astype(int)
        test_encoded[new_col] = test_encoded[col].astype(int)

        logger.info(f"  {col} -> {new_col}")

    return train_encoded, test_encoded


def apply_frequency_encoding(train_df, test_df, columns=None):
    """
    Apply frequency encoding to high cardinality categorical variables.
    Uses normalized value counts from training data.

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        columns (list): Columns to encode

    Returns:
        tuple: (train_encoded, test_encoded, freq_maps) - DataFrames and frequency mappings
    """
    if columns is None:
        # Only include columns that actually exist after enrichment
        columns = [
            "primary_type",
            "location_description",
            "fbi_code",
            "nearest_police_station_district_name",
            "beat",
            "ward",
            "community_area",
        ]

    logger.info(f"Applying frequency encoding to {len(columns)} columns...")

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    freq_maps = {}

    for col in columns:
        if col not in train_encoded.columns:
            logger.warning(f"Column '{col}' not found, skipping...")
            continue

        # Calculate frequency from training data
        freq_encoding = train_encoded[col].value_counts(normalize=True)
        freq_maps[col] = freq_encoding

        # Create new column with _freq suffix
        new_col = f"{col}_freq"
        # Use fillna(0) for both train and test to handle any NaN values consistently
        train_encoded[new_col] = train_encoded[col].map(freq_encoding).fillna(0).astype(float)

        # For test: use 0 for unseen categories
        test_encoded[new_col] = (
            test_encoded[col].map(freq_encoding).fillna(0).astype(float)
        )

        logger.info(f"  {col}: {len(freq_encoding)} unique values -> {new_col}")

    return train_encoded, test_encoded, freq_maps


def encode_data(train_df, test_df):
    """
    Apply all encoding transformations to train and test datasets.

    Encoding pipeline:
    1. Log transformation: distance_crime_to_police_station
    2. Cyclic encoding: day_of_week -> day_of_week_sin
    3. One-hot encoding: season, day_time
    4. Label encoding: domestic
    5. Frequency encoding: high cardinality categoricals
    6. Drop original categorical columns

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe

    Returns:
        tuple: (train_encoded, test_encoded) DataFrames
    """
    logger.info("Starting data encoding pipeline...")

    # 1. Log transformation
    train_df, test_df = apply_log_transformation(train_df, test_df)

    # 2. Cyclic encoding
    train_df, test_df = apply_cyclic_encoding(train_df, test_df)

    # 3. One-hot encoding
    train_df, test_df, _ = apply_onehot_encoding(train_df, test_df)

    # 4. Label encoding
    train_df, test_df = apply_label_encoding(train_df, test_df)

    # 5. Frequency encoding
    train_df, test_df, _ = apply_frequency_encoding(train_df, test_df)

    # 6. Drop original categorical columns
    columns_to_drop = [
        "season",
        "day_of_week",
        "day_time",
        "domestic",
        "primary_type",
        "location_description",
        "fbi_code",
        "nearest_police_station_district_name",
        "beat",
        "ward",
        "community_area",
    ]

    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in train_df.columns]

    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)

    logger.info(f"Encoding completed: train={train_df.shape}, test={test_df.shape}")

    return train_df, test_df
