"""
Inference Preprocessing Module for API

Lightweight preprocessing functions for raw crime data.
This module provides preprocessing without heavy dependencies like geopandas.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model feature names (must match training output)
MODEL_FEATURES = [
    "iucr_freq",
    "primary_type_freq",
    "location_description_freq",
    "day_of_week_sin",
    "x_coordinate_standardized",
    "y_coordinate_standardized",
    "distance_crime_to_police_station_standardized",
]


def apply_frequency_encoding(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Apply frequency encoding for inference.

    Args:
        df: DataFrame with categorical columns
        columns: Columns to encode

    Returns:
        DataFrame with frequency-encoded columns added
    """
    if columns is None:
        columns = ["iucr", "primary_type", "location_description"]

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        freq = result[col].value_counts(normalize=True)
        result[f"{col}_freq"] = result[col].map(freq).fillna(0).astype(float)

    return result


def apply_cyclic_encoding(
    df: pd.DataFrame,
    column: str = "day_of_week",
) -> pd.DataFrame:
    """
    Apply cyclic (sine) encoding to day of week.

    Args:
        df: DataFrame with day_of_week column (0=Monday, 6=Sunday)
        column: Column name to encode

    Returns:
        DataFrame with cyclic-encoded column added
    """
    result = df.copy()

    if column not in result.columns:
        return result

    # Convert pandas dayofweek (0=Mon, 6=Sun) to notebook mapping (Sun=1, ..., Sat=7)
    day_num = ((result[column] + 1) % 7) + 1
    result[f"{column}_sin"] = np.sin(2 * np.pi * day_num / 7)

    return result


def apply_log_transform(
    df: pd.DataFrame,
    column: str = "distance_crime_to_police_station",
) -> pd.DataFrame:
    """
    Apply log1p transformation to reduce skewness.

    Args:
        df: DataFrame with distance column
        column: Column name to transform

    Returns:
        DataFrame with log-transformed column
    """
    result = df.copy()

    if column not in result.columns:
        return result

    result[column] = np.log1p(result[column].astype(float))

    return result


def apply_standardization(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Standardize numerical features using z-score normalization.

    Args:
        df: DataFrame with numerical columns
        columns: Columns to standardize

    Returns:
        DataFrame with standardized columns added
    """
    if columns is None:
        columns = ["x_coordinate", "y_coordinate", "distance_crime_to_police_station"]

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        values = result[col].astype(float)
        mean = values.mean()
        std = values.std()

        if std > 0:
            result[f"{col}_standardized"] = (values - mean) / std
        else:
            result[f"{col}_standardized"] = 0.0

    return result


def preprocess_raw_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline for model inference.

    Expects a DataFrame with columns:
    - iucr: IUCR code
    - primary_type: Primary crime type
    - location_description: Location description
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - x_coordinate: X coordinate
    - y_coordinate: Y coordinate
    - distance_crime_to_police_station: Distance to nearest police station

    Args:
        df: Raw crime DataFrame

    Returns:
        DataFrame with model-ready features
    """
    logger.info(f"Preprocessing {len(df)} records...")

    result = df.copy()

    # 1. Frequency encoding
    result = apply_frequency_encoding(result)

    # 2. Cyclic encoding
    result = apply_cyclic_encoding(result)

    # 3. Log transform distance
    result = apply_log_transform(result)

    # 4. Standardization
    result = apply_standardization(result)

    # 5. Extract model features
    available = [f for f in MODEL_FEATURES if f in result.columns]
    result = result[available]

    logger.info(f"Preprocessing complete: {len(result.columns)} features")
    return result


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column.

    Args:
        df: DataFrame with 'date' column

    Returns:
        DataFrame with day_of_week added
    """
    result = df.copy()

    if "date" not in result.columns:
        return result

    result["date"] = pd.to_datetime(result["date"])
    result["day_of_week"] = result["date"].dt.dayofweek

    return result
