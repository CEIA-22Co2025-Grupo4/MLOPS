"""
Inference Preprocessing Module

Functions for preprocessing raw crime data for model inference.
This module provides a unified pipeline that can be used by:
- Drift monitoring DAG
- Prediction API (for raw data endpoints)
- Any other service needing to preprocess crime data

Important: Uses saved preprocessing parameters from ETL training to ensure
consistent transformations between training and inference.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .data_enrichment import enrich_crime_data
from .data_loader import download_police_stations
from .minio import check_file_exists, download_json
from . import config  # ETL config for column definitions

logger = logging.getLogger(__name__)

# MinIO configuration
BUCKET_NAME = os.getenv("DATA_REPO_BUCKET_NAME", "data")
PREFIX_PREPROCESSING = config.PREFIX_PREPROCESSING
FREQUENCY_MAPPINGS_KEY = f"{PREFIX_PREPROCESSING}frequency_mappings.json"
SCALING_PARAMS_KEY = f"{PREFIX_PREPROCESSING}scaling_params.json"

# Column definitions from config (ensures consistency with ETL)
SCALING_COLUMNS = list(config.SCALING_COLUMNS)
FREQUENCY_ENCODING_COLUMNS_INFERENCE = ["iucr", "primary_type", "location_description"]

# Cache for preprocessing parameters (avoid repeated downloads)
_preprocessing_cache: Dict[str, Any] = {}

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


def load_frequency_mappings() -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load frequency mappings from MinIO.

    Returns:
        Dictionary mapping column names to frequency dicts, or None if not found
    """
    cache_key = "frequency_mappings"
    if cache_key in _preprocessing_cache:
        return _preprocessing_cache[cache_key]

    try:
        if not check_file_exists(BUCKET_NAME, FREQUENCY_MAPPINGS_KEY):
            logger.warning(
                f"Frequency mappings not found at {BUCKET_NAME}/{FREQUENCY_MAPPINGS_KEY}. "
                "Run ETL pipeline first to generate them."
            )
            return None

        freq_maps = download_json(BUCKET_NAME, FREQUENCY_MAPPINGS_KEY)
        _preprocessing_cache[cache_key] = freq_maps
        logger.info(f"Loaded frequency mappings for {len(freq_maps)} columns")
        return freq_maps
    except Exception as e:
        logger.error(f"Error loading frequency mappings: {e}")
        return None


def load_scaling_params() -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load scaling parameters (mean/std) from MinIO.

    Returns:
        Dictionary mapping column names to {mean, std}, or None if not found
    """
    cache_key = "scaling_params"
    if cache_key in _preprocessing_cache:
        return _preprocessing_cache[cache_key]

    try:
        if not check_file_exists(BUCKET_NAME, SCALING_PARAMS_KEY):
            logger.warning(
                f"Scaling params not found at {BUCKET_NAME}/{SCALING_PARAMS_KEY}. "
                "Run ETL pipeline first to generate them."
            )
            return None

        scaling_params = download_json(BUCKET_NAME, SCALING_PARAMS_KEY)
        _preprocessing_cache[cache_key] = scaling_params
        logger.info(f"Loaded scaling parameters for {len(scaling_params)} columns")
        return scaling_params
    except Exception as e:
        logger.error(f"Error loading scaling parameters: {e}")
        return None


def clear_preprocessing_cache():
    """Clear the preprocessing parameters cache."""
    global _preprocessing_cache
    _preprocessing_cache = {}
    logger.info("Preprocessing cache cleared")


def apply_frequency_encoding_inference(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    freq_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Apply frequency encoding for inference using saved training mappings.

    Uses frequency mappings from ETL training to ensure consistent encoding
    between training and inference. Falls back to batch-computed frequencies
    if saved mappings are not available.

    Args:
        df: DataFrame with categorical columns
        columns: Columns to encode (default: iucr, primary_type, location_description)
        freq_maps: Pre-loaded frequency mappings (will load from MinIO if None)

    Returns:
        DataFrame with frequency-encoded columns added
    """
    if columns is None:
        columns = FREQUENCY_ENCODING_COLUMNS_INFERENCE

    result = df.copy()

    # Try to load saved frequency mappings if not provided
    if freq_maps is None:
        freq_maps = load_frequency_mappings()

    use_saved = freq_maps is not None

    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found for frequency encoding")
            continue

        if use_saved and col in freq_maps:
            # Use saved frequency mapping from training
            freq = freq_maps[col]
            result[f"{col}_freq"] = result[col].map(freq).fillna(0).astype(float)
            logger.debug(f"Frequency encoded '{col}' using saved training mappings")
        else:
            # Fallback: compute from current batch
            freq = result[col].value_counts(normalize=True)
            result[f"{col}_freq"] = result[col].map(freq).fillna(0).astype(float)
            logger.warning(
                f"Using batch-computed frequencies for '{col}' "
                "(saved mapping not found)"
            )

    return result


def apply_cyclic_encoding_inference(
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
        logger.warning(f"Column '{column}' not found for cyclic encoding")
        return result

    # Convert pandas dayofweek (0=Mon, 6=Sun) to notebook mapping (Sun=1, ..., Sat=7)
    day_num = ((result[column] + 1) % 7) + 1
    result[f"{column}_sin"] = np.sin(2 * np.pi * day_num / 7)
    logger.debug(f"Cyclic encoded '{column}' -> '{column}_sin'")

    return result


def apply_log_transform_inference(
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
        logger.warning(f"Column '{column}' not found for log transform")
        return result

    result[column] = np.log1p(result[column].astype(float))
    logger.debug(f"Log transformed '{column}'")

    return result


def apply_standardization_inference(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Standardize numerical features using saved training parameters.

    Uses mean/std from ETL training to ensure consistent standardization
    between training and inference. Falls back to batch-computed statistics
    if saved parameters are not available.

    Args:
        df: DataFrame with numerical columns
        columns: Columns to standardize
        scaling_params: Pre-loaded scaling params (will load from MinIO if None)

    Returns:
        DataFrame with standardized columns added
    """
    if columns is None:
        columns = SCALING_COLUMNS

    result = df.copy()

    # Try to load saved scaling params if not provided
    if scaling_params is None:
        scaling_params = load_scaling_params()

    use_saved = scaling_params is not None

    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found for standardization")
            continue

        values = result[col].astype(float)

        if use_saved and col in scaling_params:
            # Use saved mean/std from training
            mean = scaling_params[col]["mean"]
            std = scaling_params[col]["std"]
            logger.debug(
                f"Standardizing '{col}' using saved params: mean={mean:.4f}, std={std:.4f}"
            )
        else:
            # Fallback: compute from current batch
            mean = values.mean()
            std = values.std()
            logger.warning(
                f"Using batch-computed stats for '{col}' (saved params not found)"
            )

        if std > 0:
            result[f"{col}_standardized"] = (values - mean) / std
        else:
            result[f"{col}_standardized"] = 0.0
            logger.warning(f"Zero std for '{col}', setting standardized to 0")

        logger.debug(f"Standardized '{col}' -> '{col}_standardized'")

    return result


def preprocess_for_inference(
    df: pd.DataFrame,
    skip_log_transform: bool = False,
    freq_maps: Optional[Dict[str, Dict[str, float]]] = None,
    scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline for model inference.

    Uses saved preprocessing parameters from ETL training for consistent
    transformations. Parameters are loaded from MinIO if not provided.

    Pipeline:
    1. Frequency encoding for categorical features (using saved mappings)
    2. Cyclic encoding for day_of_week
    3. Log transform for distance (optional)
    4. Standardization for numerical features (using saved mean/std)

    Args:
        df: Enriched crime DataFrame (must have all required columns)
        skip_log_transform: If True, skip log transform (use if already applied)
        freq_maps: Pre-loaded frequency mappings (loads from MinIO if None)
        scaling_params: Pre-loaded scaling params (loads from MinIO if None)

    Returns:
        DataFrame with all preprocessing applied
    """
    logger.info(f"Preprocessing {len(df)} records for inference...")

    # Pre-load parameters once (caching will avoid repeated loads)
    if freq_maps is None:
        freq_maps = load_frequency_mappings()
    if scaling_params is None:
        scaling_params = load_scaling_params()

    result = df.copy()

    # 1. Frequency encoding (uses saved training mappings)
    result = apply_frequency_encoding_inference(result, freq_maps=freq_maps)

    # 2. Cyclic encoding (deterministic, no saved params needed)
    result = apply_cyclic_encoding_inference(result)

    # 3. Log transform distance (deterministic)
    if not skip_log_transform:
        result = apply_log_transform_inference(result)

    # 4. Standardization (uses saved training mean/std)
    result = apply_standardization_inference(result, scaling_params=scaling_params)

    logger.info("Preprocessing complete")
    return result


def extract_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the features expected by the model.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with only model features
    """
    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing = [f for f in MODEL_FEATURES if f not in df.columns]

    if missing:
        logger.warning(f"Missing model features: {missing}")

    return df[available].copy()


def preprocess_raw_crimes(
    crimes_df: pd.DataFrame,
    stations_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: from raw crime data to model-ready features.

    This is the main entry point for preprocessing raw data.

    Args:
        crimes_df: Raw crime data from Chicago API
        stations_df: Police stations data (will be downloaded if None)

    Returns:
        Tuple of (model_features_df, enriched_df)
        - model_features_df: Ready for model prediction (7 features)
        - enriched_df: Full enriched data (for saving ground truth, etc.)
    """
    logger.info(f"Processing {len(crimes_df)} raw crime records...")

    # 1. Download police stations if not provided
    if stations_df is None:
        logger.info("Downloading police stations...")
        stations_df = download_police_stations()

    # 2. Enrich data (add nearest station, temporal features)
    logger.info("Enriching crime data...")
    enriched_df = enrich_crime_data(crimes_df, stations_df)
    logger.info(f"Enriched data: {len(enriched_df)} records")

    # 3. Apply preprocessing
    processed_df = preprocess_for_inference(enriched_df)

    # 4. Extract model features
    features_df = extract_model_features(processed_df)

    logger.info(
        f"Ready for inference: {len(features_df)} records, {len(features_df.columns)} features"
    )

    return features_df, enriched_df


def preprocess_single_crime(crime_data: dict) -> dict:
    """
    Preprocess a single crime record for API prediction.

    This is a simplified version for single predictions where
    frequency encoding uses a default value and standardization
    uses approximate values.

    Args:
        crime_data: Dictionary with raw crime fields

    Returns:
        Dictionary with model-ready features

    Note:
        For accurate predictions on single records, consider using
        batch processing or pre-computed encoding/scaling parameters.
    """
    # Convert to DataFrame for consistent processing
    df = pd.DataFrame([crime_data])

    # Apply preprocessing
    processed = preprocess_for_inference(df)

    # Extract features
    features = extract_model_features(processed)

    return features.iloc[0].to_dict()
